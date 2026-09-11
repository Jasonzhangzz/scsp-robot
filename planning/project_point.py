import trimesh
import numpy as np
import time 
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
import open3d as o3d

class ProjectionPoint:
    def __init__(self, mesh_path, scale_factors=[1.0, 1.0, 1.0],
                 collision_hull=False, normal_stability_cos=0.90,
                 normal_stability_k=16):
        self.mesh_path = mesh_path
        self.scale_factors = scale_factors  # Default scaling factors
        # MuJoCo uses a convex collision representation for mesh geoms while
        # retaining the original mesh for rendering.  When requested, build
        # the same (single) convex hull for contact-point sampling and
        # projection so a candidate can never lie on a concavity that the
        # simulator collision model cannot reach.
        self.collision_hull = bool(collision_hull)
        # Contact candidates at very sharp tips (trunk/tail/ear ends) have a
        # rapidly changing normal.  A point contact there is ill-conditioned
        # and often leaves the spherical fingertip hovering.  Filter such
        # vertices using the normal agreement in a small metric neighbourhood.
        self.normal_stability_cos = float(normal_stability_cos)
        self.normal_stability_k = max(4, int(normal_stability_k))
        self.source_mesh = None
        self.scaled_mesh = None
        self.kdtree = None
        self.mesh_o3d = None
        self.face_normals = None
        self.faces = None
        self.vertices = None
        self.vertex_to_face = None  # Maps vertex indices to face indices
        self.vertex_normal_stability = None
        self.stable_vertex_mask = None
        self.stable_vertex_indices = None
        self.stable_kdtree = None
        self._last_projection_idx = None
        self._last_projection_normal = None

        self.load_and_scale_mesh()
        self._build_graph()

    def set_scale_factors(self, scale_x, scale_y, scale_z):
        self.scale_factors = [scale_x, scale_y, scale_z]

    def load_and_scale_mesh(self):
        """Load and scale the mesh, and compute vertex-to-face mapping."""
        # Keep trimesh's normal processing here: STL stores every triangle
        # with duplicate vertices, and retaining those duplicates would break
        # the contact graph and nearest-vertex projection.
        manipuland_mesh = trimesh.load_mesh(self.mesh_path)
        matrix = np.array([
            [self.scale_factors[0], 0, 0, 0],
            [0, self.scale_factors[1], 0, 0],
            [0, 0, self.scale_factors[2], 0],
            [0, 0, 0, 1]
        ])
        self.source_mesh = manipuland_mesh.copy()
        self.source_mesh.apply_transform(matrix)
        if self.collision_hull:
            # ``convex_hull`` returns a watertight triangular mesh with the
            # outward face winding used by MuJoCo's mesh collision path.
            # Keep this separate from the visual/source mesh for diagnostics.
            self.scaled_mesh = self.source_mesh.convex_hull
        else:
            self.scaled_mesh = self.source_mesh.copy()

        self.kdtree = cKDTree(self.scaled_mesh.vertices)
        self.face_normals = np.asarray(self.scaled_mesh.face_normals)
        self.vertex_normals = self.scaled_mesh.vertex_normals
        self.faces = self.scaled_mesh.faces
        self.vertices = self.scaled_mesh.vertices

        # Estimate local normal stability.  Incident-face averaging alone can
        # hide a sharp tip on a densely triangulated mesh, so compare each
        # vertex normal with its nearest geometric neighbours as well.
        outward_normals = np.array(self.scaled_mesh.vertex_normals,
                                   dtype=np.float64, copy=True)
        outward_normals /= np.maximum(np.linalg.norm(outward_normals, axis=1,
                                                       keepdims=True), 1e-9)
        k = min(self.normal_stability_k, len(self.vertices))
        try:
            _, nbrs = cKDTree(self.vertices).query(self.vertices, k=k)
            dots = np.sum(outward_normals[:, None, :] * outward_normals[nbrs], axis=2)
            stability = np.mean(np.clip(dots, -1.0, 1.0), axis=1)
        except Exception:
            stability = np.ones(len(self.vertices), dtype=np.float64)
        self.vertex_normal_stability = stability
        self.stable_vertex_mask = stability >= self.normal_stability_cos
        # Always retain a usable candidate set.  The fallback only matters for
        # unusually coarse meshes whose every vertex is classified as sharp.
        if int(np.count_nonzero(self.stable_vertex_mask)) < max(8, min(32, len(self.vertices))):
            keep = np.argsort(stability)[-max(8, min(32, len(self.vertices))):]
            self.stable_vertex_mask[:] = False
            self.stable_vertex_mask[keep] = True
        self.stable_vertex_indices = np.flatnonzero(self.stable_vertex_mask)
        self.stable_kdtree = cKDTree(self.vertices[self.stable_vertex_indices])

        # Build vertex-to-face mapping: for each vertex, store the faces it belongs to
        self.vertex_to_face = [[] for _ in range(len(self.vertices))]
        for face_idx, face in enumerate(self.faces):
            for vertex_idx in face:
                self.vertex_to_face[vertex_idx].append(face_idx)

    def project_point_to_mesh(self, point, contact_patch=None):
        """Find the closest point on the mesh and return its index and normal."""
        if self.scaled_mesh is None:
            self.load_and_scale_mesh()
            self._build_graph()
        
        point = np.ascontiguousarray(point).reshape(1, -1)  
        # Nearest-vertex projection is discontinuous at Voronoi boundaries;
        # a fingertip moving by a fraction of a millimetre can therefore jump
        # between sharp vertices and flip the contact frame.  Choose among a
        # small nearest-neighbour set with a normal-continuity term.
        # Restrict projection to stable vertices when possible.  This keeps
        # the physical anchor from snapping to an ill-conditioned tip.
        query_tree = self.stable_kdtree if self.stable_kdtree is not None else self.kdtree
        k = min(12, len(self.stable_vertex_indices) if self.stable_vertex_indices is not None else len(self.vertices))
        dists, indices = query_tree.query(point, k=k)
        dists = np.asarray(dists[0]).reshape(-1)
        indices = np.asarray(indices[0]).reshape(-1).astype(int)
        if query_tree is self.stable_kdtree:
            indices = self.stable_vertex_indices[indices]
        source_idx = int(indices[0])
        if self._last_projection_idx is not None and len(indices) > 1:
            prev_n = np.asarray(self._last_projection_normal)
            normals = -np.asarray(self.vertex_normals[indices])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
            # Only a modest geometric penalty is needed to prevent normal
            # chatter; distance remains dominant away from a boundary.
            radius = max(float(dists[-1]), 1e-4)
            score = dists + 0.25 * radius * (1.0 - np.clip(normals @ prev_n, -1., 1.))
            source_idx = int(indices[np.argmin(score)])

        n, t1, t2 = self.get_vertex_frame(source_idx)
        self._last_projection_idx = source_idx
        self._last_projection_normal = np.asarray(n).copy()
        return source_idx, n, t1, t2

    def get_vertex_normal(self, vertex_idx):
        """Compute the normal at a vertex by averaging the normals of adjacent faces."""
        if not self.vertex_to_face:
            raise ValueError("Vertex-to-face mapping not initialized.")
        
        normals = np.asarray(self.vertex_normals[vertex_idx], dtype=np.float64)
        norm = float(np.linalg.norm(normals))
        if norm < 1e-10:
            # A malformed/degenerate imported collision facet must not produce
            # a zero contact frame.  Use the radial hull direction as a
            # deterministic outward fallback, then negate it to preserve the
            # optimizer's inward-normal convention.
            center = np.asarray(self.scaled_mesh.vertices, dtype=np.float64).mean(axis=0)
            normals = np.asarray(self.scaled_mesh.vertices[vertex_idx], dtype=np.float64) - center
            norm = float(np.linalg.norm(normals))
            if norm < 1e-10:
                normals = np.array([0.0, 0.0, 1.0])
                norm = 1.0
        return -normals / norm

    def get_vertex_frame(self, vertex_idx):
        """Construct an orthonormal frame at the vertex using its normal.
        Returns:
            tuple: (normal, tangent1, tangent2) as 3D numpy arrays.
        """
        n = self.get_vertex_normal(vertex_idx)
        n = n / np.linalg.norm(n)  # Ensure unit length

        # Choose an arbitrary direction (avoid near-parallel cases)
        arbitrary_dir = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(arbitrary_dir, n)) > 0.9:
            arbitrary_dir = np.array([0.0, 1.0, 0.0])

        # Compute t1 (project onto tangent plane)
        t1 = arbitrary_dir - np.dot(arbitrary_dir, n) * n
        t1 = t1 / np.linalg.norm(t1)

        # Compute t2 (cross product)
        t2 = np.cross(n, t1)
        t2 = t2 / np.linalg.norm(t2)
        return n, t1, t2

    def _build_graph(self):
        """Build adjacency matrix for the mesh graph."""
        if self.scaled_mesh is None:
            self.load_and_scale_mesh()

        edges = self.scaled_mesh.edges_unique
        if len(edges) == 0:
            raise ValueError("Mesh has no edges.")

        v1 = self.scaled_mesh.vertices[edges[:, 0]]
        v2 = self.scaled_mesh.vertices[edges[:, 1]]
        edge_lengths = np.linalg.norm(v1 - v2, axis=1)
        edge_lengths[edge_lengths < 1e-10] = 1e-5

        n_vertices = len(self.scaled_mesh.vertices)
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.concatenate([edge_lengths, edge_lengths])
        
        self.graph = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
        return self.graph

    def sample_vertices(self, num_samples, strategy='poisson_disk', return_indices=False):
        """
        Sample vertices from the mesh using specified strategy to ensure good spatial distribution.
        
        Parameters:
            num_samples (int): Number of vertices to sample
            strategy (str): Sampling strategy, options: 
                'poisson_disk' - Poisson disk sampling (default)
                'farthest_point' - Farthest point sampling
                'random' - Uniform random sampling
            return_indices (bool): If True, return vertex indices instead of coordinates
        
        Returns:
            numpy.ndarray: Array of sampled vertex coordinates or indices
        """
        if self.scaled_mesh is None:
            self.load_and_scale_mesh()
        
        vertices = self.vertices
        # Sample only geometrically stable vertices.  The optimizer's contact
        # set therefore excludes tail/ear/trunk tips with unreliable normals.
        if self.stable_vertex_indices is not None and len(self.stable_vertex_indices):
            candidate_indices = self.stable_vertex_indices
        else:
            candidate_indices = np.arange(len(vertices), dtype=int)
        candidate_vertices = vertices[candidate_indices]
        n_vertices = len(candidate_vertices)
        
        if num_samples >= n_vertices:
            out = candidate_indices.copy()
            # If a caller requests more points than the stable pool contains,
            # pad deterministically with the least-unstable remaining
            # vertices.  Normal workloads (20--70 samples on the elephant)
            # never enter this branch, while large diagnostic requests still
            # preserve the requested array shape.
            if num_samples > len(out) and len(out) < len(vertices):
                rest = np.flatnonzero(~self.stable_vertex_mask)
                rest = rest[np.argsort(self.vertex_normal_stability[rest])[::-1]]
                out = np.concatenate([out, rest[:num_samples - len(out)]])
            return out if return_indices else vertices[out]
        
        if strategy == 'random':
            # Simple random sampling
            sample_indices = np.random.choice(n_vertices, num_samples, replace=False)
        
        elif strategy == 'farthest_point':
            # Farthest point sampling for good coverage
            sample_indices = np.zeros(num_samples, dtype=int)
            
            # Start with a random point
            sample_indices[0] = np.random.randint(n_vertices)
            distances = np.full(n_vertices, np.inf)
            
            for i in range(1, num_samples):
                # Update distances to the closest sample
                new_distances = np.linalg.norm(candidate_vertices - candidate_vertices[sample_indices[i-1]], axis=1)
                distances = np.minimum(distances, new_distances)
                
                # Select the farthest point
                sample_indices[i] = np.argmax(distances)
        
        elif strategy == 'poisson_disk':
            # Approximate Poisson disk sampling using Bridson's algorithm
            from sklearn.neighbors import NearestNeighbors
            
            # Initial random sample
            sample_indices = [np.random.randint(n_vertices)]
            active_list = [0]
            
            # Estimate radius based on desired sample count and surface area
            if hasattr(self.scaled_mesh, 'area'):
                area = self.scaled_mesh.area
            else:
                # Fallback area estimation
                area = 0
                for face in self.faces:
                    a, b, c = vertices[face]
                    area += 0.5 * np.linalg.norm(np.cross(b-a, c-a))
            
            radius = np.sqrt(area / (num_samples * np.pi))
            
            # Build KDTree for fast nearest neighbor queries
            kdtree = NearestNeighbors(n_neighbors=1).fit(candidate_vertices)
            
            while len(active_list) > 0 and len(sample_indices) < num_samples:
                # Randomly select an active sample
                idx = np.random.choice(active_list)
                center = candidate_vertices[sample_indices[idx]]
                
                # Generate candidates in annulus around center
                found = False
                for _ in range(30):  # Try 30 times before giving up on this sample
                    # Random direction
                    theta = np.random.uniform(0, 2*np.pi)
                    phi = np.random.uniform(0, np.pi)
                    direction = np.array([
                        np.sin(phi)*np.cos(theta),
                        np.sin(phi)*np.sin(theta),
                        np.cos(phi)
                    ])
                    
                    # Random radius between r and 2r
                    r = radius * np.random.uniform(1, 2)
                    candidate = center + direction * r
                    
                    # Find nearest vertex to candidate
                    dist, nearest_idx = kdtree.kneighbors([candidate])
                    dist = dist[0][0]
                    nearest_idx = nearest_idx[0][0]
                    
                    # Check if it's far enough from existing samples
                    if dist >= radius and nearest_idx not in sample_indices:
                        sample_indices.append(nearest_idx)
                        active_list.append(len(sample_indices)-1)
                        found = True
                        break
                
                if not found:
                    active_list.remove(idx)
            
            # If we didn't get enough samples, fill with farthest point sampling
            if len(sample_indices) < num_samples:
                remaining = num_samples - len(sample_indices)
                used = set(int(i) for i in sample_indices)
                # Work in the local candidate-index space; recursive calls
                # would return global mesh indices and corrupt the mapping
                # below when the stable set is a strict subset.
                pool = [i for i in range(n_vertices) if i not in used]
                if pool:
                    extra = np.asarray(pool, dtype=int)
                    if len(extra) > remaining:
                        # Greedy farthest completion gives better coverage
                        # than taking the first vertices in the pool.
                        chosen = []
                        dmin = np.full(len(extra), np.inf)
                        for _ in range(remaining):
                            if not chosen:
                                j = int(np.argmax(dmin))
                            else:
                                last = candidate_vertices[sample_indices[-1]]
                                dmin = np.minimum(dmin,
                                    np.linalg.norm(candidate_vertices[extra] - last, axis=1))
                                j = int(np.argmax(dmin))
                            chosen.append(int(extra[j])); dmin[j] = -np.inf
                        extra = np.asarray(chosen, dtype=int)
                    sample_indices.extend(extra[:remaining].tolist())
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        if return_indices:
            return candidate_indices[np.asarray(sample_indices, dtype=int)]
        return candidate_vertices[np.asarray(sample_indices, dtype=int)]

    def sample_vertices_with_normals(self, num_samples, strategy='farthest_point'):
        sample_indices = self.sample_vertices(num_samples, strategy, return_indices=True)
        points = np.array(self.vertices[sample_indices])
        frames = {
            'points': points,
            'normals': np.zeros((len(sample_indices), 3)),
            'tangent1': np.zeros((len(sample_indices), 3)),
            'tangent2': np.zeros((len(sample_indices), 3))
        }
        
        # Compute frames for each sampled vertex
        for i, idx in enumerate(sample_indices):
            n, t1, t2 = self.get_vertex_frame(idx)
            frames['normals'][i] = n
            frames['tangent1'][i] = t1
            frames['tangent2'][i] = t2

        return frames
    
    def to_open3d_mesh(self):
        mesh_o3d = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.vertices),
            triangles=o3d.utility.Vector3iVector(self.faces)
        )
        # ensure vertex normals exist
        if np.asarray(mesh_o3d.vertex_normals).shape[0] == 0:
            mesh_o3d.compute_vertex_normals()
        return mesh_o3d
    
    def visualize_with_normals(self, sampled_frames=None, normal_scale=None,
                               show_face_normals=False, 
                               sampled_point_color=[1.0, 0.0, .0],
                               normal_color=[1.0, 0.0, 0.0]):
        """
        Visualize mesh + sampled points and their normals.
        - sampled_frames: dict returned by sample_vertices_with_normals.
          If None, will show normals for all vertices (may be many).
        - normal_scale: length of drawn normal vectors (if None auto compute)
        """
        mesh_o3d = self.to_open3d_mesh()

        bbox = mesh_o3d.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        if normal_scale is None:
            normal_scale = diag * 0.02

        verts = np.asarray(mesh_o3d.vertices)
        vert_normals = np.asarray(mesh_o3d.vertex_normals)

        geom = [mesh_o3d]

        # axis = create_world_axis(axis_size=.05)
        # geom.append(axis)

        # If sampled_frames provided, only draw those normals and points
        if sampled_frames is not None:
            pts = sampled_frames['points']
            norms = sampled_frames['normals']
            m = pts.shape[0]
            # line points: each sampled point and sampled point + normal * scale
            line_points = np.vstack([pts, pts + norms * normal_scale])
            lines = [[i, i + m] for i in range(m)]
            colors = [normal_color for _ in range(m)]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector(lines)
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            # geom.append(line_set)

            # sampled points as small point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(np.tile(sampled_point_color, (m, 1)))
            geom.append(pcd)
        # else:
        #     # draw all vertex normals (can be heavy)
        #     n_verts = verts.shape[0]
        #     line_points = np.vstack([verts, verts + vert_normals * normal_scale])
        #     lines = [[i, i + n_verts] for i in range(n_verts)]
        #     colors = [normal_color for _ in range(n_verts)]
        #     line_set = o3d.geometry.LineSet(
        #         points=o3d.utility.Vector3dVector(line_points),
        #         lines=o3d.utility.Vector2iVector(lines)
        #     )
        #     line_set.colors = o3d.utility.Vector3dVector(colors)
        #     geom.append(line_set)

        if show_face_normals:
            faces = np.asarray(self.faces)
            face_centers = verts[faces].mean(axis=1)
            face_normals = np.asarray(self.face_normals)
            m = face_centers.shape[0]
            face_points = np.vstack([face_centers, face_centers + face_normals * normal_scale])
            face_lines = [[i, i + m] for i in range(m)]
            face_colors = [[0.0, 1.0, 0.0] for _ in face_lines]
            face_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(face_points),
                lines=o3d.utility.Vector2iVector(face_lines)
            )
            face_line_set.colors = o3d.utility.Vector3dVector(face_colors)
            geom.append(face_line_set)

        # o3d.visualization.draw_geometries(geom, mesh_show_back_face=True)
        # 用 Visualizer 才能稳定控制 point_size
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        for g in geom:
            vis.add_geometry(g)

        opt = vis.get_render_option()
        opt.point_size = 15.0        # 点显示更大：比如 3/5/8/12 自己调
        opt.mesh_show_back_face = True
        # 如果你也想让法线线更粗（对 LineSet 生效）
        # opt.line_width = 2.0

        vis.run()
        vis.destroy_window()



def create_world_axis(axis_size=0.1):
    """
    创建世界原点坐标轴 (0,0,0)
    X: 红, Y: 绿, Z: 蓝
    """
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size, origin=[-0.030, -0.025, -0.0250]
    )
    return axis      

if __name__ == "__main__":
    geo_calc = ProjectionPoint("/home/lab423/scsp/scsp-robot/envs/assets/objects/piggy_bank.stl")
    sampling_frame = geo_calc.sample_vertices_with_normals(num_samples=150, strategy='farthest_point')
    print(sampling_frame['normals'])
    # visualize mesh + sampled points + their normals + face normals
    geo_calc.visualize_with_normals(sampled_frames=sampling_frame,
                                    normal_scale=0.025,
                                    show_face_normals=False,)
