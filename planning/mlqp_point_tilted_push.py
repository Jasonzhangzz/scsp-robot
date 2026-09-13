import time
import numpy as np
import casadi as cs
import os
import sys
import shutil
import ctypes
import warnings
from scipy.spatial import cKDTree
try:
    from project_point import ProjectionPoint
except:
    from planning.project_point import ProjectionPoint


_ACADOS_TEMPLATE_SYMBOLS = None
_ACADOS_IMPORT_FAILURE = None
_ACADOS_EXPORT_VERSION = "v2"
_ACADOS_CASADI_FALLBACK_WARNED = False
_ACADOS_PRELOADED_SHARED_LIBS = []
_STAGE1_TWIST_REGULARIZATION = 0.2
_ACADOS_STATUS_LABELS = {
    -1: "ACADOS_UNKNOWN",
    0: "ACADOS_SUCCESS",
    1: "ACADOS_NAN_DETECTED",
    2: "ACADOS_MAXITER",
    3: "ACADOS_MINSTEP",
    4: "ACADOS_QP_FAILURE",
    5: "ACADOS_READY",
    6: "ACADOS_UNBOUNDED",
    7: "ACADOS_TIMEOUT",
    8: "ACADOS_QPSCALING_BOUNDS_NOT_SATISFIED",
    9: "ACADOS_INFEASIBLE",
}


def _repo_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shared_lib_ext():
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt":
        return ".dll"
    return ".so"


def _acados_root_candidates():
    from planning.acados_env import acados_root_candidates
    return acados_root_candidates(_repo_root_dir())


def _install_deprecated_sphinx_shim():
    import types

    if "deprecated.sphinx" in sys.modules:
        return

    deprecated_module = types.ModuleType("deprecated")
    sphinx_module = types.ModuleType("deprecated.sphinx")

    def _deprecated(*args, **kwargs):
        def _decorator(obj):
            return obj
        return _decorator

    sphinx_module.deprecated = _deprecated
    deprecated_module.sphinx = sphinx_module
    sys.modules.setdefault("deprecated", deprecated_module)
    sys.modules.setdefault("deprecated.sphinx", sphinx_module)


def _bootstrap_local_acados_python_interface():
    for acados_root in _acados_root_candidates():
        interface_root = os.path.join(acados_root, "interfaces", "acados_template")
        package_init = os.path.join(interface_root, "acados_template", "__init__.py")
        if not os.path.isfile(package_init):
            continue

        os.environ.setdefault("ACADOS_SOURCE_DIR", acados_root)
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

        acados_lib_dir = os.path.join(acados_root, "lib")
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        ld_entries = [entry for entry in ld_library_path.split(":") if entry]
        if acados_lib_dir not in ld_entries:
            ld_entries.append(acados_lib_dir)
            os.environ["LD_LIBRARY_PATH"] = ":".join(ld_entries)
        _preload_acados_shared_libraries(acados_root)

        if interface_root not in sys.path:
            sys.path.insert(0, interface_root)
        return


def _preload_acados_shared_libraries(acados_root):
    if os.name == "nt":
        return

    lib_dir = os.path.join(acados_root, "lib")
    if not os.path.isdir(lib_dir):
        return

    load_mode = getattr(ctypes, "RTLD_GLOBAL", None)
    shared_lib_names = [
        "libblasfeo.so.0",
        "libblasfeo.so",
        "libhpipm.so",
        "libqpOASES_e.so",
        "libdaqp.so",
        "libosqp.so",
        "libacados.so",
    ]

    for lib_name in shared_lib_names:
        lib_path = os.path.join(lib_dir, lib_name)
        if not os.path.isfile(lib_path):
            continue
        try:
            if load_mode is None:
                handle = ctypes.CDLL(lib_path)
            else:
                handle = ctypes.CDLL(lib_path, mode=load_mode)
        except OSError:
            continue
        _ACADOS_PRELOADED_SHARED_LIBS.append(handle)


def _import_acados_template_symbols():
    global _ACADOS_TEMPLATE_SYMBOLS, _ACADOS_IMPORT_FAILURE

    if _ACADOS_TEMPLATE_SYMBOLS is not None:
        return _ACADOS_TEMPLATE_SYMBOLS
    if _ACADOS_IMPORT_FAILURE is not None:
        raise ImportError(
            "Failed to import acados_template earlier. "
            "See the chained exception for the original cause."
        ) from _ACADOS_IMPORT_FAILURE

    try:
        from deprecated.sphinx import deprecated as _unused_deprecated  # noqa: F401
    except Exception:
        _install_deprecated_sphinx_shim()

    _bootstrap_local_acados_python_interface()

    try:
        from acados_template import (
            AcadosModel,
            AcadosOcp,
            AcadosOcpSolver,
            ACADOS_INFTY,
        )
    except Exception as exc:
        _ACADOS_IMPORT_FAILURE = exc
        raise ImportError(
            "Unable to import acados_template. "
            "The local acados source tree was checked, but Python still could not "
            "load the interface. Make sure the acados Python dependencies are "
            "available and that the local acados build is complete."
        ) from exc

    _ACADOS_TEMPLATE_SYMBOLS = (
        AcadosModel,
        AcadosOcp,
        AcadosOcpSolver,
        ACADOS_INFTY,
    )
    return _ACADOS_TEMPLATE_SYMBOLS


def _ensure_acados_renderer_available():
    tera_candidates = []
    tera_path = os.environ.get("TERA_PATH")
    if tera_path:
        tera_candidates.append(os.path.abspath(tera_path))
    tera_on_path = shutil.which("t_renderer")
    if tera_on_path:
        tera_candidates.append(os.path.abspath(tera_on_path))
    for acados_root in _acados_root_candidates():
        tera_candidates.append(os.path.join(acados_root, "bin", "t_renderer"))
        tera_candidates.append(
            os.path.join(
                acados_root,
                "interfaces",
                "acados_template",
                "tera_renderer",
                "target",
                "release",
                "t_renderer",
            )
        )
        tera_candidates.append(
            os.path.join(
                acados_root,
                "interfaces",
                "acados_template",
                "tera_renderer",
                "target",
                "debug",
                "t_renderer",
            )
        )

    for candidate in tera_candidates:
        if os.path.isfile(candidate):
            os.environ["TERA_PATH"] = candidate
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in tera_candidates if candidate)
    raise RuntimeError(
        "acados was selected, but the tera renderer executable is missing.\n"
        "Looked for it in:\n"
        f"{searched}\n"
        "Install the acados tera renderer or point TERA_PATH to an existing "
        "t_renderer binary before using the acados backend."
    )


def _format_acados_status(status_code, sqp_iter=None):
    label = _ACADOS_STATUS_LABELS.get(int(status_code), f"ACADOS_STATUS_{int(status_code)}")
    if sqp_iter is None:
        return label
    return f"{label} (sqp_iter={int(sqp_iter)})"


def _warn_acados_casadi_fallback(reason):
    global _ACADOS_CASADI_FALLBACK_WARNED
    if _ACADOS_CASADI_FALLBACK_WARNED:
        return
    _ACADOS_CASADI_FALLBACK_WARNED = True
    warnings.warn(
        "Falling back from the acados generated solver to the existing CasADi Opti/IPOPT path. "
        f"Reason: {reason}",
        RuntimeWarning,
    )


def _normalize_nlp_solver_name(solver_name):
    if solver_name is None:
        solver_name = os.environ.get("LCC_ROT_SOLVER", "acados")
    solver_name = str(solver_name).strip().lower()
    if solver_name == "snopt":
        solver_name = "acados"
    if solver_name not in {"ipopt", "acados"}:
        raise ValueError(
            f"Unsupported NLP solver '{solver_name}'. Expected 'ipopt' or 'acados'."
        )
    return solver_name


def _normalize_quaternion_wxyz(quat_wxyz):
    quat_norm = cs.sqrt(cs.dot(quat_wxyz, quat_wxyz) + 1e-12)
    return quat_wxyz / quat_norm


def _quat_wxyz_to_z_axis_vector(quat_wxyz):
    quat_wxyz = _normalize_quaternion_wxyz(quat_wxyz)
    w = quat_wxyz[0]
    x = quat_wxyz[1]
    y = quat_wxyz[2]
    z = quat_wxyz[3]

    # R(q) e_z: the object's local z-axis expressed in the reference frame.
    # Matching only this vector means x/y axes remain free around z.
    return cs.vertcat(
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    )


def _quat_wxyz_to_x_axis_vector(quat_wxyz):
    quat_wxyz = _normalize_quaternion_wxyz(quat_wxyz)
    w = quat_wxyz[0]
    x = quat_wxyz[1]
    y = quat_wxyz[2]
    z = quat_wxyz[3]

    return cs.vertcat(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
    )


def _normalize_vector(vec):
    return vec / cs.sqrt(cs.dot(vec, vec) + 1e-12)


def _plane_projected_x_axis_error(curr_quat_wxyz, target_quat_wxyz):
    target_z_axis = _quat_wxyz_to_z_axis_vector(target_quat_wxyz)
    curr_x_axis = _quat_wxyz_to_x_axis_vector(curr_quat_wxyz)
    target_x_axis = _quat_wxyz_to_x_axis_vector(target_quat_wxyz)

    curr_x_axis_tangent = curr_x_axis - cs.dot(curr_x_axis, target_z_axis) * target_z_axis
    target_x_axis_tangent = target_x_axis - cs.dot(target_x_axis, target_z_axis) * target_z_axis
    curr_x_axis_tangent = _normalize_vector(curr_x_axis_tangent)
    target_x_axis_tangent = _normalize_vector(target_x_axis_tangent)

    axis_alignment = cs.dot(curr_x_axis_tangent, target_x_axis_tangent)
    axis_alignment = cs.fmax(cs.fmin(axis_alignment, 1.0), -1.0)
    return 1.0 - axis_alignment


def _z_axis_alignment_error(curr_quat_wxyz, target_quat_wxyz):
    curr_z_axis = _quat_wxyz_to_z_axis_vector(curr_quat_wxyz)
    target_z_axis = _quat_wxyz_to_z_axis_vector(target_quat_wxyz)
    axis_alignment = cs.dot(curr_z_axis, target_z_axis)
    axis_alignment = cs.fmax(cs.fmin(axis_alignment, 1.0), -1.0)
    return 1.0 - axis_alignment


def _quaternion_alignment_error(curr_quat_wxyz, target_quat_wxyz):
    curr_quat_wxyz = _normalize_quaternion_wxyz(curr_quat_wxyz)
    target_quat_wxyz = _normalize_quaternion_wxyz(target_quat_wxyz)
    return 1.0 - cs.dot(curr_quat_wxyz, target_quat_wxyz) ** 2

class LambdaContactControlOptimizer:
    def __init__(self, mesh_path, obj_mass=0.01, arm_friction=0.9,
                 contact_stiffness=12.5, time_step=0.01, max_contacts=10, sample_num=70,
                 pos_coef=1, ori_coef=0.0005, scale_factors=[1.0, 1.0, 1.0],
                 nlp_solver=None, use_vertices=True):
        # 系统参数
        self.m = obj_mass
        self.mu_arm_obj = arm_friction
        self.K_contact = contact_stiffness
        self.h = time_step
        self.max_contacts = max_contacts
        self.nlp_solver = _normalize_nlp_solver_name(nlp_solver)
        self.pp = ProjectionPoint(mesh_path, scale_factors)
        # Keep the flag for API compatibility, but align the sampler with
        # planning/mlqp_point.py and always use mesh-vertex contact frames.
        self.use_vertices = True

        self.sample_num = sample_num
        self.face_centers = np.asarray(
            self.pp.scaled_mesh.vertices[self.pp.scaled_mesh.faces].mean(axis=1),
            dtype=np.float32,
        )
        self.face_normals = np.asarray(self.pp.scaled_mesh.face_normals, dtype=np.float32)
        self.face_center_tree = cKDTree(self.face_centers) if self.face_centers.size > 0 else None
        self.sampling_frame = self.pp.sample_vertices_with_normals(num_samples=self.sample_num)

        # self.pp.visualize_with_normals(sampled_frames=None,
        #                         normal_scale=None,
        #                         show_face_normals=False)
        self.sample_point = self.sampling_frame['points']
        self.normal = self.sampling_frame['normals']
        self.t1 = self.sampling_frame['tangent1']
        self.t2 = self.sampling_frame['tangent2']

        self.J_tilde = np.zeros([4 * self.max_contacts, 6])

        # 构建系统刚度矩阵Q
        self.obj_inertia = np.eye(6)
        self.obj_inertia[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia[3:, 3:] = 0.05 * np.eye(3)
        Q = np.zeros((6,6))
        Q[:6, :6] = self.obj_inertia
        self.Q_inv = np.linalg.inv(Q + 1e-8 * np.eye(Q.shape[0]))

        self.pos_coef = pos_coef
        self.ori_coef = ori_coef
        self.default_lam_upper_bound = 0.2
        self.stage1_default_lam_upper_bound = self.default_lam_upper_bound
        self.stage2_default_lam_upper_bound = self.default_lam_upper_bound
        self.point_idx = np.arange(self.sample_num)
        self.init_utils()
        self._precompile_optimization_function1()
        self._precompile_optimization_function2()

    @staticmethod
    def _build_frame_from_normal(normal):
        normal = np.asarray(normal, dtype=np.float32).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-12:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            n = normal / normal_norm

        tangent_seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(tangent_seed, n))) > 0.9:
            tangent_seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        t1 = tangent_seed - np.dot(tangent_seed, n) * n
        t1_norm = float(np.linalg.norm(t1))
        if t1_norm < 1e-12:
            tangent_seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            t1 = tangent_seed - np.dot(tangent_seed, n) * n
            t1_norm = float(np.linalg.norm(t1))
        t1 = t1 / max(t1_norm, 1e-12)
        t2 = np.cross(n, t1).astype(np.float32)
        t2 = t2 / max(float(np.linalg.norm(t2)), 1e-12)
        return n.astype(np.float32), t1.astype(np.float32), t2.astype(np.float32)

    def _sample_face_centers_with_normals(self, num_samples):
        n_faces = int(self.face_centers.shape[0])
        if n_faces == 0:
            raise ValueError("Mesh has no triangle faces to sample from.")

        if num_samples >= n_faces:
            sample_indices = np.arange(n_faces, dtype=np.int32)
        else:
            sample_indices = np.zeros((num_samples,), dtype=np.int32)
            sample_indices[0] = np.random.randint(n_faces)
            distances = np.full((n_faces,), np.inf, dtype=np.float64)

            for i in range(1, num_samples):
                new_distances = np.linalg.norm(
                    self.face_centers - self.face_centers[sample_indices[i - 1]],
                    axis=1,
                )
                distances = np.minimum(distances, new_distances)
                sample_indices[i] = int(np.argmax(distances))

        points = self.face_centers[sample_indices]
        frames = {
            "points": np.asarray(points, dtype=np.float32),
            "normals": np.zeros((len(sample_indices), 3), dtype=np.float32),
            "tangent1": np.zeros((len(sample_indices), 3), dtype=np.float32),
            "tangent2": np.zeros((len(sample_indices), 3), dtype=np.float32),
            "face_indices": np.asarray(sample_indices, dtype=np.int32),
        }

        for i, face_idx in enumerate(sample_indices):
            n, t1, t2 = self._build_frame_from_normal(-self.face_normals[int(face_idx)])
            frames["normals"][i] = n
            frames["tangent1"][i] = t1
            frames["tangent2"][i] = t2

        return frames

    def _project_point_to_contact_proxy(self, p_arm):
        p_arm = np.asarray(p_arm, dtype=np.float32).reshape(3)
        closest_idx, n, t1, t2 = self.pp.project_point_to_mesh(p_arm)
        p_obj_local = np.asarray(self.pp.scaled_mesh.vertices[closest_idx], dtype=np.float32)
        normal_obj_local = np.asarray(self.pp.scaled_mesh.vertex_normals[closest_idx], dtype=np.float32)
        return int(closest_idx), n, t1, t2, p_obj_local, normal_obj_local

    def update_Jacobian(self, J_tilde=None):
        required_rows = 4 * self.max_contacts
        """更新环境接触雅可比矩阵"""
        if J_tilde is None:
            pass
        else:
            J_tilde = J_tilde[:, :6]
            current_rows = self.J_tilde.shape[0]
            if current_rows < required_rows:
                padding = np.zeros([required_rows - current_rows, 6])
                self.J_tilde = np.concatenate(self.J_tilde, padding)
            else:
                self.J_tilde = J_tilde[:required_rows, :6]

    def _build_solver_bundle(self, use_full_pose_objective, default_lam_upper_bound):
        opti = cs.Opti()
        
        # 定义优化变量和参数
        x_d = opti.parameter(7)
        current_x = opti.parameter(7)
        J_tilde = opti.parameter(4 * self.max_contacts, 6)
        tau_o_np = opti.parameter(6)
        p_arm = opti.parameter(3)    # 接触点位置
        n_arm = opti.parameter(3)
        t1 = opti.parameter(3)
        t2 = opti.parameter(3)
        lam_upper_bound = opti.parameter(1)
        R_contact = cs.horzcat(n_arm, t1, t2)

        regularization_weight = opti.parameter()
        opti.set_value(regularization_weight, 0.01)
        opti.set_value(lam_upper_bound, float(default_lam_upper_bound))

        # 机械臂接触力作为优化变量
        lam_arm = opti.variable(3)  # fn, ft1, ft2
        
        # 初始猜测
        opti.set_initial(lam_arm, [0.01, 0, 0])
        
        # 计算世界坐标系下的接触雅可比
        J_arm_world = self.compute_contact_jacobian(p_arm)
        
        # 构建b向量
        b = tau_o_np + cs.transpose(J_arm_world) @ (R_contact @ lam_arm)
        
        # 构造接触刚度矩阵K
        K = (self.K_contact * self.h) * cs.MX.eye(4 * self.max_contacts)
        
        # 计算接触力
        Q_inv_b = cs.MX(self.Q_inv) @ b
        J_tilde_Q_inv_b = J_tilde @ Q_inv_b
        contact_force = -K @ J_tilde_Q_inv_b
        contact_force = cs.fmax(contact_force, 0)
       
        # 计算预测速度v+
        v_plus = Q_inv_b / self.h + cs.MX(self.Q_inv) @ J_tilde.T @ contact_force / self.h
        
        # 计算预测位姿x+
        x_plus = self.cs_qposInteg_(current_x, v_plus)
        
        position_error = x_plus[:3] - x_d[:3]
        orientation_error = 1 - cs.dot(x_plus[3:7], x_d[3:7]) ** 2
        objective = (
            self.pos_coef * cs.sumsqr(position_error)
            + self.ori_coef * orientation_error
        )
    
        opti.minimize(objective)
        
        # 摩擦锥约束
        mu = self.mu_arm_obj
        opti.subject_to(lam_arm[1] <= mu * lam_arm[0])
        opti.subject_to(lam_arm[1] >= -mu * lam_arm[0])
        opti.subject_to(lam_arm[2] <= mu * lam_arm[0])
        opti.subject_to(lam_arm[2] >= -mu * lam_arm[0])
        opti.subject_to(lam_arm[0] >= 0.001)
        opti.subject_to(lam_arm[0] <= lam_upper_bound)

        p_opts = {"print_time": False, "jit": False}
        s_opts = {
            "max_iter": 200,
            "tol": 1e-6,
            "linear_solver": "mumps",
            "print_level": 0,
        }
        opti.solver("ipopt", p_opts, s_opts)

        return {
            "backend": "casadi_opti",
            "default_lam_upper_bound": float(default_lam_upper_bound),
            "opti": opti,
            "x_d": x_d,
            "current_x": current_x,
            "J_tilde_param": J_tilde,
            "tau_o_param": tau_o_np,
            "p_arm_param": p_arm,
            "n_arm_param": n_arm,
            "t1_param": t1,
            "t2_param": t2,
            "lam_upper_bound_param": lam_upper_bound,
            "lam_arm_var": lam_arm,
            "x_plus_expr": x_plus,
            "v_plus_expr": v_plus,
            "objective_expr": objective,
            "last_lam_solution": np.array([0.01, 0.0, 0.0], dtype=np.float64),
        }

    def _precompile_optimization_function1(self):
        """保留阶段接口，但使用与 mlqp_point.py 一致的单阶段 cost。"""
        if self.nlp_solver == "acados":
            self._solver_bundle_stage1 = self._build_acados_solver_bundle(
                use_full_pose_objective=False,
                default_lam_upper_bound=self.stage1_default_lam_upper_bound,
            )
        else:
            self._solver_bundle_stage1 = self._build_solver_bundle(
                use_full_pose_objective=False,
                default_lam_upper_bound=self.stage1_default_lam_upper_bound,
            )

    def _precompile_optimization_function2(self):
        """保留阶段接口，但使用与 mlqp_point.py 一致的单阶段 cost。"""
        if self.nlp_solver == "acados":
            self._solver_bundle_stage2 = self._build_acados_solver_bundle(
                use_full_pose_objective=True,
                default_lam_upper_bound=self.stage2_default_lam_upper_bound,
            )
        else:
            self._solver_bundle_stage2 = self._build_solver_bundle(
                use_full_pose_objective=True,
                default_lam_upper_bound=self.stage2_default_lam_upper_bound,
            )

    def init_utils(self):
        # -------------------------------
        #    quaternion integration fn
        # -------------------------------
        quat = cs.SX.sym('quat', 4)
        H_q_body = cs.vertcat(cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
                              cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
                              cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]))
        self.cs_qmat_body_fn_ = cs.Function('cs_qmat_body_fn', [quat], [H_q_body.T])

        # -------------------------------
        #    state integration fn
        # -------------------------------
        qvel = cs.SX.sym('qvel', 6)
        qpos = cs.SX.sym('qpos', 7)
        next_obj_pos = qpos[0:3] + self.h * qvel[0:3]
        next_obj_quat = (qpos[3:7] + 0.5 * self.h * self.cs_qmat_body_fn_(qpos[3:7]) @ qvel[3:6])
        next_obj_quat = next_obj_quat / cs.norm_2(next_obj_quat)
        next_qpos = cs.vertcat(next_obj_pos, next_obj_quat)
        self.cs_qposInteg_ = cs.Function('cs_qposInte', [qpos, qvel], [next_qpos])

    @staticmethod
    def compute_contact_jacobian(p):
        """优化后的接触雅可比计算 - MX 版本"""
        J_c = cs.MX.zeros(3, 6)  # 改为 MX 类型
        J_c[:3, :3] = cs.MX.eye(3)
        # 使用 CasADi 构建斜对称矩阵
        J_c[0, 4], J_c[0, 5] = p[2], -p[1]
        J_c[1, 3], J_c[1, 5] = -p[2], p[0]
        J_c[2, 3], J_c[2, 4] = p[1], -p[0]
        return J_c

    @staticmethod
    def compute_contact_jacobian_sx(p):
        """acados 使用 SX 图，保持与 CasADi Opti 路径相同的接触雅可比。"""
        J_c = cs.SX.zeros(3, 6)
        J_c[:3, :3] = cs.SX.eye(3)
        J_c[0, 4], J_c[0, 5] = p[2], -p[1]
        J_c[1, 3], J_c[1, 5] = -p[2], p[0]
        J_c[2, 3], J_c[2, 4] = p[1], -p[0]
        return J_c

    def _pack_acados_parameter_vector(
        self,
        x_d,
        current_x,
        tau_o,
        n_arm,
        t1,
        t2,
        p_arm,
        lam_upper_bound,
    ):
        return np.concatenate(
            [
                np.asarray(x_d, dtype=np.float64).reshape(7),
                np.asarray(current_x, dtype=np.float64).reshape(7),
                np.asarray(self.J_tilde, dtype=np.float64).reshape(-1, order="F"),
                np.asarray(tau_o, dtype=np.float64).reshape(6),
                np.asarray(p_arm, dtype=np.float64).reshape(3),
                np.asarray(n_arm, dtype=np.float64).reshape(3),
                np.asarray(t1, dtype=np.float64).reshape(3),
                np.asarray(t2, dtype=np.float64).reshape(3),
                np.asarray([lam_upper_bound], dtype=np.float64).reshape(1),
            ]
        )

    def _build_acados_solver_bundle(self, use_full_pose_objective, default_lam_upper_bound):
        # 虽然接触预测的主体看起来接近 QP，但 fmax 裁剪和四元数归一化
        # 让终端映射整体变成了非线性、分段光滑 NLP。这里直接把它写成
        # acados_template 的 N=0 参数化 NLP，而不是错误地强行降成纯 QP。
        (
            AcadosModel,
            AcadosOcp,
            AcadosOcpSolver,
            ACADOS_INFTY,
        ) = _import_acados_template_symbols()

        lam_arm = cs.SX.sym("lam_arm", 3)
        param_dim = 2 * 7 + (4 * self.max_contacts) * 6 + 6 + 3 + 3 + 3 + 3 + 1
        p = cs.SX.sym("p", param_dim)

        cursor = 0

        def _take(size):
            nonlocal cursor
            chunk = p[cursor: cursor + size]
            cursor += size
            return chunk

        x_d = _take(7)
        current_x = _take(7)
        j_tilde_flat = _take((4 * self.max_contacts) * 6)
        J_tilde = cs.reshape(j_tilde_flat, 4 * self.max_contacts, 6)
        tau_o = _take(6)
        p_arm = _take(3)
        n_arm = _take(3)
        t1 = _take(3)
        t2 = _take(3)
        lam_upper_bound = _take(1)[0]

        R_contact = cs.horzcat(n_arm, t1, t2)
        J_arm_world = self.compute_contact_jacobian_sx(p_arm)
        b = tau_o + cs.transpose(J_arm_world) @ (R_contact @ lam_arm)

        K = (self.K_contact * self.h) * cs.DM.eye(4 * self.max_contacts)
        Q_inv = cs.DM(self.Q_inv)

        Q_inv_b = Q_inv @ b
        J_tilde_Q_inv_b = J_tilde @ Q_inv_b
        contact_force = -K @ J_tilde_Q_inv_b
        contact_force = cs.fmax(contact_force, 0.0)

        v_plus = Q_inv_b / self.h + Q_inv @ J_tilde.T @ contact_force / self.h
        x_plus = self.cs_qposInteg_(current_x, v_plus)
        position_error = x_plus[:3] - x_d[:3]
        orientation_error = 1 - cs.dot(x_plus[3:7], x_d[3:7]) ** 2
        objective = (
            self.pos_coef * cs.sumsqr(position_error)
            + self.ori_coef * orientation_error
        )
       
        model = AcadosModel()
        stage_name = "stage2" if use_full_pose_objective else "stage1"
        model.name = (
            f"mlqp_point_test_rot_{stage_name}_"
            f"{_ACADOS_EXPORT_VERSION}_c{self.max_contacts}_s{self.sample_num}"
        )
        model.x = lam_arm
        # The generated acados solver is configured as a terminal NLP with N=0,
        # so dynamics are never used in practice. However, the CasADi fallback
        # path still validates the integrator setup, so we provide a trivial
        # discrete identity map here for compatibility.
        model.u = cs.SX.sym("u", 0, 0)
        model.disc_dyn_expr = lam_arm
        model.p = p
        model.cost_expr_ext_cost_e = objective
        model.con_h_expr_e = cs.vertcat(
            lam_arm[1] - self.mu_arm_obj * lam_arm[0],
            -lam_arm[1] - self.mu_arm_obj * lam_arm[0],
            lam_arm[2] - self.mu_arm_obj * lam_arm[0],
            -lam_arm[2] - self.mu_arm_obj * lam_arm[0],
            lam_arm[0] - lam_upper_bound,
        )

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros((param_dim,), dtype=np.float64)
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.constraints.idxbx_e = np.array([0], dtype=np.int64)
        ocp.constraints.lbx_e = np.array([1e-3], dtype=np.float64)
        ocp.constraints.ubx_e = np.array([ACADOS_INFTY], dtype=np.float64)
        ocp.constraints.lh_e = -ACADOS_INFTY * np.ones((5,), dtype=np.float64)
        ocp.constraints.uh_e = np.zeros((5,), dtype=np.float64)

        ocp.solver_options.N_horizon = 0
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.regularize_method = "MIRROR"
        ocp.solver_options.nlp_solver_ext_qp_res = 1
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.qp_solver_iter_max = 400
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.print_level = 0

        code_export_directory = os.path.join("/tmp", f"{model.name}_codegen")
        json_file = os.path.join(code_export_directory, f"{model.name}.json")
        shared_lib_path = os.path.join(
            code_export_directory,
            f"libacados_ocp_solver_{model.name}{_shared_lib_ext()}",
        )
        os.makedirs(code_export_directory, exist_ok=True)
        ocp.code_gen_opts.code_export_directory = code_export_directory

        can_reuse_existing_solver = (
            os.path.isfile(json_file)
            and os.path.isfile(shared_lib_path)
        )
        backend = "acados"
        if can_reuse_existing_solver:
            solver = AcadosOcpSolver(
                ocp,
                json_file=json_file,
                generate=False,
                build=False,
                check_reuse_possible=False,
                verbose=False,
            )
        else:
            try:
                _ensure_acados_renderer_available()
                solver = AcadosOcpSolver(
                    ocp,
                    json_file=json_file,
                    generate=True,
                    build=True,
                    check_reuse_possible=True,
                    verbose=False,
                )
            except Exception as exc:
                _warn_acados_casadi_fallback(str(exc))
                fallback_bundle = self._build_solver_bundle(
                    use_full_pose_objective=use_full_pose_objective,
                    default_lam_upper_bound=default_lam_upper_bound,
                )
                fallback_bundle["backend"] = "casadi_opti_fallback_from_acados"
                return fallback_bundle

        eval_fun = cs.Function(
            f"{model.name}_eval",
            [lam_arm, p],
            [x_plus, v_plus, objective],
        )

        return {
            "backend": backend,
            "default_lam_upper_bound": float(default_lam_upper_bound),
            "solver": solver,
            "eval_fun": eval_fun,
            "last_lam_solution": np.array([0.01, 0.0, 0.0], dtype=np.float64),
        }
    
    def optimize_control_input(
        self,
        x_d,
        current_x,
        tau_o,
        p_arm=None,
        use_full_pose_objective=False,
        r_obj_to_world=None,
        lam_upper_bound=None,
    ):
        """优化控制输入，并在提供物体姿态时输出世界系接触力矩。"""
        if p_arm is None:
            p_arm = np.array([-1, 0, 0])

        closest_idx, n, t1, t2, p_obj_local, normal_obj_local = self._project_point_to_contact_proxy(p_arm)

        bundle = self._solver_bundle_stage2 if use_full_pose_objective else self._solver_bundle_stage1
        resolved_lam_upper_bound = float(bundle.get("default_lam_upper_bound", 1.0)) if lam_upper_bound is None else float(lam_upper_bound)

        start_time = time.time()
        lam_arm, x_plus_opt, v_plus_opt, objective_value, solver_status = self._solve_once(
            x_d=x_d,
            current_x=current_x,
            tau_o=tau_o,
            n_arm=n,
            t1=t1,
            t2=t2,
            p_arm=p_obj_local,
            use_full_pose_objective=use_full_pose_objective,
            lam_upper_bound=resolved_lam_upper_bound,
        )

        contact_basis_local = np.column_stack([n, t1, t2]).astype(np.float32)
        contact_force_local = (contact_basis_local @ lam_arm).astype(np.float32)
        contact_torque_local = np.cross(
            np.asarray(p_obj_local, dtype=np.float32).reshape(3),
            contact_force_local,
        ).astype(np.float32)
        contact_torque_world = None
        if r_obj_to_world is not None:
            r_obj_to_world = np.asarray(r_obj_to_world, dtype=np.float32).reshape(3, 3)
            contact_torque_world = (r_obj_to_world @ contact_torque_local).astype(np.float32)

        info = {
            "solve_time": time.time() - start_time,
            "control_input": lam_arm,
            "resulting_pose": x_plus_opt,
            "resulting_velocity": v_plus_opt,
            "solver_status": solver_status,
            "contact_point_local": np.asarray(p_obj_local, dtype=np.float32).reshape(3),
            "contact_basis_local": contact_basis_local,
            "contact_force_local": contact_force_local,
            "contact_torque_local": contact_torque_local,
            "use_full_pose_objective": bool(use_full_pose_objective),
        }
        if contact_torque_world is not None:
            info["contact_torque_world"] = contact_torque_world
        info["lam_upper_bound"] = float(resolved_lam_upper_bound)
        info["nlp_solver"] = self.nlp_solver
        info["solver_backend"] = str(
            bundle.get("backend", "casadi_opti")
        )
        
        return p_obj_local, -normal_obj_local, x_plus_opt, objective_value, info, contact_torque_world

    def optimize_stabilizing_torque(
        self,
        x_d,
        current_x,
        tau_o,
        p_arm=None,
        r_obj_to_world=None,
        lam_upper_bound=0.1,
        torque_norm_upper_bound=0.1,
    ):
        """单独求解抗翻倒接触力矩，使用更小的力上界。"""
        p_obj_local, normal_obj_local, x_plus_opt, objective_value, info, contact_torque_world = self.optimize_control_input(
            x_d=x_d,
            current_x=current_x,
            tau_o=tau_o,
            p_arm=p_arm,
            use_full_pose_objective=False,
            r_obj_to_world=r_obj_to_world,
            lam_upper_bound=lam_upper_bound,
        )

        if contact_torque_world is not None and torque_norm_upper_bound is not None:
            torque_norm_upper_bound = float(max(torque_norm_upper_bound, 0.0))
            torque_norm = float(np.linalg.norm(contact_torque_world))
            if torque_norm > max(torque_norm_upper_bound, 1e-8):
                torque_scale = torque_norm_upper_bound / torque_norm
                info["control_input"] = (
                    torque_scale * np.asarray(info["control_input"], dtype=np.float32).reshape(3)
                ).astype(np.float32)
                info["contact_force_local"] = (
                    torque_scale * np.asarray(info["contact_force_local"], dtype=np.float32).reshape(3)
                ).astype(np.float32)
                info["contact_torque_local"] = (
                    torque_scale * np.asarray(info["contact_torque_local"], dtype=np.float32).reshape(3)
                ).astype(np.float32)
                contact_torque_world = (
                    np.asarray(contact_torque_world, dtype=np.float32)
                    * torque_scale
                ).astype(np.float32)
        if contact_torque_world is not None:
            info["contact_torque_world"] = np.asarray(contact_torque_world, dtype=np.float32).reshape(3)
        info["torque_norm_upper_bound"] = None if torque_norm_upper_bound is None else float(torque_norm_upper_bound)

        return p_obj_local, normal_obj_local, x_plus_opt, objective_value, info, contact_torque_world

    def choose_contact_points(self, x_d, current_x, tau_o, visible_face_idx, use_full_pose_objective=False):
        if not len(visible_face_idx):
            # 确保返回不是None
            return self.sample_point[0], self.normal[0], 1, 1, 1.0
        
        error_list = np.zeros(visible_face_idx.shape[0], dtype=np.float64)
        # start_time = time.time()    
        for i, idx in enumerate(visible_face_idx):
            # start_t = time.time()
            _, _, _, objective_value, _ = self._solve_once(
                x_d=x_d,
                current_x=current_x,
                tau_o=tau_o,
                n_arm=self.normal[idx],
                t1=self.t1[idx],
                t2=self.t2[idx],
                p_arm=self.sample_point[idx],
                use_full_pose_objective=use_full_pose_objective,
            )

            error_list[i] = objective_value
            # print(f"Contact point {i+1}/{self.sample_num} evaluation time: {time.time() - start_t}")
        # print("Contact point selection time:", time.time() - start_time)

        min_error = float(np.min(error_list))
        max_error = float(np.max(error_list))
        min_idx = visible_face_idx[int(np.argmin(error_list))]

        return self.sample_point[min_idx], self.normal[min_idx], min_error, max_error, 1.0

    def _solve_once(
        self,
        x_d,
        current_x,
        tau_o,
        n_arm,
        t1,
        t2,
        p_arm,
        use_full_pose_objective=False,
        lam_upper_bound=None,
    ):
        bundle = self._solver_bundle_stage2 if use_full_pose_objective else self._solver_bundle_stage1
        if lam_upper_bound is None:
            lam_upper_bound = float(bundle.get("default_lam_upper_bound", 1.0))
        if bundle.get("backend") in {"acados", "acados_casadi"}:
            lam_upper_bound = float(max(lam_upper_bound, 1e-3))
            param_vector = self._pack_acados_parameter_vector(
                x_d=x_d,
                current_x=current_x,
                tau_o=tau_o,
                n_arm=n_arm,
                t1=t1,
                t2=t2,
                p_arm=p_arm,
                lam_upper_bound=lam_upper_bound,
            )

            solver = bundle["solver"]
            lam_initial = np.asarray(bundle["last_lam_solution"], dtype=np.float64).reshape(3).copy()
            lam_initial[0] = np.clip(lam_initial[0], 1e-3, lam_upper_bound)
            tangential_bound = self.mu_arm_obj * lam_initial[0]
            lam_initial[1] = np.clip(lam_initial[1], -tangential_bound, tangential_bound)
            lam_initial[2] = np.clip(lam_initial[2], -tangential_bound, tangential_bound)

            status = None
            sqp_iter = None
            try:
                solver.set(0, "p", param_vector)
                solver.set(0, "x", lam_initial)
                status = solver.solve()
                lam_arm = np.asarray(solver.get(0, "x"), dtype=np.float32).reshape(3)
                try:
                    sqp_iter = int(solver.get_stats("sqp_iter"))
                except Exception:
                    sqp_iter = None
            except Exception:
                lam_arm = lam_initial.astype(np.float32)
                solver_status = "ACADOS_EXCEPTION"
            else:
                solver_status = _format_acados_status(status, sqp_iter=sqp_iter)

            x_plus_val, v_plus_val, objective_val = bundle["eval_fun"](lam_arm.astype(np.float64), param_vector)
            x_plus_opt = np.asarray(x_plus_val, dtype=np.float32).reshape(7)
            v_plus_opt = np.asarray(v_plus_val, dtype=np.float32).reshape(6)
            objective_value = float(np.asarray(objective_val, dtype=np.float64).reshape(()))

            bundle["last_lam_solution"] = lam_arm.astype(np.float64)
            return lam_arm, x_plus_opt, v_plus_opt, objective_value, solver_status

        opti = bundle["opti"]
        opti.set_value(bundle["x_d"], np.asarray(x_d, dtype=np.float64).reshape(7))
        opti.set_value(bundle["current_x"], np.asarray(current_x, dtype=np.float64).reshape(7))
        opti.set_value(bundle["J_tilde_param"], np.asarray(self.J_tilde, dtype=np.float64))
        opti.set_value(bundle["tau_o_param"], np.asarray(tau_o, dtype=np.float64).reshape(6))
        opti.set_value(bundle["n_arm_param"], np.asarray(n_arm, dtype=np.float64).reshape(3))
        opti.set_value(bundle["t1_param"], np.asarray(t1, dtype=np.float64).reshape(3))
        opti.set_value(bundle["t2_param"], np.asarray(t2, dtype=np.float64).reshape(3))
        opti.set_value(bundle["p_arm_param"], np.asarray(p_arm, dtype=np.float64).reshape(3))
        lam_upper_bound = float(max(lam_upper_bound, 1e-3))
        opti.set_value(bundle["lam_upper_bound_param"], np.asarray([lam_upper_bound], dtype=np.float64))
        lam_initial = np.asarray(bundle["last_lam_solution"], dtype=np.float64).reshape(3).copy()
        lam_initial[0] = np.clip(lam_initial[0], 1e-3, lam_upper_bound)
        tangential_bound = self.mu_arm_obj * lam_initial[0]
        lam_initial[1] = np.clip(lam_initial[1], -tangential_bound, tangential_bound)
        lam_initial[2] = np.clip(lam_initial[2], -tangential_bound, tangential_bound)
        opti.set_initial(bundle["lam_arm_var"], lam_initial)

        try:
            sol = opti.solve()
            lam_arm = np.asarray(sol.value(bundle["lam_arm_var"]), dtype=np.float32).reshape(3)
            x_plus_opt = np.asarray(sol.value(bundle["x_plus_expr"]), dtype=np.float32).reshape(7)
            v_plus_opt = np.asarray(sol.value(bundle["v_plus_expr"]), dtype=np.float32).reshape(6)
            objective_value = float(sol.value(bundle["objective_expr"]))
            solver_status = str(opti.stats().get("return_status", "Solve_Succeeded"))
        except RuntimeError:
            lam_arm = np.asarray(opti.debug.value(bundle["lam_arm_var"]), dtype=np.float32).reshape(3)
            x_plus_opt = np.asarray(opti.debug.value(bundle["x_plus_expr"]), dtype=np.float32).reshape(7)
            v_plus_opt = np.asarray(opti.debug.value(bundle["v_plus_expr"]), dtype=np.float32).reshape(6)
            objective_value = float(opti.debug.value(bundle["objective_expr"]))
            solver_status = str(opti.stats().get("return_status", "Solve_Failed"))

        bundle["last_lam_solution"] = lam_arm.astype(np.float64)
        return lam_arm, x_plus_opt, v_plus_opt, objective_value, solver_status
    
    def get_availble_point_idx(self, pos, R, target_pos, threshold=0.025):
        centers_world = (R @ self.sample_point.T).T + pos
        # height_point_indices = np.where(z_coords > theshold)[0]
        # common_mask = (centers_world[:, 2] > threshold) & (centers_world[:, 2] < threshold+0.02)
        common_mask = (centers_world[:, 2] > threshold)

        direction = target_pos - pos
        dis = np.linalg.norm(direction[:2])
        
        # direction/=np.linalg.norm(direction)
        # face_normals_world = (R @ self.normal.T).T
        # face_dot_products = face_normals_world @ direction
        # common_mask = common_mask & (face_dot_products > 0.)

        # if dis > 0.05:
        #     direction/=np.linalg.norm(direction)
        #     face_normals_world = (R @ self.normal.T).T
        #     face_dot_products = face_normals_world @ direction
        #     common_mask = common_mask & (face_dot_products > 0.5)

        return np.where(common_mask)[0]
    
# 使用示例
if __name__ == "__main__":
    # 创建优化器实例 (10cm x 10cm x 10cm 的方块)
    optimizer = LambdaContactControlOptimizer(
        box_size=(0.1, 0.1, 0.1),
        obj_mass=0.01,
        arm_stiffness=200,
        ground_friction=0.9,
        arm_friction=0.9,
        contact_stiffness=10,
        time_step=0.05,
        max_contacts=5
    )
    
    # 更新接触点 (底面和机械臂接触点)
    new_contact_points = [
        {'type': 'ground', 'position': np.array([0.05, 0.05, -0.05]), 'face': 'bottom'},
        {'type': 'ground', 'position': np.array([-0.05, 0.05, -0.05]), 'face': 'bottom'},
        {'type': 'ground', 'position': np.array([0.05, -0.05, -0.05]), 'face': 'bottom'},
        {'type': 'ground', 'position': np.array([-0.05, -0.05, -0.05]), 'face': 'bottom'},
    ]
    optimizer.update_contact_points(new_contact_points)
    
    # 设置目标位姿和当前位姿
    target_pose = np.array([0.1, -0., 0.0, 0.0, 0.0, 0.5])  # [x,y,z, rx,ry,rz]
    current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    tau_o = np.array([0.0, 0.0, -0.01 * 9.81, 0.0, 0.0, 0.0])  # 重力
    p_arm = np.array([0.0, -0.05, -0.0])  # 左侧
    n_arm = np.array([-0, 1, 0])  # 法线方向指向右侧

    lam_arm, x_plus_opt, info = optimizer.optimize_control_input(
        target_pose, current_pose, tau_o, n_arm=n_arm, p_arm=p_arm
    )
    
    # 转换结果为NumPy数组
    lam_arm_np = np.array(cs.evalf(lam_arm)).flatten()
    x_plus_opt_np = np.array(cs.evalf(x_plus_opt)).flatten()
    
    # 打印结果
    print("\n优化结果:")
    print(f"求解时间: {info['solve_time']:.6f}s")
    # print(f"位置误差: {info['position_error']:.6f}")
    # print(f"姿态误差: {info['orientation_error']:.6f}")
    print(f"最优控制输入: {lam_arm_np}")
    print(f"预测位姿: {x_plus_opt_np}")
    print(f"目标位姿: {target_pose}")
