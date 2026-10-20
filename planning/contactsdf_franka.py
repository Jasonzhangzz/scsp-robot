"""Independent upstream ContactSDF controller, adapted only to one fingertip.

This module does not construct Lambda, verify trackers, staged objectives,
approach planners, MPCExplicit, or alternative initial guesses.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np

from examples.mpc.franka.scm.vendor.contactsdf.ball_lse_contact_detection import Contact
from examples.mpc.franka.scm.vendor.contactsdf.ball_mpc_lse import MPC_IPOPT
from examples.mpc.franka.scm.vendor.contactsdf.single_ball_cost import Cost


def upstream_parameters(family='cube', fingertip_count=1):
    """Original three-ball hyperparameters; only the state/control size varies."""
    if family not in ('cube', 'foambrick', 'stick'):
        raise ValueError('Unknown ContactSDF parameter family')
    extent, step = {'cube': (.03, .002), 'foambrick': (.02, .002),
                    'stick': (.01, .001)}[family]
    grid = np.arange(-extent, extent, step)
    p = SimpleNamespace(
        contact_mode_='lse', dims_=3, ftps_n_=int(fingertip_count),
        n_qpos_=7+3*fingertip_count, n_qvel_=6+3*fingertip_count,
        n_cmd_=3*fingertip_count, max_ncon_=4, h_=.1,
        gravity_=np.array([0., 0., -9.8, 0., 0., 0.]),
        con_vec_=np.ones(3), gd_points_=np.array([[x, y, 0.] for x in grid for y in grid]),
        contact_lse_smoothness_delta_=1e7, mpc_horizon_=4, ipopt_max_iter_=100,
        mpc_u_lb_=-.01, mpc_u_ub_=.01, p_weighting_=10,
        q_weighting_={'cube': 1, 'foambrick': 5, 'stick': 2}[family],
        terminal_scale_=50 if family == 'stick' else 1000)
    p.mpc_q_lb_ = np.full(p.n_qpos_, -1e7)
    p.mpc_q_ub_ = np.full(p.n_qpos_, 1e7)
    if family != 'cube':
        p.mpc_q_lb_[7:] = np.tile([-1., -1., 0.], fingertip_count)
        p.mpc_q_ub_[7:] = np.ones(3*fingertip_count)
    p.planning_cost_ = Cost(p)
    return p


def mesh_pointcloud(mesh_path, directory):
    """Provide the new task geometry in the upstream six-column pointcloud API.

    Each mesh face supplies a surface point and its outward normal. There is
    no candidate selection or local collision-response model in this adapter.
    """
    import trimesh
    mesh_path = Path(mesh_path)
    digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(digest+'.txt')
    downsampled = path.with_name(path.stem+'_subsampled.txt')
    if not path.exists() or not downsampled.exists():
        mesh = trimesh.load_mesh(mesh_path, process=False)
        data = np.c_[mesh.triangles_center, mesh.face_normals]
        data = data[np.isfinite(data).all(axis=1) & (np.linalg.norm(data[:, 3:], axis=1) > .5)]
        if not len(data):
            raise ValueError('ContactSDF needs surface points with finite normals')
        np.savetxt(path, data, delimiter=',')
        np.savetxt(downsampled, data, delimiter=',')
    return str(path)


class FrankaContactSDF:
    """Detection -> upstream planOnce -> first displacement, without strategies."""

    def __init__(self, scene, geometry_directory):
        self.param = upstream_parameters(scene.contactsdf_theta_source_)
        self.param.pointcloud_path_ = mesh_pointcloud(scene.mesh_path_, geometry_directory)
        self.theta = np.asarray(scene.contactsdf_theta_, float).copy()
        # Coordinate-only interface: the original detector's ground is z=0.
        self.origin = np.r_[scene.init_obj_qpos_[:2], scene.table_height]
        self.target_p = np.asarray(scene.target_p_, float)-self.origin
        self.target_q = np.asarray(scene.target_q_, float)
        self.contact = Contact(self.param)
        self.planner = MPC_IPOPT(self.param)
        self.sol_guess = None

    def plan(self, measured_state):
        state = np.asarray(measured_state, float).reshape(10).copy()
        state[:3] -= self.origin
        state[7:10] -= self.origin
        phi, jac_n, jac_f = self.contact.detectOnce(state)
        result = self.planner.planOnce(self.target_p, self.target_q, state,
            phi, jac_n, jac_f, self.theta, self.sol_guess)
        self.sol_guess = result['sol_guess']
        return result
