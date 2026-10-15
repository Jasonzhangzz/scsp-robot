"""Isolated CSO backends: A4 native MuJoCo QP and A5 soft-SCM/CMA-ES.

The native QP adapter reuses test_scm_mujoco's PointContactBaseline inner
force solve. An object-only MuJoCo scene is refreshed from each Isaac
observation; hypothetical candidates never modify the execution simulator.
"""
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import xml.etree.ElementTree as ET

import casadi as cs
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from planning.point_contact_baselines import PointContactBaseline


def _numbers(values):
    return ' '.join('%.17g' % float(value) for value in np.asarray(values).reshape(-1))


def qp_scene(param, optimizer):
    """Use the same actor-frame convex parts, inertia, support and friction."""
    root = ET.Element('mujoco', model='franka_a4_object_prediction')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep=str(optimizer.h), gravity=_numbers(param.gravity_[:3]),
                  cone='pyramidal', solver='Newton', iterations='100', tolerance='1e-10')
    ET.SubElement(root, 'size', njmax='4096', nconmax='512')
    assets = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    mu = str(float(param.mu_table_object_))+' 0.005 0.0001'
    ET.SubElement(world, 'geom', name='table', type='box', size='1 1 0.175',
                  pos=_numbers([1.2, 0., float(param.table_height)-.175]),
                  friction=mu, condim='6')
    body = ET.SubElement(world, 'body', name='obj', pos='0.6 0 0.5')
    ET.SubElement(body, 'freejoint', name='object_free')
    ET.SubElement(body, 'inertial', pos='0 0 0', mass=str(float(param.sim_obj_mass_)),
                  diaginertia=_numbers(param.sim_obj_inertia_diag_))
    paths = getattr(param, 'collision_mesh_paths_', []) or [param.mesh_path_]
    for index, path in enumerate(paths):
        mesh = 'object_part_%d' % index
        ET.SubElement(assets, 'mesh', name=mesh, file=str(Path(path).resolve()))
        ET.SubElement(body, 'geom', name=mesh, type='mesh', mesh=mesh,
                      friction=mu, condim='6')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


class MujocoQPCSO(PointContactBaseline):
    def __init__(self, param, maxiter=12):
        opt = param.lambda_optimizer
        model = qp_scene(param, opt)
        env = SimpleNamespace(model_=model, data_=mujoco.MjData(model),
            param_=SimpleNamespace(lambda_optimizer=opt, lambda_source_param_=param),
            force_args_=SimpleNamespace(force_apply_duration=opt.h))
        self.total_evaluations = self.total_solves = self.total_nonconverged = 0
        self.observations = 0
        super().__init__(env, 'qp', maxiter=maxiter)

    def prepare_observation(self, observation):
        # Free-joint translation velocity is world-frame; angular velocity is
        # body-frame. No predicted pose or force is written back into Isaac.
        pose = np.asarray(observation['policy_q'], dtype=float)[:7]
        data = self.env.data_
        data.qpos[:] = pose
        data.qvel[:3] = observation.get('object_linear_velocity', np.zeros(3))
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]])
        data.qvel[3:6] = rotation.inv().apply(
            np.asarray(observation.get('object_angular_velocity', np.zeros(3))))
        data.time = float(observation.get('sim_time', 0.))
        data.qacc_warmstart[:] = 0.
        mujoco.mj_forward(self.env.model_, data)
        super().prepare()
        self.observations += 1

    def predict(self, command, point, frame):
        self.total_evaluations += 1
        return super().predict(command, point, frame)

    def solve_candidate(self, **kwargs):
        self.total_solves += 1
        previous = self.inner_nonconverged
        result = super().solve_candidate(**kwargs)
        self.total_nonconverged += self.inner_nonconverged-previous
        return result

    def diagnostics(self):
        return dict(model=self.model_name, solver='slsqp',
            maxiter=self.maxiter, observations=self.observations,
            solves=self.total_solves, evaluations=self.total_evaluations,
            nonconverged_local_solves=self.total_nonconverged,
            fallback_to_scm=0)


class CMASoftSCMCSO:
    def __init__(self, optimizer):
        if optimizer.solver != 'cma-es' or optimizer.population_search is None:
            raise ValueError('A5 requires the native CMA-ES force solver')
        self.opt = optimizer
        self.solves = self.failures = self.evaluations = 0
        self.directory = tempfile.TemporaryDirectory(prefix='franka_a5_scm_')
        # Compile the unchanged objective graph. CMA-ES still runs pycma's
        # ask/tell loop and uses the same feasible-force parameterization.
        original = optimizer.force_objective_fn
        self.original_objective = original
        inputs = [cs.MX.sym(original.name_in(i), *original.size_in(i))
                  for i in range(original.n_in())]
        optimizer.force_objective_fn = cs.Function('a5_scm_force_objective', inputs,
            original.call(inputs), original.name_in(), original.name_out(),
            dict(jit=True, jit_options=dict(flags='-O3', directory=self.directory.name+'/')))
        self.original_solve = optimizer._solve_optimization

    def prepare_observation(self, observation):
        pass

    def solve_candidate(self, **kwargs):
        self.solves += 1
        result = self.original_solve(**kwargs)
        self.evaluations += self.opt.population_search.evaluations
        if result is None:
            self.failures += 1
        return result

    def diagnostics(self):
        population = self.opt.population_search
        return dict(model='soft-scm', solver='cma-es', population_size=population.size,
            generations=population.iterations, solves=self.solves,
            evaluations=self.evaluations, failures=self.failures, fallback_to_scm=0)

    def close(self):
        # Drop mapped C-function references before cleaning compiler artifacts.
        if getattr(self, 'original_objective', None) is not None:
            self.opt.population_search.maps.clear()
            self.opt.force_objective_fn = self.original_objective
            self.original_objective = None
            self.directory.cleanup()

    def __del__(self):
        self.close()


def install_cso_ablation(param, args):
    if param.ablation_mode == 'a4':
        backend = MujocoQPCSO(param, maxiter=args.a4_qp_maxiter)
        param.lambda_optimizer.solver = 'mujoco-qp-slsqp'
    elif param.ablation_mode == 'a5':
        backend = CMASoftSCMCSO(param.lambda_optimizer)
    else:
        raise ValueError('CSO backend ablation must be a4 or a5')
    def solve(instance, **kwargs):
        return backend.solve_candidate(**kwargs)
    param.lambda_optimizer._solve_optimization = MethodType(solve, param.lambda_optimizer)
    param.cso_ablation_ = backend
    return backend
