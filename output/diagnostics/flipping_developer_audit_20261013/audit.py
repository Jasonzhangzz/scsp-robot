"""Check the live flipping path against a freshly fetched developer source."""
import argparse
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path('/home/zz/scsp-robot')
REFERENCE = Path('/tmp/ik2-flipping-audit-20261013/scsp-robot')
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OBJECTS = ['foam_brick', 'stanford_bunny2', 'elephant', 'piggy_bank',
           'teapot', 'mug', 'rubber_duck']
MANIFESTS = ROOT / 'output/benchmarks/a3_rotation_20261013'


def load_reference(relative):
    module = types.ModuleType('audit_reference_' + Path(relative).stem)
    module.__file__ = str(REFERENCE / relative)
    exec(compile((REFERENCE / relative).read_text(), module.__file__, 'exec'), module.__dict__)
    return module


def tree_symbols(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def reference_functions(relative, current):
    scope = dict(current.__dict__)
    nodes = [node for node in ast.parse((REFERENCE / relative).read_text()).body
             if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REFERENCE / relative), 'exec'), scope)
    return scope


def equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    result = dict(reference_commit=subprocess.check_output(
        ['git', 'rev-parse', 'origin/developer'], cwd=ROOT, text=True).strip(),
        objects=OBJECTS, seeds=list(range(5)), parameter_cases=[], mismatches=[],
        ast_differences={}, model_checks=[])
    paths = ['examples/mpc/franka/ik2/test_mpc_isaac.py',
        'examples/mpc/franka/ik2/test_mpc_isaac_ablation.py',
        'examples/mpc/franka/ik2/params.py', 'examples/mpc/franka/ik2/params_ablation.py',
        'examples/mpc/franka/ik2/run.py', 'examples/mpc/franka/ik2/run_isaac_ablation.py',
        'examples/mpc/franka/ik2/contact_frames.py', 'examples/mpc/franka/ik2/isaac_bus.py',
        'examples/mpc/franka/ik2/physx_contact.py', 'examples/mpc/franka/ik2/test_mppi_isaac.py',
        'examples/mpc/franka/ik2/contact_policy_ablation.py', 'planning/mlqp_point.py',
        'planning/object_pose_cost.py', 'planning/franka_contact_motion.py',
        'planning/contact_transition.py', 'planning/mpc_costs.py',
        'models/cartesian_rigid_contact.py', 'models/explicit_model.py',
        'planning/mpc_explicit.py', 'planning/mpc_explicit_ablation.py',
        'models/soft_scm_contact.py', 'examples/mpc/fingertips/test/test_0902.py',
        'utils/metrics.py']
    result['file_checks'] = {}
    for name in paths:
        a, b = ROOT / name, REFERENCE / name
        current, remote = tree_symbols(a), tree_symbols(b)
        changed = [k for k in sorted(current.keys() | remote.keys()) if current.get(k) != remote.get(k)]
        result['file_checks'][name] = dict(byte_identical=a.read_bytes() == b.read_bytes(),
            current_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),
            reference_sha256=hashlib.sha256(b.read_bytes()).hexdigest(), changed_symbols=changed)
        if changed:
            result['ast_differences'][name] = changed
    model_cases = []
    for entry, mode in [('test_mpc_isaac', 'full'), ('test_mpc_isaac_ablation', 'full'),
                        ('test_mpc_isaac_ablation', 'a3')]:
        relative = 'examples/mpc/franka/ik2/' + entry + '.py'
        current = importlib.import_module('examples.mpc.franka.ik2.' + entry)
        old_functions = reference_functions(relative, current)
        param_name = 'params_ablation' if entry.endswith('_ablation') else 'params'
        new_params = importlib.import_module('examples.mpc.franka.ik2.' + param_name)
        old_params = load_reference('examples/mpc/franka/ik2/' + param_name + '.py')
        for obj in OBJECTS:
            for seed in range(5):
                for profile, flags in [
                    ('default', []), ('fixed_tilt_axis', ['--no_random_init_tilt']),
                    ('custom_tilt', ['--init_tilt_min_deg', '35', '--init_tilt_deg', '75']),
                    ('raw_mesh', ['--collision-geometry', 'hull'])]:
                    common = ['--obj', obj, *flags]
                    if entry.endswith('_ablation'):
                        common += ['--ablation-mode', mode]
                    if obj in ('mug', 'rubber_duck') and profile != 'raw_mesh':
                        common += ['--collision-manifest', str(MANIFESTS / (obj + '_collision_manifest.json'))]
                    a = current._add_rollout_policy_args(argparse.ArgumentParser()).parse_args([*common, '--flipping'])
                    b = old_functions['_add_rollout_policy_args'](argparse.ArgumentParser()).parse_args(common)
                    if profile == 'raw_mesh':
                        a.collision_hull = b.collision_hull = False
                    case = dict(entry=entry, mode=mode, object=obj, seed=seed, profile=profile)
                    if not equal(vars(a), vars(b)):
                        result['mismatches'].append(dict(case, category='parsed_args'))
                    pa = new_params.ExplicitMPCParams(a, rand_seed=seed, build_optimizer=False)
                    rng_a = np.random.get_state()
                    pb = old_params.ExplicitMPCParams(b, rand_seed=seed, build_optimizer=False)
                    rng_b = np.random.get_state()
                    keys = sorted(set(vars(pa)) | set(vars(pb)))
                    changes = [key for key in keys if not equal(vars(pa).get(key), vars(pb).get(key))]
                    if changes or not equal(rng_a, rng_b):
                        result['mismatches'].append(dict(case, category='generated_parameters',
                                                        changed_keys=changes, rng_equal=equal(rng_a, rng_b)))
                    # Execute the same physics/adaptation configuration without constructing a solver.
                    for p, args, scope in [(pa, a, vars(current)), (pb, b, old_functions)]:
                        scope['_apply_dywa_physics_to_param'](p)
                        scope['adapt_param_for_cartesian_solver'](p, args, build_optimizer=False)
                        scope['_configure_rollout_param'](p, args)
                    changes = [key for key in sorted(set(vars(pa)) | set(vars(pb)))
                               if not equal(vars(pa).get(key), vars(pb).get(key))]
                    if changes:
                        result['mismatches'].append(dict(case, category='configured_parameters', changed_keys=changes))
                    case.update(equal=not changes, parameter_count=len(vars(pa)))
                    result['parameter_cases'].append(case)
                    if entry == 'test_mpc_isaac' and seed == 0 and profile == 'default':
                        model_cases.append((obj, pa, pb))
            print('Checked %s/%s %s' % (entry, mode, obj), flush=True)
    live_mpc = importlib.import_module('planning.mpc_explicit')
    reference_mpc = load_reference('planning/mpc_explicit.py')
    # Construct the true symbolic model/costs; only skip native solver initialization.
    for module in (live_mpc, reference_mpc):
        module.MPCExplicit.init_MPC = lambda self: None
    rng = np.random.default_rng(2)
    for obj, pa, pb in model_cases:
        a, b = live_mpc.MPCExplicit(copy.copy(pa)), reference_mpc.MPCExplicit(copy.copy(pb))
        guards = dict(transports_contacts=bool(getattr(a.model, 'transports_contacts', False)),
                      residual_fn=hasattr(a, 'residual_fn'),
                      terminal_residual_fn=hasattr(a, 'terminal_residual_fn'),
                      custom_solver_profile=bool(getattr(a.param_, 'mpc_acados_profile_', None)))
        checks = {}
        for name in ('stage_prediction_fn', 'path_cost_fn', 'final_cost_fn'):
            fa, fb = getattr(a, name), getattr(b, name)
            maximum = 0.
            for index in range(20):
                inputs = []
                for i in range(fa.n_in()):
                    value = rng.uniform(-.01, .01, fa.size_in(i))
                    if value.shape in ((10, 1), (19, 1)):
                        value[:7, 0] = pa.init_obj_qpos_
                        value[7:10, 0] = pa.init_fingertip_pos_
                    inputs.append(value)
                va, vb = fa(*inputs), fb(*inputs)
                if not isinstance(va, tuple):
                    va, vb = (va,), (vb,)
                for x, y in zip(va, vb):
                    x, y = np.asarray(x), np.asarray(y)
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise RuntimeError('Nonfinite model check: ' + obj + '/' + name)
                    maximum = max(maximum, float(np.max(np.abs(x - y))))
            checks[name] = dict(samples=20, max_abs_difference=maximum)
            if maximum:
                result['mismatches'].append(dict(object=obj, category='symbolic_model', function=name,
                                                max_abs_difference=maximum))
        result['model_checks'].append(dict(object=obj, guards=guards, functions=checks))
        print('Checked symbolic model/costs: ' + obj, flush=True)
    result.update(parameter_case_count=len(result['parameter_cases']),
                  passed=not result['mismatches'],
                  limitations=['No closed-loop success-rate estimate; this audit tests branch equivalence.',
                               'Native solver construction skipped in symbolic checks; configured objective and dynamics are evaluated.'])
    (OUTPUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Audit complete: %d parameter cases, %d mismatches' % (
        result['parameter_case_count'], len(result['mismatches'])), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
