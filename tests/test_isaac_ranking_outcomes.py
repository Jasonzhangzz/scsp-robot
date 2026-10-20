"""Exercise the real trial loop with a deterministic observation bus, no Isaac."""

import ast
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from utils.ranking_metrics import discounted_terminal_reward
from examples.mpc.franka.runtime.contact import (
    resolve_nullspace_stiffness, resolve_nullspace_damping_model,
)


@pytest.mark.parametrize("initial_error,final_observation,success,steps,aborted", [
    (0.02, None, True, 0, False),  # Inclusive manuscript threshold, initially done.
    (0.03, {"error": 0.01}, True, 1, False),  # Success on the last allowed action.
    (0.03, {"error": 0.03}, False, 1, False),  # Budget exhausted.
    (0.03, None, False, 1, True),  # Bus timeout must not count as success.
    (0.03, {"viewer_closed": True}, False, 1, True),
])
def test_trial_reward_uses_observed_success(
        monkeypatch, tmp_path, initial_error, final_observation, success, steps, aborted):
    # Load the actual function without importing the simulator's heavy dependencies.
    source = Path(__file__).resolve().parents[1] / 'examples/mpc/franka/runtime/mpc.py'
    tree = ast.parse(source.read_text())
    run = next(node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == 'run_mpc_planner')

    def observation(error=0.03, **kwargs):
        return {'policy_q': np.array([error, 0, 0, 1, 0, 0, 0, 0, 0, 0]), **kwargs}

    class Bus:
        def __init__(self):
            self.actions = []

        def publish_cmd(self, command):
            self.actions.append(command)

        def wait_obs(self, **kwargs):
            if final_observation is None:
                return None
            return observation(**final_observation)

    bus = Bus()
    bus_module = ModuleType('examples.mpc.franka.runtime.bus')
    bus_module.wait_trial_obs = lambda *_: observation(initial_error)
    planner_module = ModuleType('planning.mpc_explicit')
    planner_module.planner_init_payload = lambda *_: None
    planner_module._build_mpc_planner_runtime = lambda *_: (None, param, None, {})
    planner_module.handle_mpc_request = lambda *_: {
        'rank_dt': 0.01, 'action': np.zeros(3), 'verify_chatter': False,
    }
    monkeypatch.setitem(sys.modules, bus_module.__name__, bus_module)
    monkeypatch.setitem(sys.modules, planner_module.__name__, planner_module)
    param = SimpleNamespace(
        target_p_=np.zeros(3), target_q_=np.array([1, 0, 0, 0]),
        lambda_optimizer=SimpleNamespace(contact_switch_confidence=1, sample_point=np.zeros((1, 3))), table_height=0,
        mu_fingertip_object_=1.2, mu_table_object_=0.7,
        mpc_horizon_=5, nullspace_inertia_damping_=True,
    )
    policy = {
        'value_info': {}, 'escape_on': False,
        'mpc_virtual_point': np.zeros(3), 'p_arm_world': np.zeros(3),
        'best_contact_world': np.zeros(3),
    }
    edge_info = {
        'near_edge': False, 'if_contact': False, 'opposite': False,
        'keepout': 0, 'tip_xy_r': 0, 'via_xy_r': 0, 'orbit_clips': 0,
    }
    namespace = {
        'np': np, 'os': os, 'json': json, 'time': time, 'sys': sys,
        'POLICY_INTERVAL': 0.02, 'DYWA_SIM_DT': 0.002, 'ISAAC_TABLE_CLEARANCE': 0,
        'discounted_terminal_reward': discounted_terminal_reward,
        'metrics': SimpleNamespace(
            open_trial_recorder=lambda *a, **kw: None,
            comp_pos_error=lambda p, goal: np.linalg.norm(p - goal),
            comp_quat_error=lambda q, goal: 1 - np.dot(q, goal)**2,
        ),
        'resolve_solver_backend': lambda *_: 'portable',
        'resolve_nullspace_stiffness': resolve_nullspace_stiffness,
        'resolve_nullspace_damping_model': resolve_nullspace_damping_model,
        'print_runtime_profile': lambda *a, **kw: None,
        'make_runtime_profile': lambda *a, **kw: None,
        'ExplicitMPCParams': lambda *a, **kw: param,
        '_apply_dywa_physics_to_param': lambda p: p,
        'adapt_param_for_cartesian_solver': lambda p, args: p,
        '_configure_rollout_param': lambda p, args: p,
        '_plan_from_obs': lambda obs, dwell: (None, obs['policy_q']),
        '_apply_plan_result': lambda *a, **kw: policy,
        '_print_rollout_step': lambda *_: None,
        '_pred_reduction_from_policy': lambda *_: (None, None),
        '_clip_mpc_action': lambda action, _: action,
        '_print_rank_cost_diag': lambda *_: None,
        '_analyze_edge_pose_source': lambda *_: edge_info,
    }
    exec(compile(ast.Module(body=[run], type_ignores=[]), str(source), 'exec'), namespace)
    output = tmp_path / 'trials.jsonl'
    args = SimpleNamespace(trial_start=0, trial_num=1, max_rollout_length=1,
                           control_substeps=0, obj='fixture', ranking_metrics_jsonl=str(output),
                           friction_reg_coef=0.01)
    assert namespace['run_mpc_planner'](bus, args) is aborted
    record = json.loads(output.read_text())
    assert bool(record['success']) is success
    assert record['steps'] == steps
    assert record['completion_steps'] == (steps if success else None)
    assert record['terminal_reward'] == pytest.approx(0.9**steps if success else 0)
    assert record['aborted'] is aborted
    assert record['friction_reg_coef'] == 0.01
    assert record['fingertip_object_friction'] == 1.2
    assert record['table_object_friction'] == 0.7
    assert len([cmd for cmd in bus.actions if cmd.get('kind') == 'via']) == steps
