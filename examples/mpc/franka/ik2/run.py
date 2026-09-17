"""Isaac Gym process: receive commands, env.step, publish observations."""

from __future__ import annotations

import argparse
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(current_dir)
while os.path.basename(parent_dir) != "scsp-robot":
    _next_dir = os.path.dirname(parent_dir)
    if _next_dir == parent_dir:
        raise RuntimeError("scsp-robot repo root not found from %s" % current_dir)
    parent_dir = _next_dir
# Isaac Gym/PyTorch may import a third-party ``examples`` package first.
sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir) != parent_dir]
sys.path.insert(0, parent_dir)

# gymdeps refuses to load if torch is already imported.  params -> mlqp_point
# imports torch, so isaacgym must come first.
from isaacgym import gymapi, gymtorch  # noqa: F401
import torch  # noqa: F401
import numpy as np

from examples.mpc.franka.ik2.isaac_bus import (
    IsaacBus,
    args_from_init,
    start_planner,
    stop_peer,
    viewer_draw_stride,
)
from examples.mpc.franka.ik2.params import ExplicitMPCParams
from examples.mpc.franka.ik2.test_mpc_isaac import (
    POLICY_INTERVAL,
    ContactIsaacCartesian,
    IsaacFrankaOSCSimulator,
    _apply_dywa_physics_to_param,
    _configure_rollout_param,
    _add_rollout_policy_args,
    adapt_param_for_cartesian_solver,
)


def viewer_closed(env):
    if getattr(env, "viewer_", None) is None:
        return False
    try:
        return bool(env.gym.query_viewer_has_closed(env.viewer_))
    except Exception:
        return False


def pack_state(env, seq, trial, contact_fields):
    return {
        "seq": int(seq),
        "trial": int(trial),
        "policy_q": np.asarray(env.get_policy_state(), dtype=np.float32),
        "full_q": np.asarray(env.get_state(), dtype=np.float32),
        "break_out": bool(getattr(env, "break_out_signal_", False)),
        "paused": bool(getattr(env, "dyn_paused_", False)),
        "viewer_closed": viewer_closed(env),
        **contact_fields,
    }


def refresh_contact(env, contact, table_ground):
    from examples.mpc.franka.ik2.test_mppi_isaac import _plan_payload

    payload, _curr_q, if_contact, contact_distance = _plan_payload(env, contact, table_ground)
    return {
        "phi_vec": np.asarray(payload["phi_vec"], dtype=np.float64),
        "jac_mat": np.asarray(payload["jac_mat"], dtype=np.float64),
        "jac_mat_env": np.asarray(payload["jac_mat_env"], dtype=np.float64),
        "if_contact": bool(if_contact),
        "contact_distance": float(contact_distance),
        "table_ground": float(table_ground),
    }


def apply_incoming(env, cmd, default_kind):
    if cmd is None:
        return "ok"
    name = cmd.get("cmd")
    if name in ("end_trial", "stop"):
        return name
    markers = cmd.get("markers") or {}
    target = markers.get("target")
    if target is not None:
        env.show_target(target)
    best = markers.get("best_contact")
    if best is not None:
        env.show_best_contact(best)
    kind = cmd.get("kind", default_kind)
    if kind == "via":
        via = cmd.get("via")
        if via is None:
            tip = np.asarray(env.get_policy_state()[7:10], dtype=np.float64)
            via = tip + np.asarray(cmd.get("action", np.zeros(3)), dtype=np.float64).reshape(3)
        env.set_via_action(np.asarray(via, dtype=np.float64))
    else:
        action = np.asarray(cmd.get("action", np.zeros(7)), dtype=np.float32).reshape(7)
        if float(np.linalg.norm(action)) < 1e-9:
            env.hold_current_pose()
        else:
            env.set_joint_action(action)
    return "ok"


def build_osc_param(args, trial_count):
    param = ExplicitMPCParams(
        args,
        rand_seed=trial_count,
        target_type=getattr(args, "target_type", "ground-rotation"),
        mpc_model="explicit",
    )
    param = _apply_dywa_physics_to_param(param)
    param.use_jax_contact_ = False
    param = adapt_param_for_cartesian_solver(param, args)
    param = _configure_rollout_param(param, args)
    param.osc_pos_stiffness_ = float(args.osc_pos_stiffness)
    param.osc_ori_stiffness_ = float(args.osc_ori_stiffness)
    param.nullspace_stiffness_ = float(args.nullspace_stiffness)
    stiffness = args.cartesian_stiffness
    if stiffness is None:
        from examples.mpc.franka.ik2.test_mpc_isaac import DEFAULT_CARTESIAN_STIFFNESS
        stiffness = DEFAULT_CARTESIAN_STIFFNESS.tolist()
    param.cartesian_stiffness_ = np.array(stiffness, dtype=np.float32)
    param.cartesian_damping_ = (
        None if args.cartesian_damping is None else np.array(args.cartesian_damping, dtype=np.float32)
    )
    param.effort_joint_damping_ = float(args.effort_joint_damping)
    param.show_ghost_object_ = bool(args.show_ghost_object)
    param.use_xml_texture_ = bool(getattr(args, "use_xml_texture", False))
    param.svg_screenshot_dir_ = getattr(args, "svg_screenshot_dir", None) or None
    param.svg_screenshot_interval_ = float(getattr(args, "svg_screenshot_interval", 0.2))
    param.svg_screenshot_width_ = int(getattr(args, "svg_screenshot_width", 1280))
    param.svg_screenshot_height_ = int(getattr(args, "svg_screenshot_height", 960))
    param.svg_screenshot_prefix_ = f"trial_{int(trial_count):03d}"
    param.control_substeps_ = int(getattr(args, "control_substeps", 0))
    param.physx_use_gpu_ = bool(getattr(args, "gpu_physx", False))
    param.viewer_hz_ = float(getattr(args, "viewer_hz", 60.0))
    return param


def open_env(args, trial_count):
    param = build_osc_param(args, trial_count)
    contact = ContactIsaacCartesian(param)
    env = IsaacFrankaOSCSimulator(
        param,
        headless=args.headless,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
    )
    env.show_target_object_pose(param.target_p_, param.target_q_)
    table_ground = float(param.table_height) + float(args.ground_height_threshold)
    return env, contact, table_ground


def isaac_loop(bus, args, planner):
    trial_start = int(getattr(args, "trial_start", 0))
    trial_num = max(1, int(getattr(args, "trial_num", 1)))
    default_kind = "joint" if planner == "mppi" else "via"
    contact_period = float(POLICY_INTERVAL)
    for trial_count in range(trial_start, trial_start + trial_num):
        env = None
        try:
            env, contact, table_ground = open_env(args, trial_count)
            env.hold_current_pose()
            draw_stride = viewer_draw_stride(
                env.sim_dt_, float(getattr(args, "viewer_hz", getattr(env.param_, "viewer_hz_", 60.0)))
            )
            seq = 0
            frames = 0
            last_contact = refresh_contact(env, contact, table_ground)
            last_contact_t = time.monotonic()
            bus.publish_obs(pack_state(env, seq, trial_count, last_contact))
            while True:
                if viewer_closed(env) or getattr(env, "break_out_signal_", False):
                    bus.publish_obs({
                        "trial": trial_count,
                        "break_out": True,
                        "viewer_closed": True,
                    })
                    return
                if getattr(env, "dyn_paused_", False):
                    env._simulate_once(draw=True)
                    env.sync_realtime()
                    continue
                status = apply_incoming(env, bus.take_cmd(), default_kind)
                if status == "stop":
                    return
                if status == "end_trial":
                    break
                frames += 1
                draw = (frames % draw_stride == 0)
                env.step_control_frame(draw=draw, sync_realtime=False)
                if not draw:
                    continue
                env.sync_realtime()
                now = time.monotonic()
                if now - last_contact_t >= contact_period:
                    last_contact = refresh_contact(env, contact, table_ground)
                    last_contact_t = now
                seq += 1
                bus.publish_obs(pack_state(env, seq, trial_count, last_contact))
        finally:
            if env is not None:
                env.close()
    bus.publish_obs({"cmd": "stop", "trial": trial_start + trial_num})


def isaac_worker(cmd_q, obs_q, ready_q, init, announce=True):
    if announce and ready_q is not None:
        ready_q.put({"ok": True, "pid": os.getpid()})
    args = args_from_init(init)
    isaac_loop(IsaacBus(cmd_q, obs_q), args, init.get("planner", "mppi"))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--planner", choices=("mppi", "mpc"), required=True)
    pre_args, remaining = pre.parse_known_args()
    parser = argparse.ArgumentParser(description="Isaac Gym viewer process.")
    parser.add_argument("--planner", choices=("mppi", "mpc"), required=True)
    if pre_args.planner == "mppi":
        from examples.mpc.franka.ik2.test_mppi_isaac import _add_mppi_policy_args
        _add_mppi_policy_args(parser)
    else:
        _add_rollout_policy_args(parser)
    args = parser.parse_args(["--planner", pre_args.planner, *remaining])
    args.solver = "acados"
    args.rollout = True
    print(f"isaac pid={os.getpid()} planner={args.planner}", flush=True)
    bus, proc = start_planner(args, args.planner)
    try:
        isaac_loop(bus, args, args.planner)
    finally:
        stop_peer(bus, proc)


if __name__ == "__main__":
    main()
