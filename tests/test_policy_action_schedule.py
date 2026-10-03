import numpy as np

from examples.mpc.franka.ik2.isaac_bus import PolicyActionSchedule


def _run(delay_frame):
    schedule = PolicyActionSchedule(sim_dt=0.002, policy_dt=0.02, max_step=0.005)
    schedule.submit([0.005, 0.0, 0.0], seq=0, origin=[0.4, 0.0, 0.37], table_z=0.35, clearance=0.012)
    out = []
    for frame in range(20):
        if frame == delay_frame:
            schedule.submit([0.0, 0.005, 0.0], seq=1, origin=[0.4, 0.0, 0.37], table_z=0.35, clearance=0.012)
        out.append(schedule.advance([0.4, 0.0, 0.37])[0])
    return np.asarray(out)


def test_planner_arrival_inside_interval_does_not_restart_action():
    early = _run(1)
    late = _run(9)
    np.testing.assert_allclose(early[:10], late[:10], atol=1e-8)
    np.testing.assert_allclose(early[9], [0.405, 0.0, 0.37], atol=1e-6)
    np.testing.assert_allclose(early[-1], late[-1], atol=1e-8)


def test_stale_commands_are_dropped():
    schedule = PolicyActionSchedule()
    assert schedule.submit([0.001, 0, 0], seq=4)
    assert not schedule.submit([0.005, 0, 0], seq=4)
    assert schedule.dropped == 1


def test_downward_action_is_clamped_to_table_clearance():
    schedule = PolicyActionSchedule()
    assert schedule.submit(
        [0.0, 0.0, -0.02], seq=1, origin=[0.4, 0.0, 0.37],
        table_z=0.35, clearance=0.012,
    )
    target, _velocity, _seq = schedule.advance([0.4, 0.0, 0.37])
    assert float(target[2]) >= 0.362 - 1e-7


def test_foreign_policy_period_cannot_change_executor_cadence():
    a = PolicyActionSchedule(sim_dt=0.002, policy_dt=0.02, max_step=0.005)
    b = PolicyActionSchedule(sim_dt=0.002, policy_dt=0.02, max_step=0.005)
    assert a.submit([0.005, 0.0, 0.0], seq=0, policy_dt=0.02)
    assert b.submit([0.005, 0.0, 0.0], seq=0, policy_dt=0.05)
    for _ in range(10):
        ta = a.advance([0.0, 0.0, 0.0])[0]
        tb = b.advance([0.0, 0.0, 0.0])[0]
    np.testing.assert_allclose(ta, tb, atol=1e-8)
