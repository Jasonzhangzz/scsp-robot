import numpy as np


def test_compiled_module_is_dairlib_sampling_c3():
    import dairlib_sampling_c3 as compiled

    assert compiled.planner == "dairlib.systems.SamplingC3Controller"
    assert compiled.contact_model == "dairlib.solvers.LCSFactory::LinearizePlantToLCS"


def test_compiled_controller_returns_an_end_effector_trajectory():
    import dairlib_sampling_c3 as compiled

    planner = compiled.CompiledSamplingC3()
    state = np.zeros(planner.n_x, dtype=np.float64)
    state[0:3] = [0.45, 0.0, 0.05]
    state[3:7] = [1.0, 0.0, 0.0, 0.0]
    state[7:10] = [0.50, 0.0, 0.02]
    desired = state.copy()
    desired[0:3] = [0.50, 0.0, 0.08]
    desired[7:10] = [0.55, 0.05, 0.02]
    plan = planner.step(state, desired, desired, 0.0)
    ee = np.asarray(plan["ee"], dtype=np.float64)
    assert ee.shape[0] == 3
    assert ee.shape[1] >= 1
    assert np.all(np.isfinite(ee))
    assert "is_c3" in plan
