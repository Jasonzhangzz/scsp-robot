"""A force-closed Coulomb grasp can still creep under regularized friction."""
import mujoco
import numpy as np
import pytest

from planning.grasp_diagnostics import object_contact_wrenches
from planning.grasp_verification import verify_grasp


def _pinch(noslip):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".002" cone="elliptic" impratio="10" solver="Newton"
              tolerance="1e-10" noslip_iterations="{noslip}"/>
      <default><geom friction=".7 .01 .001" condim="3"
                     solref=".02 1" solimp=".9 .95 .001"/></default>
      <worldbody>
        <body name="object" pos="0 0 .1"><freejoint/>
          <geom type="box" size=".02 .03 .08" mass=".03"/></body>
        <geom type="sphere" pos="-.028 -.015 .1" size=".01"/>
        <geom type="sphere" pos="-.028 .015 .1" size=".01"/>
        <geom type="sphere" pos=".028 -.015 .1" size=".01"/>
        <geom type="sphere" pos=".028 .015 .1" size=".01"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    _, contacts = object_contact_wrenches(model, data, model.body('object').id, range(1, 5))
    normal = np.array([c['normal_force'] for c in contacts])
    utilization = np.array([np.linalg.norm(c['force']-c['normal']*c['normal_force'])/
        (c['mu']*c['normal_force']) for c in contacts])
    certificate = verify_grasp([c['point']-data.qpos[:3] for c in contacts],
        [c['normal'] for c in contacts], .7, external_wrench=[0, 0, -.2943, 0, 0, 0],
        normal_force_limits=normal)
    assert certificate['force_closure'] and certificate['load_feasible']
    assert np.max(utilization) < .04  # Far inside the physical friction cone.
    np.testing.assert_array_equal(data.qfrc_applied, 0.)
    return abs(data.qpos[2]-.1), normal, model.geom_friction.copy()


@pytest.mark.parametrize("iterations", [3, 10])
def test_noslip_suppresses_creep_without_increasing_friction_or_pressure(iterations):
    creeping, old_normal, old_friction = _pinch(0)
    sticking, new_normal, new_friction = _pinch(iterations)
    assert creeping > .0001
    assert sticking < .00001
    assert creeping > 10*sticking
    np.testing.assert_allclose(new_normal, old_normal, rtol=1e-5)
    np.testing.assert_array_equal(new_friction, old_friction)
