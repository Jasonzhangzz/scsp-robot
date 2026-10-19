"""Scene materials must match the contact model before CSO is constructed."""
import argparse
import importlib
from types import SimpleNamespace

import pytest

from examples.mpc.franka.runtime.contact import configure_scene_friction


@pytest.mark.parametrize('suffix', ['', '_ablation'])
@pytest.mark.parametrize('backend,scene,arm_mu,table_mu', [
    ('isaac', 'isaac', 1.2, .7),
    ('mjwarp', 'isaac', 1.2, .7),
    ('mjwarp', 'legacy', 1.5, .9),
])
def test_cso_receives_scene_pair_friction_at_initial_construction(
        monkeypatch, suffix, backend, scene, arm_mu, table_mu):
    materials = (.5, .9, 1.5)
    if suffix == '' and backend == 'isaac':
        arm_mu = 1.6
        materials = (.2, 1.2, 2.)
    entry=importlib.import_module(('examples.mpc.franka.experiments.ik2.test_mpc_isaac_ablation' if suffix else 'examples.mpc.franka.runtime.mpc'))
    params=importlib.import_module(('examples.mpc.franka.experiments.ik2.params_ablation' if suffix else 'examples.mpc.franka.runtime.params'))
    args=entry._add_rollout_policy_args(argparse.ArgumentParser()).parse_args([
        '--obj','foam_brick','--physics-backend',backend,'--mjwarp-scene',scene])
    captured=[]
    def optimizer(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(params,'LambdaContactControlOptimizer',optimizer)
    p=params.ExplicitMPCParams(args,rand_seed=0)
    entry._apply_dywa_physics_to_param(p)
    assert len(captured)==1  # Physics configuration cannot leave an obsolete first CSO.
    assert captured[0]['arm_friction']==pytest.approx(arm_mu)
    assert captured[0]['friction_reg_coef']==pytest.approx(.03)
    assert captured[0]['friction_cone']=='pyramid'
    assert p.mu_fingertip_object_==pytest.approx(arm_mu)
    assert p.mu_table_object_==pytest.approx(table_mu)
    assert p.mu_object_==pytest.approx(table_mu)
    assert (p.table_friction_,p.object_friction_,p.fingertip_friction_)==materials


@pytest.mark.parametrize('suffix', ['', '_ablation'])
def test_tangent_penalty_cli_override_is_preserved(suffix):
    entry=importlib.import_module(('examples.mpc.franka.experiments.ik2.test_mpc_isaac_ablation' if suffix else 'examples.mpc.franka.runtime.mpc'))
    parser=entry._add_rollout_policy_args(argparse.ArgumentParser())
    assert parser.parse_args([]).friction_reg_coef==pytest.approx(.03)
    assert parser.parse_args(['--friction_reg_coef','1']).friction_reg_coef==1.
    assert parser.parse_args(['--friction_reg_coef','0']).friction_reg_coef==0.


def test_custom_materials_update_both_pairs_consistently():
    p=SimpleNamespace(table_friction_=.2,object_friction_=.8,fingertip_friction_=1.4)
    configure_scene_friction(p)
    assert p.mu_table_object_==pytest.approx(.5)
    assert p.mu_fingertip_object_==pytest.approx(1.1)
    p.object_friction_=.4
    configure_scene_friction(p)
    assert p.mu_table_object_==pytest.approx(.3)
    assert p.mu_fingertip_object_==pytest.approx(.9)
