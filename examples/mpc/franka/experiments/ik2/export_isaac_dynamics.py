"""Export the actual no-hand Franka properties for the MJWarp scene.

Run in the Isaac Gym environment, after changing its URDF or collision meshes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))


def main():
    from isaacgym import gymapi, gymtorch  # Must precede Torch.
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    from examples.mpc.franka.runtime.isaac import open_env
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'examples/mpc/franka/runtime/assets/isaac_franka_dynamics.json')
    output = parser.parse_args().output
    args = _add_rollout_policy_args(argparse.ArgumentParser()).parse_args(
        ['--headless', '--obj', 'foam_brick', '--trial_num', '1'])
    env, _, _ = open_env(args, 0)
    try:
        names = env.gym.get_actor_rigid_body_names(env.env, env.franka_actor)
        properties = env.gym.get_actor_rigid_body_properties(env.env, env.franka_actor)
        bodies = []
        for name, prop in zip(names, properties):
            com = prop.com if hasattr(prop.com, 'x') else prop.com.p
            bodies.append(dict(name=name, mass=prop.mass, com_pos=[com.x, com.y, com.z],
                inertia=[[getattr(getattr(prop.inertia, a), b) for b in 'xyz'] for a in 'xyz']))
        joints = env.gym.get_actor_dof_properties(env.env, env.franka_actor)
        source = 'envs/robots/assets/urdf/franka_description/robots/franka_panda.urdf'
        data = dict(schema_version=1, source_urdf=source,
                    source_urdf_sha256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),
                    source='Isaac Gym Preview 4 CPU PhysX get_actor_rigid_body_properties; '
                           'fixed base, gravity disabled, no hand + sphere',
                    reference_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                    bodies=bodies, joint_properties={key: joints[key].tolist() for key in
                        ('damping', 'friction', 'armature', 'velocity', 'effort')})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2)+'\n')
        print('Saved Isaac dynamics to '+str(output))
    finally:
        env.close()


if __name__ == '__main__':
    main()
