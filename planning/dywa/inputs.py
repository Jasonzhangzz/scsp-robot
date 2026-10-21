"""Task and end-effector inputs for unmodified upstream DyWA classes.

No policy, action, normalization, numerical operation or controller is replaced.
Reset hooks change the requested scene state before upstream observations are made.
"""
from pathlib import Path
import xml.etree.ElementTree as ET


def sphere_robot(source, destination):
    """Keep original tool frame/body ordering; put the sphere exactly at panda_tool.

The left finger retains the upstream contact filter. Its geometry is replaced
by the 10 mm benchmark sphere. Empty hand/right-finger links retain frame names
and inertials needed by upstream rigid-body indexing and numerical IK.
"""
    source, destination = Path(source), Path(destination)
    root = ET.parse(source).getroot()
    for mesh in root.findall('.//mesh'):
        name = mesh.get('filename')
        if name and not Path(name).is_absolute():
            mesh.set('filename', str((source.parent / name).resolve()))
    for name in ('panda_hand', 'panda_leftfinger', 'panda_rightfinger'):
        link = root.find("link[@name='%s']" % name)
        for child in list(link):
            if child.tag in ('collision', 'visual'):
                link.remove(child)
    finger = root.find("link[@name='panda_leftfinger']")
    tool = root.find("joint[@name='panda_hand_tool_joint']/origin")
    joint = root.find("joint[@name='panda_finger_joint1']/origin")
    tool_xyz = list(map(float, tool.get('xyz').split()))
    finger_xyz = list(map(float, joint.get('xyz').split()))
    if tool.get('rpy', '0 0 0') != '0 0 0' or joint.get('rpy', '0 0 0') != '0 0 0':
        raise ValueError('Unexpected upstream tool/finger rotation')
    offset = ' '.join(str(a-b) for a, b in zip(tool_xyz, finger_xyz))
    for tag in ('collision', 'visual'):
        shape = ET.SubElement(finger, tag)
        ET.SubElement(shape, 'origin', xyz=offset, rpy='0 0 0')
        ET.SubElement(ET.SubElement(shape, 'geometry'), 'sphere', radius='0.01')
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding='unicode')
    return destination


def configure_inputs(cfg, spec):
    from dataclasses import replace
    import trimesh

    mesh = trimesh.load(spec['mesh_path'], force='mesh')
    volume = float(mesh.volume)
    if volume <= 0:
        raise ValueError('Object mesh must have positive volume to set its mass')
    # Upstream copies its input file to a .obj filename; supply actual OBJ
    # data rather than an STL renamed to .obj. Coordinates remain unchanged.
    object_path = Path(spec['output_dir']) / 'inputs/object.obj'
    object_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(object_path)
    cfg.env.single_object_scene.mesh = replace(
        cfg.env.single_object_scene.mesh, density=float(spec['mass']) / volume,
        filename=str(object_path))
    assets = Path(spec['asset_root'])
    robot = sphere_robot(assets / 'franka_description/robots/franka_panda.urdf',
                         Path(spec['output_dir']) / 'inputs/sphere_robot.urdf')
    cfg.env.franka.asset_root = str(robot.parent)
    cfg.env.franka.robot_file = robot.name
    cfg.env.single_object_scene.asset_root = str(assets)
    return cfg


def install_task_inputs(spec):
    import torch
    from env.scene.tabletop_with_object_scene import TableTopWithObjectScene
    from env.task.push_task import PushTask
    from env.scene.mesh_object_set import MeshObjectSet

    reset_scene = TableTopWithObjectScene.reset
    reset_task = PushTask.reset
    write_urdf = MeshObjectSet._MeshObjectSet__write_urdf

    def write_object_inputs(self):
        write_urdf(self)
        parts = spec.get('collision_mesh_paths', [])
        if not parts:
            return
        # Preserve the benchmark's authored decomposition as URDF inputs.
        # Neither upstream collision handling nor mesh preprocessing is patched.
        for path in self._MeshObjectSet__urdf.values():
            tree = ET.parse(path)
            link = tree.getroot().find('link')
            for collision in list(link.findall('collision')):
                link.remove(collision)
            for part in parts:
                shape = ET.SubElement(ET.SubElement(link, 'collision'), 'geometry')
                ET.SubElement(shape, 'mesh', filename=str(Path(part).resolve()), scale='1 1 1')
            tree.write(path, encoding='unicode')

    def scene_reset(self, gym, sim, env, env_ids=None):
        ids = reset_scene(self, gym, sim, env, env_ids)
        selected = self.cur_ids if env_ids is None else self.cur_ids[env_ids]
        pose = spec['init_obj_qpos']
        root = env.tensors['root']
        root[selected.long()] = 0
        root[selected.long(), :7] = torch.as_tensor(
            pose[:3] + pose[4:7] + pose[3:4], device=root.device, dtype=root.dtype)
        return ids

    def task_reset(self, env, indices):
        reset_task(self, env, indices)
        indices = slice(None) if indices is None else indices
        quat = spec['target_q']
        self.goal[indices, :7] = torch.as_tensor(
            spec['target_p'] + quat[1:] + quat[:1],
            device=self.goal.device, dtype=self.goal.dtype)

    TableTopWithObjectScene.reset = scene_reset
    PushTask.reset = task_reset
    MeshObjectSet._MeshObjectSet__write_urdf = write_object_inputs
