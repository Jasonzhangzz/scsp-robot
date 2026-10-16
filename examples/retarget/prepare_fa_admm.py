"""Common CA postures plus original, unfiltered MANO semantic contact templates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np


def prepare(root,baseline,mano,mano_scene):
    root=Path(root).resolve();baseline=Path(baseline).resolve()
    root.mkdir(parents=True,exist_ok=True)
    for name in ['sequence','contactaware']:shutil.copytree(baseline/name,root/name,dirs_exist_ok=True)
    shutil.copy2(baseline/'forceaware.yaml',root/'forceaware.yaml')
    for path in list((root/'sequence/scene').rglob('*.xml'))+list((root/'contactaware').glob('config.json')):
        path.write_text(path.read_text().replace(str(baseline),str(root)))
    data=np.load(mano);source=mujoco.MjModel.from_xml_path(str(mano_scene))
    target=mujoco.MjModel.from_xml_path(str(root/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    names=[mujoco.mj_id2name(source,mujoco.mjtObj.mjOBJ_BODY,int(b)) for b in data['body']]
    bodies=np.array([mujoco.mj_name2id(target,mujoco.mjtObj.mjOBJ_BODY,n) for n in names],np.int32)
    assert np.all(bodies>=0)
    poses=np.load(root/'contactaware/object_pose_7.npy')[:,[4,5,6,3,0,1,2]]
    assert np.allclose(poses,data['object_pose'],atol=1e-6)
    count,slots=data['mask'].shape
    for visual in (baseline/'demonstration_visual.npz',Path(mano).parent/'demonstration_visual.npz'):
        if visual.exists():
            with np.load(visual) as saved:
                if len(saved['vertices'])!=count or not np.allclose(saved['object_pose'],poses,atol=1e-6,rtol=0.):
                    raise ValueError(f'MANO visualization does not match the reference: {visual}')
            if visual.resolve()!=(root/'demonstration_visual.npz').resolve():
                shutil.copy2(visual,root/'demonstration_visual.npz')
            break
    local=np.asarray(data['source_local'],np.float32)
    guidance=dict(contact_mask=np.asarray(data['mask'],np.float32),
        contact_weight=np.ones((count,slots),np.float32),
        contact_pos_obj=np.asarray(data['anchors'],np.float32),
        hand_query_body_ids=np.broadcast_to(bodies,(count,slots)).copy(),
        hand_query_local_pos=np.broadcast_to(local,(count,slots,3)).copy(),
        link_labels=np.asarray(names))
    shutil.copy2(root/'contactaware/contact_guidance.npz',root/'contactaware/CA_guidance_reference_only.npz')
    np.savez_compressed(root/'contactaware/contact_guidance.npz',**guidance)
    shutil.copy2(mano,root/'original_MANO_semantic_contacts.npz')
    manifest=dict(formulation='min FA(u,c)+I_surface(z), c=z',MANO_role='initialization_only',
        original_template_source=str(Path(mano).resolve()),original_mask_not_filtered=True,
        frames=count,slots=slots,hand='wuji_right',object='004_sugar_box',duration_s=5.4,
        original_FA_formulas_and_weights=True,original_Adam_called_by_ADMM=False,
        control_subsolver='independent batched L-BFGS',force_QP=False,
        methods=['fixed_MANO_FA_Adam','fixed_MANO_LBFGS','variable_MANO_ADMM_LBFGS'],
        CA_sha256=hashlib.sha256((root/'contactaware/wuji_right_qpos.npy').read_bytes()).hexdigest(),
        guidance_sha256=hashlib.sha256((root/'contactaware/contact_guidance.npz').read_bytes()).hexdigest(),
        source_bodies=names)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--mano',type=Path,required=True)
    p.add_argument('--mano-scene',type=Path,required=True);a=p.parse_args()
    prepare(a.experiment,a.baseline,a.mano,a.mano_scene)
