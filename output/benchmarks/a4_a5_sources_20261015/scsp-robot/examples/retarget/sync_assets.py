"""Copy the authorized DexForge example and Wuji dependencies into a cache."""
from pathlib import Path
import subprocess

REMOTE='/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/grasp_sonic/thirdparty'
SEQUENCE='dexycb_004_sugar_box_20200709_142802'


def sync_assets(cache,host='8.147.67.185',port=1016):
    cache=Path(cache).resolve()
    requests=[('DexForge/assets/wuji','DexForge/assets'),
              (f'DexForge/data/dexycb/{SEQUENCE}','DexForge/data/dexycb'),
              ('DexForge/contactaware/contactaware','DexForge/contactaware'),
              ('DexForge/LICENSE','DexForge'),
              ('DexForge/THIRD_PARTY_NOTICES.md','DexForge'),
              ('wuji-description/hand/body/meshes/right/','wuji-description/hand/body/meshes/right')]
    for source,destination in requests:
        target=cache/destination;target.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az','-e',f'ssh -p {int(port)} -o BatchMode=yes',
            f'{host}:{REMOTE}/{source}',str(target)+'/'],check=True)
    return cache/'DexForge'
