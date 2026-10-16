"""Publish atomic viewer snapshots; keep GLFW out of the CUDA solver process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


class ViewerProcess:
    def __init__(self, experiment, case):
        self.case=Path(case)
        environment=dict(os.environ);environment['MUJOCO_GL']='glfw'
        self.process=subprocess.Popen([sys.executable,'-m','examples.retarget.view_fa_admm',
            '--experiment',str(experiment),'--case',str(case),'--follow'],env=environment)

    def wait_if_paused(self):
        while True:
            if self.process.poll() is not None:raise KeyboardInterrupt('Viewer closed')
            path=self.case/'viewer_control.json'
            try:paused=json.loads(path.read_text()).get('paused',False)
            except (FileNotFoundError,json.JSONDecodeError):paused=False
            if not paused:return
            time.sleep(.05)

    def update(self, qpos, qvel, seconds, snapshot=None):
        self.wait_if_paused()
        arrays=dict(qpos=qpos,qvel=qvel,seconds=np.asarray(seconds))
        if snapshot is not None:arrays.update({'snapshot_'+k:v for k,v in snapshot.items()})
        temporary=self.case/'live_state_tmp.npz'
        np.savez_compressed(temporary,**arrays);temporary.replace(self.case/'live_state.npz')

    def close(self):
        # Leave the final frame open for inspection. Closing the UI during a
        # solve is detected at its next control-window boundary.
        (self.case/'viewer_finished.json').write_text(json.dumps(dict(solver_finished=True)))
