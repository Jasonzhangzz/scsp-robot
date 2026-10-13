import sys, os, json, numpy as np
from pathlib import Path
root=Path('/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget/tests/retarget_1_raw_comparison_20261013')
repo=Path('/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/ego_fast_retarget/tests/retarget_1_comparison_20261013/historical_adde321')
label='historical'
os.environ['GEORT_WUJI_MODEL_CONFIG']=str(root/'matched_model.json')
sys.path[:0]=[str(repo),str(repo/'replay'),str(repo/'utils')]
from calibrated import batch_replay as batch
from wuji_retarget import emf_offline as emf
native_original=emf.solve_native
def capture_native(*args,**kwargs):
 result=native_original(*args,**kwargs)
 np.savez_compressed(root/(label+'_native.npz'),**result[0])
 (root/(label+'_native_metadata.json')).write_text(json.dumps(result[1],ensure_ascii=False,indent=2))
 return result
emf.solve_native=capture_native
clock_original=batch._apply_publish_clock
def capture_clock(data,fps):
 np.savez_compressed(root/(label+'_raw.npz'),**data)
 result=clock_original(data,fps)
 np.savez_compressed(root/(label+'_publish_clock.npz'),**result)
 return result
batch._apply_publish_clock=capture_clock
write_original=batch._write_reference_episode
def capture_write(*args,**kwargs):
 result=write_original(*args,**kwargs)
 data=kwargs.get('replay_data')
 if data is not None:np.savez_compressed(root/(label+'_final.npz'),**data)
 return result
batch._write_reference_episode=capture_write
args=['--config',str(root/(label+'.yaml')),'--devices','cuda:3','--workers-per-gpu','1','--no-remote-fetch']
# Legacy batch locks initial_position in its preset and rejects CLI overrides.
# Adapt only the generated parameter list to match the current YAML target.
defaults_original=batch._batch_solver_defaults
def matched_defaults(*args,**kwargs):
 values=defaults_original(*args,**kwargs)
 i=values.index('--initial_position')
 values[i+1:i+4]=['0.37','0','0.245']
 return values
batch._batch_solver_defaults=matched_defaults
status=batch.main(args)
(root/(label+'_exit_status.json')).write_text(json.dumps({'status':status}))
raise SystemExit(status)
