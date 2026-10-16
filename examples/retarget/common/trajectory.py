"""Shared demonstration-clock pose interpolation and tracking statistics."""
import numpy as np


def reference(poses,times):
    frame=np.clip(times*30,0,len(poses)-1);a=np.floor(frame).astype(int);b=np.minimum(a+1,len(poses)-1);alpha=(frame-a)[:,None]
    result=(1-alpha)*poses[a]+alpha*poses[b]
    qa=poses[a,3:];qb=poses[b,3:]*np.where(np.sum(qa*poses[b,3:],axis=1,keepdims=True)<0,-1,1)
    result[:,3:]=(1-alpha)*qa+alpha*qb
    result[:,3:]/=np.linalg.norm(result[:,3:],axis=1,keepdims=True)
    return result


def stats(position,angle,mask):
    p=position[mask];a=angle[mask]
    if not len(p):return None
    return dict(samples=len(p),position_mean_mm=float(p.mean()),position_rmse_mm=float(np.sqrt(np.mean(p*p))),
        position_p95_mm=float(np.quantile(p,.95)),position_max_mm=float(p.max()),
        rotation_mean_deg=float(a.mean()),rotation_rmse_deg=float(np.sqrt(np.mean(a*a))),
        rotation_p95_deg=float(np.quantile(a,.95)),rotation_max_deg=float(a.max()),
        last_position_mm=float(p[-1]),last_rotation_deg=float(a[-1]))
