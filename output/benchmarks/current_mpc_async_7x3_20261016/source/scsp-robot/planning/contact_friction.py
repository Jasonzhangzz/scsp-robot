"""Contact frames for the MuJoCo condim=3 pyramidal force cone."""
import numpy as np


def native_tangents(normals, preferred=None):
    """Complete unit normals using MuJoCo's default makeFrame convention.

    Existing contacts can supply their actual first tangent instead. Projection
    transports that axis onto a nearby candidate normal; future contact frames
    remain a geometric prediction and must be refreshed after physical contact.
    """
    normals=np.asarray(normals,float)
    if normals.shape[-1:]!=(3,) or not np.isfinite(normals).all():
        raise ValueError('Normals must be finite 3-vectors')
    length=np.linalg.norm(normals,axis=-1,keepdims=True)
    if np.any(length<1e-12):raise ValueError('Normals must be nonzero')
    n=normals/length
    helper=np.zeros_like(n)
    helper[...,1]=(np.abs(n[...,1])<.5)
    helper[...,2]=(np.abs(n[...,1])>=.5)
    fallback=helper-n*np.sum(helper*n,axis=-1,keepdims=True)
    if preferred is not None:
        first=np.asarray(preferred,float)[...,0,:]
        first=np.broadcast_to(first,n.shape).copy()
        first-=n*np.sum(first*n,axis=-1,keepdims=True)
        usable=np.isfinite(first).all(axis=-1,keepdims=True)&(np.linalg.norm(first,axis=-1,keepdims=True)>1e-8)
        first=np.where(usable,first,fallback)
    else:first=fallback
    first/=np.linalg.norm(first,axis=-1,keepdims=True)
    return np.stack((first,np.cross(n,first)),axis=-2)
