"""CasADi external-function ABI bridge to native stage callbacks.

The OCP calls nonlinear functions at its current x/u, including on cache misses.
No affine dynamics surrogate or finite-difference derivative is installed.
The Python/Warp bridge is intentionally explicit and included in wall time.
"""
import ctypes
from pathlib import Path
import subprocess
import numpy as np


def build_bridge(directory, stages):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    source=directory/'native_bridge.c';library=directory/'libfa_native_bridge.so'
    nx=stages.nx;nu=stages.nu;ny=stages.nout;nq=stages.nq;nr=stages.nr
    parts=['/* native bridge ABI v2, with SONAME */\n#include <stddef.h>\n#include <math.h>\ntypedef long long I;\n'
        'typedef int (*Callback)(int,const double*,const double*,const double*,double*,double*,double*);\n'
        'static Callback callback=0;\nvoid fa_set_callback(Callback f) {callback=f;}\n']
    def sp(name,r,c):
        data=[r,c,*range(0,(c+1)*r,r),*list(range(r))*c] if r else [r,c,*([0]*(c+1))]
        parts.append('static const I '+name+'[]={'+','.join(map(str,data))+'};\n')
    def function(name,ins,outs,body):
        for i,(r,c) in enumerate(ins):sp(name+'_in'+str(i),r,c)
        for i,(r,c) in enumerate(outs):sp(name+'_out'+str(i),r,c)
        parts.append(f'int {name}(const double** a,double** r,I* iw,double* w,int mem){{ if(!callback)return 1; {body} }}\n')
        parts.append(f'I {name}_n_in(void){{return {len(ins)};}}\nI {name}_n_out(void){{return {len(outs)};}}\n')
        for side,shapes in [('in',ins),('out',outs)]:
            cases=''.join(f'case {i}:return {name}_{side}{i};' for i in range(len(shapes)))
            parts.append(f'const I* {name}_sparsity_{side}(I i){{switch(i){{{cases}default:return 0;}}}}\n')
        parts.append(f'int {name}_work(I* a,I* r,I* iw,I* w){{*a={len(ins)};*r={len(outs)};*iw=0;*w=0;return 0;}}\n')
    function('native_physics',[(nx,1),(nu,1),(1,1)],[(ny,1)],'return callback(0,a[0],a[1],a[2],r[0],0,0);')
    function('jac_native_physics',[(nx,1),(nu,1),(1,1),(ny,1)],[(ny,nx),(ny,nu),(ny,1)],
        'return callback(1,a[0],a[1],a[2],r[0],r[1],r[2]);')
    function('native_fa',[(nq,1),(1,1)],[(nr,1)],'return callback(2,a[0],a[1],0,r[0],0,0);')
    function('jac_native_fa',[(nq,1),(1,1),(nr,1)],[(nr,nq),(nr,1)],
        'return callback(3,a[0],a[1],0,r[0],r[1],0);')
    content=''.join(parts)
    if not source.exists() or source.read_text()!=content or not library.exists():
        source.write_text(content)
        subprocess.run(['cc','-O2','-fPIC','-shared','-Wl,-soname,libfa_native_bridge.so',str(source),'-o',str(library)],check=True)
    shared=ctypes.CDLL(str(library),mode=ctypes.RTLD_GLOBAL)
    pointer=ctypes.POINTER(ctypes.c_double)
    callback_type=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_int,pointer,pointer,pointer,pointer,pointer,pointer)
    def array(p,n):return np.ctypeslib.as_array(p,shape=(n,))
    def callback(mode,x,u,p,r0,r1,r2):
        try:
            if mode<2:
                f,j=stages.physics(array(x,nx),array(u,nu),mode==1)
                if mode==0:
                    if r0:array(r0,ny)[:]=f
                else:
                    if r0:array(r0,ny*nx)[:]=j[:,:nx].ravel(order='F')
                    if r1:array(r1,ny*nu)[:]=j[:,nx:].ravel(order='F')
                    if r2:array(r2,ny)[:]=0.
            else:
                f,g=stages.residual(array(x,nq),int(round(u[0])),mode==3)
                if mode==2:
                    if r0:array(r0,nr)[:]=f
                else:
                    if r0:array(r0,nr*nq)[:]=g.ravel(order='F')
                    if r1:array(r1,nr)[:]=0.
            return 0
        except Exception as error:
            # ctypes otherwise prints and discards callback exceptions.
            stages.error=error
            if r0:r0[0]=float('nan')
            return 1
    handle=callback_type(callback)
    shared.fa_set_callback.argtypes=[callback_type];shared.fa_set_callback(handle)
    return library,shared,handle
