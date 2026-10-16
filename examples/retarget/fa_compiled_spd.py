"""Small SPD solve with a native analytic VJP, matching FA's Cholesky solve."""
from pathlib import Path
import hashlib
import subprocess
import casadi as cs


def spd_solver(size,directory):
    name=f'fa_spd{size}'
    source=(_SOURCE.replace('SIZE',str(size)).replace('SOLVER',name))
    digest=hashlib.sha256(source.encode()).hexdigest()[:16]
    directory=Path(directory)/('spd_'+digest);directory.mkdir(parents=True,exist_ok=True)
    library=directory/'spd.so'
    if not library.exists():
        path=directory/'spd.c';path.write_text(source)
        subprocess.run(['cc','-O3','-fPIC','-shared',str(path),'-lm','-o',str(library)],check=True)
    return cs.external(name,str(library),{'enable_fd':False,'jac_penalty':-1}),library


_SOURCE=r'''
#include <math.h>
#include <string.h>
typedef long long I;
#define N SIZE
static int factor(const double*A,double*L){
 for(int j=0;j<N;j++)for(int i=j;i<N;i++){
   double value=A[i+N*j];for(int k=0;k<j;k++)value-=L[i+N*k]*L[j+N*k];
   if(i==j){if(!(value>0.0))return 1;L[i+N*j]=sqrt(value);}
   else L[i+N*j]=value/L[j+N*j];
 }
 return 0;
}
static void triangular(const double*L,const double*b,double*x){
 double y[N];
 for(int i=0;i<N;i++){double v=b[i];for(int j=0;j<i;j++)v-=L[i+N*j]*y[j];y[i]=v/L[i+N*i];}
 for(int i=N-1;i>=0;i--){double v=y[i];for(int j=i+1;j<N;j++)v-=L[j+N*i]*x[j];x[i]=v/L[i+N*i];}
}
static int solve(const double*A,const double*b,double*x){
 static _Thread_local double matrices[16][N*N],factors[16][N*N];
 static _Thread_local int valid[16],cursor;
 for(int i=0;i<16;i++)if(valid[i] && !memcmp(matrices[i],A,N*N*sizeof(double))){triangular(factors[i],b,x);return 0;}
 int slot=cursor++%16;double*L=factors[slot];valid[slot]=0;int status=factor(A,L);
 if(status){for(int i=0;i<N;i++)x[i]=NAN;return 0;}
 memcpy(matrices[slot],A,N*N*sizeof(double));valid[slot]=1;
 triangular(L,b,x);return 0;
}
int SOLVER(const double**a,double**r,I*iw,double*w,int mem){double x[N];solve(a[0],a[1],x);if(r[0])memcpy(r[0],x,sizeof(x));return 0;}
I SOLVER_n_in(void){return 2;} I SOLVER_n_out(void){return 1;}
const I* SOLVER_sparsity_in(I i){static const I matrix[]={N,N,1},vector[]={N,1,1};return i==0?matrix:vector;}
const I* SOLVER_sparsity_out(I i){static const I vector[]={N,1,1};return vector;}
int SOLVER_work(I*a,I*r,I*iw,I*w){*a=2;*r=1;*iw=0;*w=0;return 0;}
int adj1_SOLVER(const double**a,double**r,I*iw,double*w,int mem){
 double z[N];solve(a[0],a[3],z);
 if(r[0])for(int j=0;j<N;j++)for(int i=0;i<N;i++)r[0][i+N*j]=-.5*(z[i]*a[2][j]+z[j]*a[2][i]);
 if(r[1])memcpy(r[1],z,sizeof(z));return 0;
}
I adj1_SOLVER_n_in(void){return 4;} I adj1_SOLVER_n_out(void){return 2;}
const I* adj1_SOLVER_sparsity_in(I i){return SOLVER_sparsity_in(i==0?0:1);}
const I* adj1_SOLVER_sparsity_out(I i){return SOLVER_sparsity_in(i);}
int adj1_SOLVER_work(I*a,I*r,I*iw,I*w){*a=4;*r=2;*iw=0;*w=0;return 0;}
'''
