"""Native C++ discrete mesh-feature search for the symbolic FA equations.

Only discrete feature selection lives here. Distances, positions, normals and
their derivatives are expressions in fa_compiled_model, just as the original
native adjoint differentiates its selected features rather than the BVH search.
There is no Python callback in the generated function.
"""
from pathlib import Path
import hashlib
import os
import subprocess
import numpy as np
import casadi as cs


def _table(name, value, dtype='double'):
    a = np.asarray(value)
    values = ','.join(str(int(v)) if dtype == 'int' else format(float(v), '.17g')
                      for v in a.ravel())
    return f'static const {dtype} {name}[]={{' + values + '};\n'


def build_feature_search(compiled, cpu_model, directory):
    collision = compiled.base.collision
    if len(collision.batches) != 1:
        raise ValueError('Compiled FA currently supports one mesh target')
    batch = collision.batches[0]; model = batch.model; target = batch.target.target
    if target.kind != 1 or model.contact_topk != 1:
        raise ValueError('Compiled FA requires an SDF mesh target and contact_topk=1')
    vertices = target.vertices.numpy(); faces = target.faces.numpy()
    # An ordinary static BVH; selection has no derivative. Keep native source
    # ordering and its distance/source tie break. No contact template is used.
    triangles = vertices[faces]; lo = triangles.min(axis=1); hi = triangles.max(axis=1)
    nodes = []; order = []
    def tree(ids):
        index = len(nodes); nodes.append(None)
        lower = lo[ids].min(axis=0); upper = hi[ids].max(axis=0)
        if len(ids) <= 8:
            first = len(order); order.extend(ids.tolist()); children = (-1, -1, first, len(ids))
        else:
            axis = np.argmax(upper-lower); ids = ids[np.argsort((lo[ids,axis]+hi[ids,axis]),kind='stable')]
            middle = len(ids)//2; children = (tree(ids[:middle]),tree(ids[middle:]),0,0)
        nodes[index] = (*lower,*upper,*children)
        return index
    tree(np.arange(len(faces)))
    nodes = np.asarray(nodes)
    sources = model.source_centers.numpy(); slots = batch.source_slot.numpy()
    bodies = model.source_body_ids.numpy()
    slot_bodies = []
    for k in range(batch.source_target_count):
        body = np.unique(bodies[slots == k])
        if len(body) != 1:raise ValueError('Each collision slot must belong to one body')
        slot_bodies.append(int(body[0]))
    source_nodes=[];source_order=[];source_roots=[];radii=model.source_radii.numpy().astype(float)
    source_points=sources.astype(float)
    def source_tree(ids):
        index=len(source_nodes);source_nodes.append(None)
        center=(source_points[ids].min(axis=0)+source_points[ids].max(axis=0))/2
        radius=float(np.max(np.linalg.norm(source_points[ids]-center,axis=1)+radii[ids]))+1e-9
        if len(ids)==1:
            first=len(source_order);source_order.extend(ids.tolist());children=(-1,-1,first,1)
        else:
            axis=np.argmax(np.ptp(sources[ids],axis=0));ids=ids[np.argsort(sources[ids,axis],kind='stable')]
            middle=len(ids)//2;children=(source_tree(ids[:middle]),source_tree(ids[middle:]),0,0)
        source_nodes[index]=(*center,radius,*children)
        return index
    for k in range(batch.source_target_count):source_roots.append(source_tree(np.flatnonzero(slots==k)))
    source_nodes=np.asarray(source_nodes)
    plane = model.plane_geom_id
    # The generated scene uses a world-fixed support plane. Reject moving
    # planes instead of silently freezing their geometry.
    if batch.has_target_plane and cpu_model.geom_bodyid[plane] != 0:
        raise ValueError('Compiled FA requires a world-fixed support plane')
    import mujoco
    plane_rotation = np.empty(9); mujoco.mju_quat2Mat(plane_rotation,cpu_model.geom_quat[plane])
    tables = ''.join([
        _table('vertices',vertices),_table('faces',faces,'int'),
        _table('fnormal',target.face_normals.numpy()),_table('fedges',target.face_edges.numpy(),'int'),
        _table('enormal',target.edge_normals.numpy()),_table('vnormal',target.vertex_normals.numpy()),
        _table('bounds',nodes[:,:6]),_table('nodes',nodes[:,6:],'int'),_table('order',order,'int'),
        _table('sources',sources),_table('radii',model.source_radii.numpy()),
        _table('bodies',bodies,'int'),_table('slots',slots,'int'),
        _table('source_bounds',source_nodes[:,:4]),_table('source_nodes',source_nodes[:,4:],'int'),
        _table('source_order',source_order,'int'),_table('source_roots',source_roots,'int'),_table('slot_bodies',slot_bodies,'int'),
        _table('supports',model.target_spheres.numpy()),
        _table('plane_position',cpu_model.geom_pos[plane]),_table('plane_normal',plane_rotation.reshape(3,3)[:,2]),
    ])
    constants = dict(NBODY=cpu_model.nbody,NSOURCE=len(sources),NSLOT=batch.source_target_count,
                     NCONTACT=collision.contact_count,NSUPPORT=len(model.target_spheres),TARGET=model.target_body_id,
                     PLANE=int(batch.has_target_plane),OFFSET=model.target_contact_offset,NSOURCENODE=len(source_nodes),
                     SOURCE_THREADS=int(os.environ.get('FA_COMPILED_GEOMETRY_THREADS','3')))
    header = ''.join(f'#define {key} {value}\n' for key,value in constants.items())
    header += f'const double threshold={model.threshold:.17g}, offset={float(model.distance_offset.numpy()[0]):.17g};\n'
    source = _CPP.replace('/*TABLES*/',header+tables)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    directory = Path(directory)/('features_'+digest); directory.mkdir(parents=True,exist_ok=True)
    cpp = directory/'features.cpp'; library = directory/'features.so'
    if not library.exists():
        cpp.write_text(source)
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-fopenmp',str(cpp),'-o',str(library)],check=True)
    function = cs.external('fa_features',str(library),{'enable_fd':False})
    return function, slot_bodies, batch, library


_CPP = r'''
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cstdint>
using I=long long;
/*TABLES*/
struct V {double x,y,z; double operator[](int i) const {return i==0?x:(i==1?y:z);} };
V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
V operator*(V a,double s){return {a.x*s,a.y*s,a.z*s};}
double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
double length(V a){return std::sqrt(dot(a,a));}
V get(const double*a,int i){return {a[3*i],a[3*i+1],a[3*i+2]};}
V rot(const double*p,int b,V a){p+=12*b+3;return {p[0]*a.x+p[3]*a.y+p[6]*a.z,p[1]*a.x+p[4]*a.y+p[7]*a.z,p[2]*a.x+p[5]*a.y+p[8]*a.z};}
V invrot(const double*p,int b,V a){p+=12*b+3;return {p[0]*a.x+p[1]*a.y+p[2]*a.z,p[3]*a.x+p[4]*a.y+p[5]*a.z,p[6]*a.x+p[7]*a.y+p[8]*a.z};}
V pos(const double*p,int b){return {p[12*b],p[12*b+1],p[12*b+2]};}
V closest(V p,V a,V b,V c,int &r){
 V ab=b-a,ac=c-a,ap=p-a; double d1=dot(ab,ap),d2=dot(ac,ap);
 if(d1<=0 && d2<=0){r=4;return a;} V bp=p-b;double d3=dot(ab,bp),d4=dot(ac,bp);
 if(d3>=0 && d4<=d3){r=5;return b;} double vc=d1*d4-d3*d2;
 if(vc<=0 && d1>=0 && d3<=0){r=1;return a+ab*(d1/(d1-d3));}
 V cp=p-c;double d5=dot(ab,cp),d6=dot(ac,cp);
 if(d6>=0 && d5<=d6){r=6;return c;} double vb=d5*d2-d1*d6;
 if(vb<=0 && d2>=0 && d6<=0){r=3;return a+ac*(d2/(d2-d6));}
 double va=d3*d6-d5*d4;
 if(va<=0 && d4-d3>=0 && d5-d6>=0){r=2;return b+(c-b)*((d4-d3)/((d4-d3)+(d5-d6)));}
 r=0;double inv=1/(va+vb+vc);return a+ab*(vb*inv)+ac*(vc*inv);
}
double bound(V p,int node){const double*b=bounds+6*node;double d=0;for(int i=0;i<3;i++){double v=std::max({b[i]-p[i],p[i]-b[i+3],0.});d+=v*v;}return d;}
void search(V p,int node,double &best,int &face,int &region,V &point){
 if(bound(p,node)>best)return;
 const int*n=nodes+4*node;
 if(n[0]<0){for(int k=0;k<n[3];k++){int f=order[n[2]+k],r;const int*t=faces+3*f;
 V c=closest(p,get(vertices,t[0]),get(vertices,t[1]),get(vertices,t[2]),r);double d=dot(p-c,p-c);
 if(d<best || (d==best && (face<0 || f<face))){best=d;face=f;region=r;point=c;}}}
 else {int a=n[0],b=n[1];if(bound(p,b)<bound(p,a))std::swap(a,b);search(p,a,best,face,region,point);search(p,b,best,face,region,point);}
}
void feature(V p,int face,int region,V point,V &anchor,V &axis,int &code,double &distance){
 const int*t=faces+face*3;V pseudo=get(fnormal,face);
 if(region>=1 && region<=3)pseudo=get(enormal,fedges[face*3+region-1]);
 else if(region>=4)pseudo=get(vnormal,t[region-4]);
 int sign=dot(p-point,pseudo)<0?-1:1;distance=sign*length(p-point);anchor=point;axis={0,0,0};code=sign;
 if(region==0){anchor=get(vertices,t[0]);axis=get(fnormal,face);code=3;}
 else if(region<=3){anchor=get(vertices,t[region-1]);axis=get(vertices,t[region%3])-anchor;axis=axis*(1/length(axis));code=sign*2;}
}
// Signed mesh distance is 1-Lipschitz. For a source cluster with enclosing
// radius R (including sphere radii), sdf(center)-R is a rigorous lower bound
// on every source distance. This prunes whole source subtrees without changing
// the nearest contact or introducing a sampled/approximate dynamics model.
void select_source_tree(const double*p,int node,int slot,double&bestslot,int&bestsource,double*out){
 const double*b=source_bounds+4*node;const int*n=source_nodes+4*node;int body=slot_bodies[slot];
 V center={b[0],b[1],b[2]};V local=invrot(p,TARGET,pos(p,body)+rot(p,body,center)-pos(p,TARGET));
 double boxgap=std::sqrt(bound(local,0));
 if(boxgap>0 && boxgap-b[3]+offset>bestslot)return;
 double best=std::numeric_limits<double>::infinity();int face=-1,region=0;V point,anchor,axis;int code;double distance;
 static thread_local int hints[NSOURCENODE]={};
 int previous=hints[node]-1;
 if(previous>=0){const int*t=faces+3*previous;point=closest(local,get(vertices,t[0]),get(vertices,t[1]),get(vertices,t[2]),region);best=dot(local-point,local-point);face=previous;}
 search(local,0,best,face,region,point);if(face<0)return;
 hints[node]=face+1;
 feature(local,face,region,point,anchor,axis,code,distance);
 if(distance-b[3]+offset>bestslot)return;
 if(n[0]<0){
   int s=source_order[n[2]];distance+=offset-radii[s];
   if(distance<bestslot || (distance==bestslot && bestsource>=0 && s<bestsource)){
     bestslot=distance;bestsource=s;double*r=out+13*(OFFSET+slot);V src=get(sources,s);
     r[0]=src.x;r[1]=src.y;r[2]=src.z;r[3]=radii[s];r[4]=anchor.x;r[5]=anchor.y;r[6]=anchor.z;
     r[7]=0;r[8]=axis.x;r[9]=axis.y;r[10]=axis.z;r[11]=code;r[12]=1;
   }
 }else{
   int a=n[0],c=n[1];const double*ab=source_bounds+4*a;const double*cb=source_bounds+4*c;
   V al=invrot(p,TARGET,pos(p,body)+rot(p,body,{ab[0],ab[1],ab[2]})-pos(p,TARGET));
   V cl=invrot(p,TARGET,pos(p,body)+rot(p,body,{cb[0],cb[1],cb[2]})-pos(p,TARGET));
   if(dot(cl,cl)<dot(al,al))std::swap(a,c);
   select_source_tree(p,a,slot,bestslot,bestsource,out);select_source_tree(p,c,slot,bestslot,bestsource,out);
 }
}
extern "C" {
int fa_features_uncached(const double**arg,double**res,I*,double*,int){
 const double*p=arg[0];double*out=res[0];if(!out)return 0;std::fill(out,out+13*NCONTACT,0.);
 double bestslot[NSLOT];int bestsource[NSLOT];std::fill(bestslot,bestslot+NSLOT,threshold);std::fill(bestsource,bestsource+NSLOT,-1);
 V meshlo=get(vertices,0),meshhi=meshlo; // root bounds are exact mesh bounds
 meshlo={bounds[0],bounds[1],bounds[2]};meshhi={bounds[3],bounds[4],bounds[5]};
 #if 0 // Kept as a simple reference implementation of the identical search.
 for(int s=0;s<NSOURCE;s++){
   int k=slots[s];if(k<0)continue;
   V local=invrot(p,TARGET,pos(p,bodies[s])+rot(p,bodies[s],get(sources,s))-pos(p,TARGET));
   double reach=std::max(0.,bestslot[k]+radii[s]-offset),gap=0;
   for(int j=0;j<3;j++){double d=std::max({meshlo[j]-local[j],local[j]-meshhi[j],0.});gap+=d*d;}
   if(gap>reach*reach)continue;
   double best=std::numeric_limits<double>::infinity();int face=-1,region=0;V point,anchor,axis;int code;double distance;
   search(local,0,best,face,region,point);if(face<0)continue;
   feature(local,face,region,point,anchor,axis,code,distance);distance+=offset-radii[s];
   if(distance<bestslot[k] || (distance==bestslot[k] && bestsource[k]>=0 && s<bestsource[k])){
     bestslot[k]=distance;bestsource[k]=s;double*r=out+13*(OFFSET+k);V src=get(sources,s);
     r[0]=src.x;r[1]=src.y;r[2]=src.z;r[3]=radii[s];r[4]=anchor.x;r[5]=anchor.y;r[6]=anchor.z;
     r[7]=0;r[8]=axis.x;r[9]=axis.y;r[10]=axis.z;r[11]=code;r[12]=1;
   }
 }
 #endif
 #pragma omp parallel for num_threads(SOURCE_THREADS) if(SOURCE_THREADS>1)
 for(int slot=0;slot<NSLOT;slot++)select_source_tree(p,source_roots[slot],slot,bestslot[slot],bestsource[slot],out);
 if(PLANE){
   V normal=get(plane_normal,0),localnormal=invrot(p,TARGET,normal);
   double limit=-dot(pos(p,TARGET)-get(plane_position,0),normal)-offset;
   int chosen[4]={-1,-1,-1,-1},count=0;
   for(int stage=0;stage<4;stage++){
     double best=std::numeric_limits<double>::infinity();int selected=-1;
     for(int s=0;s<NSUPPORT;s++){
       V c={supports[4*s],supports[4*s+1],supports[4*s+2]};double radius=supports[4*s+3];
       double distance=dot(c,localnormal)-radius;if(distance>=limit)continue;
       bool used=false;for(int j=0;j<count;j++)used|=chosen[j]==s;if(used)continue;
       double value=distance;
       if(stage){V pt=c-localnormal*radius;double separation=std::numeric_limits<double>::infinity();
         for(int j=0;j<count;j++){int id=chosen[j];V q={supports[4*id],supports[4*id+1],supports[4*id+2]};q=q-localnormal*supports[4*id+3];V d=pt-q;separation=std::min(separation,std::max(0.,dot(d,d)-std::pow(dot(d,localnormal),2)));}value=-separation;}
       if(value<best || (value==best && s<selected)){best=value;selected=s;}
     }
     if(selected<0)break;chosen[count++]=selected;
   }
   std::sort(chosen,chosen+count);
   for(int j=0;j<count;j++){double*r=out+13*j;int s=chosen[j];for(int k=0;k<4;k++)r[k]=supports[4*s+k];r[12]=1;}
 }
 return 0;
}
int fa_features(const double**arg,double**res,I*iw,double*w,int mem){
 if(!res[0])return 0;
 struct Cache {
   double input[64][12*NBODY],output[64][13*NCONTACT];
   uint64_t hash[64]={};bool valid[64]={};int cursor=0;
 };
 static thread_local Cache cache;
 uint64_t hash=1469598103934665603ULL;
 for(int i=0;i<12*NBODY;i++){uint64_t bits;std::memcpy(&bits,arg[0]+i,8);hash=(hash^bits)*1099511628211ULL;}
 for(int i=0;i<64;i++)if(cache.valid[i] && cache.hash[i]==hash && !std::memcmp(cache.input[i],arg[0],12*NBODY*sizeof(double))){
   std::memcpy(res[0],cache.output[i],13*NCONTACT*sizeof(double));return 0;
 }
 int status=fa_features_uncached(arg,res,iw,w,mem);if(status)return status;
 int i=cache.cursor++%64;cache.valid[i]=true;cache.hash[i]=hash;
 std::memcpy(cache.input[i],arg[0],12*NBODY*sizeof(double));std::memcpy(cache.output[i],res[0],13*NCONTACT*sizeof(double));
 return 0;
}
I fa_features_n_in(){return 1;} I fa_features_n_out(){return 1;}
const I* fa_features_sparsity_in(I){static const I sp[]={12*NBODY,1,1};return sp;}
const I* fa_features_sparsity_out(I){static const I sp[]={13,NCONTACT,1};return sp;}
int fa_features_work(I*a,I*r,I*iw,I*w){*a=1;*r=1;*iw=0;*w=0;return 0;}
// Piecewise-constant discrete feature identity. Continuous feature geometry
// and its derivatives are evaluated symbolically by the caller.
int jac_fa_features(const double**,double**,I*,double*,int){return 0;}
I jac_fa_features_n_in(){return 2;} I jac_fa_features_n_out(){return 1;}
const I* jac_fa_features_sparsity_in(I i){return i==0?fa_features_sparsity_in(0):fa_features_sparsity_out(0);}
const I* jac_fa_features_sparsity_out(I){static I sp[3+12*NBODY]={13*NCONTACT,12*NBODY,0};return sp;}
int jac_fa_features_work(I*a,I*r,I*iw,I*w){*a=2;*r=1;*iw=0;*w=0;return 0;}
}
'''
