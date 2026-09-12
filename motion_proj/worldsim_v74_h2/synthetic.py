"""冻结P1程序场景；角色采用不重叠的显式PRNG种子区间。"""
import numpy as np
from scipy.spatial.transform import Rotation
from .surface_state import Surface,square,concatenate,initialize,apply
from .first_event_trace import trace,trace_mesh,metrics,objective,point_distance,boundary_neighborhood
def obs_from_surface(truth,rng,query=False):
    # 三个不同传感器原点；返回来自完整真表面最近交点，不能直接把目标采样点当首面。
    v,f=truth.mesh();points=[]
    for i in range(64):
        k=i%len(truth.center);uv=rng.uniform(-.5,.5,2)*np.min(truth.radius[k])
        points.append(truth.center[k]+truth.rotation[k,:,:2]@uv)
    points=np.array(points);o=np.zeros((80,3));o[:,2]=-5
    origins=np.array([[-1.2,0,-5],[.2,.8,-5],[1.1,-.4,-5]]) if query else np.array([[-.7,-.4,-5],[0,0,-5],[.8,.3,-5]])
    o[:]=origins[np.arange(80)%3]
    targets=np.r_[points,np.c_[rng.uniform(-3,3,(16,2)),np.full(16,3.5)]]
    d=targets-o;d/=np.linalg.norm(d,axis=1,keepdims=True)
    blank={'origins_actor_m':o,'directions_actor':d};depth=trace(truth,blank)['first'];positive=np.isfinite(depth)
    background=(3.5-o[:,2])/d[:,2];depth=np.where(positive,depth,background)
    return {**blank,'observed_first_range_m':depth,'points_actor_m':o+depth[:,None]*d,
            'positive_actor':positive,'ambiguous_owner':np.zeros(len(o),bool),'frame_ids':np.arange(80)%3}
def make_scene(family,number,seed):
    rng=np.random.default_rng(seed+number+1000*['multilayer','shared_support','missing_support','grazing_thin'].index(family))
    scale=rng.uniform(.85,1.15)
    if family=='multilayer':
        truth=square([0,0,1.4],1.15*scale)
        initial=concatenate(square([0,0,0],1.15*scale),square([0,0,.65],1.15*scale),truth)
    elif family=='shared_support':
        truth=concatenate(square([-.8,0,0],.7*scale),square([.9,0,1.1],.75*scale))
        initial=truth.copy();initial.radius[0]*=2.2;initial.center[0]+=[.1,0,-.1]
    elif family=='missing_support':
        truth=concatenate(square([-.8,0,.2],.6*scale),square([.8,0,.95],.65*scale,Rotation.from_euler('y',-.35).as_matrix()))
        initial=Surface(truth.center[:1].copy(),truth.rotation[:1].copy(),truth.radius[:1].copy())
    else:
        r=Rotation.from_euler('y',rng.uniform(.9,1.2)).as_matrix()
        truth=concatenate(square([-.2,0,.6],.9*scale,r),square([-.2,0,.63],.9*scale,r))
        initial=truth.copy();initial.center+=rng.normal(0,.04,initial.center.shape);initial.radius*=1.2
    build=obs_from_surface(truth,rng);query=obs_from_surface(truth,rng,True)
    # 对象刚体运动通过完整 SE(3) 坐标变换等变性单独检验，规范资产保持固定。
    return truth,initial,build,query
def save_obs(path,obs):np.savez_compressed(path,**obs)
def surface_quality(s,truth):
    v,f=s.mesh();tv,tf=truth.mesh()
    if not len(f):return {'surface_sample_precision_02':0.,'sampled_wrong_area_m2':0.,'surface_sample_distance_m':None}
    tri=v[f];weights=np.array([[1/3]*3,[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]])
    samples=np.einsum('ac,fcd->fad',weights,tri).reshape(-1,3);dist=point_distance(tv,tf,samples).reshape(-1,4)
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    return {'surface_sample_precision_02':float(np.average((dist<=.2).mean(1),weights=area)),
        'sampled_wrong_area_m2':float((area*(dist>.2).mean(1)).sum()),'surface_sample_distance_m':float(np.average(dist.mean(1),weights=area)),
        'surface_metric_scope':'four fixed quadrature points per triangle; sampled estimate, not exact continuous area'}
