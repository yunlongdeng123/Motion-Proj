"""H2 E0 与 E1 CPU 强控制：真实三维场景，不代替未训练 A 的必要性比较。"""
import argparse,itertools,json,os,sys,time,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74_h2.surface_state import Surface,square,concatenate,initialize,apply
from motion_proj.worldsim_v74_h2.first_event_trace import trace,trace_mesh,metrics,objective,point_distance,boundary_neighborhood
from motion_proj.worldsim_v74_h2.controls import search
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
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
def one_case(arg):
    folder,family,number,seed,steps,budget=arg;folder=Path(folder)/f'{family}-{number:02d}';folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();truth,initial,build,query=make_scene(family,number,seed)
    truth.save(folder/'truth.npz');initial.save(folder/'initial.npz');save_obs(folder/'build.npz',build);save_obs(folder/'query.npz',query)
    init_trace=trace(initial,build);np.savez_compressed(folder/'initial_build_chain.npz',**init_trace,**boundary_neighborhood(initial,build))
    variants={'INITIAL':initial,'TRUTH_ASSISTED':truth}
    tiny=initialize(build,count=min(512,int(build['positive_actor'].sum())),width=1e-4);variants['C6_TINY']=tiny
    solver={}
    for method in ['C1','C3']:
        variants[method],solver[method]=search(initial,build,method,steps,budget);write(folder/f'{method}_events.json',solver[method])
        state=initial.copy();first_deg=None;previous=metrics(state,query)
        for i,event in enumerate(solver[method]['selected_events']):
            before=state;state=apply(state,event);state.save(folder/f'{method}_step-{i:02d}.npz')
            cur=metrics(state,query)
            if first_deg is None and (cur['hit']<previous['hit']-.005 or cur['early']>previous['early']+.005):
                first_deg=i;before.save(folder/f'{method}_first_degradation_before.npz');state.save(folder/f'{method}_first_degradation_after.npz')
            previous=cur
        solver[method]['first_query_degradation_step']=first_deg
    results=[]
    for name,s in variants.items():
        s.save(folder/f'{name}.npz');np.savez_compressed(folder/f'{name}_query_chain.npz',**trace(s,query))
        results.append({'case':folder.name,'family':family,'method':name,'build':metrics(s,build),'query':metrics(s,query),
            **surface_quality(s,truth),'asset':str(folder/f'{name}.npz'),
            'evaluations':solver.get(name,{}).get('evaluations',0),'first_degradation':solver.get(name,{}).get('first_query_degradation_step')})
    write(folder/'results.json',results)
    # 真实标签由独立平面矩形面片组成；这里检验三角扇与矩形二三角导出的一致性。
    tv,tf=truth.mesh();rf=[]
    for k in range(len(truth.center)):
        rf.extend([[k*9+2,k*9+4,k*9+6],[k*9+2,k*9+6,k*9+8]])
    ideal=trace_mesh(tv,np.asarray(rf),query['origins_actor_m'],query['directions_actor'],np.repeat(np.arange(len(truth.center)),2))['first']
    fan=trace(truth,query)['first'];mask=np.isfinite(ideal)&np.isfinite(fan)
    r=Rotation.from_euler('xyz',[.3,-.2,.4]).as_matrix();t=np.array([1.1,-2.2,.7])
    shifted={**query,'origins_actor_m':query['origins_actor_m']@r.T+t,'directions_actor':query['directions_actor']@r.T}
    transformed=trace(truth.transform(r,t),shifted)['first'];valid=np.isfinite(fan)&np.isfinite(transformed)
    f32=trace(truth,query,np.float32)['first'];valid32=np.isfinite(fan)&np.isfinite(f32)
    diagnostic={'case':folder.name,'rectangle_vs_fan_miss_disagreements':int(np.sum(np.isfinite(ideal)!=np.isfinite(fan))),
        'rectangle_vs_fan_max_m':float(np.max(np.abs(ideal[mask]-fan[mask]))) if mask.any() else None,
        'rigid_query_max_m':float(np.max(np.abs(fan[valid]-transformed[valid]))) if valid.any() else None,
        'float32_max_m':float(np.max(np.abs(fan[valid32]-f32[valid32]))) if valid32.any() else None,
        'float32_miss_disagreements':int(np.sum(np.isfinite(fan)!=np.isfinite(f32))),
        'max_initial_chain':int(np.diff(init_trace['offsets']).max()),'wall_s':time.monotonic()-start}
    write(folder/'diagnostic.json',diagnostic);return results,diagnostic
def e0(out,seed):
    # 六种必要几何；精确方片铺设是达到的表示能力，绝不是整个连续问题的最优界。
    ry=Rotation.from_euler('y',.6).as_matrix()
    scenes={'plane':square([0,0,.5],1.),'fold':concatenate(square([-.8,0,.5],.8,ry),square([.8,0,.5],.8,ry.T)),
        'thin_plate':concatenate(square([0,0,.5],1),square([0,0,.52],1)),
        'layers':concatenate(square([0,0,.4],.5),square([0,0,1.2],1.)),
        'narrow_slit':concatenate(square([-1.05,0,.5],1),square([1.05,0,.5],1)),
        'partial_occlusion':concatenate(square([-.4,0,.3],.45),square([.2,0,1.],1.))}
    rows=[];rng=np.random.default_rng(seed)
    for name,s in scenes.items():
        q=obs_from_surface(s,rng,True);rows.append({'family':name,'attained':metrics(s,q),'oracle_knowledge':'known square surface components; no geometry inferred from BUILD'})
    # 两个前层后方的几何世界，前视 BUILD 完全相同，从后方 QUERY 明显不同。
    a=concatenate(square([0,0,0],1),square([0,0,1.],.8));b=concatenate(square([0,0,0],1),square([0,0,2.],.8))
    xx,yy=np.meshgrid(np.linspace(-.5,.5,7),np.linspace(-.5,.5,7));xy=np.c_[xx.ravel(),yy.ravel()]
    front={'origins_actor_m':np.c_[xy,np.full(len(xy),-4.)],'directions_actor':np.tile([0,0,1.],(len(xy),1))}
    back={'origins_actor_m':np.c_[xy,np.full(len(xy),4.)],'directions_actor':np.tile([0,0,-1.],(len(xy),1))}
    ambiguity={'build_max_difference_m':float(np.max(np.abs(trace(a,front)['first']-trace(b,front)['first']))),
        'query_mean_difference_m':float(np.mean(np.abs(trace(a,back)['first']-trace(b,back)['first']))),
        'scope':'same observed input admits two worlds; statistical priors remain open'}
    a.save(out/'ambiguity_world_a.npz');b.save(out/'ambiguity_world_b.npz');save_obs(out/'ambiguity_build_rays.npz',front);save_obs(out/'ambiguity_query_rays.npz',back)
    truth,initial,build,query=make_scene('multilayer',0,seed);library=[]
    for factors in itertools.product([1.,.05],repeat=2):
        s=initial.copy();s.radius[:2]*=np.array(factors)[:,None]
        library.append({'factors':factors,'objective':objective(s,build),'query':metrics(s,query)})
    best=min(library,key=lambda x:x['objective'])
    result={'task':'WS-V74-H2-E0-01','geometry_cases':rows,'ambiguity':ambiguity,
        'finite_library':{'size':len(library),'enumerated':library,'best':best,'claim':'exact minimum only in four frozen shrink states; not continuous bound'},
        'capacity_touched':False,'status':'attained_capacity_and_ambiguity_documented'}
    write(out/'e0.json',result);return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seed',type=int,default=7411);p.add_argument('--per-family',type=int,default=20);p.add_argument('--steps',type=int,default=8);p.add_argument('--events',type=int,default=64);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    config={'task':'WS-V74-H2-E1-CPU-01','seed':a.seed,'scenes_per_family':a.per_family,'steps':a.steps,'events_per_step':a.events,
        'families':['multilayer','shared_support','missing_support','grazing_thin'],'scale_uniform':[.85,1.15],
        'initialization':'predeclared family-specific controlled corruption; real-data PCA initialization is separate',
        'rays':'64 sampled target rays +16 wider rays, 3 BUILD /3 distinct QUERY origins, true whole-surface first hit',
        'A_trained':False,'C2_trained':False,'status':'cpu_controls_only','failure_ledger_refs':['V74-F04','V74-F09','V74-F10'],
        'failure_ledger_delta':'pending','gpu_used':False,'host':platform.node(),'source':'local CPU work then copied to AutoDL'}
    write(a.output/'manifest.json',config);e0(a.output,a.seed)
    jobs=[(str(a.output),f,i,a.seed,a.steps,a.events) for f in config['families'] for i in range(a.per_family)]
    results=[];diags=[]
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        for rows,diag in executor.map(one_case,jobs):
            results.extend(rows);diags.append(diag);print(json.dumps({'completed':len(diags),'total':len(jobs),'wall_s':time.monotonic()-start}),flush=True)
    summary=[]
    for family in config['families']:
        for method in ['INITIAL','TRUTH_ASSISTED','C6_TINY','C1','C3']:
            rr=[r for r in results if r['family']==family and r['method']==method]
            summary.append({'family':family,'method':method,'cases':len(rr),
                'build':{k:float(np.mean([r['build'][k] for r in rr])) for k in ['hit','early','miss','recall_02','free_m','any_correct']},
                'query':{k:float(np.mean([r['query'][k] for r in rr])) for k in ['hit','early','miss','recall_02','free_m','any_correct']}})
    write(a.output/'results.json',results);write(a.output/'summary.json',{'rows':summary,'diagnostics':diags,'wall_s':time.monotonic()-start,
        'A_verdict':None,'P1_pass':None,'reason':'A/C2 learned dynamics not evaluated; strong-control references ready'})
if __name__=='__main__':main()
