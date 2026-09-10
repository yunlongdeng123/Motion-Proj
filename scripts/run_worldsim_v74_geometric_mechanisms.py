"""三维机制队列：固定真表面、BUILD/QUERY分离、同候选控制和双向几何取证。"""
import argparse,json,sys,time,resource,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.common import all_intersections,save_mesh,write_json,evaluate
from motion_proj.worldsim_v73.surface_readout import closest_surface_points


def rectangle(center,u,v):
    center=np.asarray(center);u=np.asarray(u);v=np.asarray(v)
    return np.array([center-u-v,center+u-v,center+u+v,center-u+v]),np.array([[0,1,2],[0,2,3]])


def scene(family,rng):
    vs=[];fs=[]
    if family=='multi_front':
        pieces=[([-.4,0,.3],[.35,0,.08],[0,.7,0]),([.05,0,0],[.4,0,-.08],[0,.65,0]),([.5,0,-.3],[.3,0,.03],[0,.7,0])]
    elif family=='shared_conflict':
        pieces=[([-.25,0,.08],[.5,0,.1],[0,.65,0]),([.25,0,-.08],[.5,0,-.1],[0,.65,0])]
    elif family=='missing_support':
        pieces=[([x,y,.22*np.sin(x*4+y*2)],[.105,0,0],[0,.105,0]) for x in np.linspace(-.7,.7,5) for y in np.linspace(-.7,.7,5)]
    else:
        pieces=[([0,0,0],[.8,0,.06],[0,.08,0]),([.3,.3,.15],[.08,0,0],[0,.4,.1]),([-.5,-.4,-.1],[.2,0,0],[0,.16,0])]
    angle=rng.uniform(-.3,.3);R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    scale=rng.uniform(.8,1.2);shift=rng.uniform(-.03,.03,3)
    for center,u,v in pieces:
        vv,ff=rectangle(center,u,v);vv=scale*vv@R.T+shift;fs.extend((ff+len(vs)).tolist());vs.extend(vv.tolist())
    return np.asarray(vs),np.asarray(fs,int)


def observations(v,f,rng,query=False,redundant=False):
    vv,bg=rectangle([0,0,-1.2],[2,0,0],[0,2,0]);nv=len(v);allv=np.vstack([v,vv]);allf=np.vstack([f,bg+nv])
    count=96;xy=rng.uniform(-.9,.9,(count,2))
    if redundant and not query:xy[:72]=rng.normal([0,0],[.25,.035],(72,2))
    target=np.c_[xy,np.zeros(count)]
    if query:origins=np.tile([.6,-.4,2.2],(count,1))
    else:
        origins=np.tile([-.4,.2,2.2],(count,1));origins[count//2:]=[.35,.2,2.2]
    d=target-origins;d/=np.linalg.norm(d,axis=1,keepdims=True)
    hits=all_intersections(allv,allf,origins,d)
    depth=np.full(count,np.inf);owner=np.full(count,-1,int)
    for k in np.argsort(hits['t'])[::-1]:depth[hits['ray'][k]]=hits['t'][k];owner[hits['ray'][k]]=hits['face'][k]
    good=np.isfinite(depth);origins=origins[good];d=d[good];depth=depth[good];owner=owner[good]
    positive=(owner>=0)&(owner<len(f));points=origins+depth[:,None]*d
    return {'origins_actor_m':origins.astype(np.float32),'directions_actor':d.astype(np.float32),'observed_first_range_m':depth.astype(np.float32),
        'points_actor_m':points.astype(np.float32),'positive_actor':positive,'ambiguous_owner':np.zeros(len(depth),bool),
        'support_points_actor_m':points[positive].astype(np.float32),'size_lwh_m':np.array([2.4,2.4,1.8],np.float32),
        'frame_offsets':np.array([0,len(depth)//2,len(depth)]),'metadata':{'dataset':'synthetic','sensor':'ideal opaque first return','ownership':'exact mesh identity'}}


@torch.no_grad()
def geometric_distance(v,f,gtv,gtf):
    def samples(v,f):
        if not len(f):return torch.empty((0,3),device='cuda')
        tri=torch.tensor(v[f],dtype=torch.float32,device='cuda')
        return torch.cat([tri.mean(1),tri[:,0]*.6+tri[:,1]*.2+tri[:,2]*.2,tri[:,0]*.2+tri[:,1]*.6+tri[:,2]*.2,tri[:,0]*.2+tri[:,1]*.2+tri[:,2]*.6])
    if not len(f):return {'surface_to_gt_m':None,'gt_to_surface_m':None,'gt_recall_02':0.}
    pp=samples(v,f);gg=samples(gtv,gtf)
    near=closest_surface_points(torch.tensor(gtv,dtype=torch.float32,device='cuda'),torch.tensor(gtf,device='cuda'),pp)
    reverse=closest_surface_points(torch.tensor(v,dtype=torch.float32,device='cuda'),torch.tensor(f,device='cuda'),gg)
    def weights(vertices,faces):
        tri=np.asarray(vertices)[faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        return torch.tensor(np.tile(area,4)/max(area.sum()*4,1e-12),device='cuda')
    wp=weights(v,f);wg=weights(vgt,fgt)
    return {'surface_to_gt_m':float(((pp-near).norm(dim=1)*wp).sum()),'gt_to_surface_m':float(((gg-reverse).norm(dim=1)*wg).sum()),'gt_recall_02':float((((gg-reverse).norm(dim=1)<=.2).float()*wg).sum())}


parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--reuse',type=Path);parser.add_argument('--reuse-methods',nargs='*',default=[])
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cfg=json.loads((ROOT/'configs/worldsim_v74/tournament.json').read_text())
params={**cfg,**cfg['dataset_parameters']['nuscenes'],'dcs_checkpoints':cfg['dcs_checkpoints']['nuscenes']}
params['epsilon_obs_m']=.02;params['carrier_half_width_m']=.2
params['wex']={**cfg['wex'],'wall_seconds':30};params['rif']={**cfg['rif'],'initial_grid':5,'wall_seconds':30};params['dcs']={**cfg['dcs'],'initial_patches':4,'wall_seconds':30}
write_json(args.output/'manifest.json',{'task':'WS-V74-GEOMETRIC-MECHANISMS-01','seed':7401,'methods':args.methods,'config':params,
    'scope':'four geometric families x20, independently generated BUILD and QUERY rays; same full method implementations/controls',
    'limits':'shared_conflict geometry does not presume that every instance requires a joint exchange; exact mixed reference diagnoses this. Domain value scaling remains an analytic limitation. C uses fixed nuScenes FIT network on analytic geometry, with matched no-demand control.',
    'failure_ledger_refs':cfg['failure_ledger_refs']})
rows=[];start=time.monotonic();rng=np.random.default_rng(7401)
for family in ['multi_front','shared_conflict','missing_support','redundant_thin']:
    for trial in range(20):
        case=args.output/f'{family}_{trial:02d}';case.mkdir();vgt,fgt=scene(family,rng)
        build=observations(vgt,fgt,rng,redundant=family=='redundant_thin');query=observations(vgt,fgt,rng,query=True)
        save_mesh(case/'ground_truth.npz',vgt,fgt)
        for name,data in [('build',build),('query',query)]:np.savez_compressed(case/f'{name}.npz',**{k:v for k,v in data.items() if k!='metadata'})
        for method in args.methods:
            out=case/method
            previous=args.reuse/case.name/method if args.reuse and method in args.reuse_methods else None
            if previous is not None and (previous/'metrics.json').exists():
                row=json.loads((previous/'metrics.json').read_text());row['artifact_source']=str(previous)
                out.symlink_to(previous,target_is_directory=True);rows.append(row);continue
            try:
                if method=='WEX' or method.startswith('A'):from motion_proj.worldsim_v74.a_wex import reconstruct
                elif method=='RIF' or method.startswith('B'):from motion_proj.worldsim_v74.b_rif import reconstruct
                else:from motion_proj.worldsim_v74.c_dcs import reconstruct
                v,f,record=reconstruct(build,params,method,out)
                metrics=evaluate(v,f,query,out/'query_rays.npz');metrics_build=evaluate(v,f,build,out/'build_rays.npz')
                geo=geometric_distance(v,f,vgt,fgt)
                row={'family':family,'trial':trial,'method':method,'record':record,'metrics_query':metrics,'metrics_build':metrics_build,'geometry':geo}
                write_json(out/'metrics.json',row);rows.append(row)
            except Exception as exc:
                out.mkdir(exist_ok=True);(out/'error.txt').write_text(traceback.format_exc());rows.append({'family':family,'trial':trial,'method':method,'status':'engineering_error','error':str(exc)})
                print(traceback.format_exc(),flush=True)
        write_json(args.output/'results.json',rows)
        print(family,trial,'complete',len(rows),flush=True)
write_json(args.output/'resources.json',{'wall_seconds':time.monotonic()-start,'rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3})
