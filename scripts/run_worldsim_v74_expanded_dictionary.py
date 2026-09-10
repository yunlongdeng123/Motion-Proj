"""计划规定的扩大字典整数参考；只诊断候选缺口，不计新方法或部署性能。"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.c_dcs import geometry_context,patches_from_parameters,concat_patches,patch_incidence,patch_cost,master,export_patches
from motion_proj.worldsim_v74.common import write_json,save_mesh,evaluate,ray_state
parser=argparse.ArgumentParser();parser.add_argument('--mechanisms',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
manifest=json.loads((args.mechanisms/'manifest.json').read_text());cfg=manifest['config'];width=cfg['carrier_half_width_m'];eps=cfg['epsilon_obs_m'];rows=[];start=time.monotonic()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
write_json(args.output/'manifest.json',{'source':str(args.mechanisms),'task':'WS-V74-C-EXPANDED-DICTIONARY-REFERENCE-01',
    'dictionary':'all saved proposals from DCS/C1/C2/C3 plus BUILD-PCA patches at every positive input point and scales exp(-.7),1,exp(.7); all drawn from existing parameter bounds',
    'boundary':'post-hoc BUILD-only integer reference with enlarged pool; not a fourth candidate, not deployment performance, no QUERY or GT geometry in construction',
    'solver':'same HiGHS master; max300s, mip_rel_gap1e-5, actual final gap/status retained','config':cfg})
for case in sorted(p for p in args.mechanisms.iterdir() if p.is_dir() and (p/'build.npz').exists()):
    out=args.output/case.name;out.mkdir();build=arrays(case/'build.npz');context=geometry_context(build);parts=[]
    for scale in [-.7,0.,.7]:
        param=np.zeros((len(context['points']),7));param[:,5:7]=scale
        parts.append(patches_from_parameters(context,np.arange(len(param)),param,width))
    for method in ['DCS','C1_residual','C2_local_network','C3_no_demand']:
        for p in sorted((case/method).glob('pricing_*.npz')):
            z=arrays(p);parts.append(patches_from_parameters(context,z['anchors'],z['parameters'],width))
    patches=parts[0]
    for p in parts[1:]:patches=concat_patches(patches,p)
    t=time.monotonic();A,E=patch_incidence(patches,build,eps);cost=patch_cost(patches,width);solution=master(A,E,cost,cfg['face_budget'],True,seconds=300)
    selected=np.flatnonzero(solution['z']>.5);v,f=export_patches(patches,selected);save_mesh(out/'final.npz',v,f,**patches,selected=selected)
    query=arrays(case/'query.npz')
    physical=ray_state(v,f,build,eps);owned=context['owned'];row={'case_id':case.name,'candidate_count':len(cost),'selected_count':len(selected),'faces':len(f),
        'integer_objective':solution['objective'],'integer_gap':solution['gap'],'solver_status':solution['status'],'solver_seconds':time.monotonic()-t,
        'all_build_feasible':bool(physical['hit'][owned].all() and not physical['early'].any()),'unexplained_build_rays':int((~physical['hit'][owned]).sum()),'early_build_rays':int(physical['early'].sum()),
        'metrics_build':evaluate(v,f,build,out/'build_rays.npz'),'metrics_query':evaluate(v,f,query,out/'query_rays.npz')}
    write_json(out/'result.json',row);rows.append(row);write_json(args.output/'results.json',rows);print(case.name,'candidates',len(cost),'status',solution['status'],'gap',solution['gap'],flush=True)
write_json(args.output/'resources.json',{'wall_seconds':time.monotonic()-start,'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3})
