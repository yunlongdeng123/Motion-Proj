"""官方NKSR原生几何对照；BUILD点/PCA法向，固定ks先验和0.1m体素。"""
import argparse,json,sys,time,resource,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
import nksr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.data import load_build
from motion_proj.worldsim_v74.common import write_json,save_mesh,evaluate
from motion_proj.worldsim_v74.c_dcs import geometry_context
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['fit','probe'],required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic()
cfg=json.loads((ROOT/'configs/worldsim_v74/tournament.json').read_text());source=Path('/root/autodl-tmp/third_party/NKSR-v74');ckpt=source/'checkpoints/ks.pth'
if args.phase=='fit':cases=json.loads((ROOT/'docs/autoresearch/worldsim_v74/a_wex/fit_pilot_manifest.json').read_text())['cases']
else:cases=json.loads((Path(cfg['data_root'])/'probe_cohort.json').read_text())['cases']
method='NKSR_ks_native';rows=[]
write_json(args.output/'manifest.json',{'task':'WS-V74-NKSR-REFERENCE-01','phase':args.phase,'methods':[method],'cases':cases,'config':cfg,
    'implementation':str(source),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
    'checkpoint':str(ckpt),'checkpoint_origin':'https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/checkpoints/ks.pth',
    'torch_version':torch.__version__,'nksr_version':nksr.__version__,'normal':'same BUILD PCA16, oriented toward measured sensor; no heldout normal or point',
    'fixed_native_settings':{'config':'ks','voxel_size_m':.1,'detail_level':0.,'detail_level_note':'explicit voxel size overrides detail_level in official API','mise_iter':1,'solver_max_iter':2000,'solver_tol':1e-5},
    'training_boundary':'official pretrained general reconstruction prior, not retrained on either dataset; this is a common external reference, not evidence of V74 cross-dataset generalization',
    'budget_boundary':'native triangle count retained even above4096 and marked; no post-hoc decimation or hidden crop',
    'query_loaded':False,'failure_ledger_refs':cfg['failure_ledger_refs']+['V74-F06']})
model=nksr.Reconstructor('cuda',config={'parent':'ks','url':str(ckpt)})
for case in cases:
    folder=args.output/case['dataset']/case['case_id']/method;folder.mkdir(parents=True)
    b=load_build(Path(case['build_file']).parent);t=time.monotonic()
    try:
        context=geometry_context(b);xyz=torch.tensor(context['points'],dtype=torch.float32,device='cuda');normal=torch.tensor(context['frame'][:,:,2],dtype=torch.float32,device='cuda')
        np.savez_compressed(folder/'input.npz',points_actor_m=context['points'],normals_actor=context['frame'][:,:,2])
        field=model.reconstruct(xyz,normal=normal,voxel_size=.1,detail_level=0.,solver_max_iter=2000,solver_tol=1e-5)
        if field is None:v=np.empty((0,3));f=np.empty((0,3),int);status='official_reconstructor_returned_none'
        else:
            mesh=field.extract_dual_mesh(mise_iter=1);v=mesh.v.cpu().numpy();f=mesh.f.cpu().numpy();status='complete'
        torch.cuda.synchronize();seconds=time.monotonic()-t;save_mesh(folder/'final.npz',v,f)
        record={'status':status,'method':method,'wall_seconds':seconds,'faces':len(f),'vertices':len(v),'input_points':len(xyz),'face_budget_compliant':len(f)<=cfg['face_budget']}
        if args.phase=='fit':write_json(folder/'build_metrics.json',evaluate(v,f,b,folder/'build_rays.npz'))
        write_json(folder/'reconstruction.json',record)
        del field;torch.cuda.empty_cache()
    except Exception as e:
        record={'status':'engineering_error','error':str(e),'wall_seconds':time.monotonic()-t};(folder/'error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
        torch.cuda.empty_cache()
    rows.append({'dataset':case['dataset'],'log_id':case['log_id'],'case_id':case['case_id'],'method':method,'record':record,'folder':str(folder)})
    write_json(args.output/'assets.json',rows);write_json(args.output/'progress.json',{'completed':len(rows),'total':len(cases),'last':rows[-1]})
    print(case['case_id'],json.dumps(record),flush=True)
write_json(args.output/'resources.json',{'wall_seconds':time.monotonic()-start,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,'rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'cuda_peak_gib':torch.cuda.max_memory_allocated()/1024**3})
