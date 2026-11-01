import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,pathlib,shutil,gc,time
import numpy as np
import torch
torch.set_num_threads(1)
H=pathlib.Path('/root/autodl-tmp/codex_tmp/v43_case15_isolation_20261101_r1');O=H/'legacy_export';O.mkdir(exist_ok=True)
run=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k')
root=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137')
T=np.loadtxt(root/'transform_matrix.txt');s=float(np.linalg.norm(T[:3,0]));local=T.copy();local[:3]/=s
receipt={'source_run':str(run),'gauge':T.tolist(),'local_from_world_metric':local.tolist(),'meter_to_model':s,'source_role':'read-only full original 30K tensor parameters; not repaired10k platform RGB','branches':{},'torch_version':torch.__version__,'new_training_updates':0,'native_model_restored':False}
for branch,name in [('static','chkpnt30000.pth'),('dynamic','dynamic_chkpnt30000.pth')]:
 dest=O/(branch+'.npz')
 if dest.exists():continue
 p=run/name;before=p.stat();start=time.monotonic();payload,iteration=torch.load(str(p),map_location='cpu',mmap=True,weights_only=False)
 assert iteration==30000 and len(payload)==18
 a=lambda i:payload[i].detach().numpy()
 # Same native activations, converted from model units into metric local units.
 q=a(5);q=q/np.linalg.norm(q,axis=1,keepdims=True);w,x,y,z=q.T
 rot=np.empty((len(q),3,3),np.float32)
 rot[:,0,0]=1-2*(y*y+z*z);rot[:,0,1]=2*(x*y-w*z);rot[:,0,2]=2*(x*z+w*y)
 rot[:,1,0]=2*(x*y+w*z);rot[:,1,1]=1-2*(x*x+z*z);rot[:,1,2]=2*(y*z-w*x)
 rot[:,2,0]=2*(x*z-w*y);rot[:,2,1]=2*(y*z+w*x);rot[:,2,2]=1-2*(x*x+y*y)
 arrays=dict(xyz=a(1)/s,rgb=np.rint(np.clip(a(2)[:,0,:]*0.28209479177387814+.5,0,1)*255).astype(np.uint8),opacity=1/(1+np.exp(-a(6))),ids=np.arange(len(q),dtype=np.int32),scales=np.exp(a(4))/s,rotation=rot,t=a(7),sigma_t=np.exp(a(8)),velocity=a(9)/s,cycle=np.asarray(payload[16]))
 assert all(np.isfinite(v).all() for v in arrays.values())
 np.savez_compressed(dest,**arrays)
 after=p.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 receipt['branches'][branch]={'checkpoint':str(p),'iteration':iteration,'rows':len(q),'original_stat_unchanged':True,'elapsed_seconds':time.monotonic()-start,'SH_degree':payload[0],'color_export':'DC only for geometry visualization; not native view-dependent RGB'}
 print(json.dumps(receipt['branches'][branch]),flush=True)
 del payload,arrays,q,rot;gc.collect()
for i in [0,24,48]:shutil.copy2(run/'eval/train_30000_render'/('%03d.png'%i),O/('eval_%03d.png'%i))
(O/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print('LEGACY_EXPORT_COMPLETE',flush=True)
