import pathlib,json,time,traceback,torch,numpy as np,nksr,subprocess
P=pathlib.Path('/root/autodl-tmp');S=P/'runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1/secondary';out=S/'nksr';out.mkdir(exist_ok=True)
torch.set_num_threads(4);torch.manual_seed(7403)
model=nksr.Reconstructor('cuda',config={'parent':'ks','url':str(P/'third_party/NKSR-v74/checkpoints/ks.pth')});rows=[]
for e in json.loads((S/'build_manifest.json').read_text()):
 dst=out/(e['scene']+'__'+e['owner']);dst.mkdir(exist_ok=True);rf=dst/'result.json'
 if rf.exists():rows.append(json.loads(rf.read_text()));continue
 t=time.time();rec={'scene':e['scene'],'log':e['log_id'],'owner':e['owner'],'input':e['input'],'method':'nksr','input_role':'BUILD-only LiDAR + PCA16 toward nearest BUILD sensor','query_access':False}
 try:
  a=np.load(e['input']);p=a['points'];n=a['normals']
  if len(p)<16:rec.update(status='INSUFFICIENT_PCA16_INPUT',points=len(p))
  else:
   torch.cuda.reset_peak_memory_stats()
   with torch.inference_mode():
    field=model.reconstruct(torch.tensor(p,device='cuda'),normal=torch.tensor(n,device='cuda'),voxel_size=.1,detail_level=0.,solver_max_iter=2000,solver_tol=1e-5)
    if field is None:v=np.empty((0,3),'float32');f=np.empty((0,3),'uint32');status='EMPTY_NATIVE_FIELD'
    else:
     mesh=field.extract_dual_mesh(mise_iter=1);v=mesh.v.cpu().numpy();f=mesh.f.cpu().numpy();status='DONE'
   np.savez_compressed(dst/'surface.npz',vertices=v,faces=f);rec.update(status=status,points=len(p),vertices=len(v),faces=len(f),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
   del field;torch.cuda.empty_cache()
 except Exception as ex:rec.update(status='ENGINEERING_ERROR',error=str(ex));(dst/'error.txt').write_text(traceback.format_exc());torch.cuda.empty_cache()
 rec['elapsed_s']=time.time()-t;rf.write_text(json.dumps(rec,indent=2));rows.append(rec);(out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rec),flush=True)
(out/'complete.json').write_text(json.dumps({'results':len(rows),'statuses':{s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)}},indent=2))
