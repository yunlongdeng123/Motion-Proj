import pathlib,json,time,traceback,sys,os,torch,numpy as np
P=pathlib.Path('/root/autodl-tmp');S=P/'runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1/secondary';repo=P/'external/worldsim_v74_mainfig/noksr';sys.path.insert(0,str(repo));os.chdir(repo)
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from noksr.model import noksr
from noksr.model.module import Generator
torch.set_num_threads(4);torch.manual_seed(7403);np.random.seed(7403)
with initialize_config_dir(version_base=None,config_dir=str(repo/'config')):cfg=compose(config_name='config',overrides=['model=carla_model','data=carla_original','data.reconstruction.gt_mask=False','data.reconstruction.gt_sdf=False','model.network.point_transformerv3.enable_flash=False'])
out=S/'noksr';out.mkdir(exist_ok=True);(out/'config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
model=noksr(cfg);ckpt=P/'models/worldsim_v74_mainfig/noksr/Carla_Serial_best.ckpt';state=torch.load(ckpt,map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict'],strict=True);del state;model=model.eval().cuda()
generator=Generator(model.sdf_decoder,model.mask_decoder if cfg.data.reconstruction.trim else None,cfg.data.voxel_size,cfg.model.network.sdf_decoder.k_neighbors,cfg.model.network.sdf_decoder.last_n_layers,cfg.data.reconstruction)
print('MODEL_LOADED',flush=True);rows=[]
for e in json.loads((S/'build_manifest.json').read_text()):
 dest=out/(e['scene']+'__'+e['owner']);dest.mkdir(exist_ok=True);rf=dest/'result.json'
 if rf.exists():rows.append(json.loads(rf.read_text()));continue
 start=time.time();rec={'scene':e['scene'],'log':e['log_id'],'owner':e['owner'],'input':e['input'],'method':'noksr','query_access':False,'strict_checkpoint':True,'checkpoint':str(ckpt),'input_role':'Same BUILD-only LiDAR and PCA16 as NKSR; no dense GT fields supplied','attention_backend':'Official enable_flash=False configuration, FP32 PyTorch attention; installed FlashAttention CUDA ABI does not match cu118 NKSR environment'}
 try:
  a=np.load(e['input']);p=a['points'];n=a['normals']
  if len(p)<16:rec.update(status='INSUFFICIENT_PCA16_INPUT',points=len(p))
  else:
   torch.cuda.reset_peak_memory_stats();xyz=torch.tensor(p,device='cuda');feat=torch.tensor(np.concatenate([n,p],axis=1),device='cuda');offset=torch.tensor([len(p)],device='cuda',dtype=torch.int64)
   batch={'xyz':xyz,'point_features':feat,'xyz_splits':offset}
   with torch.no_grad():
    enc=model.point_transformer({'feat':feat,'offset':offset,'grid_size':.01,'coord':xyz});mesh,td=generator.generate_dual_mc_mesh(batch,enc,torch.device('cuda'))
   v=np.asarray(mesh.vertices,dtype='float32');f=np.asarray(mesh.triangles,dtype='uint32');np.savez_compressed(dest/'surface.npz',vertices=v,faces=f);rec.update(status='DONE',points=len(p),vertices=len(v),faces=len(f),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,timing=td);del enc,mesh,batch,xyz,feat,offset
 except Exception as ex:rec.update(status='ENGINEERING_ERROR',error=str(ex));(dest/'error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
 torch.cuda.empty_cache();rec['elapsed_s']=time.time()-start;rf.write_text(json.dumps(rec,indent=2));rows.append(rec);(out/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rec),flush=True)
(out/'complete.json').write_text(json.dumps({'results':len(rows),'statuses':{s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)}},indent=2))
