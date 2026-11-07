"""One Full5 checkpoint only: native KEEP/DELETE37 four-frame reference."""
from pathlib import Path
import hashlib,json,os,sys,time,shutil
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_full5_road_integration_20261107_r1';MODEL=B/'scene008_full_dynamic_edit_20261107_r1/inference10k';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');RAWS=[94,105,128,137]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np,torch
 from PIL import Image
 R.mkdir(exist_ok=True);(R/'contract').mkdir(exist_ok=True);outdir=R/'baseline4';outdir.mkdir(exist_ok=False);(outdir/'frames').mkdir()
 sys.path.insert(0,str(REPO))
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.scene_io import checkpoint_actors,load_native_models,camera_headers,make_camera,model_tensors
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.ownership import actor_ids
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.render import render_scene
 config=json.loads((MODEL/'effective_config.json').read_text());source_transform=Path(config['source_path'])/'transform_matrix.txt';transform=R/'contract/full5_transform_matrix.txt'
 if not transform.exists():shutil.copy2(source_transform,transform)
 assert transform.read_bytes()==source_transform.read_bytes()
 matrix=np.loadtxt(transform);scale=float(np.linalg.norm(matrix[:3,:3],axis=1).mean());assert scale>0 and np.isfinite(scale)
 files=[MODEL/f'{p}10000.pth' for p in ['chkpnt','dynamic_chkpnt','env_light_chkpnt']]+[MODEL/x for x in ['effective_config.json','actual_scene_binding.json','cameras.json']]+[transform,Path(__file__)]
 inputs={str(p):ident(p) for p in files};info=checkpoint_actors(MODEL,10000);actors=info['dynamic_actor_ids'];assert actors==[5,6,18,26,37]
 protocol={'status':'SEALED_BEFORE_FULL5_ONLY_BASELINE4_GPU','scene_model':str(MODEL),'source_complete_training_model':str(B/'scene008_full_dynamic_edit_20261107_r1/formal10k'),
  'iteration':10000,'raw_ids':RAWS,'states':['keep','delete'],'dynamic_actor_ids':actors,'delete_actor_ids_by_state':{'keep':[],'delete':[37]},
  'background_sources':1,'dynamic_checkpoint_sources':1,'no_v44_models_or_semantic_additive_packet':True,'group_actor_ids':[-1]+actors,'inputs':inputs,
  'native_renderer_calls_planned':8,'actual_raster_kernels_per_native_call':3,'actual_raster_kernels_planned':24,'optimizer_updates':0,'automatic_training':False,'automatic49':False,'GPU_power':'KEEP_ON'}
 write(R/'contract/full5_only_baseline4_protocol_v1.json',protocol)
 torch.set_num_threads(1);assert torch.cuda.is_available();start=time.time()
 args,models,renderer=load_native_models(MODEL,10000);headers=camera_headers(MODEL);static=models['static'];dynamic=models['dynamic'];env=models['environment']
 initial={name:v.detach().cpu().clone() for name,v in model_tensors(models).items()}
 owners=actor_ids(dynamic);static_tags=torch.zeros((len(static.get_xyz),6),device='cuda');static_tags[:,0]=1
 dynamic_tags=torch.stack([(owners==a).float() for a in actors],1);dynamic_tags=torch.cat([torch.zeros((len(owners),1),device='cuda'),dynamic_tags],1)
 source_counts={str(a):int((owners==a).sum()) for a in actors};assert all(source_counts[str(a)]>0 for a in actors)
 bg=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
 native_raster=renderer.GaussianRasterizer;counters={'actual_native_renderer_calls':0,'actual_raster_kernel_forwards':0}
 class CountedRasterizer:
  def __init__(self,*a,**kw):self.actual=native_raster(*a,**kw)
  def __call__(self,*a,**kw):
   out=self.actual(*a,**kw);counters['actual_raster_kernel_forwards']+=1;write(outdir/'kernel_progress.json',dict(counters,optimizer_updates=0));return out
 renderer.GaussianRasterizer=CountedRasterizer
 frames=[]
 try:
  for raw in RAWS:
   frame=outdir/f'frames/{raw:03d}';frame.mkdir();cam=make_camera(headers[raw]);states={}
   for state in ['keep','delete']:
    with torch.no_grad():p=render_scene(cam,static,dynamic,args,bg,dynamic_actor_ids=actors,delete_actor_ids=[] if state=='keep' else [37],renderer=renderer,
     env_map=env,other=[static_tags],dynamic_other=[dynamic_tags],is_training=False)
    counters['actual_native_renderer_calls']+=1;write(outdir/'kernel_progress.json',dict(counters,optimizer_updates=0))
    rgb=p['render'].detach().cpu().numpy();alpha=p['alpha'].detach().cpu().numpy()[0];depth=p['depth'].detach().cpu().numpy()[0];groups=p['feature'].detach().cpu().numpy()
    assert groups.shape==(6,320,480) and rgb.shape==(3,320,480) and all(np.isfinite(v).all() for v in [rgb,alpha,depth,groups])
    partition=float(np.abs(groups.sum(0)-alpha).max());assert partition<2e-5
    if state=='delete':assert float(np.abs(groups[actors.index(37)+1]).max())==0.
    np.savez_compressed(frame/(state+'_float.npz'),rgb=rgb,alpha=alpha,depth_sum=depth,depth_m=depth/np.maximum(alpha,1e-8)/scale,group_alpha_T=groups,group_actor_ids=np.asarray([-1]+actors,np.int64))
    Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(frame/(state+'.png'))
    states[state]={'group_partition_max_abs':partition,'a37_joint_max':float(np.abs(groups[actors.index(37)+1]).max()),'owner_alpha_sums':{str(a):float(groups[i+1].astype(np.float64).sum()) for i,a in enumerate(actors)},
     'K':{k:headers[raw][k] for k in ['width','height','fx','fy','cx','cy']},'W_model_to_camera':headers[raw]['W'].tolist(),'timestamp':headers[raw]['timestamp']}
    del p
   frames.append({'raw':raw,'states':states});write(frame/'input_guard.json',frames[-1]);write(outdir/'progress.json',dict(completed=[x['raw'] for x in frames],**counters));print(json.dumps({'raw':raw,'status':'FULL5_ONLY_BOTH_RENDERED',**counters}),flush=True)
   del cam;torch.cuda.empty_cache()
 finally:renderer.GaussianRasterizer=native_raster
 current=model_tensors(models);assert set(current)==set(initial) and all(torch.equal(v,current[k].detach().cpu()) for k,v in initial.items())
 for path,value in inputs.items():assert ident(path)==value,path
 assert counters['actual_native_renderer_calls']==8 and counters['actual_raster_kernel_forwards']==24
 receipt={'status':'PASS_FULL5_ONLY_NATIVE_BASELINE4_ENGINEERING_PENDING_QUALITY','model':str(MODEL),'iteration':10000,'raw_ids':RAWS,'states':['keep','delete'],
  'dynamic_actor_ids':actors,'delete_actor_ids_by_state':{'keep':[],'delete':[37]},'group_actor_ids':[-1]+actors,'owner_source_row_counts':source_counts,
  'static_gaussian_rows':len(static.get_xyz),'dynamic_gaussian_rows':len(dynamic.get_xyz),'model_to_world_scale':scale,'transform_sha256':sha(transform),'frames':frames,
  **counters,'backwards':0,'optimizer_updates':0,'parameters_and_owners_unchanged':True,'inputs':inputs,'seconds':time.time()-start,'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),
  'single_background_and_five_actor_scene_source':True,'reconstruction_quality_verdict':None,'delete_quality_verdict':None,'automatic_training':False,'automatic49':False,'shutdown':False,'reset_card':False}
 write(outdir/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ['frames','inputs']}),flush=True)
if __name__=='__main__':main()
