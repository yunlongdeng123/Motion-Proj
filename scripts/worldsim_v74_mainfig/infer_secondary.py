"""只读取预登记 BUILD RGB；保存官方前向的原生输出，评价在独立进程进行。"""
import os,sys,json,time,argparse,pathlib,traceback,subprocess
import numpy as np
import torch
from safetensors.torch import load_file
P=pathlib.Path('/root/autodl-tmp');R=P/'runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1'
REPOS={'dggt':P/'external/worldsim_v81/dggt','dvgt2':P/'external/worldsim_v81/DVGT','vggt':P/'external/worldsim_v72/vggt','dvgt1':P/'external/worldsim_v81/DVGT','pi3x':P/'external/worldsim_v72/Pi3','omega512':P/'external/worldsim_v74_mainfig/vggt-omega'}
WEIGHTS={'dggt':P/'models/worldsim_v81/model_latest_nuscenes.pt','dvgt2':P/'models/worldsim_v74_mainfig/dvgt2.pt','vggt':P/'models/eas_vggt/vggt/model.safetensors','dvgt1':P/'models/worldsim_v81/dvgt1.pt','pi3x':P/'models/eas_vggt/pi3x/model.safetensors','omega512':P/'models/worldsim_v74_mainfig/vggt_omega_1b_512.pt'}
def main():
 a=argparse.ArgumentParser();a.add_argument('--method',choices=REPOS,required=True);a.add_argument('--scenes',nargs='*');a=a.parse_args()
 torch.set_num_threads(4);torch.manual_seed(7403)
 assert torch.cuda.is_available()
 repo=REPOS[a.method];sys.path.insert(0,str(repo));os.chdir(repo)
 if a.method=='dggt':
  from dggt.models.vggt import VGGT
  from dggt.utils.load_fn import load_and_preprocess_images
  from dggt.utils.pose_enc import pose_encoding_to_extri_intri
  model=VGGT();state=torch.load(WEIGHTS[a.method],map_location='cpu',weights_only=True,mmap=True)
 elif a.method=='vggt':
  from vggt.models.vggt import VGGT
  from vggt.utils.load_fn import load_and_preprocess_images
  from vggt.utils.pose_enc import pose_encoding_to_extri_intri
  model=VGGT();state=load_file(str(WEIGHTS[a.method]))
 elif a.method in ['dvgt1','dvgt2']:
  from dvgt.models.architectures.dvgt1 import DVGT1
  from dvgt.models.architectures.dvgt2 import DVGT2
  from dvgt.utils.load_fn import load_and_preprocess_images
  original=torch.hub.load
  def hub(*args,**kwargs):
   if args and 'dinov3' in str(args[0]):kwargs['pretrained']=False;kwargs.pop('weights',None)
   return original(*args,**kwargs)
  torch.hub.load=hub
  try:
   if a.method=='dvgt1':model=DVGT1(dino_v3_weight_path=None,frames_chunk_size=1)
   else:
    from load_dvgt2_full import build
    model=build(WEIGHTS[a.method])
  finally:torch.hub.load=original
  state=torch.load(WEIGHTS[a.method],map_location='cpu',weights_only=True,mmap=True)
 elif a.method=='pi3x':
  from pi3.models.pi3x import Pi3X
  from pi3.utils.basic import load_images_as_tensor
  model=Pi3X();state=load_file(str(WEIGHTS[a.method]))
 else:
  from vggt_omega.models import VGGTOmega
  from vggt_omega.utils.load_fn import load_and_preprocess_images
  from vggt_omega.utils.pose_enc import encoding_to_camera
  model=VGGTOmega();state=torch.load(WEIGHTS[a.method],map_location='cpu',weights_only=True,mmap=True)
 state=state['model'] if 'model' in state else state
 model.load_state_dict(state,strict=True);del state
 if a.method=='pi3x':model.disable_multimodal(free_cuda_cache=False)
 model=model.eval().cuda()
 print(json.dumps({'stage':'MODEL_LOADED','method':a.method}),flush=True)
 for f in sorted((R/'inputs').glob('*.json')):
  m=json.loads(f.read_text())
  if a.scenes and m['scene'] not in a.scenes:continue
  for variant,n in [('six',6),('twelve',12)]:
   out=R/'secondary'/'predictions'/a.method/m['scene']/variant;out.mkdir(parents=True,exist_ok=True)
   if (out/'result.json').exists():continue
   views=m['views'][:n];paths=[v['image'] for v in views]
   try:
    if a.method in ['vggt','dggt']:images=load_and_preprocess_images(paths,mode='crop').cuda()
    elif a.method=='omega512':images=load_and_preprocess_images(paths,image_resolution=512).cuda()
    else:
     inp=out/'native_input';inp.mkdir(exist_ok=True)
     for i,v in enumerate(views):
      d=inp/f'frame_{i//6}' if a.method in ['dvgt1','dvgt2'] else inp;d.mkdir(exist_ok=True)
      link=d/f'{i:02d}_{v["camera"]}.jpg'
      if not link.exists():link.symlink_to(v['image'])
     images=load_and_preprocess_images(str(inp),mode='crop').cuda() if a.method in ['dvgt1','dvgt2'] else load_images_as_tensor(str(inp),verbose=False)[None].cuda()
    torch.cuda.reset_peak_memory_stats();t=time.time()
    print(json.dumps({'stage':'FORWARD','method':a.method,'scene':m['scene'],'variant':variant,'shape':list(images.shape)}),flush=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(images)
    torch.cuda.synchronize();elapsed=time.time()-t
    keys={'dggt':['depth','depth_conf','world_points','world_points_conf','pose_enc','gs_map','gs_conf','dynamic_conf','semantic_logits'],'dvgt2':['points','points_conf'],'vggt':['depth','depth_conf','world_points','world_points_conf','pose_enc'],'omega512':['depth','depth_conf','pose_enc'],'dvgt1':['points','points_conf','absolute_ego_pose_enc'],'pi3x':['points','local_points','camera_poses','conf']}[a.method]
    saved={k:pred[k].detach().float().cpu().numpy() for k in keys if k in pred}
    if a.method in ['vggt','dggt']:
     with torch.inference_mode():E,K=pose_encoding_to_extri_intri(pred['pose_enc'],images.shape[-2:])
     saved['extrinsics']=E.float().cpu().numpy();saved['intrinsics']=K.float().cpu().numpy()
    if a.method=='omega512':
     with torch.inference_mode():E,K=encoding_to_camera(pred['pose_enc'],images.shape[-2:])
     saved['extrinsics']=E.float().cpu().numpy();saved['intrinsics']=K.float().cpu().numpy()
    np.savez_compressed(out/'native_outputs.npz',**saved)
    rec={'status':'DONE','method':a.method,'scene':m['scene'],'variant':variant,'input':str(f),'n_images':n,'network_hw':list(images.shape[-2:]),'elapsed_s':elapsed,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'checkpoint':str(WEIGHTS[a.method]),'repository':str(repo),'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'strict_checkpoint':True,'precision':'BF16','seed':7403,'extra_input_to_model':None,'heldout_access':False,'shapes':{k:list(v.shape) for k,v in saved.items()}}
    if a.method=='omega512':rec.update(checkpoint_source='user-provided 1kaiser/vggt-omega-jax mirror',checkpoint_variant='original512; not416 reproduction; training overlap not excluded')
    (out/'result.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)
    if a.method=='dggt':
     try:
      from render_dggt_native import render
      render(model,pred,images,out)
     except Exception as ex:(out/'render_error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    del images,pred,saved
   except Exception as e:
    (out/'error.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()},indent=2));print(traceback.format_exc(),flush=True)
   torch.cuda.empty_cache()
if __name__=='__main__':main()
