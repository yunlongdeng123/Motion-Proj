"""Actual native all-vehicle30K full/static/dynamic/delete-all, full frame49."""
from pathlib import Path
import os,sys,json,copy
E=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/drivingeditor_repro')
ROOT=BASE/'campaign/r15_baseline_allvehicles_20261018';SID='scene0008_089_137'
def main():
 sys.path.insert(0,str(E/'direct_implementation'))
 from priority import require_gpu_slot
 lock,_=require_gpu_slot(str(BASE/'campaign/gpu_slot.lock'))
 try:
  import torch,numpy as np
  from PIL import Image
  from omegaconf import OmegaConf
  run=ROOT/'runs'/f'{SID}_hybrid_30000_seed0_GN11'
  assert json.loads((run/'status.json').read_text())['iteration']==30000
  out=E/'all_vehicle_full49';out.mkdir(exist_ok=False)
  source=ROOT/'training_source'/SID
  assert (source/'scene/__init__.py').exists(),source
  sys.path.insert(0,str(source));from scene import Scene,GaussianModel,Dynamic_GaussianModel,EnvLight
  from gaussian_renderer import render
  cfg=json.loads((run/'effective_config.json').read_text());cfg.update(model_path=str(out),training=True,including_dynamic=True,adding=False)
  args=OmegaConf.create(cfg);s=GaussianModel(args);d=Dynamic_GaussianModel(args);scene=Scene(args,s,d,shuffle=False)
  env=EnvLight(args.env_map_res).cuda()
  originals=[]
  for name,m in [('chkpnt30000.pth',s),('dynamic_chkpnt30000.pth',d),('env_light_chkpnt30000.pth',env)]:
   p=run/name;originals.append((p,p.stat().st_size,p.stat().st_mtime_ns));state,step=torch.load(p,map_location='cuda',weights_only=False);assert step==30000;m.restore(state,args)
  def png(p,t):
   a=t.detach().clamp(0,1).cpu();a=a.permute(1,2,0) if a.shape[0]==3 else a[0]
   Image.fromarray(np.rint(a.numpy()*255).astype(np.uint8)).save(p)
  rows=[];bg=torch.zeros(3,device='cuda');staticargs=copy.deepcopy(args);staticargs.including_dynamic=False
  with torch.no_grad():
   for cam in scene.getTrainCameras():
    fid=Path(cam.image_name).stem;folder=out/fid;folder.mkdir()
    other=[s.get_scaling_t.clamp_max(2),s.get_inst_velocity];dynamic=[d.get_scaling_t.clamp_max(2),d.get_inst_velocity]
    full=render(cam,s,d,args,bg,env_map=env,other=other,dynamic_other=dynamic,is_training=False)
    removed=render(cam,s,None,staticargs,bg,env_map=env,other=other,is_training=False)
    delta=float((full['render_s']-removed['render']).abs().max());assert delta<2e-4
    for n,key in [('full','render'),('dynamic','render_d'),('static','render_s'),('dynamic_alpha','dynamic_alpha'),('static_alpha','static_alpha')]:png(folder/(n+'.png'),full[key])
    png(folder/'gt.png',cam.original_image);png(folder/'delete_all.png',removed['render'])
    rows.append(dict(frame_id=fid,delete_all_dynamic_point_count=0,static_replay_max_abs=delta))
    (out/'progress.json').write_text(json.dumps(dict(frames=len(rows),total=49)))
  assert len(rows)==49 and all((p.stat().st_size,p.stat().st_mtime_ns)==(n,t) for p,n,t in originals)
  (out/'receipt.json').write_text(json.dumps(dict(state='COMPLETE_NATIVE_ALL_VEHICLE_FULL49_REQUIRES_VISUAL_REVIEW',frames=49,records=rows,parameter_updates=0,ROI=False,all_full_frame=True,all_inputs_are_training_views=True,human_verdict=None,unobserved_background_completion_not_added=True),indent=2))
 finally:lock.close()
if __name__=='__main__':main()
