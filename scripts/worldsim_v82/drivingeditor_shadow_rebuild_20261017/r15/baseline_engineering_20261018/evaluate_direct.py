"""Zero-update exact restore and full-frame49 five-ID/deletion export."""
from pathlib import Path
import os,sys,json,random,datetime
E=Path(__file__).resolve().parent
sys.path.insert(0,str(E/'direct_implementation'))
from priority import require_gpu_slot
def write(p,v):p.write_text(json.dumps(v,indent=2))
def main():
 slot,admission=require_gpu_slot('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
 try:
  import torch,numpy as np
  from PIL import Image
  from runtime import load_models
  from renderer import render
  from formal_support import EpochSampler,restore_flags
  from restore_contract import verify_runtime_restore
  registry=E/'direct_active_run.json'
  active=Path(json.loads(registry.read_text())['output']) if registry.exists() else E/'direct_instance30k'
  ck=active/'checkpoint_030000.pt';out=E/'direct_full49'
  assert not out.exists();out.mkdir()
  before=(ck.stat().st_size,ck.stat().st_mtime_ns)
  saved=torch.load(ck,map_location='cuda',weights_only=False)
  assert saved['step']==30000
  cfg,scene,static,m,opt,env,ex=load_models(saved['spec'],out,freeze_background=False,restore=saved)
  restore_flags(static,saved['formal_runtime'],30000,cfg)
  sampler=EpochSampler(0,scene.getTrainCameras());sampler.setstate(saved['sampler'])
  random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
  golden=torch.load(ck,map_location='cpu',weights_only=False)
  receipt=verify_runtime_restore(golden,static,m,opt,env,sampler)
  write(out/'restore_receipt.json',receipt);del golden,saved
  def png(p,t):
   a=t.detach().clamp(0,1).cpu()
   a=a.permute(1,2,0) if a.shape[0]==3 else a[0]
   Image.fromarray(np.rint(a.numpy()*255).astype(np.uint8)).save(p)
  tracks=list(m.instance_ids);actors=[5,26,18,6,37];assert len(tracks)==5
  bg=torch.zeros(3,device='cuda');rows=[]
  with torch.no_grad():
   for idx,c in enumerate(scene.getTrainCameras()):
    d=out/Path(c.image_name).stem;d.mkdir()
    full=render(c,static,m,cfg,bg,env_map=env,instance_ids=tracks)
    for n,key in [('full','render'),('dynamic','raw_render_d'),('static','render_s'),('dynamic_alpha','dynamic_alpha'),('static_alpha','static_alpha')]:png(d/(n+'.png'),full[key])
    png(d/'gt.png',c.original_image)
    for a,t in zip(actors,tracks):
     rem=render(c,static,m,cfg,bg,env_map=env,instance_ids=[t],edits={t:{'delete':True}})
     assert rem['instance_visible_contribution'].count_nonzero()==0
     assert torch.equal(rem['render_s'],full['render_s'])
     iso=render(c,static,m,cfg,bg,env_map=env,instance_ids=[t],edits={x:{'delete':True} for x in tracks if x!=t})
     for n,v in [('delete',rem['render']),('dynamic_delete',rem['raw_render_d']),('dynamic_delete_alpha',rem['dynamic_alpha']),('isolated',iso['raw_render_d']),('alpha',iso['dynamic_alpha'])]:png(d/(f'a{a:02d}_'+n+'.png'),v)
     rows.append(dict(frame_id=Path(c.image_name).stem,actor=a,track_id=t,deleted_visible_alpha_exact_zero=True,static_exactly_unchanged=True))
    allout=render(c,static,m,cfg,bg,env_map=env,instance_ids=tracks,edits={t:{'delete':True} for t in tracks})
    assert allout['dynamic_alpha'].count_nonzero()==0
    png(d/'delete_all.png',allout['render'])
    write(out/'progress.json',dict(frames=idx+1,total=49,all_full_frame=True))
  assert before==(ck.stat().st_size,ck.stat().st_mtime_ns)
  write(out/'receipt.json',dict(state='COMPLETE_FULL_FRAME49_REQUIRES_QUALITY_REVIEW',frames=49,actors=actors,records=rows,parameter_updates=0,full_frame=True,ROI_crop=False,checkpoint_unchanged=True,human_verdict=None,quality_accepted=False))
 finally:slot.close()
if __name__=='__main__':main()
