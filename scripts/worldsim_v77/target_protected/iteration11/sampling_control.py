"""一个预登记的采样强控制：同r7权重，只取消CFG条件外推，不扫参数。"""
from pathlib import Path
import os,sys,json,time,signal,gc,fcntl
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P.parent));sys.path.insert(0,str(P/'iteration8'));sys.path.insert(0,str(P/'iteration9'))
from temporal_factory import T,read,dump
from evaluate_data_control import inputs,metrics
import numpy as np
from PIL import Image
O=T/'r13'

def main():
 plan=read(O/'evaluation_plan.json');dest=O/'evaluation';dest.mkdir(exist_ok=True)
 lock=open(dest/'evaluate.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=read(dest/'state.json') if (dest/'state.json').exists() else {'completed':[]};done={r['eval_id'] for r in state['completed']}
 from repair_drive import Engine,set_seed,torch
 from safetensors.torch import load_file
 torch.set_num_threads(4);e=Engine();ck=T/'r7/encoder_fixed_lowres/training';patch=load_file(str(ck/'attention_step_0160.safetensors'));cfg=read(ck/'config.json');assert set(patch)==set(cfg['trainable_tensors']) and len(patch)==80
 _,extra=e.model.load_state_dict(patch,strict=False);assert not extra;del patch
 guider=e.model.sampler.guider;oldscale=guider.scale.detach().cpu().tolist();assert len(oldscale)==1 and len(oldscale[0])==10 and np.allclose(oldscale[0],np.linspace(1.2,2.,10),atol=1e-6)
 guider.scale=torch.ones_like(guider.scale);guider.min_scale=1.;guider.max_scale=1.
 dump(dest/'sampling_condition.json',{'r7_weights_unchanged':True,'checkpoint':str(ck/'attention_step_0160.safetensors'),'80_keys_exact':True,'original_scales':oldscale,'control_scales':guider.scale.cpu().tolist(),'3d_guider_unchanged':True,'seed':42,'steps':25,'previous_condition':False,'no_training':True,'operator':'x_u + 1*(x_c-x_u) == x_c; no guidance extrapolation'})
 def timeout(*_):raise TimeoutError('300sec per window')
 signal.signal(signal.SIGALRM,timeout)
 for c in plan['cases']:
  cid=c['eval_id']
  if cid in done:continue
  out=dest/cid;out.mkdir(exist_ok=True);x,y,h,b,alphas=inputs(c)
  for role in ['input','GT','mask','r7_cfg1','r7_cfg1_native']:(out/role).mkdir(exist_ok=True)
  # 固定基线的输入逐像素认证，不能比较不同删除范围。
  prev=T/'r10/evaluation'/cid
  for i,(xx,hh) in enumerate(zip(x,h)):
   assert np.array_equal(xx,np.asarray(Image.open(prev/'input'/f'{i:05}.png'))) and np.array_equal(hh,np.asarray(Image.open(prev/'mask'/f'{i:05}.png'))>0)
  set_seed(42);torch.cuda.reset_peak_memory_stats();start=time.monotonic();e.im=x;e.masks=h;e.previous_segment_last_frame=None;e.im_result=[];signal.alarm(300)
  try:e.predict(1,False,'Deletion')
  finally:signal.alarm(0)
  assert len(e.im_result)==10;scores=[]
  for i,(xx,hh,bb,a,raw) in enumerate(zip(x,h,b,alphas,e.im_result)):
   comp=np.rint(raw*a[...,None]+xx*(1-a[...,None])).clip(0,255).astype('uint8');assert np.array_equal(comp[~hh],xx[~hh])
   for role,im in [('input',xx),('mask',hh.astype('uint8')*255),('r7_cfg1',comp),('r7_cfg1_native',raw)]:Image.fromarray(im).save(out/role/f'{i:05}.png')
   if y is not None:Image.fromarray(y[i]).save(out/'GT'/f'{i:05}.png');scores.append({'frame':i,**metrics(raw,y[i],hh,bb)})
   else:scores.append({'frame':i,'actor_free_GT_available':False,'RGB_recovery_MAE':None})
  row={'eval_id':cid,'arm':'r7_cfg1','scores':scores,'seconds':time.monotonic()-start,'seed':42,'steps':25,'previous_condition':False,'input_matches_r10_exact':True,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'human_verdict':None};dump(out/'r7_cfg1_metrics.json',row)
  state['completed'].append({'eval_id':cid,'seconds':row['seconds']});done.add(cid);state.update(stage='running',completed_count=len(done),total=len(plan['cases']),pid=os.getpid());dump(dest/'state.json',state);print(cid,round(row['seconds'],1),flush=True)
 state.update(stage='complete',all_cases_complete=True);dump(dest/'state.json',state)
if __name__=='__main__':main()
