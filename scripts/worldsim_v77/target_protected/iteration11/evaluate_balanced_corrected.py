"""六臂同输入，复用旧三臂必须核对每帧实际RGB/H/Y。"""
from pathlib import Path
import sys,os,time,gc,fcntl,signal,shutil
import numpy as np
from PIL import Image
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected');sys.path.insert(0,str(P.parent));sys.path.insert(0,str(P/'iteration7'));sys.path.insert(0,str(P/'iteration8'));sys.path.insert(0,str(P/'iteration9'))
from evaluate_data_control import inputs,metrics
from temporal_factory import T,read,dump
O=T/'r18';R8=T/'r14'
def link(src,dst):
 dst.parent.mkdir(exist_ok=True,parents=True)
 if not dst.exists():os.link(src,dst)
def reuse(c,arm,out):
 oldarm=arm;prev=R8/'evaluation'/c['eval_id'];meta=prev/(oldarm+'_metrics.json')
 if not meta.exists() or arm=='r18':return None
 row=read(meta);assert row['seed']==42 and row['steps']==25 and row['previous_condition'] is False;x,y,h,b,a=inputs(c)
 assert all(np.array_equal(xx,np.asarray(Image.open(prev/'input'/f'{i:05}.png'))) and np.array_equal(hh,np.asarray(Image.open(prev/'mask'/f'{i:05}.png'))>0) for i,(xx,hh) in enumerate(zip(x,h)))
 if y is not None:assert all(np.array_equal(yy,np.asarray(Image.open(prev/'GT'/f'{i:05}.png'))) for i,yy in enumerate(y))
 for role in ['input','GT','condition','mask',oldarm,oldarm+'_native']:
  mapped=arm+role[len(oldarm):] if role in [oldarm,oldarm+'_native'] else role
  for p in (prev/role).glob('*.png'):link(p,out/mapped/p.name)
 row.update(arm=arm,reused_from=str(prev),seconds=0.,query_inputs_checked_exact=True);dump(out/(arm+'_metrics.json'),row);return row
def main():
 plan=read(O/'evaluation_plan.json');dest=O/'evaluation';dest.mkdir(exist_ok=True);lock=open(dest/'evaluate.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=read(dest/'state.json') if (dest/'state.json').exists() else {'completed':[]};done={(r['eval_id'],r['arm']) for r in state['completed']}
 def record(c,arm,row):
  state['completed'].append({'eval_id':c['eval_id'],'arm':arm,'seconds':row['seconds'],'reused':bool(row.get('reused_from'))});done.add((c['eval_id'],arm));state.update(stage=arm,completed_count=len(done),total=len(plan['cases'])*6,pid=os.getpid());dump(dest/'state.json',state);print(state['completed'][-1],flush=True)
 for c in plan['cases']:
  out=dest/c['eval_id'];out.mkdir(exist_ok=True)
  for arm in ['base','r7','r8','r10','r14']:
   if (c['eval_id'],arm) in done:continue
   row=reuse(c,arm,out)
   if row is not None:record(c,arm,row)
 from repair_drive import Engine,set_seed,torch
 from safetensors.torch import load_file
 torch.set_num_threads(4);e=None
 def timeout(*_):raise TimeoutError('fixed 300sec inference deadline')
 signal.signal(signal.SIGALRM,timeout)
 for arm in plan['arms']:
  todo=[c for c in plan['cases'] if (c['eval_id'],arm) not in done]
  if not todo:continue
  if e is not None:del e;gc.collect();torch.cuda.empty_cache()
  e=Engine()
  if arm!='base':
   ck={'r7':T/'r7/encoder_fixed_lowres/training','r8':T/'r8/training','r10':T/'r10/training','r14':R8/'training','r18':O/'training'}[arm];assert read(ck/'state.json')['stage']=='complete';cfg=read(ck/'config.json');patch=load_file(str(ck/'attention_step_0160.safetensors'));assert len(patch)==80 and set(patch)==set(cfg['trainable_tensors']);_,extra=e.model.load_state_dict(patch,strict=False);assert not extra;del patch;dump(dest/f'{arm}_patch_loaded.json',{'checkpoint':str(ck/'attention_step_0160.safetensors'),'80_keys_exact':True,'fresh_original_engine':True})
  for c in todo:
   cid=c['eval_id'];out=dest/cid;x,y,h,b,alphas=inputs(c)
   for role in ['input','GT','condition','mask',arm,arm+'_native']:(out/role).mkdir(exist_ok=True)
   set_seed(42);torch.cuda.reset_peak_memory_stats();start=time.monotonic();e.im=x;e.masks=h;e.previous_segment_last_frame=None;e.im_result=[];signal.alarm(300)
   try:e.predict(1,False,'Deletion')
   finally:signal.alarm(0)
   raws=e.im_result;assert len(raws)==10;scores=[]
   for i,(xx,hh,bb,alpha,raw) in enumerate(zip(x,h,b,alphas,raws)):
    comp=np.rint(raw*alpha[...,None]+xx*(1-alpha[...,None])).clip(0,255).astype('uint8');assert np.array_equal(comp[~hh],xx[~hh]);Image.fromarray(comp).save(out/arm/f'{i:05}.png');Image.fromarray(raw).save(out/(arm+'_native')/f'{i:05}.png')
    Image.fromarray(xx).save(out/'input'/f'{i:05}.png');Image.fromarray(hh.astype('uint8')*255).save(out/'mask'/f'{i:05}.png');cp=xx.copy();cp[hh]=127;Image.fromarray(cp).save(out/'condition'/f'{i:05}.png')
    if y is not None:Image.fromarray(y[i]).save(out/'GT'/f'{i:05}.png');scores.append(dict(frame=i,**metrics(raw,y[i],hh,bb)))
    else:scores.append({'frame':i,'actor_free_GT_available':False,'RGB_recovery_MAE':None,'outside_write_mask_pixels_changed':0})
   row={'eval_id':cid,'arm':arm,'scores':scores,'seconds':time.monotonic()-start,'seed':42,'steps':25,'previous_condition':False,'reused_from':None,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'human_verdict':None};dump(out/(arm+'_metrics.json'),row);record(c,arm,row)
 state.update(stage='complete',all_six_arms_complete={(c['eval_id'],a) for c in plan['cases'] for a in plan['arms']}<=done);dump(dest/'state.json',state)
if __name__=='__main__':main()
