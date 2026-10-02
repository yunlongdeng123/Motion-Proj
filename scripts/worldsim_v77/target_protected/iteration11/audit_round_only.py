"""零训练舍入对照：只检查P019幻觉及A061真实收益，不扩样或调参。"""
from pathlib import Path
import sys,os,time,gc,signal,fcntl,shutil
os.environ['DRIVEEDITOR_SEQUENTIAL_CFG']='1';os.environ['OMP_NUM_THREADS']='4'
P=Path('/root/autodl-tmp/motion_proj_v77');S=P/'scripts/worldsim_v77/target_protected'
sys.path.insert(0,str(S.parent));sys.path.insert(0,str(S/'iteration8'));sys.path.insert(0,str(S/'iteration9'))
from temporal_factory import T,read,dump
from evaluate_data_control import inputs
import numpy as np
from PIL import Image
O=T/'r20'
PLAN='''# r20：零训练初始化舍入对照

task WS-V77-TARGET-PROTECTED-20260929/r20，wm-3090-1001，failure_ledger_refs [V77-F02]。

```mermaid
flowchart LR
 B[原始DriveEditor FP32权重] --> Q[仅80张量 FP32→BF16→FP32]
 B --> E[同P019与A061输入／seed42]
 Q --> E
 F[旧r7／r14实际训练权重] --> E
 E --> V[原生＋写回／固定f5与十帧]
```

用户要求暂停盲目迭代、先排查工程/数据。r19 CPU逐张量确认model_init会先将整个模型转BF16，再将训练参数转FP32；舍入变化混入了最终patch。本轮只作初始化精度的零训练诊断，不训练、不换mask或数据、不恢复r18评价。

事前固定temporal_P019（用户指出微调新增幻觉）与A061_w08（用户指出真实收益）。原始、已训练r7、已训练r14的旧输出仅核对逐帧输入相同后复用；新增r7范围舍入、r14范围舍入两臂，共4个10帧窗口。两臂均重新加载原模型，仅将对应80个张量舍入到BF16再恢复FP32，优化步数0，不能命名为微调模型。其余条件完全相同：576×1024、10帧、seed42、25steps、原默认CFG、previous=false、原alpha写回，300秒/窗口，异常保留并停止、不换seed救结果。

只判断舍入本身是否足以复现已知幻觉/可见改动；模型间像素差不是质量指标，也不能由两例推出所有失败原因。GT、人工原文排序与助手固定帧观察分开。全部输入和输出保留，关机条件false，诊断后回到工程报告，不继续自主训练迭代。
'''

def register():
 O.mkdir(exist_ok=True);E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r20';E.mkdir(parents=True,exist_ok=True)
 if (O/'run.json').exists():return
 plan=read(T/'r14/evaluation_plan.json');cases=[next(c for c in plan['cases'] if c['eval_id']==cid) for cid in ['temporal_P019','A061_w08']]
 run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r20','stage':'registered_zero_training_rounding_diagnosis','training_steps':0,'new_windows_budget':4,'cases':[c['eval_id'] for c in cases],'arms':['r7_round_only','r14_round_only'],'seed':42,'sampler_steps':25,'previous_condition':False,'human_verdict':None,'automatic_shutdown':False,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending'}
 for r in [O,E]:dump(r/'run.json',run);(r/'plan.md').write_text(PLAN)
 dump(O/'evaluation_plan.json',{'cases':cases,'arms':run['arms'],'reuse':['base','r7','r14']})
 b=O/'docs_before_r20';b.mkdir(exist_ok=True)
 for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md']:shutil.copy2(P/rel,b/Path(rel).name)
 p=P/'docs/EXPERIMENTS.md';ls=p.read_text().splitlines();i=next(i for i,l in enumerate(ls) if l.startswith('|---'));ls.insert(i+1,'| WS-V77-TARGET-PROTECTED-20260929 / r20 | 两例、两种80张量范围的零训练舍入控制；4窗有界工程诊断 | [预案](autoresearch/worldsim_v77/target_protected_20260929/r20/plan.md) |');p.write_text('\n'.join(ls)+'\n')
 p=P/'docs/RESEARCH_STATUS.md';p.write_text(p.read_text().replace('当前唯一新run r19 CPU只读工程与合成合同诊断，无训练/生成任务。','r19 CPU合同已完成，发现训练参数先BF16再FP32的初始化舍入；当前仅r20两例四窗零训练精度对照，不恢复训练迭代。'))

def main():
 register();lock=open(O/'diagnostic.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=read(O/'evaluation_plan.json');dest=O/'evaluation';dest.mkdir(exist_ok=True)
 if (O/'state.json').exists():raise RuntimeError('已有状态，须先检查，禁止重复启动')
 state={'stage':'input_check','pid':os.getpid(),'completed':[],'training_steps':0,'human_verdict':None};dump(O/'state.json',state)
 for c in plan['cases']:
  x,y,h,b,alphas=inputs(c);prev=T/'r14/evaluation'/c['eval_id'];out=dest/c['eval_id'];out.mkdir(exist_ok=True)
  assert all(np.array_equal(xx,np.asarray(Image.open(prev/'input'/f'{i:05}.png'))) and np.array_equal(hh,np.asarray(Image.open(prev/'mask'/f'{i:05}.png'))>0) for i,(xx,hh) in enumerate(zip(x,h)))
  for role in ['input','GT','condition','mask','base','base_native','r7','r7_native','r14','r14_native']:
   (out/role).mkdir(exist_ok=True)
   for p in (prev/role).glob('*.png'):os.link(p,out/role/p.name)
 from repair_drive import Engine,set_seed,torch
 from safetensors import safe_open
 torch.set_num_threads(4);engine=None
 def timeout(*_):raise TimeoutError('单窗300秒诊断预算')
 signal.signal(signal.SIGALRM,timeout)
 for arm in plan['arms']:
  if engine is not None:del engine;gc.collect();torch.cuda.empty_cache()
  state.update(stage='loading_'+arm);dump(O/'state.json',state);engine=Engine();family=arm.split('_')[0];cfg=read((T/'r7/encoder_fixed_lowres/training' if family=='r7' else T/'r14/training')/'config.json');patch={}
  with safe_open('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor/checkpoints/model.safetensors',framework='pt',device='cpu') as f:
   for key in cfg['trainable_tensors']:patch[key]=f.get_tensor(key).bfloat16().float()
  assert len(patch)==80;_,extra=engine.model.load_state_dict(patch,strict=False);assert not extra;dump(O/(arm+'_load.json'),{'80_keys':list(patch),'optimizer_steps':0,'full_original_engine':True,'only_change':'FP32 -> BF16 -> FP32 of selected tensors'});del patch
  for c in plan['cases']:
   cid=c['eval_id'];out=dest/cid;x,y,h,b,alphas=inputs(c)
   for role in [arm,arm+'_native']:(out/role).mkdir(exist_ok=True)
   set_seed(42);engine.im=x;engine.masks=h;engine.previous_segment_last_frame=None;engine.im_result=[];state.update(stage=arm,current_case=cid);dump(O/'state.json',state);start=time.monotonic();signal.alarm(300)
   try:engine.predict(1,False,'Deletion')
   finally:signal.alarm(0)
   assert len(engine.im_result)==10;scores=[]
   for i,(raw,xx,hh,alpha) in enumerate(zip(engine.im_result,x,h,alphas)):
    comp=np.rint(raw*alpha[...,None]+xx*(1-alpha[...,None])).clip(0,255).astype('uint8');assert np.array_equal(comp[~hh],xx[~hh]);Image.fromarray(raw).save(out/(arm+'_native')/f'{i:05}.png');Image.fromarray(comp).save(out/arm/f'{i:05}.png');base=np.asarray(Image.open(out/'base_native'/f'{i:05}.png'))
    scores.append({'frame':i,'native_difference_from_base_inside_H_MAE':float(np.abs(raw.astype(float)-base.astype(float))[hh].mean()/255),'scope':'输出差值不是质量分数'})
   row={'eval_id':cid,'arm':arm,'seconds':time.monotonic()-start,'scores':scores,'seed':42,'sampler_steps':25,'training_steps':0,'human_verdict':None};dump(out/(arm+'_metrics.json'),row);state['completed'].append({'eval_id':cid,'arm':arm,'seconds':row['seconds']});dump(O/'state.json',state);print('DIAGNOSTIC',cid,arm,row['seconds'],flush=True)
 state.update(stage='complete_pending_visual_review',new_windows=4);dump(O/'state.json',state)
if __name__=='__main__':main()
