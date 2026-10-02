"""同r14数据/模块/预算，仅将洞内两类恢复损失作为等权宏平均。"""
from pathlib import Path
import sys,copy,shutil,argparse,math,json
P=Path('/root/autodl-tmp/motion_proj_v77');S=P/'scripts/worldsim_v77/target_protected';sys.path.insert(0,str(S/'iteration7'));sys.path.insert(0,str(S/'iteration9'));sys.path.insert(0,str(S/'iteration11'))
from temporal_factory import T,read,dump
from temporal_scope import selected
import train_control as trainer
import torch,torch.nn.functional as F
O=T/'r17'
PLAN='''# r17：固定数据的恢复区域宏平均损失控制

task WS-V77-TARGET-PROTECTED-20260929/r17，wm-3090-1001，failure_ledger_refs [V77-F02]。

```mermaid
flowchart LR
 Y[同r14真实Y／50train25world] --> U[原架构时间self attention80]
 H[先擦X的同一条件H] --> U
 U --> L[全局＋洞内背景＋洞内保护车损失]
 B[真实保护mask，仅训练标签] -.-> L
 L --> W[同160步／49.57M]
 W --> E[同11GT＋8真实DEV，原生／写回]
```

r14有限新过程GT改善但真实关键失败仍在。r10实际被遮保护像素仅画面约.344%，原blank mask_fuse全零使二维扩散损失按整幅均匀平均；这是监督稀疏证据，不是已证明的梯度原因。本轮只改已有损失归约，不把B/C加到网络输入，不改DriveEditor架构、encoder、数据、模块、采样条件或预算。

保持r14同50train/25world、11GT/5world与8已曝光真实DEV、原初始化、官方106目标encoder、80时间self attention/49,574,080参数、160步、320×576、AdamW1e-5/wd.01、seed6201，实际160次case/window顺序必须一致。r15/r16新数据仍因验证过程缺额不混入训练。final未用。

二维epsilon加权平方误差仍采用官方sigma权重；每帧先对通道平均，再求全图均值、H内非保护像素均值、H内保护像素均值，对存在的项等权平均。空区域不除零、不伪造loss。H与B采用原保守latent maxpool标签，H-bg=H且非B。无可调倍率、权重网格或附加网络；三维损失不改。全局项保留，GT标签仅BUILD监督，QUERY不需要B真值。

同默认CFG1.2→2.0、seed42、25steps、previous=false、10帧576×1024。旧五臂95窗仅输入逐帧相同才复用，r17新19窗；GT指标与真实单帧/视频保持分开。教师loss新旧公式不同，不能用数值直接比较。一次160步，无效则停止该具体归约控制，不加倍率或步数救结果。

数据AI2、合成误差或训练loss下降均不满足关机条件；必须真实跨至少两scene的删净/保护对象收益、无新严重误删，独立视频帧复核并扩三秒确认，保存交付推送且无其它作业后才关机。人工评分空，全部反例和原模型保留。
'''

def register():
 O.mkdir(exist_ok=True);E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r17';E.mkdir(parents=True,exist_ok=True)
 if (O/'run.json').exists():return
 for name in ['dataset_catalog.json','admission_result.json']:dump(O/name,read(T/'r14'/name))
 plan=copy.deepcopy(read(T/'r14/evaluation_plan.json'));plan['arms']=['base','r7','r8','r10','r14','r17'];plan.update(selection_before_r17_outputs=True,only_loss_reduction_changed=True);dump(O/'evaluation_plan.json',plan)
 run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r17','stage':'registered_equal_region_macro_loss_control','host':'wm-3090-1001','training_steps':160,'dataset_exact_same_as':'r14','modules_exact_same_as':'r14','only_change':'2D loss region reduction','architecture_unchanged':True,'B_labels_train_loss_only':True,'human_verdict':None,'final_used':False,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','automatic_shutdown':False,'real_cross_case_benefit_demonstrated':False}
 for root in [O,E]:dump(root/'run.json',run);(root/'plan.md').write_text(PLAN)
 b=O/'docs_before_r17';b.mkdir(exist_ok=True)
 for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md']:shutil.copy2(P/rel,b/Path(rel).name)
 idx=P/'docs/EXPERIMENTS.md';lines=idx.read_text().splitlines();at=next(i for i,l in enumerate(lines) if l.startswith('|---'));lines.insert(at+1,'| WS-V77-TARGET-PROTECTED-20260929 / r17 | 固定r14数据/80时间张量/160步，仅恢复区域宏平均；真实收益待验证 | [预案](autoresearch/worldsim_v77/target_protected_20260929/r17/plan.md) |');idx.write_text('\n'.join(lines)+'\n')
 status=P/'docs/RESEARCH_STATUS.md';status.write_text(status.read_text().replace('当前唯一新run r16延长为约3秒真实连续曝光并补隔离验证过程，来源冻结最多48／每world最多2，训练0。','r16实际40来源1200帧及47条SAM2（44连续性pass），10候选正在完整生成/独立审核，val扫过仍0，训练0。当前唯一GPU控制r17固定r14全部数据与模块，仅改变恢复区域宏平均损失，160步一次；新数据不混入。'))
 print('REGISTERED_R17',flush=True)

def region_reduction(error,hole,protected):
 """每帧等权存在的全局、背景洞、保护洞；保持官方返回的一维帧loss形状。"""
 assert error.ndim==hole.ndim==protected.ndim==3 and error.shape==hole.shape==protected.shape
 v=error.flatten(1).mean(1);den=torch.ones_like(v)
 for mask in [hole&~protected,hole&protected]:
  count=mask.flatten(1).sum(1);term=(error*mask).flatten(1).sum(1)/count.clamp_min(1);present=count>0;v=v+torch.where(present,term,torch.zeros_like(term));den=den+present
 return v/den

def train():
 register();trainer.selected=selected;original_loss=trainer.loss;original_save=trainer.save_json
 def write(p,obj):
  if Path(p).name=='config.json':obj=dict(obj,loss='official sigma-weighted epsilon MSE; equal per-frame mean of global/H-bg/H-protected nonempty regions',loss_change_only=True,B_labels_train_loss_only=True)
  original_save(p,obj)
 def loss(model,prepared,seed,protected=None):
  original=model.loss_fn.get_loss
  def reduced(output,target,weight,mask):
   assert torch.all(mask==0),'控制要求原blank mask_fuse不变'
   err=((output-target)**2*weight).mean(1);h=F.adaptive_max_pool2d(prepared[2][:,None].float().cuda(),err.shape[-2:])[:,0]>0
   pb=torch.zeros_like(h) if protected is None else F.adaptive_max_pool2d((protected&prepared[2])[:,None].float().cuda(),err.shape[-2:])[:,0]>0
   return region_reduction(err,h,pb)
  model.loss_fn.get_loss=reduced
  try:val=original_loss(model,prepared,seed,protected)
  finally:model.loss_fn.get_loss=original
  model.last_loss_regions['official_mask_fuse_uniform']=model.last_loss_regions.pop('loss_weights_uniform');model.last_loss_regions['effective_loss_weights_uniform']=False;model.last_loss_regions['objective']='equal mean of present global/H-bg/H-protected frame means'
  return val
 trainer.loss=loss;trainer.save_json=write
 trainer.main(argparse.Namespace(root=O,steps=160,size=[320,576],modules='temporal_self',encoder=T/'r7/encoder_recovery/official_svd_encoder.safetensors'))
 cfg=read(O/'training/config.json');old=read(T/'r14/training/config.json');fields=['lr','weight_decay','seed','steps_requested','optimizer','dtype','training_resolution','trainable_tensors','trainable_parameters'];assert all(cfg[k]==old[k] for k in fields)
 new=[json.loads(l) for l in (O/'training/steps.jsonl').read_text().splitlines()];prev=[json.loads(l) for l in (T/'r14/training/steps.jsonl').read_text().splitlines()];assert len(new)==len(prev)==160;assert all((n['last_case'],n['last_window'])==(p['last_case'],p['last_window']) for n,p in zip(new,prev));assert all(math.isfinite(n['loss']) and math.isfinite(n['grad_norm']) for n in new)
 dump(O/'training/objective_control.json',{'same_recipe_fields':fields,'same_160_case_window_order':True,'same_80_tensors_49574080_parameters':True,'all_finite_loss_and_grad_norm':True,'network_input_B_channels':0,'architecture_unchanged':True,'human_verdict':None})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--register-only',action='store_true');a=p.parse_args();register() if a.register_only else train()
