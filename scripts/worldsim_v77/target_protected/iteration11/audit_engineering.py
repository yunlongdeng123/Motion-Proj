"""只读CPU诊断：真实输入合同、mask分布、写回、去重和权重加载。"""
from pathlib import Path
import os,sys,json,ast,shutil,math,time
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4'
P=Path('/root/autodl-tmp/motion_proj_v77');S=P/'scripts/worldsim_v77/target_protected'
sys.path.insert(0,str(S/'iteration7'));sys.path.insert(0,str(S/'iteration8'));sys.path.insert(0,str(S/'iteration9'))
from train_control import arrays,resized,sample
from audit import protected_mask
from evaluate_data_control import inputs
from temporal_factory import T,read,dump
import numpy as np,torch,torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
O=T/'r19';OFF=Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor')
PLAN='''# r19：人工反例驱动的工程与合成合同诊断

WS-V77-TARGET-PROTECTED-20260929/r19，CPU只读审计，failure_ledger_refs [V77-F02]。

```mermaid
flowchart LR
 H[用户14条人工排序] --> D[同case源图、洞、原生与写回]
 Y[真实Y／合成X／H／保护标签] --> C[逐帧条件泄漏、缩放、latent mask]
 C --> Q[官方blank与推理字段对齐]
 W[原权重与80张量patch] --> Q
 D --> A[工程错误与任务分布分开]
 Q --> A
 A --> R[诊断报告／暂不追加训练]
```

用户最新要求先排查、不要持续盲迭代。已停止r18后续评价，保留160步checkpoint与监督证据，尚无r18新评价窗；控制器的-15保留为用户中止，不当科学负结果。关机前真实收益条件仍未满足，不关机；本诊断不加载GPU模型，不造新训练数据，不更换架构或启动训练。

保存人工原文排序，不代填0/1/2，也不把部分排序当全体通过率。r9名称歧义原样保留、按实际checkpoint追溯。逐帧复查当前r14固定61数据例（50train/11val）与8真实DELETE窗口；同输入比较官方字段、条件/监督路由、mask覆盖及缩放；按actual160训练step计算分布。核对Y原始RGB来源、M006/M009实际输入重复、两臂80张量shape/finite/update与部署记录。统计mask分布差异不直接等于因果；SAM2输出不是像素GT，未mask的SAM像素只能当漏目标疑点。
'''

def register():
 O.mkdir(exist_ok=True);E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r19';E.mkdir(parents=True,exist_ok=True)
 if (O/'run.json').exists():return
 b=O/'docs_before_r19';b.mkdir(exist_ok=True)
 for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:shutil.copy2(P/rel,b/Path(rel).name)
 run={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r19','stage':'registered_CPU_contract_and_distribution_diagnosis','host':'wm-3090-1001','new_training_steps':0,'new_GPU_forwards':0,'human_source':'本轮用户部分排序','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','architecture_changed':False,'automatic_shutdown':False}
 for root in [O,E]:dump(root/'run.json',run);(root/'plan.md').write_text(PLAN)
 dump(E/'human_review.json',read(O/'human_review.json'));dump(E/'r18_user_hold.json',read(T/'r18/user_hold.json'))
 cs=read(T/'r18/controller_state.json');dump(T/'r18/controller_state_before_user_hold_annotation.json',cs);cs.update(stage='held_by_user_pending_CPU_diagnosis',termination_returncode_preserved=cs['returncode'],training_complete=True,new_evaluation_windows=0,scientific_result=None);dump(T/'r18/controller_state.json',cs)
 idx=P/'docs/EXPERIMENTS.md';lines=idx.read_text().splitlines();at=next(i for i,l in enumerate(lines) if l.startswith('|---'));lines.insert(at+1,'| WS-V77-TARGET-PROTECTED-20260929 / r19 | 人工排序驱动CPU工程/合成诊断，不追加训练 | [预案](autoresearch/worldsim_v77/target_protected_20260929/r19/plan.md) |');idx.write_text('\n'.join(lines)+'\n')
 status=P/'docs/RESEARCH_STATUS.md';status.write_text('''# 当前研究状态

2026-10-01，v77，wm-3090-1001。用户最新要求先排查工程/合成问题，暂停追加迭代。人工部分排序已保存：真实A061_w08的r7>r14>原模型；其它五真实无明显区别、A022/A041全失败；P019的r7/r14道路改善但新增车幻觉。因此有局部收益，尚无稳定跨例真实收益，不关机。

r18修正B标签路由后160步已完成，100步启用保护项，80梯度/全部loss finite；评价在0新窗时按用户要求停止，权重/中止状态保留，非模型失败。当前唯一新run r19 CPU只读工程与合成合同诊断，无训练/生成任务。r16三秒10候选300帧合同＋独立AI2，40视频交付；val扫过0，训练0。r14/r15及所有旧对照、拒绝与人工证据保留，final未用，无新自动化。

参见[r19预案](autoresearch/worldsim_v77/target_protected_20260929/r19/plan.md)、[r14](v77/TARGET_PROTECTED_TEMPORAL_SCOPE_R14.md)、[r16](v77/TARGET_PROTECTED_LONG_DATA_R16.md)，failure_ledger_refs [V77-F02]。
''');print('REGISTERED_R19_CPU_ONLY',flush=True)

def image(p):return np.asarray(Image.open(p).convert('L'))
def stats(h,b):
 yy,xx=np.where(h);bb=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1]);wh=bb[2:]-bb[:2]
 return {'hole_fraction':float(h.mean()),'box_fill':float(h.sum()/np.prod(wh)),'touches_edge':bool(h[0].any() or h[-1].any() or h[:,0].any() or h[:,-1].any()),'box_wh':wh.tolist(),'hidden_B_canvas_fraction':float((h&b).mean()),'hidden_B_fraction':float((h&b).sum()/max(1,b.sum())) if b.any() else None}
def latent_stats(h):
 hh=F.interpolate(torch.from_numpy(h[:,None].astype('float32')),size=(320,576),mode='nearest');mask=F.interpolate(hh,size=(40,72),mode='nearest')>0;allcover=F.adaptive_max_pool2d(hh,(40,72))>0
 native=F.interpolate(torch.from_numpy(h[:,None].astype('float32')),size=(72,128),mode='nearest')>0;nativecover=F.adaptive_max_pool2d(torch.from_numpy(h[:,None].astype('float32')),(72,128))>0
 return {'train_latent_H_cells':int(mask.sum()),'train_latent_cells_touching_H':int(allcover.sum()),'train_touching_cells_omitted_by_nearest':int((allcover&~mask).sum()),'query_latent_H_cells':int(native.sum()),'query_touching_cells_omitted_by_nearest':int((nativecover&~native).sum()),'interpretation':'边界单点采样差异；maxpool是诊断覆盖代理，不等于应直接替换官方mask，不代表合成RGB泄漏'}
def agg(rows):
 flat=[v for r in rows for v in r['frame_statistics']]
 return {'cases':len(rows),'frames':len(flat),'mean_hole_fraction':float(np.mean([r['hole_fraction'] for r in flat])),'mean_bbox_fill':float(np.mean([r['box_fill'] for r in flat])),'edge_frames':sum(r['touches_edge'] for r in flat),'edge_frame_fraction':float(np.mean([r['touches_edge'] for r in flat])),'mean_hidden_B_canvas_fraction':float(np.mean([r['hidden_B_canvas_fraction'] for r in flat]))}
def patches():
 out={}
 with safe_open(str(OFF/'checkpoints/model.safetensors'),framework='pt') as base:
  for name,folder in [('r7',T/'r7/encoder_fixed_lowres/training'),('r14',T/'r14/training')]:
   cfg=read(folder/'config.json');keys=cfg['trainable_tensors'];different=0;nonfinite=0;diffsq=0.;basesq=0.;maxabs=0.
   with safe_open(str(folder/'attention_step_0160.safetensors'),framework='pt') as patch:
    assert set(patch.keys())==set(keys) and len(keys)==80
    for k in keys:
     a=base.get_tensor(k).float();b=patch.get_tensor(k).float();assert a.shape==b.shape
     nonfinite+=int(not torch.isfinite(b).all());d=b-a;different+=int(torch.any(d!=0));diffsq+=float(d.double().square().sum());basesq+=float(a.double().square().sum());maxabs=max(maxabs,float(d.abs().max()))
   out[name]={'80_keys_exact':True,'nonfinite_tensors':nonfinite,'changed_tensors':different,'parameter_count':cfg['trainable_parameters'],'relative_weight_L2_change':math.sqrt(diffsq/basesq),'max_abs_change':maxabs,'strict_encoder_restore':cfg['encoder_restore'],'patch_loading_record':read(T/'r14/evaluation'/f'{name}_patch_loaded.json') if (T/'r14/evaluation'/f'{name}_patch_loaded.json').exists() else '按实际r10/r8部署记录复用，未伪造r14新部署证据'}
 return out

def main():
 register();torch.set_num_threads(4);start=time.time();catalog=read(T/'r14/dataset_catalog.json')['cases'];plan=read(T/'r14/evaluation_plan.json');source={c['source_id']:c for c in read(T/'r8/factory/source_manifest.json')['clips']};rows=[];bydataset={}
 for c in catalog:
  folder=Path(c['folder']);y,x,h=arrays(c);pair=read(folder/'pair_manifest.json');yy,cc,hh=resized(y,x,h,(320,576))
  ty=torch.from_numpy(y.copy()).permute(0,3,1,2).float()/127.5-1;ty.masked_fill_(torch.from_numpy(h[:,None]),0);ref=F.interpolate(ty,size=(320,576),mode='bilinear',align_corners=False,antialias=True)
  equal=torch.equal(cc,ref);assert equal;leak=int(((x!=y).any(-1)&~h).sum());assert leak==0
  bs=[protected_mask(folder,i,(576,1024),c['type']) for i in range(10)];fs=[stats(a,b) for a,b in zip(h,bs)];original=source.get(pair['source_id']);gt_exact=0;gt_unknown=0
  for i in range(10):
   if original is None:gt_unknown+=1;continue
   src=original['frames'][i];expected=pair['frames'][i]
   if isinstance(expected,dict) and expected.get('timestamp') is not None and expected['timestamp']!=src['timestamp']:gt_unknown+=1;continue
   with Image.open(T/'r8/factory/rgb'/src['filename']) as im:raw=np.asarray(im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))
   assert np.array_equal(y[i],raw),(c['dataset_id'],i,'GT不等于真实源RGB');gt_exact+=1
  r={'dataset_id':c['dataset_id'],'case_id':c['case_id'],'scene':c['receiver_scene'],'split':c['split'],'type':c['type'],'source_id':pair['source_id'],'actual_frames':10,'synthetic_RGB_outside_H_pixels':leak,'masked_X_equals_masked_Y_before_and_after_resize':equal,'GT_real_redecode_exact_frames':gt_exact,'GT_source_unknown_frames':gt_unknown,'frame_statistics':fs,'latent_mask':latent_stats(h),'process':pair.get('temporal_process'),'duration_s':(pair['frames'][-1]['timestamp']-pair['frames'][0]['timestamp'])/1e6 if isinstance(pair['frames'][0],dict) and 'timestamp' in pair['frames'][0] else None}
  rows.append(r);bydataset[c['dataset_id']]=r;print('AUDIT_PAIR',c['dataset_id'],gt_exact,flush=True)
 real=[]
 for c in plan['cases']:
  if c['kind']=='synthetic':continue
  x,y,h,bs,alpha=inputs(c);folder=Path(c['folder']);paths=sorted((folder/'sam').glob('*.png'));checks=[]
  for i,j in enumerate(c['frames']):
   sam=image(paths[j])>0;aa=alpha[i];checks.append({'frame':j,'SAM_pixels':int(sam.sum()),'SAM_pixels_outside_model_H':int((sam&~h[i]).sum()),'SAM_pixels_with_nonunit_write_alpha':int((sam&(aa<.999)).sum()),'SAM_mean_original_RGB_fraction_retained_by_writeback':float((1-aa)[sam].mean()) if sam.any() else None,'SAM_pixels_excluded_by_protection':int((sam&bs[i]).sum()),'alpha_outside_H':int(((aa>0)&~h[i]).sum())})
  real.append({'eval_id':c['eval_id'],'scene':c['scene'],'frame_statistics':[stats(a,b) for a,b in zip(h,bs)],'mask_checks':checks,'latent_mask':latent_stats(np.stack(h)),'SAM_scope':'原SAM不是像素GT；其外部像素可能是误分，不能自动称漏掉目标车'})
 train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation'];steps=[json.loads(l) for l in (T/'r14/training/steps.jsonl').read_text().splitlines()];weighted=[]
 for step in steps:weighted+=bydataset[step['last_case']]['frame_statistics']
 a=next(c for c in plan['cases'] if c['eval_id']=='synthetic_M006');b=next(c for c in plan['cases'] if c['eval_id']=='synthetic_M009');xa,ya,ha,_,_=inputs(a);xb,yb,hb,_,_=inputs(b);dup={'same_real_Y_all_10':all(np.array_equal(u,v) for u,v in zip(ya,yb)),'same_X_all_10':all(np.array_equal(u,v) for u,v in zip(xa,xb)),'same_H_all_10':all(np.array_equal(u,v) for u,v in zip(ha,hb)),'mean_H_IoU':float(np.mean([(u&v).sum()/(u|v).sum() for u,v in zip(ha,hb)])),'same_receiver_scene':a['receiver_scene']==b['receiver_scene']}
 # 执行当前推理get_deletion未修改AST，无需导入UI或加载GPU模型。
 tree=ast.parse((P/'scripts/worldsim_v77/repair_drive.py').read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine');method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='get_deletion');ns={'torch':torch,'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),'repair_drive.py','exec'),ns)
 obj=type('CPUFields',(),{})();obj.im=xa;obj.masks=ha;obj.to_tensor=lambda a:torch.from_numpy(a.copy()).permute(2,0,1).float()/255;obj.transform_mask=lambda a:F.interpolate(a,size=(72,128),mode='nearest')*2-1
 got=ns['get_deletion'](obj);prep=resized(np.stack(ya),np.stack(xa),np.stack(ha),(576,1024));batch=sample(*prep,seed=42)
 equality={'masked_RGB':torch.allclose(got[0],prep[1],atol=1e-7),'first_frame_CLIP_image':torch.allclose(got[0][0],batch['cond_frames_without_noise'][0],atol=1e-7),'latent_H':torch.equal(got[2],batch['mask_concat']),'mask_fuse':torch.equal(got[3],batch['mask_fuse']),'blank_object_reference':torch.equal(got[8],batch['cond_frames_without_noise'][1]),'depth_no_3D_boxes':torch.equal(got[9],batch['depth']),'valid_mask':torch.equal(got[10],batch['valid_mask']),'obj_ratio':torch.equal(got[11],batch['obj_ratio'])};assert all(equality.values())
 summary={'train':agg(train),'validation':agg(val),'real_DEV':agg(real),'actual_training_steps':len(steps),'actual_training_frames_with_repeat':len(weighted),'step_weighted_hole_fraction':float(np.mean([r['hole_fraction'] for r in weighted])),'step_weighted_hidden_B_canvas_fraction':float(np.mean([r['hidden_B_canvas_fraction'] for r in weighted])),'all_610_frames_masked_X_equals_masked_Y':all(r['masked_X_equals_masked_Y_before_and_after_resize'] for r in rows),'synthetic_RGB_leak_pixels':sum(r['synthetic_RGB_outside_H_pixels'] for r in rows),'GT_redecoded_exact_frames':sum(r['GT_real_redecode_exact_frames'] for r in rows),'GT_source_unknown_frames':sum(r['GT_source_unknown_frames'] for r in rows),'official_blank_vs_current_deletion_AST_fields':equality,'M006_M009':dup,'patches':patches(),'new_GPU_forwards':0,'new_training_steps':0,'seconds':time.time()-start,'human_numeric_verdict':None,'causal_attribution_proven':False}
 dump(O/'audit_result.json',{'summary':summary,'synthetic':rows,'real':real});print('AUDIT_SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
