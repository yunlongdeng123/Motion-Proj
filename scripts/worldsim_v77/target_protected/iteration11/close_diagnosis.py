"""保存用户原文、工程证据与有界零步诊断，不将局部收益扩大成任务成功。"""
from pathlib import Path
import sys,shutil,json,time
S=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected');sys.path.insert(0,str(S/'iteration9'))
from temporal_factory import T,read,dump
import numpy as np
from PIL import Image
P=S.parents[2];O=T/'r19';E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r19';E20=E.parent/'r20'

def main():
 a=read(O/'audit_result.json');d=read(O/'process_and_duplicates.json');q=read(O/'precision_initialization_audit.json');human=read(O/'human_review.json');state=read(T/'r20/state.json');assert len(state['completed'])==4
 duplicate_outputs={}
 for arm in ['base','r7','r8','r10','r14']:
  equal=all(np.array_equal(np.asarray(Image.open(T/'r14/evaluation/synthetic_M006'/f'{arm}_native'/f'{i:05}.png')),np.asarray(Image.open(T/'r14/evaluation/synthetic_M009'/f'{arm}_native'/f'{i:05}.png'))) for i in range(10));assert equal;duplicate_outputs[arm]=equal
 dump(O/'duplicate_output_check.json',{'all_10_native_frames_equal':duplicate_outputs,'inference_state_leak_evidence':False})
 r20={'stage':'complete_bounded_engineering_diagnosis','new_windows':4,'reused_windows':6,'new_training_steps':0,'seed':42,'steps':25,'assistant_review':[
  {'eval_id':'temporal_P019','review_frames':list(range(10)),'observation':'两个零步舍入臂仍有原模型的灰色轮廓/块状残留，未复现实际r7/r14在道路上新增的深色车辆；新增车不能仅由这次BF16舍入解释。','temporal_pass':None,'human_verdict':None},
  {'eval_id':'A061_w08','review_frames':[5],'observation':'零步舍入臂固定f5接近原模型；实际训练版有其它道路/边缘变化。用户原文r7>r14>base保留，不把像素差当质量分或将单帧升为时序通过。','temporal_pass':None,'human_verdict':None}],
  'conclusion':'初始化精度偏差已确认并修复；此两例零训练控制不足以将微调后的幻觉/收益归于舍入。停止该精度诊断，不扩seed或继续训练。','rounding_is_sufficient_cause_of_P019_hallucination':False,'general_image_quality_causality_proven':False,'human_verdict':None,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02','automatic_shutdown':False}
 dump(T/'r20/closeout.json',r20);dump(E20/'closeout.json',r20);dump(O/'rounding_diagnostic.json',r20)
 findings={'date':'2026-10-02','confirmed_engineering_issues':[
  {'id':'real_target_mask_clipping','evidence':'mask_batch.py: core=largest(SAM & GT_hull_pad3), write_mask &= target；A022 f0有1314个原SAM像素在模型H外，图上可见车头边缘残留。','scope':'原SAM不是像素GT；A022车头有视觉支持，其它case的未覆盖SAM不能自动全称漏车。A041全部原SAM被遮仍失败。','fixed':False,'next':'核准实例mask后保证H及write alpha完整覆盖；GT作提示/一致性证据，避免硬裁真实车身。旧70例冻结产物不改写。'},
  {'id':'trainable_initialization_rounding','evidence':'r7/r14/r18的80个原FP32训练张量先转BF16再转回FP32，零步已经变化。','fixed':True,'validation':'在两组真实80张量上逐值验证FP32原值恢复、冻结参数BF16、梯度可用；无新DriveEditor训练。','causal_scope':'r20两例四窗零训练对照未复现P019新增车，不能把全部失败归因此项。'},
  {'id':'duplicate_model_visible_tasks','evidence':'50train只有46个不同Y/H/条件；11合成验证只有10个。M006/M009的X/Y/H和五臂输出十帧完全相同。','fixed':'已识别并另报去重统计；未静默删除旧数据或重训。','metric_scope':'旧验证按case均值/分母变化，按scene等权MAE在本例未变。'}],
  'confirmed_data_mismatch':{'train_hole_mean_percent':a['summary']['train']['mean_hole_fraction']*100,'real_hole_mean_percent':a['summary']['real_DEV']['mean_hole_fraction']*100,'train_largest_hole_percent':max(v['hole_fraction'] for r in a['synthetic'] if r['split']=='train' for v in r['frame_statistics'])*100,'A022_hole_mean_percent':float(np.mean([v['hole_fraction'] for r in a['real'] if r['eval_id']=='A022' for v in r['frame_statistics']]))*100,'train_edge_frames':0,'real_edge_frames':39,'real_frames':80,'train_shape':'车辆轮廓，bbox fill平均0.814','real_shape':'矩形，bbox fill=1','process_counts':d['summary'],'causal_scope':'差异是真实统计，不是充分因果；A061洞也超过训练最大值但人眼有收益。'},
  'passed_contracts':{'actual_frames':610,'GT_equals_real_RGB':True,'synthetic_RGB_leak_pixels':0,'masked_X_equals_masked_Y_after_resize':True,'deletion_condition_fields_match':True,'r7_r14_80_updated_finite_tensors':True,'encoder_missing':0,'encoder_unexpected':0},
  'not_established':['可训练模块选择错误是主因','补更多scene就会改善','合成MAE下降意味着不新增车辆','GT cuboid表面格支持等于真实纹理在其它帧可见'],
  'next_priority':['修真实入口目标覆盖并检查write alpha，先保住完整DELETE定义','冻结目录时按真实Y/H/模型条件去重，合成与真实H的形状/尺度/截边/时序证据分别对齐','入口合格后保持模块范围做一次固定对照；新增车与保护车身份必须独立记失败，不用全洞MAE代替'],
  'training_iteration_on_hold_by_user':True,'r18_new_eval_windows':0,'new_training_steps_after_user_review':0,'shutdown_eligible':False,'human_numeric_scores_inferred':False}
 dump(O/'findings.json',findings)
 report='''# v77 工程与合成数据诊断

2026-10-02。WS-V77-TARGET-PROTECTED-20260929/r19（CPU合同）＋r20（两例四窗零训练精度控制），failure_ledger_refs [V77-F02]。

**已确认工程问题与数据问题同时存在；现有证据不足以判定“微调模块选错”是主因。** 用户最新要求先排查，因此已停止持续训练迭代，r18在160步完成、0新评价窗时中止后续推理，保留权重。收到本次人工review后没有新增训练；仅运行了事前登记的四个零步诊断窗口。

```mermaid
flowchart LR
 Y[真实视频Y] --> X[合成遮挡X＋车辆轮廓H]
 X --> C[先擦除，再resize／编码]
 R[真实DELETE RGB] --> M[SAM→GT硬裁→矩形H]
 M --> C
 W[原FP32权重] --> P[修正：先保留训练参数，再降冻结参数精度]
 P --> D[原架构DriveEditor]
 C --> D
 Y -. 仅训练监督 .-> D
 D --> N[原生补景]
 N --> A[alpha写回]
 A --> V[新增车／保护车身份／道路／时序审核]
```

## 1. 真实DELETE入口存在可见的目标覆盖问题

当前冻结mask代码先取 `core = largest(SAM & GT_hull_pad3)`，膨胀后的write mask再与GT投影相交，随后用其外包矩形作为模型H。GT投影在这里成了硬裁边界。A022的f0有1314个原SAM像素落在H外，接触图中可见目标车头边缘仍在条件图里。十帧累计1501个原SAM像素在H外，2693个像素的写回alpha不为1。原车边缘可能既进入条件，也在写回时保留。

这不是把所有SAM输出都当真值：A042等例的未覆盖像素可能含误分；必须看具体实例。A041_w10十帧原SAM全被H覆盖、写回alpha均为1，仍然被用户判失败，因此漏mask不能解释所有失败。

应先核准目标实例mask，再检查H完整覆盖、目标像素alpha=1及保护对象冲突。GT作为提示和一致性证据，避免直接截断可见车身。本次没有改写旧70例冻结mask或偷偷重新生成结果。

## 2. 初始化混入了非训练造成的参数变化，已修代码

旧训练入口先对全模型 `.to(bfloat16)`，再将可训练参数 `.float()`。原FP32精度在第一步已经丢失。r7与r14的80个训练张量全部受影响。r7舍入差的L2范数0.855，实际训练相对舍入初值的差为0.969；r14分别0.398与0.941。范数不能换算成画面损伤百分比。

已将两个有效训练入口改成：先保存选中参数的原FP32值，再转换冻结权重，最后恢复FP32训练参数。两组真实80张量逐值验证通过，冻结参数BF16、可训练参数FP32且梯度可用。旧权重与旧代码备份保留，尚未使用修复入口重训。

r20固定P019和A061，分别只对r7范围与r14范围的80张量执行FP32→BF16→FP32，优化0步，其余输入/seed42/25采样步完全不变，共4新窗口。P019的0–9帧接触图中，两种round-only仍接近原模型的块状残留，没有复现实际r7/r14新增的深色车。因此这个工程偏差需要修，但它本身不足以解释该幻觉。A061的固定f5也未显示round-only复现训练版的全部道路/边缘改动；不以这个单帧替代用户视频排序。

## 3. 存在重复训练任务和重复验证任务

按真实Y、完整H和实际遮后条件比较，50个train只有46个不同任务；11个合成验证只有10个。四个训练重复对是M018/M036、M041/M056、M042/M057、M013/M031。M006/M009为验证重复，X/Y/H与五臂原生输出的十帧全部完全相同。

这会改变训练采样权重，也夸大独立case分母。已保留原统计并另存去重统计。旧七个合成验证变为六个；本次按scene等权MAE恰好未改变，但按case均值及胜例数改变。没有跨train/val的相同任务。

## 4. 当前训练任务与真实删除入口未对齐

| 实际输入 | 合成训练50例／500帧 | 真实开发8例／80帧 |
|---|---:|---:|
| H平均画面占比 | 1.82% | 5.60% |
| H形状 | 车辆轮廓，外包框填充率0.814 | 矩形，填充率1.000 |
| 触及画面边缘 | 0/500 | 39/80 |
| 模型分辨率 | 320×576训练 | 576×1024推理 |

训练中最大洞仅5.08%；A022平均21.57%，范围7.55%–28.21%。A061平均9.15%也超过训练范围，却获人眼局部收益，所以大小差是覆盖缺口，不能单独作为失败判据。真实保护区域来自邻车GT投影包络，合成B标签主要是实例SAM，两种面积不能直接当同口径车身像素占比比较。

目前50个训练case实际只有4个扫过例、来自3个world；验证仅1个扫过例、1个world。世界静止A＋运动ego已有11例。所有这些训练窗实际只有0.85–0.90秒。准入检查了连续性和每帧保留一部分B，却未保证“当前被遮的车身部分在其它输入帧真实出现”。GT视频Y里看得到，不能推出遮后的条件里也看得到。

GT cuboid表面格的近似对应统计提示，P006其它帧支持均值约60.5%；P019仅15.8%；P012两个B约47.0%与11.4%。这只是几何代理，不是真实纹理可见性证明。应把有证据恢复、证据不足生成、未知证据分开；不能只用连续/无漏mask的AI2代替任务覆盖。

实际160步按样本计，被遮保护车像素平均仅占画面0.344%。旧blank损失全图均匀，保护区域监督稀疏。它是值得检查的训练目标问题，尚未证明是唯一瓶颈。r17另有“训练调用未传B标签”错误，已在87步隔离；旧uniform训练仍通过Y监督B，不因此作废。r18修正该路由后100/160步启用B项，但用户要求中止评价，不能宣称新损失有效。

## 5. 已排除哪些明显工程错误

当前61个合成case的610帧全部重新解码：Y精确等于原始真实RGB；X的变化全部在H内；先遮再resize后masked-X与masked-Y完全一致。当前首帧CLIP图像、latent H、空物体参考、depth、valid mask等八项字段在同尺寸下与实际DELETE入口一致。没有证据表明完整synthetic-X或其材质/光照伪影泄漏进条件。

r7/r14的80个权重都实际更新且有限，patch路径及键集合匹配；106个目标encoder权重严格恢复、0缺失/0额外。更新并非没有执行。nearest latent mask边界存在单点采样遗漏（训练触洞格中7850/34002），但这是官方nearest语义，不能直接等同合成RGB泄漏或擅自改成maxpool。

另外，r7与r14的数据并不相同，二者人眼排序不能单独归因更新模块；同数据的r10与r14才是模块范围控制。文档中“epsilon MSE”的旧用词应纠正为官方sigma加权的去噪latent MSE，源码监督的是clean Y latent，公式未因此变化。

## 人工review如何改变判断

人工原文14条完整保存在human_review.json：真实A061明确r7>r14>原模型，五例没明显区别，A022/A041全失败。这是局部真实收益，不能再笼统说完全无收益。P006说明某些遮挡过程可改善；P019说明路面误差下降与新增车幻觉可以同时发生；P012两辆车融合仍失败。MAE必须与新增车、保护身份损坏分开报告。

M009原文写“有效r9”，原样保存。实际r14页面的权重只有base/r7/r8/r10/r14，r9是造数阶段，没有独立微调checkpoint，因此没有擅自把该词改成r8或据此制造r9成绩。

## 下一步的最短顺序

先修真实目标覆盖和alpha写回、加入目录去重准入；再使合成训练覆盖实际模型H的形状、大小、截边及遮挡—显露过程，并明确跨帧证据。之后保持更新模块范围，使用固定对照判断收益。先不扩大网络范围，也不继续堆同配方步数。

本轮停在诊断交付。r18仍保留但未纳入效果排名；r20仅零步工程控制。人工0/1/2未代填，未曝光final未用。真实跨例DELETE＋补景收益条件仍未达到，因此未执行关机。
'''
 (P/'docs/v77/TARGET_PROTECTED_ENGINEERING_AUDIT_R19.md').write_text(report)
 (O/'report.md').write_text(report)
 for name in ['audit_result.json','process_and_duplicates.json','precision_initialization_audit.json','precision_fix_validation.json','duplicate_output_check.json','human_review.json','findings.json','rounding_diagnostic.json']:
  dump(E/name,read(O/name))
 close={'stage':'diagnosis_complete_training_iteration_on_hold_by_user','new_training_steps_after_user_review':0,'CPU_checked_RGB_frames':610,'real_mask_frames':80,'new_zero_step_diagnostic_windows':4,'real_benefit_scope':'用户A061单例收益，尚无稳定跨例','shutdown_eligible':False,'human_scores_inferred':False,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02','report':'docs/v77/TARGET_PROTECTED_ENGINEERING_AUDIT_R19.md'}
 dump(O/'closeout.json',close);dump(E/'closeout.json',close)
 b=O/'docs_before_closeout';b.mkdir(exist_ok=True)
 for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
  p=P/rel;dest=b/p.name
  if not dest.exists():shutil.copy2(p,dest)
 p=P/'docs/EXPERIMENTS.md';ls=p.read_text().splitlines();replacements={
  'r18':'160步完成／100步保护项，用户要求0新评价窗时停止；权重保留、效果未知',
  'r19':'610帧合同；GT硬裁漏目标、初始化舍入、4train＋1val重复；CPU修复与人工记录',
  'r20':'两例四窗零训练舍入对照；未复现P019新增车，停止该工程诊断'}
 for i,l in enumerate(ls):
  for rid,note in replacements.items():
   if l.startswith('| WS-V77-TARGET-PROTECTED-20260929 / '+rid+' |'):ls[i]='| WS-V77-TARGET-PROTECTED-20260929 / '+rid+' | '+note+' | [诊断与证据](v77/TARGET_PROTECTED_ENGINEERING_AUDIT_R19.md) |'
 p.write_text('\n'.join(ls)+'\n')
 p=P/'docs/RESEARCH_STATUS.md';p.write_text('''# 当前研究状态

2026-10-02，v77，wm-3090-1001。按用户最新要求停止持续迭代，r19工程/合成诊断已完成。确认A022车头边缘未被当前GT硬裁后的mask覆盖；50train实际46任务、11合成val实际10任务；训练参数初始化先BF16再FP32有舍入，代码已修并验证160真实张量原值保留。610帧真实Y及遮后X合同通过。两例四窗r20零步舍入控制未复现P019新增车，不能用精度偏差解释全部失败。

用户14条部分排序原文保存：真实A061 r7>r14>base，五例没区别、A022/A041全失败；P019道路改善但新增车。真实任务有局部收益，仍未证实稳定跨例；不关机。先修真实入口覆盖/写回和去重，随后对齐训练/真实H及跨帧证据，再考虑固定模块对照；目前没有获准后自行启动的新训练或造数队列。

r18在160步完成、0新评价窗时依用户要求中止，权重保留不作效果结论。r16三秒10候选300帧合同＋独立AI2，40视频已交付但val扫过0、训练0。所有原始/适配/失败产物保留，final未用，无新自动化。

参见[诊断报告与组件图](v77/TARGET_PROTECTED_ENGINEERING_AUDIT_R19.md)、[原文人工记录](autoresearch/worldsim_v77/target_protected_20260929/r19/human_review.json)，failure_ledger_refs [V77-F02]。
''')
 p=P/'docs/research_failures/entries/V77-F02.md';title='## r19/r20：人工排序驱动的工程与合成合同诊断'
 if title not in p.read_text():p.write_text(p.read_text()+'\n\n'+title+'\n\n610帧合同无合成RGB泄漏、Y精确真实；发现真实入口GT硬裁导致A022首帧车头边缘仍可见，50train/11val仅46/10不同任务。初始化全模型BF16后回转FP32带来非梯度更新，已修训练参数原值保存并用160个真实张量验证。两例四窗零训练舍入控制未复现P019新增车，不能归因全部幻觉。人工A061有局部真实收益，P019道路改善与新增车并存；更新此前“无稳定跨例收益”的范围，不抹掉人眼正例。真实矩形H平均5.60%对训练轮廓1.82%、训练无截边/实际39/80触边；扫过验证仅1world、跨帧纹理证据未认证。先整改入口/去重/任务覆盖，按用户要求停止盲迭代，r18零新评价窗保留。报告见[组件图与证据](../../v77/TARGET_PROTECTED_ENGINEERING_AUDIT_R19.md)。failure_ledger_delta: updated V77-F02。\n')
 print('DIAGNOSIS_CLOSED',close)
if __name__=='__main__':main()
