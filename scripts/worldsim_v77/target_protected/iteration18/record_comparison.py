"""仅在72例结果、独立图像评分和本地交付全部完成后收口。"""
from pathlib import Path
import json, shutil, subprocess, re
from collections import Counter

R=Path('/root/autodl-tmp/motion_proj_v77')
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
O=T/'r51';C=O/'r47_comparison'
TASK='WS-V77-TARGET-PROTECTED-20260929'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    plan=read(C/'manifest.json');b=read(O/'drive_state.json');e=read(C/'evaluation_state.json')
    qa=read(C/'comparison_image_review_56sol.json');delivery=read(C/'local_delivery_check.json')
    workbook=read(C/'workbook_delivery_check.json');overlap=read(C/'training_overlap.json')
    ids={c['case_id'] for c in plan['cases']}
    assert len(ids)==72 and e['stage']=='complete' and len(e['completed'])==72
    assert b['stage']=='DELETE_complete_pending_structure_review' and len(b['completed'])==32
    assert {q['case_id'] for q in qa['cases']}==ids and len(qa['cases'])==72
    assert all(q['human_verdict'] is None and q['temporal_verdict'] is None for q in qa['cases'])
    assert delivery['videos']==504 and delivery['actual_decoded_frames']==5040 and not delivery['missing_local_links']
    assert workbook['cases']==72 and workbook['human_scores_filled']==0
    assert all(x['CFG_prior_calls']=={'conditional':25,'unconditional':25} for x in e['completed'])
    # GPU结束已实际核对空进程；交付CPU文件时允许用户切回无卡实例。
    assert read(C/'chain_state.json')['stage']=='GPU_complete'
    source=read(C/'source_mask_validation.json');gate=read(O/'mask_visual_review.json')
    assert gate['rejected']==['R093'] and len(gate['approved'])==32
    scores=['official_native_score','r46_writeback_score','r47_native_score','r47_writeback_score']
    acceptable={key:sum(q[key]>=2 for q in qa['cases']) for key in scores}
    native=dict(Counter(q['native_comparison'] for q in qa['cases']))
    final=dict(Counter(q['writeback_comparison'] for q in qa['cases']))
    stats={'task_id':TASK,'run_id':'r51','host_alias':'wm-vgpu-1008','cases':72,'scenes':52,
        'new_baseline_windows':32,'r47_windows':72,'reused_r50_baseline_windows':40,'training_steps':0,
        'sam_rejected':['R093'],'seed':42,'steps':25,'frames':10,'resolution':[1024,576],
        'branch_checkpoint':plan['branch_checkpoint'],'branch_and_backbone_frozen':True,
        'reference_policy':plan['reference_policy'],'source_reference_checks':len(source['checks']),
        'disabled_reference_slots':sum(not q['passed'] for q in source['checks']),
        'baseline_seconds_sum':sum(r['seconds'] for r in b['completed']),
        'r47_seconds_sum':sum(r['seconds'] for r in e['completed']),
        'r47_peak_allocated_GiB':max(r['peak_allocated_GiB'] for r in e['completed']),
        'AI_native_comparison':native,'AI_writeback_comparison':final,'AI_f05_scores_ge_2':acceptable,
        'AI_score_scope':'single f05 image, requested gpt-5.6-sol xhigh, no fast; not human or temporal verdict',
        'training_overlap':overlap,'local_videos':504,'decoded_local_frames':5040,
        'GPU_jobs':0,'human_verdict':None,'temporal_verdict':None,
        'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02'}
    if (C/'reference_pool_audit.json').exists():
        stats['reference_pool_audit']=read(C/'reference_pool_audit.json')
        stats['reference_policy']+='; additionally selected sample tokens include available same-scene query exposures (see reference_pool_audit)'
    dump(C/'gpu_comparison_result.json',stats)
    evidence=R/'docs/autoresearch/worldsim_v77/target_protected_20260929/r51'
    for name,value in [('gpu_comparison_result.json',stats),('comparison_image_review_56sol.json',qa),
                       ('mask_visual_review.json',gate),('training_overlap.json',overlap),
                       ('workbook_delivery_check.json',workbook)]:dump(evidence/name,value)
    record=read(C/'review/delivery_index.json')
    dump(evidence/'condition_coverage.json',{'cases':[{k:q[k] for k in ['case_id','source_batch','scene','O_N_U_hole_coverage','references']} for q in record['cases']]})
    backup=Path('/root/autodl-tmp/backups/v77_r51_gpu_20261008');backup.mkdir(parents=True,exist_ok=True)
    paths=['docs/RESEARCH_STATUS.md','docs/v77/REAL_DELETE_NATIVE_R51.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']
    for path in paths:
        dest=backup/Path(path).name
        if not dest.exists():shutil.copy2(R/path,dest)
    cpu_archive=R/'docs/v77/REAL_DELETE_NATIVE_R51_CPU.md'
    if not cpu_archive.exists():shutil.copy2(backup/'REAL_DELETE_NATIVE_R51.md',cpu_archive)
    report=f'''# r51：官方、r46 与 r47 的真实 DELETE 对照

任务 `{TASK}/r51`，`wm-vgpu-1008`。旧r50的40例补跑r47，新r51的32例同时完成官方基线和r47：共 **52景72例**，新推理104窗，旧基线复用40窗，新增训练0步。全部输入、失败与写回对照保留。

```mermaid
flowchart LR
 X[真实RGB] --> M[完整SAM<br/>固定独立编辑洞]
 M --> D[官方DriveEditor<br/>原始权重]
 D --> N[官方原生DELETE]
 N --> W[r46固定写回]
 P[真实邻帧/多相机RGB<br/>GT轨迹与LiDAR] --> C[参考RGB＋BEV＋2D条件]
 C --> A[r47 step320条件分支<br/>冻结DriveEditor]
 M --> A
 A --> RN[r47原生DELETE]
 RN --> RW[r47固定写回]
```

## 对照定义

官方原生列也使用统一r21完整SAM，不是重新采用官方1.9倍框挖洞；r46是同一原生输出的固定写回。r47加载已有`branch_0320.safetensors`，RGB编码、BEV与2D融合复用原r47实现，主干与分支均不训练。查询RGB、mask、alpha、seed42、25步、10帧、1024×576固定。原生与最终分别评分，原生DELETE不能称为factual重建。

当前候选参考池使用10个查询SAM帧，加查询开始前约1秒/开始/结束/结束后约1秒六相机关键帧；旧r47为26个SAM帧。此外，8例候选池带入了同scene其他查询窗关联到所选sample的36个非关键曝光；最终R003/R009/R051采用其中5槽，全部通过源SAM擦除检查。实际来源见reference_pool_audit.json。沿用固定4保护车槽＋2上下文槽选择，不按输出挑参考。额外RGB/LiDAR提取1034文件约173MiB；109个已选源mask检查，2槽未通过后停用、不重选。目标A的源像素编码前擦除，无隐藏GT进入条件。

绿O是保留车GT 3D框投影proxy，蓝N是实测背景落点，灰U未知；BEV是80m世界网格。不是精确silhouette、不是完整自由空间、不是隐藏真值。页面每例提供实际OCC/BEV视频、六槽真实参考及逐帧洞内占比。

## 样本与评分边界

新40景候选输入独立复核后33例通过，7例0/1分剔除；SAM再拒绝R093，三帧均串入右侧邻车，故32景32例生成。旧r50保留原准入40例20景。新32景与r50审核过的38景分离，但与r47分支训练同场景的7例为 {', '.join(overlap['overlapping_case_ids'])}，共6景；官方预训练重叠未核实。它们是开发对照，不是严格最终泛化评测。

按用户指定使用5.6-sol/xhigh、未启用fast逐例直接审核固定f05，各列给一位小数0.x/1.x/2.x分数；2.0表示可接受但不完美、不作为满分。AI原生比较：{native}；AI最终写回比较：{final}。四个输出列达到2.0的抽帧数：{acceptable}。完整像素观察逐例留存；这不是人工视频通过率或统计显著性结论。旧人工分数、旧r50 AI记录均保留，人工和时序verdict为空。

先看原生判断生成错误，再看写回新增损伤。保持用户给出的r50原生错误11例/仅写回损伤8例归档，不因本轮AI重评覆盖人工意见。未来匹配遮挡数据只能使用真实视频Y；失败生成不能变成训练GT。本轮未造新数据或追加微调。

## 执行与交付

72例条件数组、目标RGB擦除、无效参考擦除、CFG两路各25次、洞外像素保持检查通过；504本地视频实际解码5040帧，无缺失链接。推理时间求和：新基线{stats['baseline_seconds_sum']/60:.1f}分钟、r47 {stats['r47_seconds_sum']/60:.1f}分钟；r47峰值分配{stats['r47_peak_allocated_GiB']:.2f}GiB。时间求和不含数据准备、编码与人工/AI审核。

本地 `outputs/v77-real-delete-r51/index.html` 提供原视频、官方原生、r46、r47原生、r47写回五列同步视频，mask/OCC与参考在每例展开区。用户原 `C:/Users/dengyunlong/Downloads/打分记录.xlsx` 新增 `r50+r51（官方-r46-r47）`，72行六列AI分数和备注；保留原6页及WPS内嵌图片、原生链接，修改前备份。

对照资产在 `{C}`；复用权重仍在 `{plan['branch_checkpoint']}`。完整评分表和本地交付脚本分别归档为`review/scores_with_r51.xlsx`与`review/local_delivery_source.zip`。轻量记录见 [运行摘要](../autoresearch/worldsim_v77/target_protected_20260929/r51/gpu_comparison_result.json)、[独立图像评分](../autoresearch/worldsim_v77/target_protected_20260929/r51/comparison_image_review_56sol.json)、[逐帧条件覆盖](../autoresearch/worldsim_v77/target_protected_20260929/r51/condition_coverage.json)。入口为iteration18的`compare_r47.py`、`run_comparison.py`、`comparison_review.py`，沿用iteration14条件/分支与iteration17原生基线。保留[CPU准备原文](REAL_DELETE_NATIVE_R51_CPU.md)及其中的原始选样、输入审核和用户失败拆分入口。

`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`。GPU已用完，未执行电源操作。
'''
    (R/'docs/v77/REAL_DELETE_NATIVE_R51.md').write_text(report)
    status=f'''# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 `wm-vgpu-1008`。

唯一run `{TASK}/r51` 已完成：新32例官方基线、旧40例＋新32例的原r47 step320补跑，共52景72例对照；104新窗、40旧窗复用、训练0步。R093双车SAM拒绝，保留原图证据。

独立5.6-sol/xhigh（未开fast）已按固定f05给72例六列小数分数；原生r47相对官方 {native}，最终r47相对r46 {final}。这是AI单帧开发观察，不代替人工/时序验收。7例与r47分支训练同场景，来源偏差明确；不宣称独立泛化或自动升级默认方法。

本地 `outputs/v77-real-delete-r51/index.html` 已更新五列视频和实际OCC/BEV、RGB参考，504视频/5040帧完整解码。原 `打分记录.xlsx` 新增72行对照页，旧6页和图片保留。原生模型错误与写回工程损伤分别记录，未来Y仍真实视频，未启动新数据/训练。

GPU与顺序控制器已完成、无后续GPU作业，通知用户可切CPU；无本轮关机/自动化授权。下一步由本次图像对照和用户review选择具体原生失败，再定匹配数据；不自动修改mask或扫参数。[报告与组件图](v77/REAL_DELETE_NATIVE_R51.md)，[V77-F02](research_failures/entries/V77-F02.md)。
'''
    (R/'docs/RESEARCH_STATUS.md').write_text(status)
    exp=R/'docs/EXPERIMENTS.md';text=exp.read_text()
    row=f'| {TASK} / r51 | 52景72例官方/r46/r47对照；104新窗，AI六列抽帧评分，504视频解码；训练0 | [报告](v77/REAL_DELETE_NATIVE_R51.md) |'
    text,n=re.subn(rf'^\| {TASK} / r51 \|.*$',lambda _:row,text,flags=re.M);assert n==1;exp.write_text(text)
    card=R/'docs/research_failures/entries/V77-F02.md';text=card.read_text();title='### r51 GPU：原生与写回分列的r47迁移对照'
    assert title not in text
    text+=f'''\n\n{title}

用户要求旧r50的40例补r47，新r51经输入/SAM门槛留下32例（R093双车串实例拒绝）。52景72例、104新推理窗，冻结官方主干与已有r47 step320分支，训练0步。7例与r47分支训练同scene，RGB参考池时窗与旧实验不同，均为DEV，不当独立泛化结论。

5.6-sol/xhigh直接审核固定f05六列：r47原生相对官方 {native}；最终相对r46 {final}。原生达到2.0的抽帧：官方{acceptable['official_native_score']}/72、r47 {acceptable['r47_native_score']}/72；写回分数单列。完整失败像素说明和正反例保留，不用GT误差替代审核；人工/视频时序null。不能把写回损伤拿去训练diffusion，也不凭这组单帧均值宣布架构或数据根因。504本地视频解码通过。

这是同一失败卡的新迁移证据，不追加微调、mask调参或新方法。实际OCC/BEV与六槽参考可逐例复核，灰U仍未知。见[组件图与逐例记录](../../v77/REAL_DELETE_NATIVE_R51.md)。failure_ledger_delta: updated V77-F02；无新ID。
'''
    card.write_text(text)
    subprocess.run(['/root/autodl-tmp/envs/motionproj/bin/python','scripts/build_research_failure_index.py'],cwd=R,check=True)
    dump(C/'chain_state.json',{'stage':'delivered_GPU_idle','GPU_jobs':0,'training_steps':0,'cases':72,'human_verdict':None})
    print('RECORDED_COMPARISON',native,final)

if __name__=='__main__':main()
