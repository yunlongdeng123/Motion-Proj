"""依据实际产物收口同一task/run，保留质量分母与未完成范围。"""
import argparse,datetime,shutil
from pathlib import Path
from collections import Counter
from geometry_factory import read,dump

def counts(rows,key):return dict(Counter(r[key] for r in rows))
def main(root,repo):
    out=root/'synthetic_delivery';manifest=read(out/'synthetic_manifest.json');qa=read(out/'independent_synthetic_reviews.json');val=read(out/'delivery_validation.json');reviews={r['case_id']:r for r in qa['clips']}
    rows=manifest['clips'];assert len(rows)==len(reviews)==len(val['cases']);assert all(r['pass'] for r in val['cases'])
    assert all(r.get('human_verdict') is None for r in reviews.values())
    passed=[r for r in rows if reviews[r['case_id']]['synthetic_status']=='pass']
    assembly=root/'expanded_factory';src=read(assembly/'subagent_source_reviews.json')['clips'];mq=read(assembly/'mask_review/independent_mask_reviews.json')['clips'];numeric=read(assembly/'mask_review/mask_audit.json')['clips'];secondary=read(assembly/'mask_review_secondary/independent_secondary_mask_reviews.json')['clips']
    evidence=repo/'docs/autoresearch/worldsim_v77/target_protected_20260929/gpu';evidence.mkdir(parents=True,exist_ok=True)
    cohorts=[]
    for label in dict.fromkeys(r['cohort'] for r in rows):
        rr=[r for r in rows if r['cohort']==label];pp=[r for r in rr if reviews[r['case_id']]['synthetic_status']=='pass']
        cohorts.append({'cohort':label,'rendered':len(rr),'receiver_scenes':len({r['scene'] for r in rr}),'frames_per_case':sorted({len(r['frames']) for r in rr}),'frames':sum(len(r['frames']) for r in rr),'types':counts(rr,'type'),'independent_quality':counts([reviews[r['case_id']] for r in rr],'synthetic_status'),'passed_types':counts(pp,'type')})
    summary={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'status':'pilot_delivery_pending_human_full_frame_review; approximate_50_qualified_goal_not_met',
      'source_RGB_reviewed':len(src),'receiver_source_quality':counts(src,'receiver_status'),'donor_source_quality':counts(src,'donor_status'),
      'primary_SAM2_jobs':len(numeric),'primary_numeric_pass':sum(r['numeric_gate_pass'] for r in numeric),'primary_independent_receiver_mask':counts(mq,'receiver_mask_status'),'primary_independent_donor_mask':counts(mq,'donor_mask_status'),
      'secondary_SAM2_jobs':len(secondary),'secondary_independent_mask':counts(secondary,'protected_mask_status'),'SAM2_total_frames':30*(len(numeric)+len(secondary)),
      'rendered_cases':len(rows),'rendered_receiver_scenes':len({r['scene'] for r in rows}),'rendered_frames':sum(len(r['frames']) for r in rows),'cohorts':cohorts,
      'independent_synthetic':counts(qa['clips'],'synthetic_status'),'independent_pass_cases':len(passed),'independent_pass_case_ids':[r['case_id'] for r in passed],
      'independent_pass_receiver_scenes':len({r['scene'] for r in passed}),'independent_pass_types':counts(passed,'type'),
      'human_review_received':0,'training_ready':0,'formal_training_runs':0,'DriveEditor_forward_runs':0,'GT_generated':False,'architecture_changed':False,
      'reviewer_model':'gpt-6-sol','reasoning_effort':'xhigh','fast':False,
      'validation':{k:v for k,v in val.items() if k!='cases'},'native_window_ego_guard_rejected':read(root/'native_window_factory/pair_candidates.json')['pre_render_ego_guard_rejected'],
      'dense_strict_selected':len(read(root/'dense_window_factory/pair_candidates_strict.json')['selected']),'dense_unreviewed_diagnostic_selected':len(read(root/'dense_window_factory/pair_candidates.json')['selected']),
      'run_root':str(root),'local_html':'C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-synthetic/index.html',
      'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: mask contamination, synthesis lighting/foreground-order failures and dense-pair data gap; no model-family scientific rejection',
      'known_limits':['Boston logs and repeated receiver/donor scenes; not generalization','10-frame fixed windows are short samples, not 3s or long-time validation','2.5D real cutouts, not relit 3D render','no qualified dense multi-actor family in this pool','single-camera visible evidence only; no new SV3D/multiview condition','sampling QA does not certify temporal image quality','current condition preview is data recipe, not full DriveEditor loader/forward'],
      'resource_note':'GPU used for official frozen SAM2; CPU synthesis and QA. No power action or automation in this phase.'}
    dump(root/'gpu_stage_summary.json',summary);dump(evidence/'summary.json',summary)
    for srcpath,name in [(out/'independent_synthetic_reviews.json','independent_synthetic_reviews.json'),(out/'delivery_validation.json','delivery_validation.json'),(assembly/'mask_review/independent_mask_reviews.json','independent_primary_mask_reviews.json'),(assembly/'mask_review_secondary/independent_secondary_mask_reviews.json','independent_secondary_mask_reviews.json'),(assembly/'subagent_source_reviews.json','source_reviews_combined.json'),(assembly/'geometry_positive_controls.json','geometry_positive_controls.json'),(root/'dense_window_factory/pair_candidates_strict.json','dense_strict_proposals.json'),(root/'dense_window_factory/pair_candidates.json','dense_diagnostic_proposals.json'),(root/'native_window_factory/pair_candidates.json','native_window_proposals.json')]:
        shutil.copy2(srcpath,evidence/name)
    for part,label in [('', 'P'),('expanded_factory','D'),('native_window_factory','W')]:
        folder=root/part/'synthetic_review';light=[]
        for r in read(folder/'synthetic_manifest.json')['clips']:
            light.append({k:r.get(k) for k in ['case_id','source_id','donor_source_id','scene','donor_scene','type','placement_mode','source_window','donor_window','review_frames','contacts','videos','frame_count','edge_mode','extra_motion_blur_px','mask_dilation_px','min_GT_clearance_m','max_view_yaw_delta_deg','max_view_pitch_delta_deg','minimum_actual_actor_size_px']})
        dump(evidence/f'{label}_case_index.json',{'full_remote_manifest':str(folder/'synthetic_manifest.json'),'cases':light})
    table='\n'.join(f"| {c['cohort']} | {c['rendered']} | {c['receiver_scenes']} | {c['frames_per_case']} | {c['independent_quality']} |" for c in cohorts)
    report=f'''# Target + Protected Actors：GPU造数试产与逐帧审核

`WS-V77-TARGET-PROTECTED-20260929/r1`。实际生成 **{len(rows)} 例 / {summary['rendered_receiver_scenes']} 个receiver scene**，共{summary['rendered_frames']}帧、{val['video_count']}段四列视频。独立抽帧结果为 **{summary['independent_synthetic']}**；只有{len(passed)}例进入用户逐帧全检，来自{summary['independent_pass_receiver_scenes']}个receiver scene。人工评分全部留空，训练准入0。**尚未形成约50例合格数据，密集多车类别仍缺额。**

```mermaid
flowchart LR
 Y[nuScenes train 真实 Y] --> Q[来源质量 / SAM2]
 A[另一真实 donor A] --> Q
 Q --> G[相机 + GT轨迹 + 地图 / LiDAR]
 G --> X[连续合成 X + A删除洞]
 Y --> S[真实监督 Y 原样保留]
 X --> V[全帧合同检查 + Sol独立抽帧]
 S --> V
 V --> H[仅通过例：用户全部帧打分]
 H --> T[人工合格后才讨论微调]
```

## 实际产物与范围

| 分支 | 已渲染case | receiver scenes | 每例帧数 | 独立抽帧结果 |
|---|---:|---:|---|---|
{table}

所有来源仅为train；旧9例DEV、70例val审计及隔离final不参与。原70例用户CSV已由上一阶段原样归档，见[用户评审记录](DELETE_AUDIT_USER_REVIEW.md)。本轮来源集中少数Boston日志，重复scene/track和短窗均公开，不能把49左右的case数称为49个独立scene或泛化结果。当前通过类型：`{summary['independent_pass_types']}`。

来源RGB独立检查{len(src)}段；主SAM2 {len(numeric)}段、次要保护车{len(secondary)}段，共{summary['SAM2_total_frames']}帧。数值通过不能替代实例身份检查：T028把路边白色矩形也分进pickup，已独立拒绝；完整mask保留。所有本轮新独立检查使用gpt-6-sol / xhigh，不启用fast，原先已完成的来源记录不改写。

## 规则变化、反例与缺额

P保留初始30帧固定世界位移控制。随后明确测试camera-relative donor轨迹，并将旧0.65下采样代理下界替换为最终真实alpha至少72×40px；最大放大1.5、视角、形变、地面、全GT避碰继续检查。不是所有分支同一冻结配置，详见[质量协议v2](TARGET_PROTECTED_QUALITY.md)与各分支manifest。

官方本地`DriveEditor/configs/train.yaml`的data.num_frames=10，故W使用固定0:10/10:20/20:30三个窗口，不按渲染好坏选时间。W每例约1秒、10次实际曝光；它不替代30帧或长时序验证。

D003/D004的亮面与建筑阴影明显不相容；D006/D009覆盖了真实自车机盖。它们是数据合成问题，未送入训练。后续W采用底部64px保守禁入带，渲染前拒绝5个候选；不是精确自车分割。弱接触影单独列风险，不要求第一阶段完美投射阴影；明显悬浮、照明冲突或遮挡错仍拒绝/待定。所有原始正反对照保留。

对已经有两个真实实例mask的receiver做有界更密放置搜索仍0个密集合格候选；再仅诊断未审查GT包络影响，仍0个满足两保护车可见比例/遮挡次序的候选。停止在同一来源池扫同类格点。**这说明当前来源配对与2.5D放置的产量不足，不是DriveEditor能力否定。**后续密集素材应先按可投放空间和两个protected的相对几何采样，再提取RGB/分割；不靠降低质量阈值凑数。

## 数据合同与审核方式

远端每case保存`Y/`、`X/`、`alpha/`、`influence/`、`model_hole/`、`protected/`，以及`target.json`、`protected_actors.json`、`sample.json`和完整pair manifest。Y为真实JPEG的确定性解码/缩放无损PNG；网页JPEG/MP4仅供看图，不能充当训练GT。

四列依次是真实Y、合成X、A/B/C轮廓、masked-X条件预览。最后一列洞内归一化0显示为中灰，**不是DriveEditor生成结果**。本轮未运行DriveEditor前向/微调，未改变架构。所有合成影响都在洞内，masked X与masked Y相等；所以donor外观变化不能被夸大为deletion网络已经看到了这些变化。

保护车完整状态只供监督和审核；reference provenance限定X中`protected AND NOT hole`可见区域，未把Y隐藏区作为reference。没有新增其他相机reference/SV3D prior或新网络输入通道。后续需把样本逐字段接入官方loader并做零训练对照，当前不声称已完成训练适配。

正式列表只含工程检查和独立抽帧都pass的case；拒绝/待定留在诊断列表。按每例全部帧逐个打0/1分，支持同步视频、逐帧、放大及JSON导入导出，没有一键全通过。抽帧不认证时序；所有人工分数初始为空。

## 验证、资源与恢复

实际全帧核对Y重新解码一致、X-Y影响域、洞覆盖、protected剩余可见性与隐藏Y扰动不改变condition。{val['video_count']}段MP4全部实际解码；{val['preview_frame_count']}张预览JPEG检查通过。7项有意义的几何/条件合同测试通过，Python编译与网页脚本检查另存交付证据。浏览器file页面未做交互实测，不能把静态/解码检查当点击播放验证。

模型为冻结SAM2.1 Hiera Large，GPU RTX3090 24GiB；几何和合成为CPU。数据盘本轮末约146GiB空闲，无扩盘需求。没有新建自动化、没有关机。原始模型/数据与所有对照保留。

远端：`{root}`。本地：[打开逐帧审核页](file:///C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-synthetic/index.html)。[机器可读证据](../autoresearch/worldsim_v77/target_protected_20260929/gpu/summary.json)。`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；人工verdict=null。
'''
    (repo/'docs/v77/TARGET_PROTECTED_GPU.md').write_text(report,encoding='utf-8')
    status=f'''# 当前研究状态

2026-09-29，v77；`WS-V77-TARGET-PROTECTED-20260929/r1`。用户已授权GPU造数，本轮SAM2和试产完成，进入用户逐帧全检；未训练、未运行新DriveEditor前向、架构未改。[当前报告](v77/TARGET_PROTECTED_GPU.md)。

实际{len(rows)}个合成case / {summary['rendered_receiver_scenes']}个receiver scene，独立抽帧{summary['independent_synthetic']}，仅{len(passed)}例可进入人工全检；通过类型{summary['independent_pass_types']}。尚未收到人工全检回传、训练准入0。约50例合格目标未达、密集多车0；不得把候选数/重复短窗当独立合格scene数。

P/D是30帧控制；W是依据官方训练长度的固定10帧短窗，规则变化在质量协议v2保留。49左右全部实际产物、失败和未确定例可在本地`outputs/v77-target-protected-synthetic/index.html`查阅，正式人工列表只含pass。此前来源审核页独立保留，来源通过不等于合成通过。

来源/实例污染、光照与自车前景顺序错误已更新同一V77-F02。当前池密集提案有界细化及未知包络诊断均0，不继续相同扫描；下一步用可投放空间与保护车相对几何先选新来源，再做RGB/分割，以补齐类型和场景多样性。人工逐帧确认前不正式微调；只保留X可观测reference、真实Y仅作监督。

run：`{root}`。GPU任务/CPU渲染均结束；没有新建自动化或电源操作，当前阶段不继承旧审计关机授权。subagent默认gpt-6-sol / xhigh，禁止fast。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
'''
    (repo/'docs/RESEARCH_STATUS.md').write_text(status,encoding='utf-8')
    index=repo/'docs/EXPERIMENTS.md';lines=index.read_text().splitlines();lines=[f'| WS-V77-TARGET-PROTECTED-20260929 / r1 | {len(rows)}合成case/{summary["rendered_receiver_scenes"]}个receiver scene；独立{summary["independent_synthetic"]}，仅通过例人工逐帧全检；密集类缺额、零训练 | [GPU试产](v77/TARGET_PROTECTED_GPU.md) · [CPU来源](v77/TARGET_PROTECTED_CPU.md) · [质量协议v2](v77/TARGET_PROTECTED_QUALITY.md) |' if l.startswith('| WS-V77-TARGET-PROTECTED-20260929 / r1 |') else l for l in lines];index.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    card=repo/'docs/research_failures/entries/V77-F02.md';marker='## Target + Protected Actors GPU试产：实例污染与合成入口失效'
    text=card.read_text();assert marker not in text,'do not duplicate same failure section'
    text+=f'''\n{marker}

`WS-V77-TARGET-PROTECTED-20260929/r1`，零DriveEditor前向/训练。{len(rows)}个实际合成case经全帧工程验证及gpt-6-sol xhigh独立指定帧检查，结果{summary['independent_synthetic']}；只有{len(passed)}例进入人工全检、人工null，密集类仍0。真实Y始终来自train视频，不生成GT，不增加架构输入通道。

T028的SAM2数值连续性通过，但实图把路边白色矩形加入pickup，已拒绝，说明数值与GT包络不能认证实例身份。P001/P003及D003/D004暴露合成受光/接地问题；D006/D009明确覆盖真实自车机盖，属于缺自车图像遮挡的工程错误，不能由米制包络避碰通过抵消。后续W加底部64px保守禁入带并提前拒绝5候选；所有旧对照保留。完整逐例结果见[独立QA](../../autoresearch/worldsim_v77/target_protected_20260929/gpu/independent_synthetic_reviews.json)。

固定10帧窗口增加了短窗候选，但不证明30帧/长时序；光照和可投放空间仍限制产量。已有双实例来源的有限细化与未审查包络诊断均没有得到合格密集样本，停止同池重复扫格，转向几何可行性优先的来源采样。当前缺额是数据工厂问题，不构成DriveEditor科学否定，也不据此修改人工结论。[报告与组件图](../../v77/TARGET_PROTECTED_GPU.md)、[实际计数与限制](../../autoresearch/worldsim_v77/target_protected_20260929/gpu/summary.json)。`failure_ledger_delta: updated V77-F02`。
'''
    card.write_text(text,encoding='utf-8');print('RECORDED',len(rows),'PASS',len(passed),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();main(a.root,a.repo)
