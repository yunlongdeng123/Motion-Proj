"""r50固定批次收口：复用同一报告/失败卡，只保存轻量结果。"""
from common import *
from collections import Counter
import re, statistics


def main():
    m=read(O/'manifest.json');gate=read(O/'mask_visual_review.json')
    masks=read(O/'mask_state.json');drive=read(O/'drive_state.json')
    qa=read(O/'assistant_structure_review.json');runtime=read(O/'gpu_runtime.json')
    assert drive['stage']=='DELETE_complete_pending_structure_review'
    rows=drive['completed'];ids={r['case_id'] for r in rows}
    assert ids==set(gate['approved'])=={r['case_id'] for r in qa['cases']}
    assert len(rows)==40 and len(ids)==40 and len(masks['completed'])==42
    assert all(not r['empty_frames'] for r in masks['completed'])
    assert all(r['seed']==42 and r['steps']==25 and not r['adapter'] and not r['previous_condition']
               and len(r['checks'])==10 and all(x['outside_H_exact'] and x['full_SAM_native_exact'] for x in r['checks']) for r in rows)
    assert all(r['human_verdict'] is None and r['temporal_verdict'] is None and r['frame']==5 for r in qa['cases'])
    assert runtime['GPU_processes_idle']
    tags=Counter(tag for r in qa['cases'] for tag in r['labels'])
    active=[c for c in m['cases'] if c['case_id'] in ids]
    b=Counter(c['B_evidence_status'] for c in active)
    summary={'task_id':TASK,'run_id':'r50','host_alias':'wm-vgpu-1008','runtime':runtime,
        'input_cases':42,'input_scenes':20,'input_quality_candidate_count':87,'input_low_quality_rejected':36,
        'SAM_cases':42,'SAM_empty_cases':0,'mask_approved':40,'mask_rejected':gate['rejected'],
        'DELETE_windows':40,'DELETE_scenes':len({c['scene'] for c in active}),
        'generated_targets_by_scene':dict(Counter(c['scene'] for c in active)),
        'SAM_seconds_sum':sum(r['seconds'] for r in masks['completed']),
        'DELETE_seconds_sum':sum(r['seconds'] for r in rows),'DELETE_seconds_median':statistics.median(r['seconds'] for r in rows),
        'peak_gib':max(r['peak_gib'] for r in rows),'checkpoint':m['model_checkpoint'],
        'peak_memory_metric':'torch.cuda.max_memory_allocated；不包含reserved/driver',
        'seed':42,'steps':25,'mask_policy':'sam_full_v2','adapter':False,'training_steps':0,
        'writeback_frames_checked':400,'outside_H_exact':True,'full_SAM_native_exact':True,
        'B_input_evidence':dict(b),'single_frame_labels':dict(tags),'labels_can_overlap':True,
        'human_verdict':None,'temporal_verdict':None,'hidden_GT':None,
        'engineering_issue':'首次gate使用approved_case_ids而入口要求approved，加载模型前停止；已对齐字段，原日志保留。没有改mask、seed或采样。',
        'cases':[{k:r[k] for k in ['case_id','seconds','peak_gib','human_verdict','temporal_verdict']} for r in rows],
        'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02'}
    dump(O/'gpu_results.json',summary)
    state=read(O/'controller_state.json');state.update(stage='r50_review_complete_GPU_idle',host_alias='wm-vgpu-1008',
        GPU_jobs=0,mask_cases=42,windows=40,training_steps=0,human_verdict=None)
    dump(O/'controller_state.json',state)
    E=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r50'
    for name,value in [('gpu_results.json',summary),('mask_visual_review.json',gate),
                       ('assistant_structure_review.json',qa),('controller_state.json',state)]:dump(E/name,value)
    for name in ['gpu_results.json','assistant_structure_review.json','controller_state.json']:
        dump(O/'review'/name,read(O/name))
    status=f'''# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 `wm-vgpu-1008`。

唯一run `WS-V77-TARGET-PROTECTED-20260929/r50` 已完成固定真实DELETE扩大排查。独立输入检查87例，36个0/1分剔除，20个官方train开发场景42个输入2分目标。42例SAM无空mask；独立实例检查40例通过，R013/R066因吞入邻车拒绝，不算补景模型失败。40个DELETE全部完成、20景仍在，其中两景各1例，其余2–3例。

固定r46官方DriveEditor原权重+r21完整SAM，seed42/25steps/10帧/1024×576；无Adapter、训练、参数扫描或时间模块修改。400帧写回验证通过，原生与最终视频均保留。独立subagent每例只抽f05粗分类，标签可重叠：{dict(tags)}。它不是人工0/1/2、隐藏区域真值或视频通过率。后车可见证据充分9例、薄弱8例、无候选21例、不确定2例，不能当完整隐藏车身已知。

同一页面 `outputs/v77-real-delete-r50/index.html` 展示40例四列原视频/mask/原生DELETE/固定写回；两例mask拒绝及全部旧低质量输入另存归档。主干没有更新，因此本轮是问题排查，不能宣称模型收益。后续按有真实证据的结构失败匹配遮挡数据，Y必须是真实视频，再单独验证内部空间层；不自动训练、不改时间层。5个隔离train景与旧val隔离不动，本轮缓存来源不是总体/最终评测。

GPU计算已结束、进程退出，用户可切CPU；本轮无自动关机或定时任务。GPU报告32GB RTX 4080 SUPER，CPU16核，推理累计{summary['DELETE_seconds_sum']/60:.1f}分钟、PyTorch峰值已分配显存{summary['peak_gib']:.2f}GiB；数据盘余量见run资源记录，无清理。[报告与组件图](v77/REAL_DELETE_STRUCTURE_R50.md)，[同一失败卡V77-F02](research_failures/entries/V77-F02.md)。
'''
    (REPO/'docs/RESEARCH_STATUS.md').write_text(status,encoding='utf-8')
    report=REPO/'docs/v77/REAL_DELETE_STRUCTURE_R50.md';text=report.read_text(encoding='utf-8')
    text=text.replace('主机 `wm-3090-1001`','主机 `wm-vgpu-1008`').replace('当前只完成CPU准备；SAM、生成和训练均为0。',
        '42例SAM已完成，40例实例准入后完成官方DELETE，仍覆盖20景；训练0步。')
    text=text.replace('V -. 未开始 .-> F','V --> F')
    text=text.replace('F -. 后续 .-> Y[真实视频Y + 匹配遮挡]',
        'F -. 仅取布局与洞 .-> Y[匹配遮挡数据]\n R[真实可见视频Y] --> Y\n Y -. 后续 .-> S[内部空间层验证]')
    results=f'''## GPU结果与单帧结构粗分类

用户开启GPU并指定 `wm-vgpu-1008`。42例SAM无空mask；独立f00/f05/f09检查40例2分通过，R013吞邻黑SUV及行人、R066吞邻黑车，均1分拒绝。没有逐例改prompt、mask或seed补齐数字。20景仍覆盖，两景各剩1个生成目标，其余2–3个。输入2分与mask2分不表示补景合格。

40个固定DELETE全部完成，累计{summary['DELETE_seconds_sum']/60:.1f}分钟，中位{summary['DELETE_seconds_median']:.1f}秒/窗，PyTorch峰值已分配显存{summary['peak_gib']:.2f}GiB，不含reserved/driver。官方checkpoint加载0 missing/0 unexpected；无Adapter、previous-window条件或训练。400帧洞外原像素、完整SAM内原生像素检查全部通过；800张原生/写回PNG保留在run。首次gate字段名不匹配在加载模型前停止，已对齐并保存旧日志；不是模型失败，也没有更换推理配置。

独立subagent每例仅审f05四列和同ROI近景，必要时核对真实B crop，未逐输出帧审查。多标签统计：`{dict(tags)}`，标签可重叠，包含任一列的可见现象；不能当最终任务失败率。原生/写回差别在逐例说明中，`none_visible`只表示抽取帧未见明显结构错误。车辆再生标签是单帧迹象，隐藏身份未知时保留不确定性。人工与时序verdict均空，不能当整段视频通过率、隐藏区域GT或模型根因证据。完整逐例依据见下方JSON。

40例B可见证据：充分9、薄弱8、无候选21、不能确定2。可见片段充分不等于全部隐藏车身已知；优先看有充分真实证据而仍有薄膜/变形的失败，再按布局与洞造遮挡，训练Y仍用真实视频。本轮没有训练或新方法收益，不自动恢复r49或开始参数扫描。

GPU已空、用户可切CPU。实际设备32GB RTX 4080 SUPER，CPU cgroup16核；数据盘余量与结束时进程证据保存在gpu_results.json/runtime，无清理或电源操作。

'''
    text=re.sub(r'## 下一步与资源\n.*?(?=## 交付与证据)',lambda _:results,text,flags=re.S)
    text=text.replace('显示20景42个2分目标，原图黄框A、绿框B；尚未推理的列明确空缺。',
        '显示20景40个DELETE：原视频、SAM洞、官方原生DELETE、固定写回。两例mask拒绝在独立页保留原图与理由，不让用户重复审核低质入口。黄框A、绿框B。原生DELETE不是factual重建。')
    text=text.replace('`failure_ledger_delta: none`（本次输入准入调整，尚无新生成或科学失败）',
        '`failure_ledger_delta: updated V77-F02`（新增真实结构现象，输入失败与生成失败分开）')
    text+='\n[GPU结果与资源](../autoresearch/worldsim_v77/target_protected_20260929/r50/gpu_results.json)、[SAM独立准入](../autoresearch/worldsim_v77/target_protected_20260929/r50/mask_visual_review.json)、[逐例单帧粗分类](../autoresearch/worldsim_v77/target_protected_20260929/r50/assistant_structure_review.json)。\n'
    report.write_text(text,encoding='utf-8')
    exp=REPO/'docs/EXPERIMENTS.md';text=exp.read_text(encoding='utf-8')
    row='| WS-V77-TARGET-PROTECTED-20260929 / r50 | 20景40个固定真实DELETE完成；2例SAM吞邻车拒绝；独立f05结构粗分类，训练0 | [报告](v77/REAL_DELETE_STRUCTURE_R50.md) |'
    text=re.sub(r'^\| WS-V77-TARGET-PROTECTED-20260929 / r50 \|.*$',lambda _:row,text,flags=re.M);exp.write_text(text,encoding='utf-8')
    card=REPO/'docs/research_failures/entries/V77-F02.md';text=card.read_text(encoding='utf-8')
    text+=f'''\n\n### r50 GPU：固定基线真实结构排查

20个train开发景42个输入2分目标，SAM无空但独立实例检查拒绝2例吞邻车，40个r46官方DELETE全部完成。独立f05粗分类标签可重叠：{dict(tags)}；完整逐例像素依据、B证据与原生/写回差异已保存。输入2分、mask2分不能代替生成质量；真实隐藏区无GT，车辆再生仅记迹象，人工及视频时序未判。400帧写回合同通过不表示结构正确。

这是更大开发池的问题定位，不是新方法收益或总体评测。先看真实证据充分的结构失败再匹配遮挡数据，Y仍真实视频；训练0步，不改时间层。gate字段名工程错误在加载模型前修正，未换seed/mask。GPU已空，通知用户切CPU，不关机。[报告与组件图](../../v77/REAL_DELETE_STRUCTURE_R50.md)。failure_ledger_delta: updated V77-F02；无新ID。
'''
    card.write_text(text,encoding='utf-8')
    print('RECORDED_GPU',len(rows),dict(tags),flush=True)


if __name__=='__main__':main()
