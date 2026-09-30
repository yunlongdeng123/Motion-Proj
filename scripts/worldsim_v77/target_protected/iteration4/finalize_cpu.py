"""归档CPU预检及明确的GPU边界；人工训练准入保持空白。"""
import argparse,html,shutil
from collections import Counter
from pathlib import Path
import json

def read(p):return json.loads(p.read_text())
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main(root,repo):
    out=root/'review';manifest=read(out/'preflight_manifest.json');q=read(root/'independent_preflight_reviews.json')
    byid={x['case_id']:x for x in q['cases']};assert set(byid)=={c['case_id'] for c in manifest['cases']}
    for c in manifest['cases']:
        r=byid[c['case_id']]
        assert r['reviewed_frame']==c['frame']==5 and r['human_verdict'] is None and r['training_ready'] is False
        c['independent_source_status']=r['source_status']
        c['status']='source_preflight_'+r['source_status']+'_pending_GPU_masks'
    dump(out/'preflight_manifest.json',manifest)
    queue=read(root/'gpu_queue.json')
    for j in queue['jobs']:
        status=byid[j['source_id']]['source_status'];j['independent_source_status']=status
        j['eligible_after_GPU_authorization']=status=='pass';j['blockers']=['explicit_GPU_authorization'] if status=='pass' else ['independent_source_quality_'+status,'explicit_GPU_authorization']
    dump(root/'gpu_queue.json',queue)
    s=read(root/'selected_pair_plan.json')['summary'];sampling=read(root/'native10_factory/sampling_control.json');old=read(root/'factory/source_manifest_30.json');prep=read(root/'preparation_native10.json')
    assert s['complete'] and not prep['missing_files'],'CPU全量检查未完成，不能收口'
    s.update(task_id='WS-V77-TARGET-PROTECTED-20260929',run_id='r4',cpu_only=True,initial_selection=32,
             old_outer_30_pass=len(old['clips']),nearest_fixed_ten_pass=sampling['nearest_fixed_ten_pass'],ordered_fixed_ten_pass=sampling['ordered_fixed_ten_pass'],
             independent_preflight_counts=dict(Counter(x['source_status'] for x in q['cases'])),
             queued_mask_jobs=sum(j['eligible_after_GPU_authorization'] for j in queue['jobs']),GPU_model_calls=0,actual_synthetic_cases=0,training_ready=0)
    dump(root/'summary.json',s);dump(out/'summary.json',s);dump(out/'independent_preflight_reviews.json',q);dump(out/'gpu_queue.json',queue)
    cards=[]
    for c in manifest['cases']:
        r=byid[c['case_id']];cards.append(f'<section id="{c["case_id"]}"><h2>{c["case_id"]} · {c["scene"]} · {c["camera"]}</h2><p>预案类型：{c["planned_type"]}；供体：{c["donor"]}；独立CPU来源检查：<b>{r["source_status"]}</b>。{html.escape(r["note"])}</p><p>GT米制最小净距 {c["min_clearance_m"]:.2f}m；LiDAR支持最大距离 {c["max_ground_support_m"]:.2f}m；最大视角差 {c["max_view_delta_deg"]:.2f}°。真实protected mask尚缺，包络交叠不能认证真实遮挡比例。</p><img src="{c["image"]}" alt="{c["case_id"]} CPU空间预案"></section>')
    a=[('真实train视频','新receiver窗口'),('有序曝光匹配','固定10帧 / 不复制'),('已有真实供体','地图+LiDAR+GT配对'),('CPU预案与独立QA','不生成合格训练对'),('待开GPU','SAM2 → 精确检查')]
    boxes=''.join(f'<rect x="{10+i*240}" y="10" width="205" height="80" rx="9"/><text x="{112+i*240}" y="42">{t}<tspan x="{112+i*240}" dy="25">{u}</tspan></text>' for i,(t,u) in enumerate(a))
    arrows=''.join(f'<path d="M{217+i*240} 50h29" marker-end="url(#arrow)"/>' for i in range(4))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 r4 CPU配对与GPU准备</title><style>body{{max-width:1550px;margin:30px auto;padding:0 24px;background:#101827;color:#edf3fb;font:16px/1.7 system-ui}}section{{background:#1b293d;padding:20px;border-radius:12px;margin:20px 0}}a{{color:#8cd2ff}}img,svg{{max-width:100%;height:auto}}svg rect{{fill:#244b70;stroke:#639dcc}}svg text{{fill:white;stroke:none;text-anchor:middle;font-size:16px}}svg path{{stroke:#8cc5e8;fill:none}}pre{{white-space:pre-wrap}}small{{color:#aec0d6}}</style>
<h1>v77 r4：CPU几何配对与采样修正</h1><p>本页是GPU前的来源与空间预案。不是新合成训练数据，不是DriveEditor生成结果。<a href="../v77-target-protected-r3/data_review.html">r3实际训练输入逐帧打分</a> · <a href="../v77-target-protected-r3/index.html">r3报告</a></p>
<section><svg viewBox="0 0 1210 105" role="img" aria-label="CPU造数准备组件"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8" style="fill:#8cc5e8"/></marker></defs>{boxes}{arrows}</svg></section>
<section><h2>这轮实际完成</h2><p>32个未处理来源窗口 / 31个场景，其中26个此前未进入来源池。使用相同10Hz目标时刻、55ms最大匹配误差和180ms最大曝光间隔：旧“检查完整30帧再截取10帧”通过{s['old_outer_30_pass']}例；固定10帧最近邻通过{s['nearest_fixed_ten_pass']}例；有序且不重复的全局匹配通过{s['ordered_fixed_ten_pass']}例。保留失败及对照，不复制帧、不插帧。</p><p>真实曝光几何通过{s['receiver_windows']}例，真实LiDAR地面拟合通过{s['real_lidar_ground_pass']}例。配对得到{s['planned_cases']}个预案 / {s['planned_scenes']}个scene，类型{html.escape(str(s['planned_types']))}；独立来源抽帧{html.escape(str(s['independent_preflight_counts']))}。准备了{s['queued_mask_jobs']}个受保护实例mask任务，尚未执行。</p><p>CPU仅半核、2GB内存：单线程、流式元数据、按需提取RGB/LiDAR、最多缓存8个供体mask。所需文件{prep['required_files']}个，复用{prep['reused_files']}个，缺失{len(prep['missing_files'])}个。本轮GPU模型调用0，新增合格训练对0，训练0。</p></section>
<section><h2>哪些结论还不能下</h2><p>候选沿用r3同一相机轨迹回放、位移集合和物理门槛。前后次序、净距、地图、LiDAR支持逐帧检查；下图B/C框是GT包络，不是精确实例mask。包络交叠仅排序GPU前候选，不能替代最终A→B/C的真实像素遮挡比例、邻车保护、洞边界与实际masked-X检查。黄色是已审供体轮廓投影与6px洞上界；尚未作合成写回。</p><p>独立检查每例固定第6帧，gpt-6-sol / xhigh，未使用fast；只评价来源可辨认与明显空间问题，不从一帧认证时序。人工verdict全部留空。下一步必须新SAM2分割后继续原全帧标准，不能直接把本页预案拿去训练。</p></section>
{''.join(cards)}<section><h2>未通过的几何原因</h2><pre>{html.escape(json.dumps(s['rejection_counts'],ensure_ascii=False,indent=2))}</pre><small>拒绝计数是有限候选放置尝试数，不是scene失败率。原结果均保留。run：{root}</small></section></html>'''
    (out/'index.html').write_text(page)
    comparison=f'<section><h2>供体实例去重对照：没有收益，保留原排序</h2><p>前6个候选窗口在19/29个receiver里只覆盖1–3个真实实例。保持每轮每receiver最多6个供体×35个位移、相同物理门槛，增加一次按实例去重的对照：原排序得到{s["baseline_planned_cases"]}个预案，去重排序得到{s["distinct_instance_planned_cases"]}个，丢失C012且没有新增scene。此次不采用去重排序替换原规则，保留原4个预案和完整负对照；密集类仍为0。</p></section>'
    page=page.replace('<section><h2>哪些结论还不能下</h2>',comparison+'<section><h2>哪些结论还不能下</h2>')
    (out/'index.html').write_text(page)
    ev=repo/'docs/autoresearch/worldsim_v77/target_protected_20260929/r4';ev.mkdir(parents=True,exist_ok=True)
    for f in ['summary.json','run_config.json','planner_config.json','preparation_native10.json','gpu_queue.json','independent_preflight_reviews.json','long_window_control_stopped.json','donor_rank_diversity.json']:(ev/f).write_text((root/f).read_text())
    (ev/'distinct_instance_planner_config.json').write_text((root/'diverse_control/planner_config.json').read_text())
    (ev/'sampling_control.json').write_text((root/'native10_factory/sampling_control.json').read_text())
    (ev/'preflight_manifest.json').write_text((out/'preflight_manifest.json').read_text())
    dump(ev/'source_selection.json',{'clips':[{'source_id':c['source_id'],'scene':c['scene'],'camera':c['camera'],'start_keyframe':c['start_keyframe'],'actor_tokens':[a['instance_token'] for a in c['actors']]} for c in read(root/'factory/source_selection.json')['clips']]})
    dump(ev/'plan_inventory.json',{'plans':[{'case_id':p['case_id'],'source_id':p['source_id'],'donor_source_id':p['donor_source_id'],'planned_type':p['planned_type'],'planning_control':p['planning_control'],'min_GT_clearance_m':p['min_GT_clearance_m'],'max_ground_support_distance_m':p['max_ground_support_distance_m'],'max_view_yaw_delta_deg':p['max_view_yaw_delta_deg'],'required_protected_instances':p['required_protected_instances']} for p in read(root/'selected_pair_plan.json')['plans']]})
    (ev/'review_link.md').write_text('[CPU预检报告](C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r4/index.html)\n')
    report=f'''# v77 r4：CPU有序曝光与空间配对

`WS-V77-TARGET-PROTECTED-20260929/r4`；用户明确仅CPU继续。完整run `{root}`。保持DriveEditor架构、真实Y监督与五项训练合同。

```mermaid
flowchart LR
 X[新train来源] --> T[固定10帧有序曝光匹配]
 T --> G[真实LiDAR / 地图 / 全帧GT几何]
 D[已有已审供体mask] --> G
 G --> Q[CPU预案 / Sol固定一帧QA]
 Q --> M[等待GPU SAM2精确保护mask]
 M --> H[原全帧合成合同 / 人工全检]
```

32个未处理窗口、31个scene，26个是新的来源scene；限定现有03/07公共分片、Boston地图。旧30帧外层门槛仅{s['old_outer_30_pass']}例通过；本轮实际需要固定网格10:20。修正后最近邻{s['nearest_fixed_ten_pass']}例、有序唯一曝光匹配{s['ordered_fixed_ten_pass']}例，通过实际曝光几何{s['receiver_windows']}例。匹配误差55ms、间隔180ms未放宽，目标时刻未变；四项控制测试通过。保留原30帧对照，停止不必要的旧提取作业，已提取原件复用。

在真实LiDAR地面/地图/净距/视角/连续性门槛下，{s['real_lidar_ground_pass']}个地面检查通过，{s['planned_cases']}个预案/{s['planned_scenes']}个scene，类型{s['planned_types']}。来源独立单帧{s['independent_preflight_counts']}，GPU队列{s['queued_mask_jobs']}个实例，disabled。包络覆盖比例只是CPU规划代理，精确mask未获得，不能宣称密集遮挡训练样本已合格。固定每例第6帧独立检查、全帧机器几何，不认证全视频时序。GPU调用0、新合成0、训练0、人工null。

两分片仅按需解包{prep['required_files']}个文件，其中复用{prep['reused_files']}个；缺失{len(prep['missing_files'])}。半核/2GB，单线程、流式元数据与8供体mask缓存。GPU后须执行真实SAM2保护mask、原精确遮挡/可见比例与写回合同、独立QA、人工全检。并非改模型或训练的证据。

[轻量证据](../autoresearch/worldsim_v77/target_protected_20260929/r4/summary.json) · [CPU报告](../autoresearch/worldsim_v77/target_protected_20260929/r4/review_link.md)。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
'''
    report+=f'\n供体多样性控制：19/29个receiver的前6个窗口只覆盖1–3个真实实例。相同6供体×35位移预算，按实例去重对照得到{s["distinct_instance_planned_cases"]}例，原排序{s["baseline_planned_cases"]}例；去重丢失C012、新增scene为0，不采用替换，保留原排序。两组原始结果均保留。此有限对照未解决密集类缺额，不继续在本轮重复扫格；不能由此否定所有多样性采样。\n'
    report+=f'''\n## 复现与下一步入口

CPU检查：`CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/envs/motionproj/bin/python scripts/worldsim_v77/target_protected/iteration4/test_sampling.py`；GPU队列预检运行同目录`segment_receivers.py --root {root}`，默认只校验真实RGB、曝光、GT prompt与独立QA，不导入torch或加载模型。

只有用户重新明确开启GPU后，才使用`/root/autodl-tmp/envs/worldsim-v77-sam2/bin/python scripts/worldsim_v77/target_protected/iteration4/segment_receivers.py --root {root} --execute`。入口有独占锁、来源字段和全部PNG解码续跑检查。输出`segmented_receivers/`，仍须独立mask检查与原精确合成关卡。此命令本轮未执行。
'''
    (repo/'docs/v77/TARGET_PROTECTED_CPU_R4.md').write_text(report)
    (repo/'docs/RESEARCH_STATUS.md').write_text(f'''# 当前研究状态

2026-09-30，v77；`WS-V77-TARGET-PROTECTED-20260929/r4`。用户要求只用CPU。新来源准备及几何预案完成，[本轮报告](v77/TARGET_PROTECTED_CPU_R4.md)。32窗口/31scene，26新来源scene；固定10帧唯一曝光匹配{s['ordered_fixed_ten_pass']}例，通过实际曝光几何{s['receiver_windows']}例。旧30帧门槛与最近邻对照保留，未复制帧或放宽时间限制。

真实LiDAR与旧r3物理规则得到{s['planned_cases']}个CPU预案/{s['planned_scenes']}scene：{s['planned_types']}。独立来源抽帧{s['independent_preflight_counts']}，待GPU精确SAM2共{s['queued_mask_jobs']}个保护实例；队列disabled，未运行模型。包络交叠不能认证精确遮挡，本轮新合成0、训练0、人工null。

本地`outputs/v77-target-protected-r4/index.html`展示预案；r3真实合成与人工逐帧页不覆盖。下一步必须用户开GPU后获取新保护mask，再检查完整五项合同；候选不因规划通过自动准入。真实Y不变、DriveEditor架构不变，旧27技术候选与r3的24例保持独立。CPU工作已收口，GPU不可自启，未设自动化或关机。subagent默认gpt-6-sol/xhigh，禁fast。

run：`{root}`。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
''')
    exp=repo/'docs/EXPERIMENTS.md';t=exp.read_text();marker='WS-V77-TARGET-PROTECTED-20260929 / r4';assert marker not in t
    exp.write_text(t.replace('|---|---|---|','|---|---|---|\n'+f'| {marker} | CPU固定10帧{s["ordered_fixed_ten_pass"]}/32；几何预案{s["planned_cases"]}例；SAM2队列{s["queued_mask_jobs"]}实例，未推理/训练 | [报告](v77/TARGET_PROTECTED_CPU_R4.md) · [证据](autoresearch/worldsim_v77/target_protected_20260929/r4/summary.json) |',1))
    fail=repo/'docs/research_failures/entries/V77-F02.md';t=fail.read_text();marker='## r4：实际窗口曝光匹配与新receiver空间预案';assert marker not in t
    fail.write_text(t.rstrip()+f'''\n\n{marker}

`WS-V77-TARGET-PROTECTED-20260929/r4`。旧入口先验30帧再截取10帧，错误让窗口外异常拒绝合法短窗；独立最近邻还会重复选曝光。固定同一10Hz时刻、55ms误差、180ms间隔，有序唯一匹配后{s['ordered_fixed_ten_pass']}/32（固定10帧最近邻{s['nearest_fixed_ten_pass']}/32，旧外层{s['old_outer_30_pass']}/32），保留对照、无插帧。不是模型能力改进。

不再重复旧receiver池密集扫格，选32个未处理窗口做真实LiDAR与原物理条件配对，得到{s['planned_cases']}个预案，类型{s['planned_types']}；独立来源{s['independent_preflight_counts']}。包络是CPU规划代理，精确保护mask与最终实际遮挡尚未验证，GPU队列{s['queued_mask_jobs']}、未执行；新合格合成0。必须继续SAM2原标准与人工全检，不能用预案统计充当密集类成功。[报告](../../v77/TARGET_PROTECTED_CPU_R4.md)、[证据](../../autoresearch/worldsim_v77/target_protected_20260929/r4/summary.json)。failure_ledger_delta: updated V77-F02。

按真实供体实例去重的有限对照使用相同6供体×35位移预算，原排序4例、去重3例，丢失C012且新增scene为0。停止此替换，保留原排序及负对照，不据此宣称多样性采样整体无效。密集类仍0；此轮新知识是先修正实际短窗采样接口能够增加可检查来源，但仅去重供体排序不足以增加合法空间配对。
''')
    print('FINALIZED',s)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();main(a.root,a.repo)
