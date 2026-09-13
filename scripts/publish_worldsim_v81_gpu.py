"""发布GPU探索证据；大原生tensor留在runs，仓库只收关键图与小统计。"""
import json,shutil,subprocess,datetime,tarfile,html
from pathlib import Path
RUN=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01');REPO=Path('/root/autodl-tmp/motion_proj');ANA=RUN/'analysis';DOC=REPO/'docs';DEL=RUN/'delivery';DEL.mkdir(exist_ok=True)

def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s.rstrip()+'\n',encoding='utf-8')
jobs=[json.loads(l) for name in ['execution_queue.jsonl','texture_queue.jsonl'] for l in (RUN/name).read_text().splitlines()]
expected={(j['method'],j['window_id'],j.get('output_key',j['variant'])) for j in jobs};actual={}
for method in ['dvgt','vggt']:
    for rp in (RUN/method).rglob('result.json'):
        r=json.loads(rp.read_text());actual[(method,r['window'],rp.parent.name)]=r
metrics=list(map(json.loads,(RUN/'evaluation/metrics.jsonl').read_text().splitlines()))
errors=list(RUN.glob('dvgt/**/error.json'))+list(RUN.glob('vggt/**/error.json'))
audit={'task_id':'WS-V81-GPU-P2-01','run_id':'20260913-dual3090-r1','status':'P2_PRIMARY_DISCOVERY_COMPLETE','expected_jobs':len(expected),'actual_jobs':len(actual),'missing_jobs':[list(x) for x in sorted(expected-set(actual))],'unexpected_jobs':[list(x) for x in sorted(set(actual)-expected)],'errors':[str(x) for x in errors],'roles':{role:sum(r['role']==role for r in actual.values()) for role in ['DISCOVERY','VIEW_DIAGNOSTIC','SYNTHETIC_DIAGNOSTIC']},'dvgt_fixed_export_contract':sum(r.get('export_contract')=='dvgt_rdf_scale0.1_camera_ego_v1' for r in actual.values()),'metric_rows':len(metrics),'main_full6_coverage':{},'validation':'7 geometry checks and 1 native coordinate/scale/timestamp regression passed','promotion':'NO_GO_PENDING_RELIABLE_FAILURE','human_verdict':None,'failure_ledger_refs':['V81-F01','V81-F02','V81-F03'],'failure_ledger_delta':'V81-F04','completed_at':datetime.datetime.now().isoformat(),'automatic_resume':False,'shutdown_performed':False}
for method in ['dvgt','vggt']:
    rr=[r for r in metrics if r['method']==method and r['role']=='DISCOVERY'];audit['main_full6_coverage'][method]={'rows':len(rr),'coverage_below_0_9':sum(r['coverage']<.9 for r in rr),'mae_unavailable':sum(r['mae_m'] is None for r in rr)}
audit['gpu_snapshot']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True)
proc=subprocess.check_output(['ps','-eo','pid,stat,args'],text=True);audit['active_research_workers']=[l for l in proc.splitlines() if ('scripts/run_worldsim_v81_' in l or 'scripts/evaluate_worldsim_v81.py' in l) and 'python ' in l]
if audit['missing_jobs'] or audit['unexpected_jobs'] or len(actual)!=440 or audit['dvgt_fixed_export_contract']!=220:raise RuntimeError('Delivery count/contract mismatch')
write(ANA/'completion_audit.json',json.dumps(audit,indent=2));write(RUN/'status.json',json.dumps(audit,indent=2))

keyfigs=['architecture.png','paired_view_effects.png','same_scene_views.png','texture_diagnostic.png','outside_roi_scale_control.png','case_scene-0071_cd3039e0_CAM_BACK_LEFT_01.png','case_scene-0139_7e27d5c0_CAM_BACK_LEFT_12.png','case_scene-0632_1b3e964e_CAM_FRONT_RIGHT_13.png']
repofig=DOC/'figures/worldsim_v81_gpu';repofig.mkdir(exist_ok=True)
for name in keyfigs:shutil.copy2(ANA/'figures'/name,repofig/name)
evidence=DOC/'autoresearch/worldsim_v81/gpu_p2';evidence.mkdir(parents=True,exist_ok=True)
small=['completion_audit.json','cohort_descriptive.json','paired_view_effects.json','runtime_summary.json','scale_summary.json','exploratory_selection.json','posthoc_visual_reviews.json','anchor_diagnostics.json','outside_roi_scale_control.json']
for name in small:shutil.copy2(ANA/name,evidence/name)
shutil.copy2(RUN/'contract_repair.json',evidence/'contract_repair.json')
write(evidence/'README.md','# V8.1 GPU P2证据\n\n440项推理，6324条ROI评价；原生输出与全31图位于`'+str(RUN)+'`。8张关键图入库，完整交付包见run内delivery。自然四格匹配=0，V8.2 NO_GO_PENDING_RELIABLE_FAILURE。旧错误DVGT导出只在retired_contract_r0中，不能加入当前评价。\n')

reviews=json.loads((ANA/'posthoc_visual_reviews.json').read_text());review_by={r['roi_id']:r for r in reviews};explore=json.loads((ANA/'exploratory_selection.json').read_text())
atlas_md='# V8.1 GPU Failure / Goodcase Atlas\n\n![Architecture components](figures/worldsim_v81_gpu/architecture.png)\n\n已完成两模型各220项真实推理。31张图包括24张逐案例对比、2张探索筛选复核图和5张机制/统计/架构图。完整gallery在run交付目录；所有案例为DISCOVERY，人工verdict=null。\n\n| 案例 | 复核 | 观察 |\n|---|---|---|\n'
for r in reviews:atlas_md+=f'| {r["roi_id"]} | {r["review"]} | {r["observation"]} |\n'
atlas_md+='\n8个CPU冻结卡、3个冻结视角干预目标以及posthoc选择规则均保留来源；不因结果不好删图。框选错绑参考的卡车/栅栏不能作为科学badcase，植被小误差不能作为goodcase。详见[科学报告](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md)和[V8.2决策](WORLDSIM_V8_1_TO_V8_2_DECISION.md)。\n\n'+''.join(f'![{n}](figures/worldsim_v81_gpu/{n})\n\n' for n in keyfigs[1:])
write(DOC/'WORLDSIM_V8_1_FAILURE_ATLAS.md',atlas_md)
handoff='# V8.1 GPU P2完成与后续入口\n\n![Architecture components](figures/worldsim_v81_gpu/architecture.png)\n\n状态P2_PRIMARY_DISCOVERY_COMPLETE；2×3090足够，当前推理worker均已结束，无自动恢复器，不关机。V8.2 NO_GO_PENDING_RELIABLE_FAILURE。\n\n全部输入/原生输出/指标/日志/图：`'+str(RUN)+'`。正式导出使用本run的input_manifests，补充首输入相机ego时间戳；旧CPU manifest仍保留，但缺此字段时新的DVGT runner会报错，先运行repair_worldsim_v81_dvgt_export.py生成补充输入。不要重新启动已完成440项任务。\n\n复现CPU评价：\n\n```bash\ncd /root/autodl-tmp/motion_proj\nPYTHONPATH=. OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/envs/motionproj/bin/python scripts/evaluate_worldsim_v81.py \\\n  --atlas /root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2 \\\n  --prediction-root '+str(RUN)+' \\\n  --out '+str(RUN)+'/evaluation_recheck --no-figures\n```\n\n新增实验使用新的task/run与输出路径，先冻结输入来源。下一步是补干净高重叠自然对照和可靠残余failure；GPU显存不是当前限制。实际runtime、完成数、缺失分母和合同修正见autoresearch/worldsim_v81/gpu_p2。\n'
write(DOC/'WORLDSIM_V8_1_GPU_HANDOFF.md',handoff)
write(DOC/'WORLDSIM_V8_1_PLAN.md','# V8.1执行入口\n\n原计划：[稀疏视角×低纹理失效发现](WorldSim_V81_SparseView_LowTexture_Failure_Discovery_Plan.md)。当前P2主模型阶段已完成，未进入V8.2。\n\n读取[科学报告](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md) → [图谱](WORLDSIM_V8_1_FAILURE_ATLAS.md) → [V8.2决策](WORLDSIM_V8_1_TO_V8_2_DECISION.md) → [GPU交接](WORLDSIM_V8_1_GPU_HANDOFF.md)。CPU r2和旧交接为历史，无需重复无卡准备。\n')
for name in ['RESEARCH_STATUS.md','EXPERIMENTS.md','RESEARCH_FAILURES.md']:
    p=DOC/name;s=p.read_text();head='# 当前：V8.1 P2_PRIMARY_DISCOVERY_COMPLETE（2026-09-13）\n\nWS-V81-GPU-P2-01：DVGT/GPU0、VGGT/GPU1各220项真实推理，共440；6324条ROI评价，31张图，15个posthoc唯一ROI视觉复核。峰值12.60/8.20GiB，2×3090足够。7项几何+1项原生坐标尺度时间戳检查通过；无训练、无自动恢复、未关机。\n\nDVGT公共相机对照没有一致稀疏化退化；最大4个非地面C11日志候选含卡车/网格/植被/遮挡混杂，灰墙有0.103m goodcase。raw VGGT大误差部分被区域外INPUT LiDAR全局尺度控制解释：招牌3图22.130→1.105m，低纹理墙7.678→0.413m。自然完整四格=0，未触发V8.2，NO_GO_PENDING_RELIABLE_FAILURE。H2–H5方法假设未检验，独立AV2质量未查看，人工verdict=null。\n\nfailure_ledger_refs=V81-F01/F02/F03；failure_ledger_delta=V81-F04。[科学报告与架构](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md)、[图谱](WORLDSIM_V8_1_FAILURE_ATLAS.md)、[接续](WORLDSIM_V8_1_GPU_HANDOFF.md)。以下为冻结历史。\n\n';write(p,head+s)

for name in ['WORLDSIM_V8_1_SCIENTIFIC_REPORT.md','WORLDSIM_V8_1_TO_V8_2_DECISION.md','WORLDSIM_V8_1_FAILURE_ATLAS.md','WORLDSIM_V8_1_GPU_HANDOFF.md','WORLDSIM_V8_1_METHOD_AUDIT.md']:
    s=(DOC/name).read_text().replace('figures/worldsim_v81_gpu/','figures/');write(DEL/name,s)
for n in ['V81-F03.md','V81-F04.md']:shutil.copy2(DOC/'research_failures/entries'/n,DEL/n)
s=(DEL/'WORLDSIM_V8_1_SCIENTIFIC_REPORT.md').read_text().replace('research_failures/entries/V81-F03.md','V81-F03.md');write(DEL/'WORLDSIM_V8_1_SCIENTIFIC_REPORT.md',s)
shutil.copytree(ANA/'figures',DEL/'figures',dirs_exist_ok=True)
for n in small:shutil.copy2(ANA/n,DEL/n)
shutil.copy2(RUN/'evaluation/metrics.jsonl',DEL/'metrics.jsonl')
shutil.copy2(RUN/'contract_repair.json',DEL/'contract_repair.json')
shutil.copy2(RUN/'execution_queue.jsonl',DEL/'execution_queue.jsonl');shutil.copy2(RUN/'texture_queue.jsonl',DEL/'texture_queue.jsonl')
cards=[]
for p in sorted((DEL/'figures').glob('case_*.png')):
    rid=p.stem[5:];rv=review_by.get(rid,{});note=rv.get('observation','CPU冻结案例；参考混杂和科学接受状态见CPU卡及当前报告。');status=rv.get('review','FROZEN_DISCOVERY_CANDIDATE')
    cards.append(f'<details class="case"><summary>{html.escape(rid)}<span>{html.escape(status)}</span></summary><p>{html.escape(note)}</p><img loading="lazy" src="figures/{p.name}"></details>')
body=''.join(f'<section><h2>{title}</h2><p>{note}</p><img src="figures/{name}"></section>' for name,title,note in [('architecture.png','两模型独立推理与评价','RGB只进入模型；held-out LiDAR只进入评价，INPUT LiDAR用于普通控制。'),('paired_view_effects.png','先看公共相机配对','DVGT变化集中在零附近；VGGT的深度差异需要先审计尺度。误差条以日志bootstrap，候选不是完整自然四格。'),('same_scene_views.png','保留同一个目标相机','3个CPU冻结目标；18图是三个时刻×六相机。相机数本身不等价于真实几何重叠。'),('outside_roi_scale_control.png','普通全局尺度控制解释多少','招牌3图22.130→1.105m；低纹理墙3图7.678→0.413m。只使用当前帧、目标ROI外的INPUT LiDAR，没有held-out拟合。'),('texture_diagnostic.png','合成纹理诊断','单个参考平面辅助mask；不能推广为自然低纹理因果结论。')])
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>V8.1 GPU科学发现图谱</title><style>body{margin:0;background:#eef2f6;color:#172a3c;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1450px;margin:auto;padding:30px}header,section,details{background:white;border:1px solid #dce4ec;border-radius:12px;padding:24px;margin:18px 0}h1{font-size:32px;margin:0}h2{font-size:23px}p{max-width:1050px}a{color:#086f83}img{display:block;width:100%;height:auto}.kpis{display:flex;gap:18px;flex-wrap:wrap}.kpis b{font-size:26px;color:#097e81}.notice{background:#fff1cf;padding:14px;border-radius:8px}summary{cursor:pointer;font-weight:650}summary span{display:block;color:#7e5430;font-size:13px}details p{color:#526273}.links a{margin-right:22px}footer{color:#667c8f;font-size:13px}</style><main><header><h1>V8.1 · 稀疏视角 × 低纹理</h1><p>双3090阶段 · 科学发现、badcase与goodcase边界 · 2026-09-13</p><div class="kpis"><div><b>440</b> 真实推理</div><div><b>27</b> 探索日志</div><div><b>6324</b> ROI评价记录</div><div><b>31</b> 可视化</div></div><p class="notice">V8.2：NO_GO_PENDING_RELIABLE_FAILURE。DVGT未显示一致稀疏化退化；raw VGGT大误差部分由普通尺度控制解释。自然完整四格=0，没有独立确认或真实渲染结论。</p><p class="links"><a href="WORLDSIM_V8_1_SCIENTIFIC_REPORT.md">完整科学报告</a><a href="WORLDSIM_V8_1_TO_V8_2_DECISION.md">V8.2决策</a><a href="completion_audit.json">执行与资源记录</a></p></header>'''+body+'<section><h2>逐案例对比：包括混杂与goodcase</h2><p>展开查看RGB、held-out支撑、DVGT/VGGT深度、误差和侧视图。15个唯一posthoc候选的选择与排除全部保留；未代填人工verdict。</p></section>'+''.join(cards)+'<footer>当前为P2主模型发现阶段完成；不是V8.1全部假设通过。原生tensor与所有run保留远端，无自动恢复、没有关机操作。</footer></main></html>'
write(DEL/'index.html',page)
with tarfile.open(RUN/'v81_gpu_report.tar','w') as tar:
    for p in DEL.rglob('*'):
        if p.is_file():tar.add(p,arcname=str(p.relative_to(DEL)))
print(json.dumps({'audit':audit,'delivery_files':len(list(DEL.rglob('*'))),'archive':str(RUN/'v81_gpu_report.tar')},indent=2))
