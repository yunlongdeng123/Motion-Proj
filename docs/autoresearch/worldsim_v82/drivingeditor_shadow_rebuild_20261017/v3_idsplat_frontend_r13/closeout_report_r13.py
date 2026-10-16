from pathlib import Path
import json,datetime,re,ast,shutil,zipfile
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/shadow_rebuild_v3';D=O/'idsplat_frontend_r13';now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
review=dict(status='ROOT_POINTSET_REVIEW_NOT_ACCEPTED_FOR_TRAINING',reviewer='root agent, no subagents',human_verdict=None,problem_solved=False,numerical_gate=False,scene_training_updates=0,
 reviewed_actors=[5,6,12,18,25,26,37],reviewed_images=[f'a{a:02d}_pointset.png' for a in [5,6,12,18,25,26,37]],
 positives=['a5/a26 flat road-base point layers greatly reduced while visible body surface stays represented','a37 large base reduced, finite low points remain','6 of7 meet predeclared body and lowerbody retention proxy'],
 limitations=['a12 visible mask is a partial sliver behind other parked vehicles; QUERY body proxy not exact manual ownership','a12 common-mask lift loses most coverage before DBSCAN; no positive road conflicts for that Actor','body coverage proxies cannot certify unobserved opposite surfaces or train-time split/owner correctness','DINOv3 gives47 transforms from56 planned pairs;9 have insufficient distinct feature correspondences; several GT disagreements are large','no new learned assets or edit videos; current result is initialization only'],
 decision='Do not start training or①②④. Preserve finite useful front-end evidence and failed gate; obey prior stop/shutdown condition after push and live task/queue idle verification.',at=now)
(D/'root_pointset_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf8')
for name in ['state.json','experiment_schedule.json']:
    p=W/name;v=json.loads(p.read_text(encoding='utf8'));v.update(status='IDSPLAT_R13_POINTSET_GATE_NOT_ACCEPTED_CLOSING',updated_at=now,active_campaign='idsplat_frontend_r13',next_action='Save/push complete R13 evidence, verify no owned or other GPU/queue work, AutoDL shutdown; no further training authorized within this failed route.',training_controller='NONE: B/C stopped, all R13 pointset/feature processes completed',scheduled_or_background_controller_started=False,
        next_visual_gate_global_step=None,real_scene_training_updates=19500,charged_training_updates_range=[19500,19549],formal_budget_remaining_conservative=25451,historical_monitor_status='b-c-15 to be paused after finite failed pointset closeout',IDSplat_r13_root_review=review,actual_problem_solved=False)
    v['shutdown_policy'].update(condition='Latest final IDSplat pointset gate not passed for all7; no new SGD; closeout/push/actual-all-idle then shutdown',authorized=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8');(O/('experiment_state.json' if name=='state.json' else name)).write_bytes(p.read_bytes())
p=D/'report.html';s=p.read_text(encoding='utf8')
s=s.replace('7 辆车各取 8 个跨帧配对，共 56 对，调用官方余弦匹配＋三点 Umeyama / 100000 次 RANSAC。','7 辆车各规划 8 个跨帧配对，共 56 对；其中 9 对不同源点的特征对应不足，47 对得到官方余弦匹配＋三点 Umeyama / 100000 次 RANSAC 的变换估计。')
s=s.replace('<th>目标覆盖≥50%的配对</th>','<th>覆盖≥50% / 有效估计数</th>')
s=s.replace('<p id="closeout">等待根代理视觉复核和保存退出交接。</p>','<p id="closeout">本代理已检查全部 7 辆车的 RGB、共同轮廓、真实点投影及局部点集图。a5/a26 的平坦路面基座明显减少；a37 大片基座减少但仍有低处点。a12 只显示其他停车车辆后方的局部可见片段，覆盖代理仍未保住，不能据此验收整个实例。按既定全部 7 车门槛停止训练，先保存推送，再核对空闲关机。</p><p><a href="root_pointset_review.json">本代理视觉复核记录</a> · <a href="actual_data_audit.json">原始来源 / 坐标 / 共同输入核验</a> · <a href="source_r13.zip">本轮代码与结果包</a> · <a href="evidence_and_interface.zip">实际点集接口、待定来源与 GT</a></p>')
p.write_text(s,encoding='utf8')
unified=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-15/file-c-users-dengyunlong-documents-codex/outputs/DrivingEditor_unified_report.html');t=unified.read_text(encoding='utf8');marker='<!-- IDSPLAT_R13_ENTRY_START -->'
entry=f'''{marker}<section><h2>⑥ IDSplat 前端点集对照 R13（最新）</h2><p>已停止 B/C；固定 GT 与共同 mask，实际 mask / 深度 / DBSCAN 和 DINOv3＋RANSAC。6 车保体达标，a12 未通过；本轮新增训练 0 步，不能宣称编辑已修好。</p><p><a href="{(D/'report.html').as_uri()}">打开七车点集、分步诊断、组件图和完整指标</a> · <a href="{(O/'ownership_controls_r12_sam3/A_quick_report/report.html').as_uri()}">A 与旧 18k 比较</a></p></section><!-- IDSPLAT_R13_ENTRY_END -->'''
if marker in t:t=re.sub(r'<!-- IDSPLAT_R13_ENTRY_START -->.*?<!-- IDSPLAT_R13_ENTRY_END -->',lambda _:entry,t,flags=re.S)
else:t=t.replace('</body>',entry+'</body>') if '</body>' in t else t+entry
unified.write_text(t,encoding='utf8')
for path in [O/'gpu_experiment.html',O/'ownership_controls_r12_sam3/report.html',O/'ownership_controls_r12_sam3/A_quick_report/report.html',O/'controls49_comparison.html']:
    t=path.read_text(encoding='utf8');notice=f'<p style="padding:15px;background:#fff0e8;border:1px solid #dba184"><strong>2026-10-17 最新：</strong>B/C 已按要求停止，C 未运行；IDSplat 前端点集对照 6 车保体达标、a12 未通过，新训练 0 步。<a href="{(D/"report.html").as_uri()}">查看最新点集报告</a>。</p>'
    if 'IDSPLAT_R13_LATEST_NOTICE' not in t:t=t.replace('</h1>','</h1><!-- IDSPLAT_R13_LATEST_NOTICE -->'+notice,1)
    path.write_text(t,encoding='utf8')
scripts=['stop_BC_for_IDSplat_r13.py','register_IDSplat_r13.py','idsplat_frontend_r13.py','evaluate_IDSplat_r13.py','dinov3_registration_r13.py','export_pointset_review_r13.py','audit_frontend_r13.py','monitor_IDSplat_r13.py','report_IDSplat_r13.py','closeout_report_r13.py']
for name in scripts:ast.parse((W/name).read_text(encoding='utf8'))
with zipfile.ZipFile(D/'source_r13.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in scripts:z.write(W/name,'campaign/'+name)
    for p in D.glob('*.json'):z.write(p,'records/'+p.name)
    z.write(D/'architecture.svg','architecture.svg')
    z.write(D/'report.html','report.html')
    for p in D.glob('*_pointset.png'):z.write(p,p.name)
print(json.dumps({'root_review':'not accepted for SGD','scripts_AST_passed':len(scripts),'unified_entry_updated':True,'new_training_updates':0},ensure_ascii=False))
