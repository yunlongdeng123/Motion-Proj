from pathlib import Path
import json,subprocess,datetime
W=Path(__file__).resolve().parent;P=W.parent
current=json.loads((P/'state.json').read_text(encoding='utf8'))
if current.get('status')=='PAUSED_BY_USER_BACKEND_BASELINE_AUDIT':
    print(json.dumps(dict(status=current['status'],automatic_monitor_recording=False,reason='Latest explicit user pause; use read-only audit without rewriting canonical status')))
    raise SystemExit(0)
r=subprocess.run(['C:/Program Files/Python311/python.exe','-X','utf8',str(P/'remote.py'),str(W/'monitor_current.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
v=json.loads(r.stdout.decode());(W/'latest_monitor.json').write_text(json.dumps(v,indent=2),encoding='utf8')
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
pipeline=v['pipeline']['status'];failed='FAILURE' in pipeline
status='R14_ENGINEERING_FAILURE_REQUIRES_REPAIR' if failed else 'R14_FULL_TRAINING_RUNNING'
if pipeline=='ACTUAL_OWNER_REVIEW':status='R14_ACTUAL_OWNER_AUDIT_AND_REPAIR_RUNNING'
controller=v['controller'].get('script','full_controller.py');active_log=v.get('active_log','controller.log')
protocol=json.loads((W/'protocol.json').read_text());protocol['safety']['immutable_code']='implementation_r14_final';(W/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
for name in ['state.json','experiment_schedule.json']:
    p=P/name;s=json.loads(p.read_text(encoding='utf8'))
    s.update(status=status,updated_at=now,code_snapshot='implementation_r14_final',data='ownership_rebuild_r14/idsplat_frontend/data_IDSplat',training_output='ownership_rebuild_r14/main',training_controller='ownership_rebuild_r14/'+controller,training_log='ownership_rebuild_r14/'+active_log,training_launch='ownership_rebuild_r14/'+('full_controller_recovery_launch.json' if controller!='full_controller.py' else 'full_controller_launch.json'),scheduled_or_background_controller_started=bool(v['owned_live_processes']),last_monitor=v,historical_monitor_status='b-c-15 ACTIVE',next_action='Repair actual controller failure from preserved checkpoint; no replay or shutdown' if failed else 'Continue full22400 updates, actual owner/split audits at3k/9k, full49x65 edit evaluation and root review. Current mask drawing-only fix does not change frozen labels.',R14_protocol=protocol)
    s['real_scene_training_updates']=19500+v['progress']['global_step']
    s['actual_new_run_updates']=v['progress']['global_step'];s['historical_updates_range']=[19500,19549]
    s['estimated_training_remaining_hours_initial_size']=v['training_only_remaining_hours_initial_probe']
    s['estimated_duration_hours']=None
    s['formal_budget_remaining']=22400-v['progress']['global_step']
    s['charged_completed_updates_range']=[19500+v['progress']['global_step'],19549+v['progress']['global_step']]
    if s['github_publish'].get('run_id')!='ownership_rebuild_r14':s['github_publish']['scope']='R13 only; R14 not yet published'
    if name=='experiment_schedule.json':
        s['stages']=[dict(id='R14-input',name='区域有效域融合＋IDSplat真实点集',status='COMPLETE'),dict(id='R14-probe',name='真实CUDA ABI/梯度/吞吐探测',status='COMPLETE_ZERO_SGD'),dict(id='R14-full',name='G13000/M7000/J2400与持续归属约束',status='ENGINEERING_REPAIR_REQUIRED' if failed else pipeline),dict(id='R14-evaluation',name='49帧65流与逐车编辑/阴影反事实',status='PENDING'),dict(id='R14-report',name='根代理复核、报告、Git保存及空闲关机',status='PENDING')]
        s['input']=json.loads((W/'input_preflight.json').read_text())
    p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(step=v['progress']['global_step'],pipeline=v['pipeline']['status'],mask_frames=v['mask_visualization_frames'],controller=v['controller']['pid'],training_only_remaining_hours_initial_size=v['training_only_remaining_hours_initial_probe'])))
