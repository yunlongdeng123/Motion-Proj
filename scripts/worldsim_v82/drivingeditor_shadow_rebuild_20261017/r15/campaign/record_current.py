from pathlib import Path
import subprocess,json,datetime
W=Path(__file__).resolve().parent;P=W.parent
r=subprocess.run(['C:/Program Files/Python311/python.exe','-X','utf8',str(P/'remote.py'),str(W/'monitor_remote.py')],check=True,stdout=subprocess.PIPE)
v=json.loads(r.stdout.decode('utf8'));(W/'latest_monitor.json').write_text(json.dumps(v,indent=2),encoding='utf8')
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat();step=v['progress']['global_step'];status=v['pipeline']['status']
protocol=json.loads((W/'protocol.json').read_text(encoding='utf8')) if (W/'protocol.json').exists() else None
for name in ['state.json','experiment_schedule.json']:
    p=P/name
    if p.exists():
        old=json.loads(p.read_text(encoding='utf8'))
        if old.get('run_id')!='backend_restore_r15' and not (W/('prior_'+name)).exists():(W/('prior_'+name)).write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf8')
    else:old={}
    state=dict(task_id='WS-V82-DRIVINGEDITOR-SHADOW-REBUILD-20261017',run_id='backend_restore_r15',status='R15_'+status,updated_at=now,no_subagents=True,GPU_started=True,code_snapshot='implementation_r15_train_v3',data='backend_restore_r15/data',remote_root='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3',training_output='backend_restore_r15/main',training_controller='backend_restore_r15/controller.py',actual_new_run_updates=step,actual_updates_semantics='last logged minimum; log interval50; bootstrap1 is included; discarded early-topology tail199..248 separately charged; main restored before early topology',historical_updates_range=[25049,25196],charged_completed_updates_range=[25049+step,25196+step+49],planned_updates=dict(geometry=30000,material=10000,joint=5000,total=45000),formal_budget_remaining=45000-step,last_monitor=v,R15_protocol=protocol,CPU_tests=22,actual_GPU_selfcheck_passed=True,actual_nonempty_Adam_restore_passed=True,eligible_delete_actor_indices=[5,6,12,18,25,26,37],legacy_actor_mapping={'0':5,'1':26,'2':18},historical_monitor_status='b-c-15 ACTIVE to be verified by automation receipt',previous_R14='paused/retained at recoverable5000, last log5350; no R14 resume',next_action='Monitor registered restored-backend run; root inspect1k/3k/9k/30k/40k/45k previews, review gates before material, then full49x65 outputs',shutdown_policy=dict(authorized=True,condition='Only after complete registered training, full49 edits/root review/report/source push and all own/otherGPU/queued tasks idle; temporary idle is not completion'),human_verdict=None,actual_problem_solved=None,failure_ledger_refs=['V82-F01'],failure_ledger_delta='none',github_publish=old.get('github_publish'),report='outputs/shadow_rebuild_v3/r15/report.html')
    rates={x['stage']:x['seconds_forward_backward'] for x in v['selfcheck']['stages']}
    state['training_only_remaining_hours_initial_size']=(max(30000-step,0)*rates['geometry']+max(40000-max(step,30000),0)*rates['material']+max(45000-max(step,40000),0)*rates['joint'])/3600
    state['ETA_scope']='initial-seed fwd/back only; Adam, densification, teacher, audits, IO and full49 rendering excluded; update from actual logs'
    if name=='experiment_schedule.json':state['stages']=[dict(id='R15-input',status='COMPLETE'),dict(id='R15-selfcheck',status='CPU22_AND_ACTUAL_GPU_PASSED'),dict(id='R15-training',status=status),dict(id='R15-full49',status='PENDING'),dict(id='R15-delivery',status='PENDING')]
    p.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(step=step,status=status,controller=v['controller']['pid'],GPU=v['GPU'],training_only_remaining_hours_initial_size=state['training_only_remaining_hours_initial_size'])))
