"""Close the observed storage wait using actual saved before/after samples."""
from pathlib import Path
import datetime,json,shutil
W=Path(__file__).parent;O=W.parents[1]/'outputs/baseline_engineering_20261018'
rows=[json.loads(line) for line in (W/'runtime_io_health_history.jsonl').read_text(encoding='utf-8').splitlines()]
blocked=[x for x in rows if x['status']=='BLOCKED_HOST_STORAGE_WRITE_WAIT']
last=blocked[-1]
after=next(x for x in rows if x['at']>last['at'] and x['status']=='NO_PERSISTENT_MD_WRITE_WAIT_OBSERVED' and x['array_state'] in ['active','clean'] and x['A_reported_status']['step']>last['A_reported_status']['step'])
assert last['A_worker']['pid']==after['A_worker']['pid']==79624
assert last['A_worker']['cmdline']==after['A_worker']['cmdline']
assert 'md_write_start' not in after['A_worker']['wchan']
assert float(after['gpu_csv'].split(',')[-1])>0
b=json.loads((W/'live_B_native_status.json').read_text(encoding='utf-8'));assert b['status']=='complete' and b['iteration']==30000
disk=json.loads((W/'remaining_disk_forecast.json').read_text(encoding='utf-8'));assert disk['at']>after['at'] and disk['fits_this_forecast']
r=dict(status='OBSERVED_STORAGE_RECOVERY_WITH_CONTINUED_A_PROCESS',recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 last_confirmed_blocked_at=last['at'],first_confirmed_recovered_at=after['at'],exact_platform_recovery_time_known=False,
 last_blocked_A_step=last['A_reported_status']['step'],first_recovered_A_step=after['A_reported_status']['step'],
 original_A_pid=79624,same_pid_and_command_observed=True,original_guardian_pid=79622,
 first_recovered_GPU_csv=after['gpu_csv'],platform_root_cause_known=False,agent_restart_performed=False,agent_replay_updates=0,agent_training_updates=0,
 blocked_forecast_helper=dict(pid=252862,observed_absent_after_recovery=True,original_command_exit_code=0,old_forecast_timestamp='2026-10-18T23:15:34.556411+00:00',old_result_not_treated_as_current=True),
 refreshed_forecast=disk,B_complete_step=30000,final_A_B_delete_evaluations_complete=False,shutdown_performed=False,human_verdict=None,
 last_blocked_sample=last,first_recovered_sample={k:v for k,v in after.items() if k!='interpretation'})
for base in [W,O]:(base/'runtime_io_recovery_receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:r[k] for k in ['status','last_confirmed_blocked_at','first_confirmed_recovered_at','first_recovered_A_step','agent_replay_updates']}))
