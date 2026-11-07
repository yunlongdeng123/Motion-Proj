from pathlib import Path
import os,signal,json,time
ROOT=Path(__file__).resolve().parent.parent
status=json.loads((ROOT/'queue_status.json').read_text())
assert status['controller_pid']==2816 and status['stage']=='train20k',status
pid=status['controller_pid'];child=status['child_pid']
parent_cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
child_cmd=Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert str(ROOT/'source/run_single_longrun.py') in parent_cmd
assert str(ROOT/'source/train_longrun.py') in child_cmd and '--stop-iteration 20000' in child_cmd
assert not (ROOT/'continuation30k').exists()
receipt=dict(status='SCOPE_CHANGED_STOP_AT_COMPLETE20K_NO30K_RELEASE',unix=time.time(),
    reason='User switched mainline to full dynamic_actor_ids=[5,6,18,26,37] and delete_actor_ids=[37] query filtering',
    cancelled_future_iterations=30000,final_native_iteration=20000,planned_actual_new_update_cap=8023,
    actual_new_updates_not_precounted=True,old_engineering6599_unchanged=True,
    terminated_controller_pid=pid,preserved_running_trainer_pid=child,
    prior_controller_status=status,parent_command=parent_cmd,trainer_command=child_cmd,
    signal_scope='SIGTERM to this exact controller PID only; no process group or child signal',
    training_code_and_method_modified=False,shutdown=False,reset_card=False)
(ROOT/'scope_change_stop20k.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
os.kill(pid,signal.SIGTERM)
for _ in range(20):
    if not Path(f'/proc/{pid}').exists():break
    stat=Path(f'/proc/{pid}/stat').read_text().split()[2]
    if stat=='Z':break
    time.sleep(.1)
assert Path(f'/proc/{child}').exists()
receipt['trainer_still_alive_after_parent_signal']=True
receipt['controller_state_after_signal']=Path(f'/proc/{pid}/stat').read_text().split()[2] if Path(f'/proc/{pid}').exists() else 'EXITED'
assert receipt['controller_state_after_signal'] in ['EXITED','Z']
assert not (ROOT/'continuation30k').exists()
(ROOT/'scope_change_stop20k.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
(ROOT/'queue_status.json').write_text(json.dumps(dict(stage='TRAIN20K_ONLY_PARENT30K_RELEASE_TERMINATED',child_pid=child,
    terminated_controller_pid=pid,shutdown=False,reset_card=False),indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
