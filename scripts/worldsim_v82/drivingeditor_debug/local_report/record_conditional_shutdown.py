"""Persist user authorization, without touching any training files or power state."""
import datetime,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
p=root/'task_manifest.json'
m=json.loads(p.read_text(encoding='utf-8-sig'))
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
policy={
 'recorded_utc':stamp,
 'user_authorized':True,
 'host':'wm-vgpu-1011',
 'scope':'Power off only after the entire vGPU paper-version debugging task is genuinely complete and verified.',
 'completion_not_yet_met':True,
 'shutdown_now':False,
 'if_bugs_remain':'Continue debugging; keep hourly heartbeat ACTIVE.',
 'before_shutdown':['Complete algorithm/effect acceptance and required reproducibility checks; do not treat a short improvement as success.','Validate final dark Chinese HTML, architecture diagram, actual comparison media and metrics.','Sync deliverables locally and preserve remote source/data/checkpoints.','All task GPU jobs, evaluators, encoders and transfers finished; no unrelated user work interrupted.'],
 'power_off_only_no_instance_release_or_data_deletion':True,
 'confirm_shutdown_state_not_just_ssh_loss':True,
 'after_verified_delivery_and_shutdown':'Pause drivingeditor-50k heartbeat.',
 'additional_user_confirmation_required':False,
}
m['completion_shutdown_policy']=policy
m['updated_utc']=stamp
m['latest_runtime_check']={'checked_utc':'2026-10-11T19:50:29.161572+00:00','experiment':'original30k_to32000','last_completed_step':30257,'status':'controlled_training','worker_pid':36255,'launcher_pid':36252,'actual_commands_verified':True,'GPU_unique_worker_verified':True,'exit_record_present':False,'live_read_required_for_next_action':True}
p.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
policy_path=root/'completion_shutdown_policy.json'
policy_path.write_text(json.dumps(policy,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
for f in [p,policy_path]:
 subprocess.run(['scp',str(f),'wm-vgpu-1011:/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/'+f.name],check=True)
print(json.dumps({'status':'authorization_recorded','shutdown_executed':False,'continue_debugging':True}))
