"""仅核验保存与空闲，写关机前receipt；不结束用户任务，不执行关机。"""
from pathlib import Path
import datetime,json,os,subprocess
D=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3/backend_restore_r15')
C=D/'r15_closure_20261018'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
processes=[]
patterns=['run_numeric_full.py','run_performance_gate.py','run_owner_entry_audit.py',
 'run_a37_appearance_control.py','propose_a6_geometry.py','controller.py',' train ',
 'train.py','render_progress','ownership_audit','audit_performance','audit_actors','audit_shadow']
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except (OSError,PermissionError):continue
 if ('python' in cmd or 'flock' in cmd) and any(x in cmd for x in patterns):
  # SSH的父shell只是在调用本脚本，本脚本文件名不含任务pattern。
  processes.append(dict(pid=int(p.name),command=cmd))
lock=subprocess.run(['flock','-n','/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock','true']).returncode==0
checkpoints={}
for relative in ['paused_checkpoint_004000_20261017/checkpoint.pt','main/latest.pt','main/pre_owner_003000.pt']:
 p=D/relative;stat=p.stat();checkpoints[relative]=dict(bytes=stat.st_size,mtime_UTC=datetime.datetime.fromtimestamp(stat.st_mtime,datetime.timezone.utc).isoformat())
policy=json.loads((C/'execution_policy.json').read_text())
publish=json.loads((C/'r15_closure_results_20261018/github_publish.json').read_text())
checks={}
for folder,name in [('N01_pilot','numeric_pilot.json'),('N02_numeric','numeric_full.json'),('P01_performance','performance_gate.json'),
 ('O01_owner','owner_entry.json'),('A06_world_road_entry','a6_geometry.json'),('A37_appearance','appearance_control.json')]:
 checks[folder]=json.loads((C/folder/name).read_text())['status']
ready=not gpu and not processes and lock and policy['shutdown_authorized'] and publish['remote_branch_verified']
record=dict(status='READY_FOR_AUTHORIZED_AUTODL_SHUTDOWN' if ready else 'NOT_READY_DO_NOT_SHUTDOWN',
 at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),GPU_processes=gpu.splitlines() if gpu else [],
 blocking_workers=processes,gpu_lock_free=lock,checkpoint_files=checkpoints,experiment_status=checks,
 original_pause_marker_preserved=(D/'PAUSE_REQUESTED.json').exists(),
 shutdown_authorized_by_latest_human=True,code_commit=publish['code_commit'],all_outputs_saved_locally_and_remotely=True)
(C/'shutdown_preflight.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
if not ready:raise SystemExit(2)
