"""Refresh this transition from live evidence; runs on wm-vgpu-1011, CPU only."""
import datetime, hashlib, json, os, pathlib, subprocess

BASE=pathlib.Path('/root/autodl-tmp/drivingeditor_repro')
D=BASE/'campaign/vgpu_bugfix_20261012'
P=D/'loss_audit/long_pair_30k_32000/materialized_plan'
B1=D/'continuation_diagnostic/original30k_to32000'
B2=D/'continuation_diagnostic/original30k_to32000_noise_reference'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(P/'continuation_launcher_status.json')['actual_exit_code']==0
summary=read(B1/'receipts/full49_summary.json')
assert summary['status']=='all_long_original_receipts_verified_CPU'
launcher=read(P/'baseline_noise_reference/continuation_launcher_status.json')
status=read(B2/'status.json')
rows=[json.loads(x) for x in (B2/'step_losses.jsonl').read_text().splitlines()]
assert rows and rows[0]['iteration']==30001
proc=pathlib.Path('/proc')/str(launcher['worker_pid'])/'cmdline'
cmd=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
runtime={'checked_utc':now,'experiment':B2.name,'last_completed_step':rows[-1]['iteration'],'recorded_steps':len(rows),'status':status['status'],'launcher_status':launcher['status'],'worker_pid':launcher['worker_pid'],'launcher_pid':launcher['launcher_pid'],'actual_worker_cmd':cmd,'GPU_compute_apps':apps,'exit_record_present':(P/'baseline_noise_reference/continuation_launcher.exit').exists(),'live_read_required_for_next_action':True}
manifest_path=D/'task_manifest.json';m=read(manifest_path)
m.update(updated_utc=now,current_experiment=B2.name,current_experiment_status='B2_running' if cmd else 'B2_requires_completion_audit',gpu_owner='vgpu_component_diagnostic_solmax',gpu_owner_scope='Only root-approved original-repeat B2 30001..32000',gpu_status_path=str(B2.relative_to(D)/'status.json'),root_receipt=str((P/'baseline_noise_reference/root_coordination_receipt.json').relative_to(D)),next_experiment='After B2 actual complete: real paired sampling/GT/noise receipts; newly bind long V, CPU prepare, independent review, exact root approval, exclusive GPU launch.')
m['long_three_arm'].update(B1='complete_exit0_all49_fourstate_verified',B2='running' if cmd else 'awaiting_completion_audit',V='materialized_source_only_real_B2_reference_pending',B1_completion_receipt_sha256=sha(B1/'receipts/full49_summary.json'),B1_terminal_manifest_sha256=sha(B1/'terminal_32000/diagnostic_snapshot_manifest.json'),B2_root_receipt_sha256=sha(P/'baseline_noise_reference/root_coordination_receipt.json'),historical_launcher_pid=launcher['launcher_pid'],historical_worker_pid=launcher['worker_pid'],started_utc=launcher['started_utc'],iteration_at_verification=rows[-1]['iteration'])
m['long_B1_result']={'composite_full49_mean_frame_psnr_db':summary['final49_regions']['full']['composite']['mean_frame_psnr_db'],'observed_region':summary['final49_regions']['non_sky_outside_actor'],'receipt':str((B1/'receipts/full49_summary.json').relative_to(D)),'interpretation':'Composite recovered by32000; static road confetti and dynamic background persist. Not only reset transient; not normal separation.','root_visual_check':'Viewed actual raw10 30001/30300/31000/32000 six-channel recovery sheet.'}
m['latest_runtime_check']=runtime
m['acceptance_not_yet_met']=True
tmp=manifest_path.with_suffix('.next.json');tmp.write_text(json.dumps(m,indent=2,allow_nan=False)+'\n');os.replace(tmp,manifest_path)
receipt=D/'heartbeat_checks/20261011T2051_B2_transition.json';receipt.parent.mkdir(exist_ok=True)
receipt.write_text(json.dumps(runtime,indent=2)+'\n')
print(json.dumps(runtime))
