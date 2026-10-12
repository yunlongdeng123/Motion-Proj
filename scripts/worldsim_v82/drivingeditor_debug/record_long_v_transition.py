"""Live CPU-only metadata refresh for the approved long V transition."""
import datetime,hashlib,json,os,pathlib,subprocess
BASE=pathlib.Path('/root/autodl-tmp/drivingeditor_repro');D=BASE/'campaign/vgpu_bugfix_20261012'
P=D/'loss_audit/long_pair_30k_32000/materialized_plan';V=D/'continuation_diagnostic/opacity_outside30k_to32000'
B2=D/'continuation_diagnostic/original30k_to32000_noise_reference'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
release=read(B2/'receipts/gpu_release_handoff.json');assert release['worker_exit_code']==release['launcher_exit_code']==0
launcher=read(P/'opacity_outside_launcher_status.json')
status=read(V/'status.json') if (V/'status.json').exists() else {'status':'initializing_before_status'}
rows=[json.loads(x) for x in (V/'step_losses.jsonl').read_text().splitlines()] if (V/'step_losses.jsonl').exists() else []
proc=pathlib.Path('/proc')/str(launcher['worker_pid'])/'cmdline';cmd=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
first=read(V/'first_backward_reference_comparison.json') if (V/'first_backward_reference_comparison.json').exists() else None
runtime={'checked_utc':now,'experiment':V.name,'last_completed_step':rows[-1]['iteration'] if rows else 30000,'recorded_steps':len(rows),'status':status['status'],'launcher_status':launcher['status'],'worker_pid':launcher['worker_pid'],'launcher_pid':launcher['launcher_pid'],'actual_worker_cmd':cmd,'GPU_compute_apps':apps,'exit_record_present':(P/'opacity_outside_launcher.exit').exists(),'actual_first_backward_original10_exact':all(v['exact_equal'] for v in first['comparisons'].values()) if first else None,'original_first_backward_returned':first.get('original_backward_returned') if first else None,'live_read_required_for_next_action':True}
path=D/'task_manifest.json';m=read(path)
m.update(updated_utc=now,current_experiment=V.name,current_experiment_status='V_running' if cmd else 'V_requires_completion_audit',gpu_owner='vgpu_component_diagnostic_solmax',gpu_owner_scope='Only root-approved single outside-opacity V 30001..32000; coefficient .05',gpu_status_path=str(V.relative_to(D)/'status.json'),root_receipt=str((P/'opacity_outside_root_coordination_receipt.json').relative_to(D)),next_experiment='After actual V complete: final four states/49 frames/2000 paired inputs and original pins; compare separately to B1 and B2 with observed repeat distribution, actor preservation and real branch/locality images. Decide next isolated causal or from-zero validation from evidence. No whole-task completion from V alone.')
m['long_three_arm'].update(B1='complete_exit0_all49_fourstate_verified',B2='complete_exit0_all49_fourstate_noise_verified',V='running' if cmd else 'awaiting_completion_audit',B2_completion_receipt_sha256=sha(B2/'receipts/full49_summary.json'),B2_terminal_manifest_sha256=sha(B2/'terminal_32000/diagnostic_snapshot_manifest.json'),B1_B2_noise_sha256=sha(B2/'receipts/B1_B2_observed_noise.json'),V_root_receipt_sha256=sha(P/'opacity_outside_root_coordination_receipt.json'),V_binding_sha256=sha(P/'experiment_binding.json'),V_prepared_plan_sha256=sha(P/'opacity_outside_prepared_plan.json'),V_final_review_sha256=sha(D/'sgcm_data_audit/long_pair_review/final_review_V.json'),historical_launcher_pid=launcher['launcher_pid'],historical_worker_pid=launcher['worker_pid'],started_utc=launcher['started_utc'],iteration_at_verification=runtime['last_completed_step'])
m['latest_runtime_check']=runtime;m['acceptance_not_yet_met']=True;m['paper_complete_reproduction_claimed']=False
m['long_B2_result']={'composite_full49_mean_frame_psnr_db':read(B2/'receipts/full49_summary.json')['final49_regions']['full']['composite']['mean_frame_psnr_db'],'noise_receipt':str((B2/'receipts/B1_B2_observed_noise.json').relative_to(D)),'interpretation':'Replicates poor static/dynamic separation; mean differences and per-frame error distribution both required; one repeat is not a universal bound.'}
tmp=path.with_suffix('.next.json');tmp.write_text(json.dumps(m,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
receipt=D/'heartbeat_checks/20261011T2155_V_transition.json';receipt.parent.mkdir(exist_ok=True);receipt.write_text(json.dumps(runtime,indent=2)+'\n')
print(json.dumps(runtime))
