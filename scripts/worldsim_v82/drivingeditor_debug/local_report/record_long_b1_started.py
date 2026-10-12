import datetime,hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
p=R/'task_manifest.json';m=json.loads(p.read_text(encoding='utf8'))
P=R/'loss_audit/long_pair_30k_32000/materialized_plan'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),current_experiment='original30k_to32000',current_experiment_status='B1_running_actual_restore_and_first_step_exact',gpu_owner='vgpu_component_diagnostic_solmax',gpu_owner_scope='Only root-approved original-objective B1 30001..32000',gpu_status_path='continuation_diagnostic/original30k_to32000/status.json',next_experiment='B2 independent original repeat from original30k, then V same outside-opacity hypothesis. Separate exact approvals/idle GPU required; V long binding not yet ready.')
m['long_three_arm']={
 'B1':'running','B2':'CPU_prepared_reviewed_not_GPU_authorized','V':'materialized_source_only_real_long_reference_binding_pending',
 'historical_launcher_pid':36252,'historical_worker_pid':36255,'started_utc':'2026-10-11T19:45:30Z',
 'iteration_at_verification':30027,'progress_read_policy':'Read last complete step_losses.jsonl row; status.json updates only at30001/30300/31000/32000 nodes.',
 'first_actual_restore':'actual_restore_pass','first_actual_scalar_comparisons':'all10 exact_equal true, abs_delta0','first_camera':'image_0/0017034','first_loss':0.2980644106864929,'first_psnr':8.590572357177734,
 'fresh_restore_iteration':30000,'stop_iteration':32000,'lr_horizon':50000,'seed':0,
 'evaluation_protocol_sha256':sha(P/'evaluation_protocol_locked.json'),
 'B1_prepared_plan_sha256':sha(P/'prepared_plan.json'),'B1_root_receipt_sha256':sha(P/'root_coordination_receipt.json'),
 'B1_plan_directory':'loss_audit/long_pair_30k_32000/materialized_plan',
 'B2_plan_directory':'loss_audit/long_pair_30k_32000/materialized_plan/baseline_noise_reference',
 'source_review_directory':'sgcm_data_audit/long_pair_review',
 'acceptance':'Three long trajectories are a recovery/mechanism test; not author-formula certification or from-zero validation.',
}
m['local_report']={'path':'outputs/drivingeditor_vgpu_debug/index.html','short_paired_video':'49 actual frames,10fps,4.9s,one encoded mosaic for synchronization','local_references_validated':45,'browser_playback_checked':False,'display_request_status':'queued'}
p.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
subprocess.run(['scp',str(p),'wm-vgpu-1011:/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/task_manifest.json'],check=True)
print(json.dumps({'status':m['current_experiment_status'],'goal_complete':False}))
