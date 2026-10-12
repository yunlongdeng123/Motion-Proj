"""Record reviewed S evidence and report delivery, without touching experiments."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

B = Path('/root/autodl-tmp/drivingeditor_repro')
D = B / 'campaign/vgpu_bugfix_20261012'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bundle = read(D / 'root_review/S_report_delivery_20261012.json')
assert bundle['whole_task_complete'] is False
assert bundle['local_validation_pass'] is True
assert bundle['actual_S_all294_png_verified'] is True
assert bundle['S_synchronous_video_all49_decoded'] is True
assert bundle['browser_playback_verified'] is False
t = read(D / 'task_manifest.json')
assert t['current_experiment'] == 'observed_static_rgb30k_to32000', 'Future experiment already selected'
out = D / 'continuation_diagnostic/observed_static_rgb30k_to32000'
s = read(out / 'receipts/full49_summary.json')
assert s['harness_exit_code'] == s['wrapper_exit_code'] == 0
assert s['step_count'] == 2000 and s['all4_node_eval_RNG_preserved'] and s['all_protected_SHA_unchanged']
apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
assert not apps, 'Do not overwrite GPU ownership while another task is live'
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
t.update(updated_utc=stamp,
         current_experiment_status='complete_reviewed_tradeoffs_not_repaired_report_delivered',
         gpu_owner=None,
         gpu_owner_scope='S completed and reviewed. No subsequent GPU task approved.',
         acceptance_not_yet_met=True, paper_complete_reproduction_claimed=False)
t['observed_static_rgb_S'].update(effect_review_complete=True, whole_debug_task_complete=False,
    interpretation='Static observed reconstruction improves, but composite and actor errors rise and dynamic background remains. S is an extra diagnostic objective, not a certified paper repair.',
    local_delivery=bundle)
audit = D / 'loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu'
for key, name in [('static_initialization_visibility','front_visibility_followup_cpu.json'),
                  ('dynamic_initialization_native_road','dynamic_actor_native_semantics_followup_cpu.json')]:
    p = audit / name
    if p.exists():
        t[key] = {'receipt':str(p), 'sha256':sha(p), 'audit_only':True, 'training_inputs_modified':False}
build = D / 'preprocessing_gap_audit/mdetector_adapter_candidate/build_attempt_01/native_build_receipt.json'
if build.exists():
    n = read(build)
    assert n['full_native_core_and_adapter_compilation_passed']
    assert not n['detector_initialized'] and not n['real_scans_processed']
    t['mdetector_native_build'] = {'receipt':str(build), 'sha256':sha(build),
        'status':'isolated_compile_pass_only', 'vendor_mutations':n['vendor_source_config_mutations'],
        'initialization_or_inference_approved_by_this_record':False}
t['latest_root_review'] = {'at':stamp,'receipt':str(D/'root_review/S_report_delivery_20261012.json'),
    'sha256':sha(D/'root_review/S_report_delivery_20261012.json'), 'gpu_apps_at_review':[]}
tmp = D / 'task_manifest.S_review.tmp'
tmp.write_text(json.dumps(t,ensure_ascii=False,indent=2)+'\n')
tmp.replace(D / 'task_manifest.json')
print(json.dumps({'status':t['current_experiment_status'],'updated_utc':stamp,'whole_task_complete':False}))
