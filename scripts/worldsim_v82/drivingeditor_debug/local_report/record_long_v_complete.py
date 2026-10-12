"""Record actual completed V evidence without modifying any pinned experiment receipt."""
import datetime, hashlib, json, os, pathlib, subprocess
BASE = pathlib.Path('/root/autodl-tmp/drivingeditor_repro')
D = BASE / 'campaign/vgpu_bugfix_20261012'
V = D / 'continuation_diagnostic/opacity_outside30k_to32000'
P = D / 'loss_audit/long_pair_30k_32000/materialized_plan'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = read(V/'receipts/gpu_release_handoff.json')
assert r['worker_exit_code'] == r['launcher_exit_code'] == 0
assert read(V/'status.json')['status'] == 'complete'
assert int((P/'opacity_outside_launcher.exit').read_text()) == 0
assert sha(V/'receipts/three_arm_locked_metrics.json') == '922a3dea16a427539ec944765680b315e9b69b8e7a217af4c5d990b7f7679a9d'
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
path = D/'task_manifest.json'
m = read(path)
assert m['current_experiment'] == V.name, 'Do not overwrite a later experiment'
m.update(updated_utc=now, current_experiment_status='V_complete_exit0_verified_tradeoff_not_repaired', gpu_owner=None if not apps else m.get('gpu_owner'), gpu_owner_scope='No new GPU task approved. Read-only fixed-pose temporal diagnostic preparation is in progress.', next_experiment='Read-only B1/V32000 fixed-pose temporal diagnostic and CPU paper/loss-domain audit; then select one evidence-grounded causal intervention. No formal50k or other scenes.')
m['long_three_arm']['V'] = 'complete_exit0_all49_fourstate_paired_verified'
m['long_three_arm']['V_release_sha256'] = sha(V/'receipts/gpu_release_handoff.json')
m['long_three_arm']['V_metrics_sha256'] = sha(V/'receipts/three_arm_locked_metrics.json')
m['long_V_result'] = {'receipt':str((V/'receipts/three_arm_locked_metrics.json').relative_to(D)), 'static_background_psnr_approx':27.33124, 'composite_psnr_approx':35.90406, 'dynamic_outside_alpha_approx':0.113536, 'dynamic_outside_rgb_support_approx':0.793688, 'all49_static_MSE_vs_each_original':'improved', 'all49_composite_and_actor_MSE_vs_each_original':'worse', 'interpretation':'Strong branch-allocation recovery, with persistent background leakage and actor/composite reconstruction cost. Not normal separation or author-certified formula.', 'root_visual_review':'Actual paired raw10 B1/B2/V six-channel image inspected: road static recovery but residual dynamic road texture/colored highlights.'}
m['latest_runtime_check'] = {'checked_utc':now, 'experiment':V.name, 'status':'complete', 'last_completed_step':32000, 'recorded_steps':2000, 'actual_worker_exit':0, 'actual_launcher_exit':0, 'GPU_compute_apps':apps, 'live_read_required_for_next_action':True}
m['acceptance_not_yet_met'] = True
m['paper_complete_reproduction_claimed'] = False
tmp = path.with_suffix('.next.json'); tmp.write_text(json.dumps(m, indent=2, allow_nan=False)+'\n'); os.replace(tmp,path)
receipt = D/'heartbeat_checks/20261011T2252_V_complete.json'
receipt.parent.mkdir(exist_ok=True)
receipt.write_text(json.dumps({'checked_utc':now, 'release_sha256':sha(V/'receipts/gpu_release_handoff.json'), 'metrics_sha256':sha(V/'receipts/three_arm_locked_metrics.json'), 'GPU_compute_apps':apps, 'shutdown':False, 'whole_task_complete':False},indent=2)+'\n')
print(json.dumps(m['latest_runtime_check']))
