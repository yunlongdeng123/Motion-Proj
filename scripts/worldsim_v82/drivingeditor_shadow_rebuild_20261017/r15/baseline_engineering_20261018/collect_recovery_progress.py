from pathlib import Path
import json, os
E = Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018')
NEW = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018')
names = ['A_recovery_queue_status.json', 'GN11_resume_GPU_receipt.json',
         'GN11_resume_sequence_receipt.json', 'all_vehicle_train_queue_status.json',
         'A_replay_authorization.json', 'GN11_recovery_audit.json',
         'GN11_timing_fix_GPU_receipt.json', 'native_runner_timing_GPU_receipt.json',
         'intersection_input_passed.json', 'intersection_input_review_receipt.json',
         'GN11_25k_durability_receipt.json','direct_queue_status.json']
out = {n: json.loads((E/n).read_text()) for n in names if (E/n).exists()}
if (E/'parallel_queue_status.json').exists():
    out['A_recovery_queue_status.json']=json.loads((E/'parallel_queue_status.json').read_text())
    out['current_A_queue_source']='parallel_queue_status.json'
run = NEW/'runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k'
if (run/'status.json').exists():
    out['live_GN11_recovery_status.json'] = json.loads((run/'status.json').read_text())
active=Path(json.loads((E/'direct_active_run.json').read_text())['output']) if (E/'direct_active_run.json').exists() else E/'direct_instance30k'
for path,name in [(active/'status.json','live_direct_instance_status.json'),
                  (E/'direct_full49/progress.json','live_direct_full49_progress.json'),
                  (NEW.parent/'r15_baseline_allvehicles_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11/status.json','live_B_native_status.json')]:
    if path.exists():out[name]=json.loads(path.read_text())
out['recovery_processes'] = {}
for pid in [52748,52417]:
    p = Path('/proc')/str(pid)/'cmdline'
    out['recovery_processes'][str(pid)] = p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
out['B_training_ready_released'] = (E/'all_vehicle_ready.json').exists()
out['latest_recovery_log_tail'] = (E/'A_recovery_GN11_RESUME20K_TO30K.log').read_bytes()[-1800:].decode(errors='replace')
print(json.dumps(out))
