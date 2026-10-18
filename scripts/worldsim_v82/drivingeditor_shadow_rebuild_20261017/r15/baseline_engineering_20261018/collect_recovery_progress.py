from pathlib import Path
import json, os
E = Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018')
NEW = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018')
names = ['A_recovery_queue_status.json', 'GN11_resume_GPU_receipt.json',
         'GN11_resume_sequence_receipt.json', 'all_vehicle_train_queue_status.json',
         'A_replay_authorization.json', 'GN11_recovery_audit.json',
         'GN11_timing_fix_GPU_receipt.json', 'native_runner_timing_GPU_receipt.json',
         'intersection_input_passed.json', 'intersection_input_review_receipt.json']
out = {n: json.loads((E/n).read_text()) for n in names if (E/n).exists()}
run = NEW/'runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k'
if (run/'status.json').exists():
    out['live_GN11_recovery_status.json'] = json.loads((run/'status.json').read_text())
out['recovery_processes'] = {}
for pid in [52748,52417]:
    p = Path('/proc')/str(pid)/'cmdline'
    out['recovery_processes'][str(pid)] = p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
out['B_training_ready_released'] = (E/'all_vehicle_ready.json').exists()
out['latest_recovery_log_tail'] = (E/'A_recovery_GN11_RESUME20K_TO30K.log').read_bytes()[-1800:].decode(errors='replace')
print(json.dumps(out))
