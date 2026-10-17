"""暂停已核实的 R15 进程，保留原检查点及恢复状态；不执行 SGD。"""
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import torch

torch.set_num_threads(2)
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3/backend_restore_r15')
receipt_path = BASE / 'pause_for_diagnosis_20261017.json'
receipt = json.loads(receipt_path.read_text())
checkpoint = BASE / 'main/latest.pt'
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert saved['global_step'] == 4000 and saved['stage'] == 'geometry'
assert saved['optimizer']['state'], 'Checkpoint must retain nonempty Adam states'
assert all(k in saved for k in ['generator', 'torch_rng', 'numpy_rng', 'python_rng', 'cuda_rng'])
destination = BASE / 'paused_checkpoint_004000_20261017'
destination.mkdir(exist_ok=True)
pinned = destination / 'checkpoint.pt'
if not pinned.exists():
    os.link(checkpoint, pinned)
for name in ['config.json', 'execution_policy.json', 'pipeline_state.json']:
    shutil.copy2(BASE / name, destination / name)
for name in ['progress.json', 'resolved_config.json', 'input_contract.json']:
    shutil.copy2(BASE / 'main' / name, destination / name)
policy = json.loads((BASE / 'execution_policy.json').read_text())
policy.update(training_allowed=False, automatic_resume_allowed=False,
              shutdown_allowed=False, status='PAUSED_BY_LATEST_HUMAN_REQUEST',
              latest_authorization='Pause and preserve; diagnose performance and a6/a37; publish code to v82; no training resume authorized')
(BASE / 'execution_policy.json').write_text(json.dumps(policy, indent=2))
marker = dict(status='PAUSED_BY_USER', checkpoint=str(pinned), global_step=4000,
              last_logged_step=receipt['progress']['global_step'],
              unsaved_progress='After checkpoint 4000, last logged step 4100; exact in-flight step unknown and not claimed saved',
              resume_training_authorized=False, shutdown_authorized=False)
(BASE / 'PAUSE_REQUESTED.json').write_text(json.dumps(marker, indent=2))
stopped = []
for row in receipt['processes']:
    process = Path('/proc') / str(row['pid'])
    if not process.exists():
        continue
    assert (process / 'stat').read_text().split()[21] == row['start_ticks']
    cmd = (process / 'cmdline').read_bytes().replace(b'\x00', b' ').decode()
    assert cmd == row['cmdline']
    os.kill(row['pid'], signal.SIGKILL)
    stopped.append(row['pid'])
(BASE / 'pipeline_state.json').write_text(json.dumps(marker, indent=2))
receipt.update(marker)
receipt.update(stopped_pids=stopped, status='PAUSED_CHECKPOINT_VERIFIED_WORKERS_TERMINATED',
               checkpoint_bytes=pinned.stat().st_size,
               checkpoint_optimizer_entries=len(saved['optimizer']['state']),
               checkpoint_stage_step=saved['stage_step'], checkpoint_RNG_retained=True,
               completed_at_singapore=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
receipt_path.write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
