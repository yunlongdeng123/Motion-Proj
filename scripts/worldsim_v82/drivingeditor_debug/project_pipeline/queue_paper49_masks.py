"""One-shot dependency queue for49-frame instance masks, not a recurring monitor."""
import datetime
import json
from pathlib import Path
import subprocess
import time

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
STATUS = BASE / 'logs/paper49_mask_queue.json'
record = {
    'stage': 'waiting_for_existing_training',
    'started': datetime.datetime.now().astimezone().isoformat(),
    'scene': str(BASE / 'data/waymo_paper_scenes/scene0017_000_048'),
    'purpose': 'Extend real instance tracking to49frames for paper-method preparation.',
    'limitations': 'DEVA/SAM instance masks do not complete the paper Grounded-SAM/shadow/M-detector pipeline. No automatic actor identity declaration or formal50k run is made here.'}

def update(**values):
    record.update(values)
    record['updated'] = datetime.datetime.now().astimezone().isoformat()
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(record), flush=True)

deadline = time.monotonic() + 6 * 3600
try:
    update()
    dependencies = [BASE / 'logs/formal_demo.exit', BASE / 'logs/paper_method_smoke1000.exit']
    while not all(path.is_file() for path in dependencies):
        if time.monotonic() >= deadline:
            raise TimeoutError('Training dependency wait exceeded6hours')
        time.sleep(15)
    for path in dependencies:
        if path.read_text().strip() != '0':
            raise RuntimeError(f'Dependency failed: {path}')
    update(stage='waiting_for_gpu_headroom')
    # Avoid colliding with any subsequently started user workload.
    while True:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=used_gpu_memory',
                                 '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
        if sum(values) <= 1024:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('GPU remained occupied beyond queue deadline')
        time.sleep(15)
    update(stage='running_deva_sam49')
    result = subprocess.run(['bash', str(BASE / 'scripts/run_paper_deva49.sh')], check=False)
    if result.returncode:
        raise RuntimeError(f'DEVA preparation exited{result.returncode}; retain outputs for diagnosis')
    update(stage='instance_masks_validated', completed=True,
           next='Inspect49-frame actor tracks, then prepare actor LiDAR and shadow/motion preprocessing.')
except Exception as error:
    update(stage='failed', error=repr(error))
    raise
