"""Preserve the stopped no-training runtime import failure; CPU filesystem only."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
PLAN = BASE / 'campaign/vgpu_bugfix_20261012/component_diagnostic/continuation_plan'
FAILED = BASE / 'campaign/vgpu_bugfix_20261012/continuation_diagnostic/opacity_outside30k_to30300'
ATTEMPT = FAILED.parent / 'attempts/opacity_outside_01_runtime_image_utils_import'


def main():
    status = json.loads((FAILED / 'status.json').read_text())
    if status['status'] != 'failed' or status['actual_exit_code'] != 1 or status['gpu_lock_held'] or 'utils.image_utils' not in status['error'] or (FAILED / 'step_losses.jsonl').exists():
        raise RuntimeError('Expected stopped before-training import failure evidence differs')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU compute still active; do not archive failed attempt')
    if FAILED.resolve().parent != (BASE / 'campaign/vgpu_bugfix_20261012/continuation_diagnostic').resolve() or ATTEMPT.exists():
        raise RuntimeError('Unexpected isolated attempt archive target')
    with (BASE / 'campaign/gpu_slot.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ATTEMPT.mkdir(parents=True)
        runner = ATTEMPT / 'runner'
        runner.mkdir()
        for name in ('continuation_variant_harness.py', 'launch_opacity_outside_continuation.py', 'opacity_outside_prepared_plan.json', 'opacity_outside_root_coordination_receipt.json', 'opacity_outside_cpu_preflight.log', 'opacity_outside_cpu_preflight_receipt.json', 'opacity_outside_harness_delta.patch', 'opacity_outside_launcher_status.json', 'opacity_outside_launcher.exit', 'opacity_outside_training.log', 'opacity_outside_launcher_bootstrap.log', 'opacity_outside_worker.pid'):
            shutil.copy2(PLAN / name, runner / name)
        FAILED.rename(ATTEMPT / 'output')
        fcntl.flock(lock, fcntl.LOCK_UN)
    files = {str(path.relative_to(ATTEMPT)): {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in ATTEMPT.rglob('*') if path.is_file()}
    receipt = {'status': 'failed_attempt_evidence_preserved', 'actual_exit_code': 1, 'cause': "Diagnostic check imported nonexistent utils.image_utils; frozen train binds PSNR and SSIM from utils.loss_utils.", 'restoration_training_backward_optimizer_steps_executed': False, 'actual_first_step_not_reached': True, 'original_algorithm_binding_guard_observer_unchanged': True, 'GPU_apps_empty': True, 'GPU_lock_available_and_released': True, 'canonical_output_ready_for_root_reviewed_bounded_retry': str(FAILED), 'archive': str(ATTEMPT), 'files': files}
    (ATTEMPT / 'failure_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'status': receipt['status'], 'archive': str(ATTEMPT), 'file_count': len(files)}))


if __name__ == '__main__':
    main()
