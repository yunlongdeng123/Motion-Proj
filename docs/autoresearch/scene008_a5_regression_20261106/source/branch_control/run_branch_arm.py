"""Explicit phase entry only. No automatic launch, queue, waiter, or full49 expansion."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a5_support_ablation_20261106_r1/support')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run(label, *args):
    with (ROOT / f'{label}.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', *map(str, args)], stdout=log, stderr=subprocess.STDOUT)
        write(ROOT / 'phase_status.json', dict(stage=label, pid=process.pid, updated_unix=time.time(),
              shutdown_allowed=False, gpu_reset_allowed=False, keep_only=True))
        assert process.wait() == 0, (label, process.returncode)


def sources():
    files = json.loads((ROOT / 'engine_source_manifest.json').read_text())
    files.update(json.loads((ROOT / 'input_source_manifest.json').read_text()))
    changed = []
    for name, receipt in files.items():
        path = Path(name)
        if path.stat().st_size != receipt['bytes'] or path.stat().st_mtime_ns != receipt['mtime_ns'] or hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
            changed.append(name)
    assert not changed, changed
    return dict(status='PASS_UNCHANGED_BYTES_MTIME_SHA256', checked_files=len(files))


def paired(step):
    fields = ['iteration', 'frame_id', 'time_shift', 'learning_rates']
    def read(root):
        result = []
        for phase in ['formal1k'] + (['formal5k'] if step == 5000 else []):
            for line in (root / phase / 'paired_steps.jsonl').read_text().splitlines():
                item = json.loads(line)
                result.append({key: item[key] for key in fields})
        return result
    old, new = read(BASELINE), read(ROOT)
    assert len(old) == len(new) == step
    mismatch = [i + 1 for i, (a, b) in enumerate(zip(old, new)) if a != b]
    write(ROOT / f'paired_camera_shift_lr_{step}.json', dict(status='PASS' if not mismatch else 'FAIL',
          baseline=str(BASELINE), steps=step, fields=fields, mismatch_iterations=mismatch,
          global_rng_or_identical_topology_claim=False))
    assert not mismatch, mismatch[:10]


def pilot(step, model):
    output = ROOT / f'keep_pilot_{step // 1000}k'
    run(f'keep_pilot_{step // 1000}k', ROOT / 'source/evaluate_branch_keep.py',
        '--model', model, '--iteration', step, '--output', output, '--scope', 'pilot')
    result = json.loads((output / 'summary.json').read_text())
    assert result['status'] == 'PASS_NATIVE_KEEP_ONLY_PILOT' and result['raw_ids'] == [97, 105, 109]
    assert result['frame_count'] == 3 and result['optimization_updates'] == 0 and not result['delete_produced']
    assert result['source_integrity'] == 'PASS_UNCHANGED_BYTES_MTIME_SHA256'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['first1k', 'finish5k'], required=True)
    args = parser.parse_args()
    assert ROOT.name == 'scene008_a5_branch_control_20261107_r1'
    write(ROOT / f'protected_before_{args.phase}.json', sources())
    if args.phase == 'first1k':
        run('formal1k', ROOT / 'source/train_branch_arm.py', '--mode', 'formal',
            '--stop-iteration', 1000, '--output', ROOT / 'formal1k')
        result = json.loads((ROOT / 'formal1k/returned.json').read_text())
        assert result['source_iteration'] == 0 and result['iteration'] == result['actual_additional_iterations'] == 1000
        estimate = result['elapsed_seconds'] * 5 + 300
        assert estimate < 3600 or result['elapsed_seconds'] >= 540, 'Long-smoke requirement not covered'
        paired(1000)
        checked = pilot(1000, ROOT / 'formal1k')
        stage = 'COMPLETE_FRESH1K_KEEP_PILOT_READY_FOR_ROOT_VISUAL_REVIEW'
        count = 1000
    else:
        first = json.loads((ROOT / 'keep_pilot_1k/summary.json').read_text())
        assert first['status'] == 'PASS_NATIVE_KEEP_ONLY_PILOT'
        run('formal5k', ROOT / 'source/train_branch_arm.py', '--mode', 'extend',
            '--checkpoint', ROOT / 'formal1k/chkpnt1000.pth', '--stop-iteration', 5000,
            '--output', ROOT / 'formal5k')
        result = json.loads((ROOT / 'formal5k/returned.json').read_text())
        audit = json.loads((ROOT / 'formal5k/restore_audit.json').read_text())
        assert result['source_iteration'] == 1000 and result['iteration'] == 5000 and result['actual_additional_iterations'] == 4000
        assert audit['status'] == 'PASS_EXACT_FULL_RESTORE' and not audit['differences']
        paired(5000)
        checked = pilot(5000, ROOT / 'formal5k')
        stage = 'COMPLETE_FRESH5K_KEEP_PILOT_FULL49_NOT_AUTOMATICALLY_RUN'
        count = 5000
    write(ROOT / f'protected_after_{args.phase}.json', sources())
    write(ROOT / 'phase_status.json', dict(stage=stage, formal_updates=count, learning_rate_horizon=30000,
        independent_evaluation_updates=0, extra_engineering_updates=0, pilot_frames=3,
        pilot_summary=str(ROOT / f'keep_pilot_{count // 1000}k/summary.json'), full49_automatically_run=False,
        keep_only=True, native_delete_removes=[5,37], a37_only_delete_solution=False,
        no_waiter=True, shutdown_allowed=False, gpu_reset_allowed=False,
        quality_accepted=False, human_verdict=None))


if __name__ == '__main__':
    main()
