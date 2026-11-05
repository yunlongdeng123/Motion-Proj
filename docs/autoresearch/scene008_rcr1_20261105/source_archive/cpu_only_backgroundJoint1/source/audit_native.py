"""Audit the paired train schedule and hash the resumable final checkpoint."""
import hashlib
import json
from pathlib import Path
import sys

from native_ownership import CAMPAIGN, OBS, REF
from evaluate_dual import ENGINE


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def load(path):
    return json.loads(path.read_text())


def verify(run, destination):
    result = {}
    old = CAMPAIGN / 'scene008_a6_a37_dynamic_20261104_r1/formal_dual_actor_r1'
    formal = run / 'formal'
    with (old/'paired_steps.jsonl').open() as a, (formal/'paired_steps.jsonl').open() as b:
        updates = 0
        for lines in zip(a, b):
            rows = [json.loads(value) for value in lines]
            updates += 1
            for field in ('iteration', 'frame_id', 'time_shift', 'learning_rates'):
                assert rows[0][field] == rows[1][field], (updates, field)
            assert rows[1]['iteration'] == updates
        assert updates == 10000
        assert next(b, None) is None
    smoke = load(run/'smoke/returned.json')
    resume = load(run/'resume/returned.json')
    assert smoke['seconds'] >= 540 and smoke['status'] == 'PASS_COMPLETE_BOUNDARY'
    assert resume['updates'] == 2 and resume['iteration'] == smoke['iteration'] + 2
    assert (run/'smoke/smoke_evaluation.json').is_file()
    for name in ('native_ownership.py', 'integrate_native.py', 'readout_native.py', 'early_accept.py'):
        path = f'executed_{name}'
        assert (run/'smoke'/path).read_bytes() == (formal/path).read_bytes(), name
    continued = [json.loads(line) for line in (run/'resume/paired_steps.jsonl').read_text().splitlines()]
    with (old/'paired_steps.jsonl').open() as file:
        for i, line in enumerate(file, 1):
            if smoke['iteration'] < i <= resume['iteration']:
                original, restored = json.loads(line), continued[i - smoke['iteration'] - 1]
                assert all(original[k] == restored[k] for k in ('iteration','frame_id','time_shift','learning_rates'))
    assert load(run/'resume/restore_audit.json')['status'] == 'PASS_EXACT_FULL_RESTORE'
    assert load(formal/'early_accept_01000.json')['status'] == 'PASS'
    assert load(formal/'returned.json')['updates'] == 10000
    for step in (100, 500, 1000, 3000, 3300, 6000, 6300, 10000):
        assert (formal / f'probe_{step:05d}/summary.json').is_file()
    a = load(run/'control/probe_10000/summary.json')
    b = load(formal/'probe_10000/summary.json')
    assert a['frames'] == b['frames'] == list(range(89, 138))
    for old_row, row in zip(a['rows'], b['rows']):
        assert old_row['raw'] == row['raw']
        for key, value in row['metrics'].items():
            assert old_row['metrics'][key]['count'] == value['count']
    historical = load(old.parent/'formal_49_r1_probe/summary.json')['pooled']['static_front/a6_measured_visible']
    paired = a['pooled']['a6_static_front']
    assert historical['pixels'] == paired['count']
    assert abs(historical['mean'] - paired['mean']) < 1e-6
    supplemental = {}
    for arm in ('native', 'union'):
        edit = load(run/f'edit_{arm}/delete_domains.json')
        assert [r['raw'] for r in edit['rows']] == b['frames']
        supplemental[arm] = {'max_rgb_difference': max(r['native_rgb_max_abs'] for r in edit['rows'])}
    paths = [formal / f'{name}10000.pth' for name in ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state')]
    result.update(status='PASS', paired_updates=updates, paired_fields=['iteration', 'frame_id', 'time_shift', 'learning_rates'],
                  metrics_frames=b['frames'], checkpoint=[digest(path) for path in paths],
                  smoke=smoke, resume=resume, restore=load(run/'resume/restore_audit.json'),
                  formal=load(formal/'returned.json'),
                  inputs=[digest(REF), digest(ENGINE/'train.py'), digest(ENGINE/'gaussian_renderer/__init__.py'),
                          *[digest(path) for path in sorted(OBS.glob('*.npz'))]],
                  historical_fixed_probe=historical, native_raster_parity=supplemental,
                  scope='Native training through step 10000; no 1K road repair or PVD geometry')
    destination.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    verify(Path(sys.argv[1]), Path(sys.argv[2]))
