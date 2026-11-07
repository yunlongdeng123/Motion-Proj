"""GPU queue for one isolated bug replay; call after the dual queue finishes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
def status(stage, **kw):
    write(ROOT / 'queue_status.json', dict(stage=stage, controller_pid=os.getpid(), updated_unix=time.time(), shutdown=False, **kw))
def run(label, script, *args):
    with (ROOT / (label + '.log')).open('x') as log:
        process = subprocess.Popen([PY, '-B', str(ROOT / 'source' / script)] + list(map(str, args)), stdout=log, stderr=subprocess.STDOUT)
        status(label, pid=process.pid)
        assert process.wait() == 0, (label, process.returncode)
def evaluate(label, model, step):
    run(label, 'evaluate_a5.py', '--model', model, '--iteration', step, '--engine', ROOT / 'engine',
        '--output', ROOT / label, '--dynamic-aliases', 37, '--delete-name', 'delete_a37')
    result = json.loads((ROOT / label / 'summary.json').read_text())
    assert result['status'] == 'PASS_FRESH_NATIVE49_A5_EVALUATION'
    assert result['fixed_denominator']['pixels'] == 121128
    assert result['replay']['complete_all49_keep_delete_replay'] and result['replay']['max_uint8_difference'] <= 1
    assert result['source_integrity'] == 'PASS_UNCHANGED_BYTES_MTIME_SHA256'
    return result
def guard_original(contract, label):
    rows = {name: dict(expected=value['sha256'], current=sha(Path(name))) for name, value in contract['parent_files'].items()}
    assert all(row['expected'] == row['current'] for row in rows.values())
    original = Path(contract['original_engine'])
    current = {str(p.relative_to(original)): sha(p) for p in original.rglob('*.py') if '__pycache__' not in p.parts}
    assert current == contract['source_sha256'], 'Original engine source changed'
    isolated = {str(p.relative_to(ROOT / 'engine')): sha(p) for p in (ROOT / 'engine').rglob('*.py')}
    assert isolated == contract['isolated_engine_sha256'], 'Isolated engine source changed'
    write(ROOT / (label + '_parent_integrity.json'), dict(status='PASS_PARENT_WEIGHTS_AND_ENGINE_SOURCES_UNCHANGED', files=rows,
        original_source_files=len(current), isolated_source_files=len(isolated)))
def main():
    contract = json.loads((ROOT / 'isolated_fix_contract.json').read_text())
    guard_original(contract, 'before')
    parent = Path(contract['parent_model']) / 'chkpnt10000.pth'
    run('smoke32', 'train_isolated_fix.py', '--mode', 'smoke', '--checkpoint', parent, '--output', ROOT / 'smoke32')
    evaluate('smoke_a5_evaluation', ROOT / 'smoke32', 10032)
    run('smoke_resume2step', 'train_isolated_fix.py', '--mode', 'resume2step', '--checkpoint', ROOT / 'smoke32/chkpnt10032.pth', '--output', ROOT / 'smoke_resume2step')
    assert json.loads((ROOT / 'smoke_resume2step/restore_audit.json').read_text())['status'] == 'PASS_EXACT_FULL_RESTORE'
    assert json.loads((ROOT / 'smoke_resume2step/resume_entry.json').read_text())['completed_extra_updates'] == 2
    write(ROOT / 'smoke_summary.json', dict(status='PASS_REAL_GPU_SAVE_EVAL_FULL_RESTORE_NEXT_ENTRY', smoke_updates=32,
        resume_extra_updates=2, formal_updates_added=0, replay_updates=0, long_run=False,
        rule='1977-update debugging replay is below one hour; real short smoke covers changed graph and complete lifecycle'))
    run('replay1977', 'train_isolated_fix.py', '--mode', 'replay', '--checkpoint', parent, '--output', ROOT / 'replay1977')
    fixed = evaluate('a5_evaluation', ROOT / 'replay1977', 11977)
    baseline_dir = ROOT.parent / 'scene008_a5_old_single30k_20261106_r1'
    baseline = json.loads((baseline_dir / 'smoke_a5_evaluation/summary.json').read_text())
    def sampling(path):
        return [(r['iteration'], r['frame_id'], r['time_shift']) for r in
                (json.loads(line) for line in path.read_text().splitlines() if line.strip())]
    a = sampling(ROOT / 'replay1977/paired_steps.jsonl')
    b = sampling(baseline_dir / 'smoke_retry1/paired_steps.jsonl')
    assert len(a) == len(b) == 1977 and a == b
    assert fixed['fixed_denominator'] == baseline['fixed_denominator']
    guard_original(contract, 'after')
    result = dict(status='COMPLETE_CONTROLLED_1977_UPDATE_BUG_REPLAY', baseline_endpoint=11977, fixed_endpoint=11977,
                  same_initial_complete_state=True, identical_camera_time_shift_schedule=True, compared_updates=1977,
                  baseline_all49=baseline['all49'], fixed_all49=fixed['all49'],
                  baseline_boundary=baseline['boundary_all49'], fixed_boundary=fixed['boundary_all49'],
                  added_formal_updates=0, additional_engineering_updates=2011,
                  engineering_accounting=dict(real_smoke=32, restore_probe=2, paired_replay=1977),
                  improvement_is_not_assumed=True, bitwise_determinism_not_claimed=True,
                  only_semantic_engine_change='Remove two unconditional static/split time_shift=None assignments',
                  automatic_shutdown_executed=False)
    write(ROOT / 'paired_result.json', result)
    status(result['status'], result=str(ROOT / 'paired_result.json'))
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('FAILED', error=repr(exc))
        raise
