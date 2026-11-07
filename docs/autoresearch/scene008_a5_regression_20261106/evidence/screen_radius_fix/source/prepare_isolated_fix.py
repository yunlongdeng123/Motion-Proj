"""Prepare only the validated screen-radius bookkeeping candidate; CPU only."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a5_screen_radius_bug_20261106_r1')
ORIGINAL = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
PARENT = ROOT.parent / 'scene008_a37_only_B10k_20261103_r1/formal10k'
ORIGINAL_GMODEL = 'e5da10984f8836da1ac51c975823c3ca89bbf8c56f45ec4d4dcb071c06aafa72'
CANDIDATE_GMODEL = '3b3c5f43ea46342f0e36b461af04fd1d2e52afead24d9953673894dff788ce1a'
ORIGINAL_RENDERER = '35b2863da1bf9173812207dbd06fbbecdb88929f95b50a7968db3b5b2aa60ced'
PATCH_SHA = '939cad8c1568eb0bbf5a70faf350dceb2f2962b37260066181cc1713ff72f58e'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert ROOT.name == 'scene008_a5_screen_radius_bug_20261106_r1'
    destination = ROOT / 'engine'
    assert not destination.exists()
    before = {str(p.relative_to(ORIGINAL)): sha(p) for p in ORIGINAL.rglob('*.py') if '__pycache__' not in p.parts}
    assert before['scene/gaussian_model.py'] == ORIGINAL_GMODEL
    assert before['gaussian_renderer/__init__.py'] == ORIGINAL_RENDERER
    patch = ROOT / 'source/screen_radius_candidate.patch'
    assert sha(patch) == PATCH_SHA
    receipt_path = ROOT / 'source/screen_radius_candidate_cpu_test.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'PASS_STRICT_CPU_SCREEN_RADIUS_CANDIDATE' and receipt['case_count'] == 18
    assert receipt['candidate_sha256'] == CANDIDATE_GMODEL and receipt['patch_sha256'] == PATCH_SHA
    assert not receipt['GPU_used'] and not receipt['cuda_initialized']
    # Reuse the exact CPU-tested context-checking patch parser, without importing
    # its torch test harness or creating any CUDA context.
    parser_source = ROOT / 'source/probe_screen_radius_candidate_cpu.py'
    fn = next(node for node in ast.parse(parser_source.read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == 'apply_unified')
    scope = dict(re=re)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(parser_source), 'exec'), scope)
    original_bytes = (ORIGINAL / 'scene/gaussian_model.py').read_bytes()
    candidate = scope['apply_unified'](original_bytes.decode('utf-8'), patch.read_bytes().decode('utf-8')).encode('utf-8')
    assert hashlib.sha256(candidate).hexdigest() == CANDIDATE_GMODEL
    shutil.copytree(ORIGINAL, destination, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', 'smoke_baseline', 'smoke_no_sgcm', 'Tracking-Anything-with-DEVA'))
    (destination / 'scene/gaussian_model.py').write_bytes(candidate)
    after = {str(p.relative_to(ORIGINAL)): sha(p) for p in ORIGINAL.rglob('*.py') if '__pycache__' not in p.parts}
    assert before == after
    copied = {str(p.relative_to(destination)): sha(p) for p in destination.rglob('*.py')}
    different = [name for name in copied if copied[name] != before[name]]
    assert different == ['scene/gaussian_model.py']
    assert copied['gaussian_renderer/__init__.py'] == ORIGINAL_RENDERER
    checkpoints = [PARENT / f'{prefix}10000.pth' for prefix in ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state')]
    contract = dict(
        status='PREPARED_ISOLATED_SCREEN_RADIUS_BOOKKEEPING_CANDIDATE', original_engine=str(ORIGINAL), engine=str(destination),
        original_engine_files_unchanged=True, source_sha256=before, isolated_engine_sha256=copied,
        changed_files=different, original_gaussian_model_sha256=ORIGINAL_GMODEL,
        candidate_gaussian_model_sha256=CANDIDATE_GMODEL, renderer_sha256=ORIGINAL_RENDERER,
        candidate_patch_sha256=PATCH_SHA, cpu_candidate_test_sha256=sha(receipt_path), cpu_candidate_test_cases=18,
        time_shift_fix_included=False,
        candidate_change='Carry old-row max_radii2D through append, append zero for new rows; defer existing normal generic window reset until after pruning; preserve prune_only survivor history.',
        normal_window_length_changed=False, max_screen_size_threshold_changed=False,
        static_screen_threshold=20, dynamic_screen_threshold=10, loss_or_sampling_changed=False,
        line_endings=dict(original_LF=original_bytes.count(b'\n'), original_CRLF=original_bytes.count(b'\r\n'),
                          candidate_LF=candidate.count(b'\n'), candidate_CRLF=candidate.count(b'\r\n'), normalization_applied=False),
        parent_model=str(PARENT), parent_iteration=10000, replay_updates=1977, replay_endpoint=11977,
        untouched_baseline=str(ROOT.parent / 'scene008_a5_old_single30k_20261106_r1/smoke_retry1'),
        baseline_role='Existing unpatched1977-update branch from the same original10K complete state; not the time-shift candidate.',
        learning_rate_horizon=30000, formal_updates_added=0,
        smoke_updates=128, smoke_endpoint=10128, smoke_required_covered_nodes=[10050, 10100], restore_probe_updates=2,
        additional_engineering_updates=2107, engineering_accounting=dict(real_smoke=128, restore_probe=2, paired_replay=1977),
        input_config_changes=['model_path', 'start_checkpoint'],
        causal_limit='CPU bookkeeping correctness and an isolated engineering candidate; a5 benefit is not assumed. No time-shift patch is combined.',
        parent_files={str(p): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in checkpoints},
        preparation_GPU_used=False, preparation_training_updates=0, automatic_shutdown_allowed=False)
    (ROOT / 'isolated_fix_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(json.dumps(dict(status=contract['status'], root=str(ROOT), changed_files=different,
                          candidate_sha256=CANDIDATE_GMODEL, renderer_unchanged=True, GPU_started=False)))


if __name__ == '__main__':
    main()
