from pathlib import Path
import ast
import hashlib
import json
import shutil

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a5_single_bug_20261106_r1')
ORIGINAL = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
PARENT = ROOT.parent / 'scene008_a37_only_B10k_20261103_r1/formal10k'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    destination = ROOT / 'engine'
    assert not destination.exists()
    before = {str(p.relative_to(ORIGINAL)): sha(p) for p in ORIGINAL.rglob('*.py') if '__pycache__' not in p.parts}
    shutil.copytree(ORIGINAL, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'smoke_baseline', 'smoke_no_sgcm', 'Tracking-Anything-with-DEVA'))
    renderer = destination / 'gaussian_renderer/__init__.py'
    text = renderer.read_bytes().decode('utf-8')
    assert sha(renderer) == '35b2863da1bf9173812207dbd06fbbecdb88929f95b50a7968db3b5b2aa60ced'
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_rasterization')
    dropped = {n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'time_shift' for t in n.targets)
               and isinstance(n.value, ast.Constant) and n.value.value is None}
    assert dropped == {265, 341}
    renderer.write_bytes(''.join(line for i, line in enumerate(text.splitlines(True), 1) if i not in dropped).encode('utf-8'))
    after = {str(p.relative_to(ORIGINAL)): sha(p) for p in ORIGINAL.rglob('*.py') if '__pycache__' not in p.parts}
    assert before == after
    copied = {str(p.relative_to(destination)): sha(p) for p in destination.rglob('*.py')}
    different = [name for name in copied if copied[name] != before[name]]
    assert different == ['gaussian_renderer/__init__.py']
    checkpoints = [PARENT / f'{prefix}10000.pth' for prefix in ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state')]
    contract = dict(status='PREPARED_ISOLATED_STATIC_TIME_SHIFT_FIX', original_engine=str(ORIGINAL), engine=str(destination),
                    original_engine_files_unchanged=True, source_sha256=before, isolated_engine_sha256=copied,
                    changed_files=different, removed_original_lines=sorted(dropped),
                    parent_model=str(PARENT), parent_iteration=10000, replay_updates=1977, replay_endpoint=11977,
                    untouched_baseline=str(ROOT.parent / 'scene008_a5_old_single30k_20261106_r1/smoke_retry1'),
                    baseline_role='Existing independently recorded unpatched 1977-update smoke from the same original10K complete state',
                    learning_rate_horizon=30000, formal_updates_added=0,
                    engineering_updates='32-update real smoke plus2-step restore probe; replay1977 updates separately counted',
                    input_config_changes=['model_path', 'start_checkpoint'], renderer_fix='Honor existing static time_shift argument; no new loss or coefficient',
                    causal_limit='Bug fix counterfactual, not a claim that all a5 defects are caused by this single bug',
                    parent_files={str(p): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in checkpoints}, automatic_shutdown_allowed=False)
    (ROOT / 'isolated_fix_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(json.dumps(dict(status=contract['status'], changed_files=different, removed_lines=sorted(dropped))))

if __name__ == '__main__':
    main()
