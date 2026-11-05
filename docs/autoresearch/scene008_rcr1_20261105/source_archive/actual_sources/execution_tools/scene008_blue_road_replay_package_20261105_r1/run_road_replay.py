"""Private engineering replay of the unchanged original road1K method.

Default is a read-only plan. --execute creates a NEW campaign-child run only.
It never changes the old run/source/inputs, native parent, or fine-road files.
Preflight4K and formal10K must use independent run directories/parent bindings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD_ROAD = CAMPAIGN/'scene008_native_road1k_20261105_r1'
REFERENCE_NATIVE = CAMPAIGN/'scene008_native_ownership_20261105/run_r3/formal'
TEXT_SUFFIXES = {'.py', '.sh'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def steps(phase):
    result = [
        ('capture_parent', 'native_binding.py', []),
        ('prepare_inputs', 'prepare_road37.py', []),
        ('selection', 'repair37.py', ['--select']),
        ('smoke', 'repair37.py', ['--output', 'smoke', '--steps', '32']),
        ('restore_a', 'repair37.py', ['--output', 'restore_a', '--steps', '33', '--resume', 'smoke/checkpoint000032.pt']),
        ('restore_b', 'repair37.py', ['--output', 'restore_b', '--steps', '33', '--resume', 'smoke/checkpoint000032.pt']),
    ]
    if phase == 'formal':
        result += [
            ('formal', 'repair37.py', ['--output', 'formal', '--steps', '1000']),
            ('evaluation', 'repair37.py', ['--output', 'evaluation', '--resume', 'formal/checkpoint001000.pt', '--evaluate']),
        ]
    return result


def plan(args):
    return dict(
        execute=bool(args.execute), phase=args.phase, native_parent=str(args.native_parent),
        native_iteration=args.native_iteration, run_dir=str(args.run_dir),
        original_method=str(OLD_ROAD), original_source_snapshot='original_source/', patched_source='source/',
        pipeline=[dict(stage=label, script=script, arguments=arguments) for label, script, arguments in steps(args.phase)],
        formal_updates=1000 if args.phase == 'formal' else 0, smoke_updates=32, restore_updates=2,
        total_updates=1034 if args.phase == 'formal' else 34,
        unchanged=['road representation and geometry construction', 'settings.json bytes', 'loss and weights',
                   'Adam LR/eps', 'gate initialization', 'sampler/seed', '1000 formal updates',
                   'original observations and 199-frame source evidence'],
        fresh=['native_delete all49', 'prepared road_reference and fit/protection inputs',
               'contributors on current static topology', 'opacity gate parameters'],
        old_static_gate_loaded=False, old_fine_road_coefficients_loaded=False,
        quality_accepted=False, shutdown=False)


def validate_parent(args):
    parent = args.native_parent.resolve()
    files = [parent/f'{prefix}{args.native_iteration}.pth' for prefix in
             ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state')]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = read(parent/'effective_config.json')
    ref = read(REFERENCE_NATIVE/'effective_config.json')
    if cfg.get('dual_actor_schema_version') != 1:
        raise ValueError('The frozen parent must retain the original dual-actor source schema')
    if Path(cfg['source_path']).resolve() != Path(ref['source_path']).resolve():
        raise ValueError('This replay requires the original scene008 coordinate/data binding')
    current = {r['img_name']: r for r in read(parent/'cameras.json')}
    original = {r['img_name']: r for r in read(REFERENCE_NATIVE/'cameras.json')}
    if set(current) != set(original) or len(current) != 49:
        raise ValueError('Parent camera identities differ from the fixed original49 views')
    keys = ('rotation', 'position', 'fx', 'fy', 'cx', 'cy')
    for name, row in current.items():
        if any(row[key] != original[name][key] for key in keys):
            raise ValueError(f'Parent camera/coordinate contract differs at {name}')
    if read(parent/'actual_scene_binding.json')['timestamps'] != read(REFERENCE_NATIVE/'actual_scene_binding.json')['timestamps']:
        raise ValueError('Parent49 timestamps differ from the original protocol')
    identity = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
    metadata = {str(p): sha(p) for p in
                (parent/'effective_config.json', parent/'cameras.json', parent/'actual_scene_binding.json')}
    return parent, identity, metadata


def copy_text_source(source, destination):
    destination.mkdir()
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            shutil.copy2(path, destination/path.name)


def original_files_identity():
    paths = [p for p in (OLD_ROAD/'source').iterdir()
             if p.is_file() and p.suffix in TEXT_SUFFIXES]
    paths += [p for p in (OLD_ROAD/'inputs').rglob('*') if p.is_file()]
    paths.append(OLD_ROAD/'settings.json')
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


def check_preflight(root):
    for name in ('restore_a', 'restore_b'):
        audit = read(root/name/'restore.json')
        if not audit.get('exact_full_state') or audit.get('differences'):
            raise RuntimeError(f'Original complete-state restoration failed: {name}')
    same = {str(raw): (root/'restore_a/step000033'/f'{raw:03d}_repaired.png').read_bytes()
                     == (root/'restore_b/step000033'/f'{raw:03d}_repaired.png').read_bytes()
            for raw in (101, 105, 113)}
    if not all(same.values()):
        raise RuntimeError(f'Original next-update repeat PNG check failed: {same}')
    summary = dict(status='PASS_ORIGINAL_ROAD_PREFLIGHT', smoke_updates=32, restore_updates=2,
                   exact_full_model_Adam_sampler_RNG_restore=True, repeated_next_step_PNG_exact=same,
                   native_parent_frozen=True, formal_starts_fresh=True)
    write(root/'smoke_summary.json', summary)
    return summary


def execute(args):
    root = args.run_dir.resolve()
    if root.parent != CAMPAIGN.resolve():
        raise ValueError('Use a fresh direct child of campaign; original relative evidence paths remain unchanged')
    if root.exists():
        raise FileExistsError(f'Never overwrite or silently resume an existing replay: {root}')
    parent, identity, metadata = validate_parent(args)
    if root == parent or root in parent.parents:
        raise ValueError('Replay output must be independent of its frozen native parent')
    source_hashes = {p.name: sha(p) for p in (PACKAGE/'source').iterdir()
                     if p.is_file() and p.suffix in TEXT_SUFFIXES}
    snapshot_hashes = {p.name: sha(p) for p in (PACKAGE/'original_source').iterdir()
                       if p.is_file() and p.suffix in TEXT_SUFFIXES}
    # Every loss/geometry/sampling module except the reviewed three-file
    # compatibility patch must remain byte-identical to the original snapshot.
    changed = sorted(name for name, digest in source_hashes.items() if snapshot_hashes.get(name) != digest)
    if changed != ['editor_common.py', 'native_binding.py', 'repair37.py']:
        raise RuntimeError(f'Unexpected source changes: {changed}')
    if sha(PACKAGE/'original_settings.json') != sha(OLD_ROAD/'settings.json'):
        raise RuntimeError('Original road settings changed')
    preserved_original = original_files_identity()
    root.mkdir()
    copy_text_source(PACKAGE/'source', root/'source')
    copy_text_source(PACKAGE/'original_source', root/'original_source')
    shutil.copy2(PACKAGE/'original_settings.json', root/'settings.json')
    binding = dict(native_parent=str(parent), native_iteration=args.native_iteration,
                   parent_identity=identity, parent_metadata_sha256=metadata,
                   original_source_package=str(CAMPAIGN/'scene008_native_ownership_20261105/source'))
    write(root/'replay_binding.json', binding)
    write(root/'preserved_original_files.json', preserved_original)
    contract = plan(args)
    contract.update(patched_source_sha256=source_hashes, original_source_sha256=snapshot_hashes,
                    settings_sha256=sha(root/'settings.json'), parent_binding=binding)
    write(root/'replay_contract.json', contract)
    started = time.monotonic()
    completed = []

    def frozen_parent():
        now = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in identity}
        assert now == identity, 'Frozen native model files changed'
        assert all(sha(path) == digest for path, digest in metadata.items()), 'Frozen native metadata changed'
        assert original_files_identity() == preserved_original, 'Original road source/inputs changed'

    try:
        for label, script, arguments in steps(args.phase):
            frozen_parent()
            command = [sys.executable, str(root/'source'/script), *arguments]
            write(root/'runner_progress.json', dict(status='RUNNING', stage=label, completed=completed, command=command))
            with (root/f'{label}.log').open('x') as stream:
                subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
            frozen_parent()
            completed.append(label)
            if label == 'restore_b':
                check_preflight(root)
        ledger = dict(status='PASS_COMPLETE_ENGINEERING_REPLAY', phase=args.phase,
                      native_parent=str(parent), native_iteration=args.native_iteration,
                      formal_updates=1000 if args.phase == 'formal' else 0,
                      smoke_updates=32, restore_updates=2,
                      total_updates=1034 if args.phase == 'formal' else 34,
                      wall_seconds=time.monotonic()-started, completed=completed,
                      parent_unchanged=True, original_source_and_inputs_unchanged=True,
                      old_static_gate_loaded=False, old_fine_road_coefficients_loaded=False,
                      quality_accepted=False, human_verdict=None, shutdown=False)
        write(root/'replay_receipt.json', ledger)
        write(root/'runner_progress.json', ledger)
        print(json.dumps(ledger, ensure_ascii=False))
    except BaseException as error:
        write(root/'runner_progress.json', dict(status='FAILED_PRESERVED_FOR_REVIEW',
              completed=completed, wall_seconds=time.monotonic()-started, error=repr(error),
              no_automatic_overwrite=True, quality_accepted=False, shutdown=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-parent', type=Path, required=True)
    parser.add_argument('--native-iteration', type=int, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('preflight', 'formal'), default='preflight')
    parser.add_argument('--execute', action='store_true', help='Create a new private run and invoke the original GPU method')
    args = parser.parse_args()
    if args.native_iteration < 1:
        parser.error('native iteration must be positive')
    if args.execute:
        execute(args)
    else:
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
