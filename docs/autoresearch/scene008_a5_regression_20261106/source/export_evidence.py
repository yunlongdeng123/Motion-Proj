"""Package only this a5 diagnostic's results; checkpoints stay on AutoDL."""
from pathlib import Path
import hashlib
import json
import tarfile
import time

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
SINGLE = CAMPAIGN / 'scene008_a5_old_single30k_20261106_r1'
DUAL = CAMPAIGN / 'scene008_a5_old_dual30k_20261106_r1'
DEST = SINGLE / 'delivery_exports'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    DEST.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    archive = DEST / f'a5_results_{stamp}.tar.gz'
    evaluations = [
        (DUAL / 'a5_evaluation_5k', 'dual5k'),
        (DUAL / 'a5_evaluation_10k', 'dual10k'),
        (SINGLE / 'reference_a5_single10k', 'reference_single10k'),
        (SINGLE / 'reference_a5_full5_B30k', 'reference_full5_B30k'),
        (SINGLE / 'reference_a5_full5_B10k', 'reference_full5_B10k'),
    ]
    files = {}
    selected = []
    for directory, label in evaluations:
        candidates = sorted(directory.parent.glob(directory.name + '*'))
        passing = []
        for candidate in candidates:
            summary = candidate / 'summary.json'
            if candidate.is_dir() and summary.is_file():
                data = json.loads(summary.read_text())
                if data.get('status') == 'PASS_FRESH_NATIVE49_A5_EVALUATION':
                    passing.append(candidate)
        if not passing:
            continue
        directory = passing[-1]
        selected.append(dict(label=label, source=str(directory)))
        if directory.is_dir():
            for path in sorted(directory.rglob('*')):
                if path.is_file():
                    files[f'{label}/{path.relative_to(directory).as_posix()}'] = path
    for root, label in [(SINGLE, 'single'), (DUAL, 'dual')]:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or DEST in path.parents:
                continue
            relative = path.relative_to(root)
            if any(part in {'checkpoint_events', 'visualization', 'eval', 'evaluation', 'point_cloud', 'inputs', '__pycache__'}
                   or 'evaluation' in part or part.startswith('reference_') for part in relative.parts[:-1]):
                continue
            if path.suffix.lower() not in {'.json', '.csv', '.py', '.jsonl'}:
                continue
            files[f'run_receipts/{label}/{relative.as_posix()}'] = path
        for path in sorted((root / 'source').glob('*.py*')):
            if path.is_file():
                files[f'run_receipts/{label}/source/{path.name}'] = path
    manifest = dict(created_unix=time.time(), checkpoint_files_included=False,
                    model_weights_retained_remote=True, automatic_shutdown_executed=False,
                    evaluations_available=selected,
                    files={name: dict(bytes=path.stat().st_size, sha256=sha(path), source=str(path))
                           for name, path in files.items()})
    manifest_path = DEST / f'a5_results_{stamp}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    with tarfile.open(archive, 'w:gz', compresslevel=4) as tar:
        for name, path in files.items():
            tar.add(path, arcname=name, recursive=False)
        tar.add(manifest_path, arcname='retrieval_manifest.json', recursive=False)
    result = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
                  file_count=len(files), evaluations_available=manifest['evaluations_available'])
    (DEST / 'latest_export.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
