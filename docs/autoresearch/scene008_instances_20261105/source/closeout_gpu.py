"""Check completed Scene008 runs and archive the inputs and final states created here."""
import csv
import hashlib
import json
import math
import sys
import tarfile
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def training(path, stage, first, last):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert [r['step'] for r in rows] == list(range(first, last + 1)), path
    assert all(r['stage'] == stage and 89 <= r['raw'] <= 137 and math.isfinite(r['loss'])
               and all(math.isfinite(x) for x in r['terms'].values()) for r in rows), path
    return dict(updates=len(rows), elapsed_s=rows[-1]['elapsed_s'],
                peak_cuda_bytes=max(r['max_cuda_bytes'] for r in rows),
                raw_count=len({r['raw'] for r in rows}))


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main(root, smoke_name, formal_name, destination):
    smoke, formal = root / smoke_name, root / formal_name
    settings = read(root / 'source/settings.json')
    receipt = read(smoke / 'receipt.json')
    assert receipt['status'] == 'PASS' and receipt['settings'] == settings
    assert all(receipt[k] for k in ('restored_b', 'restored_c', 'd_entry'))
    baseline = read(smoke / 'baseline_replay.json')
    assert set(baseline) == {'105', '113', '130'} and max(baseline.values()) <= 1
    n_b, n_c = settings['steps_b'], settings['steps_c']
    b = training(formal / 'b/training.jsonl', 'B', 1, n_b)
    c = training(formal / 'c/training.jsonl', 'C', 1, n_c)
    attr = read(formal / 'ledger_all49.json')
    assert attr['raw_ids'] == list(range(89, 138)) and attr['eligible'] > 0
    eval_dir = formal / 'eval_all49'
    evaluation = read(eval_dir / 'summary.json')
    assert sorted(evaluation['raw_ids']) == list(range(89, 138))
    assert set(evaluation['groups']) == {'strong_baseline', 'objects', 'replacement', 'pvd'}
    assert set(evaluation['max_group_conservation_error']) == set(evaluation['groups'])
    assert max(evaluation['max_group_conservation_error'].values()) <= 1e-4
    with (eval_dir / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        metrics = list(csv.DictReader(stream))
    expected_frames = {f'{n:03d}' for n in range(89, 138)}
    assert {d.name for d in (eval_dir / 'frames').iterdir()} == expected_frames
    for name in expected_frames:
        assert len(list((eval_dir / 'frames' / name).glob('*.png'))) == 31
    for group in evaluation['groups']:
        for state in ('unedited', 'delete_a37'):
            key = f'{group}/{state}/a6_visible/rgb_mae'
            assert evaluation['pooled'][key]['count'] == 39308, key
    b_final = formal / f'b/checkpoint{n_b:06d}.pt'
    c_final = formal / f'c/checkpoint{n_c:06d}.pt'
    d_final = formal / 'd/checkpoint000000.pt'
    paths = [b_final, c_final, d_final, formal / 'ledger_all49.json', formal / 'ledger_all49.npz',
             formal / 'b/training.jsonl', formal / 'c/training.jsonl', eval_dir / 'summary.json',
             eval_dir / 'metrics.csv', smoke / 'receipt.json', smoke / 'baseline_replay.json',
             smoke / 'ledger_pilots.json', root / 'preflight_cpu.json', root / 'source/settings.json',
             root / 'inputs/ledger_cpu.npz', root / 'inputs/ledger_cpu_projections.npz', root / 'inputs/ledger_cpu.json',
             root / f'{formal_name}.log', root / f'{smoke_name}.log', root / 'observations/receipt.json',
             root / 'observations/a6.npz', root / 'observations/a37.npz',
             *sorted((root / 'observations/frames').glob('*.npz'))]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    destination.mkdir(exist_ok=False)
    run_receipt = destination / 'run_receipt.json'
    run_receipt.write_text(json.dumps(dict(engineering_status='PASS', smoke=receipt, baseline_replay=baseline,
        formal=dict(b=b, c=c, d_updates=0, attribute=attr, eval_frames=49, metric_rows=len(metrics),
                    max_group_conservation_error=evaluation['max_group_conservation_error']),
        human_verdict=None, quality_accepted=False), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    archive = destination / 'scene008_replay_inputs.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=1, format=tarfile.PAX_FORMAT) as bundle:
        for path in paths + [run_receipt]:
            bundle.add(path, arcname=path.relative_to(root).as_posix())
    summary = dict(file=archive.name, bytes=archive.stat().st_size, sha256=digest(archive), members=len(paths) + 1)
    (destination / 'archive_receipt.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], Path(sys.argv[4]).resolve())
