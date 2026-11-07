"""CPU packaging only; keeps all49 numeric records and four native RGB anchors."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a5_static_rgb5_ablation_20261107_r2')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['control', 'support'], required=True)
    parser.add_argument('--step', choices=[1000, 5000], type=int, required=True)
    args = parser.parse_args()
    arm = ROOT / args.arm
    source = arm / f'a5_evaluation_{args.step // 1000}k'
    summary = json.loads((source / 'summary.json').read_text())
    assert summary['status'] == 'PASS_FRESH_NATIVE49_A5_EVALUATION'
    assert summary['frame_count'] == 49 and summary['fixed_denominator']['pixels'] == 121128
    assert summary['source_integrity'] == 'PASS_UNCHANGED_BYTES_MTIME_SHA256'
    alias = ('loss5_control' if args.arm == 'control' else 'loss5_full_a5_') + f'{args.step // 1000}k'
    files = [source / name for name in ['summary.json', 'source_integrity.json',
                                       'source_manifest_before.json', 'fixed_domain.json', 'comparison.csv']]
    files += sorted((source / 'a5_float').glob('*.npz'))
    files += sorted((source / 'anchors').glob('*.png'))
    for raw in [89, 97, 105, 109]:
        files += [source / f'frames/{raw:03d}/{name}.png' for name in ['gt', 'keep', 'delete_a37']]
    assert len(list((source / 'a5_float').glob('*.npz'))) == 49
    manifest = {str(path.relative_to(source)): dict(bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files}
    archive = ROOT / f'{alias}_light.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for path in files:
            tar.add(path, arcname=alias + '/' + str(path.relative_to(source)), recursive=False)
    receipt = dict(status='PACKAGED_COMPLETE_ALL49_NUMERICS_FOUR_NATIVE_ANCHORS', alias=alias,
                   source=str(source), archive=str(archive),
                   archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=manifest,
                   frame_count=49, optimization_updates=0)
    (ROOT / f'{alias}_collection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ['status', 'alias', 'archive', 'archive_sha256']}))


if __name__ == '__main__':
    main()
