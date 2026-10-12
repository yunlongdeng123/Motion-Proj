"""Hash the real49-frame experiment inputs against the saved formal manifest.

No model, renderer, Torch, CUDA, or training imports. Evidence only.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binding', type=Path, required=True)
args = parser.parse_args()
binding = json.loads(args.binding.read_text(encoding='utf8'))
source = Path(binding['parent_training_config']['source_path']).resolve()
parent = Path(binding['parent_static_checkpoint']).parent
manifest = parent / 'input_files_before_training.json'
expected_manifest_sha = binding['protected_original_file_sha256'][str(manifest)]
assert sha(manifest) == expected_manifest_sha
rows = json.loads(manifest.read_text(encoding='utf8'))
actual = {}
for row in rows:
    relative = PurePosixPath(row['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = source / Path(*relative.parts)
    assert source in path.resolve().parents and path.is_file() and not path.is_symlink()
    assert path.stat().st_size == row['bytes'], row['path']
    digest = sha(path)
    assert digest == row['sha256'], row['path']
    assert row['path'] not in actual
    actual[row['path']] = digest
required = []
for i in range(49):
    frame = f'0017{i:03d}'
    for prefix, suffix in [('calib', 'txt'), ('pose', 'txt'), ('image/image_0', 'png'), ('sky/sky_0', 'png'), ('dynamic_masks/dynamic_mask_0', 'png')]:
        relative = f'{prefix}/{frame}.{suffix}'
        assert relative in actual, relative
        required.append(relative)
assert not (source / 'dynamic_masks/shadow_mask_0').exists()
report = {'status': 'PASS', 'scope': 'Real source-data complete saved input manifest SHA validation; CPU only', 'binding_sha256': sha(args.binding), 'input_manifest': str(manifest), 'input_manifest_sha256': expected_manifest_sha, 'data_source': str(source), 'full_input_file_count': len(rows), 'all_original_input_sha_match': True, 'all49_calibration_pose_RGB_sky_actor_required_file_count': len(required), 'all49_calibration_pose_RGB_sky_actor_sha_match': True, 'new_shadow_mask_absent': True, 'file_sha256': actual, 'GPU_used': False, 'formal_input_modified': False, 'camera_receipt_limit': 'This establishes original data/calibration/pose inputs plus the separately pinned unchanged loader/camera/normalization source. It does not claim an independent all49 actualScene pose receipt or actual variant runtime equality.'}
output = Path(__file__).resolve().parent / 'common49_input_manifest_review_cpu.json'
output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
print(json.dumps({key: report[key] for key in ('status', 'binding_sha256', 'input_manifest_sha256', 'full_input_file_count', 'all49_calibration_pose_RGB_sky_actor_required_file_count')}, indent=2))
