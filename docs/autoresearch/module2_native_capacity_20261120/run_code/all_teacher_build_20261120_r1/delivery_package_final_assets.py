"""CPU-only byte-preserving final asset delivery. No model/native imports."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import time

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/native_capacity_20261120_r1')
BUILD = B / 'all_teacher_build_20261120_r1'
SEEN = B / 'numerical_completion/formal_r1'
DELIVERY = BUILD / 'delivery/final_assets_r1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    DELIVERY.mkdir(parents=True, exist_ok=False)
    payload = DELIVERY / 'package'
    payload.mkdir()
    selected = []
    for role, folder in [('FIELD', BUILD / 'field/formal_r1'), ('NUMERIC', BUILD / 'numerical/formal_r1')]:
        for path in sorted((folder / 'asset').rglob('*')):
            if path.is_file():
                selected.append((path, Path('BUILD') / role / 'asset' / path.relative_to(folder / 'asset')))
        names = ['oracle_colors.pt', 'receipt.json', 'endpoint.json']
        names += ['field.pt', 'final_checkpoint.pt'] if role == 'FIELD' else ['final_checkpoint.npz', 'final_checkpoint.rng.pt', 'source_binding.json']
        selected += [(folder / name, Path('BUILD') / role / name) for name in names]
    for name in ['protocol.json', 'results.json', 'execution_accounting.json', 'closure_receipt.json',
                 'owned_source_inventory.json', 'source_freeze.json', 'CPU_inputs.json', 'CPU_tests_stdout.txt', 'README.md']:
        selected.append((BUILD / name, Path('BUILD') / name))
    for name in ['receipt.json', 'common_loss_snapshot.json', 'source_freeze.json']:
        selected.append((BUILD / 'common' / name, Path('BUILD/common') / name))
    for role in ['field', 'numerical']:
        selected.append((BUILD / role / 'smoke_r1/receipt.json', Path('BUILD') / role / 'smoke_receipt.json'))
    for name in ['oracle_colors.pt', 'final_checkpoint.npz', 'final_checkpoint.rng.pt',
                 'receipt.json', 'endpoint.json', 'source_binding.json']:
        selected.append((SEEN / name, Path('seen_G_NUMERIC') / name))
    for source, target in [
        (B / 'protocol.json', Path('history/main_protocol.json')),
        (B / 'numerical_completion/protocol.json', Path('history/seen_numerical_protocol.json')),
        (B / 'public1920/numeric_export_r1/authority_checkpoint_audit_r1.json', Path('seen_G_NUMERIC/authority_checkpoint_audit.json')),
        (BUILD / 'public1920/export_r1/receipt.json', Path('BUILD/public1920/receipt.json')),
        (BUILD / 'public1920/export_r1/packets_manifest.json', Path('BUILD/public1920/packets_manifest.json')),
        (BUILD / 'public1920/F0_precheck_r1/F0_initial_exact.json', Path('BUILD/public1920/F0_initial_exact.json')),
    ]:
        if source.is_file():
            selected.append((source, target))
    if not json.loads((BUILD / 'results.json').read_text()).get('completed'):
        raise RuntimeError('BUILD formal not complete')
    rows, unique = [], {}
    for source, relative in selected:
        if source.is_symlink() or not source.is_file():
            raise ValueError('required original regular file missing: ' + str(source))
        destination = payload / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        size = source.stat().st_size
        key = (digest, size)
        duplicate = unique.get(key)
        if duplicate is None:
            shutil.copy2(source, destination)
            unique[key] = relative
        else:
            os.link(payload / duplicate, destination)
        if sha(destination) != digest or source.stat().st_size != size:
            raise RuntimeError('copied bytes differ from original')
        rows.append(dict(original_path=str(source), package_path=relative.as_posix(),
                         bytes=size, sha256=digest, hardlink_to=None if duplicate is None else duplicate.as_posix()))
    tests = [
        ('code/test_native_capacity_loss.py', 6, 'motionproj Python3.10, pytest'),
        ('code/test_registered_controller.py', 7, 'motionproj Python3.10, pytest'),
        ('gs_oracle/code/test_gs_color_oracle.py', 9, 'motionproj Python3.10, pytest'),
        ('shared_field/code/test_shared_field_gs_read.py', 5, 'motionproj Python3.10, pytest'),
        ('shared_field/code/test_shared_training_state.py', 6, 'motionproj Python3.10, pytest'),
        ('numerical_completion/code/test_numerical_completion.py', 3, 'drivingeditor Python3.9 / SciPy1.13.1, stdlib unittest; complete private Fortran ABI'),
        ('all_teacher_build_20261120_r1/code/test_build_inputs.py', 5, 'motionproj Python3.10, stdlib unittest'),
    ]
    test_manifest = dict(prior_executed_pass_counts_not_rerun_for_packaging=True, expected_total=41,
        CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=1,
        files=[dict(path=str(B / path), expected=count, environment=env, sha256=sha(B / path)) for path, count, env in tests],
        isolation='Keep each test alongside its selected module/helpers or add the code directories to PYTHONPATH. Numerical ABI tests require the pinned real Python3.9 environment; no x-only substitute.')
    write(payload / 'CPU_TEST_FILES.json', test_manifest)
    dependencies = dict(
        source_campaign=str(B), fixed_parent_scene='/root/autodl-tmp/drivingeditor_repro/campaign/road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded/persistent/scene.json',
        native_repository='/root/autodl-tmp/motion_proj_v82_gn11_v42',
        native_environment='/root/autodl-tmp/envs/drivingeditor (Python3.9, Torch/CUDA raster and SciPy1.13.1)',
        world_read_operator='/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/permission/avg_sparse_gh31_r1/operator',
        world_field_runtime='/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/code/world_texture_runtime.py',
        target_and_split_bindings='BUILD/source_freeze.json and source_binding.json retain exact original paths and SHA256',
        standalone=False,
        restore='Preserved scene/proof and checkpoint bytes retain original absolute remote paths. Restore at those paths or create a separately validated explicit rebind. Do not edit archived proof/checkpoint to imply self-containment.',
        actual1920_public='Public loader needs registered1920 camera binding; original scene defaults480. Actual1920 source/public checks are in BUILD/public1920/receipt.json.',
        originals_not_repacked='Main four-arm433MB bundle, intermediate checkpoints, raw native full-resolution NPZ, original models/data/native checkpoints and training images are excluded.')
    write(payload / 'DEPENDENCIES.json', dependencies)
    (payload / 'README.txt').write_text(
        'Final fixed-road asset delivery (CPU packaging; no new optimization/native/GPU execution).\n'
        'BUILD/FIELD includes the complete derived GS asset, authoritative field.pt, derived Boundedci and final full Adam/RNG checkpoint.\n'
        'BUILD/NUMERIC includes complete asset, last-accepted Boundedci and full Fortran workspace/RNG.\n'
        'seen_G_NUMERIC is point/workspace only; its third large asset is intentionally excluded.\n'
        'Original source/known/unknown/observed ledgers and frozen geometry were preserved. No default scene promotion.\n'
        'Training completion is not a strict capacity/quality certificate; numerical FW gates were not reached.\n'
        'See DEPENDENCIES.json: original native scene/models/read operators are required.\n'
        'SOURCE_MANIFEST.json gives original file paths, exact SHA256 and package paths. Identical complete files are hard-linked only inside this new package; original research files were copied and never linked for writing.\n', encoding='utf-8')
    manifest = dict(schema='fixed_road_final_assets_delivery.v1', completed=True, GPU_executed=False,
        optimizer_updates=0, native_calls=0, original_bytes_preserved=True, files=rows,
        original_file_count=len(rows), logical_original_bytes=sum(r['bytes'] for r in rows),
        unique_original_bytes=sum(size for digest, size in unique), exact_duplicate_files=len(rows) - len(unique),
        source_models_data_not_included=True, tests_expected=41)
    write(payload / 'SOURCE_MANIFEST.json', manifest)
    shutil.copy2(payload / 'SOURCE_MANIFEST.json', DELIVERY / 'source_manifest.json')
    archive = DELIVERY / 'build_assets_final_20261120_r1.tar.gz'
    with archive.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=raw, mtime=0) as gz:
            with tarfile.open(mode='w', fileobj=gz, dereference=False) as tar:
                tar.add(payload, arcname='road_native_final_assets', recursive=True)
    digest = sha(archive)
    receipt = dict(completed=True, archive=str(archive), archive_bytes=archive.stat().st_size,
        archive_sha256=digest, manifest=str(DELIVERY / 'source_manifest.json'), manifest_sha256=sha(DELIVERY / 'source_manifest.json'),
        original_file_count=len(rows), logical_original_bytes=manifest['logical_original_bytes'],
        unique_original_bytes=manifest['unique_original_bytes'], exact_duplicate_files=manifest['exact_duplicate_files'],
        source_file_copy_SHA_all_verified=True, elapsed_seconds=time.monotonic() - started,
        expected_CPU_tests=41, GPU_executed=False, native_calls=0, optimizer_updates=0, GPU_keep_on=True)
    write(DELIVERY / 'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
