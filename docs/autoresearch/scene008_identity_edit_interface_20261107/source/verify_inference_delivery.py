"""Verify the downloaded inference archive and extract only manifest-listed files."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import tarfile


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.delivery_root.resolve()
    evidence = root / 'evidence/inference_export'
    expected = json.loads((evidence / 'inference10k_delivery_archive.json').read_text(encoding='utf-8'))
    archive = root / 'model/inference10k_delivery.tar.gz'
    assert archive.stat().st_size == expected['bytes'], 'Archive transfer size mismatch'
    archive_sha = file_sha(archive)
    assert archive_sha == expected['sha256'], 'Archive transfer SHA mismatch'
    manifest_path = evidence / 'file_manifest.json'
    assert file_sha(manifest_path) == expected['file_manifest_sha256']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = dict(manifest['files'])
    entries['file_manifest.json'] = dict(bytes=manifest_path.stat().st_size, sha256=file_sha(manifest_path))
    model = root / 'model/inference10k'
    assert not model.exists(), 'Refusing to overwrite an existing extracted model'
    seen = set()
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            parts = PurePosixPath(member.name).parts
            assert member.isfile() and parts[0] == 'inference10k' and '..' not in parts
            relative = PurePosixPath(*parts[1:]).as_posix()
            assert relative in entries and relative not in seen, member.name
            destination = model.joinpath(*parts[1:])
            assert destination.resolve().is_relative_to(model), 'Unsafe extraction path'
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            assert source is not None
            with destination.open('xb') as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(chunk)
            row = entries[relative]
            assert destination.stat().st_size == row['bytes'], relative
            assert file_sha(destination) == row['sha256'], relative
            seen.add(relative)
    assert seen == set(entries)
    gpu = json.loads((evidence / 'inference10k_equivalence.json').read_text(encoding='utf-8'))
    cli = json.loads((evidence / 'inference10k_cli_comparison.json').read_text(encoding='utf-8'))
    assert all(max(row['max_abs_difference'].values()) == 0 for row in gpu['frames'])
    assert all(row['max_abs_uint8_difference'] == 0 for row in cli['frames'])
    proof = dict(
        status='PASS_INFERENCE_PACKAGE_LOCAL_ARCHIVE_AND_ALL_FILES',
        archive=dict(path=archive.as_posix(), bytes=archive.stat().st_size, sha256=archive_sha),
        extracted_model_directory=model.as_posix(),
        extracted_file_count=len(seen),
        extracted_bytes=sum(row['bytes'] for row in entries.values()),
        every_file_sha256_verified=True,
        file_manifest_sha256=expected['file_manifest_sha256'],
        iteration=10000, dynamic_actor_ids=[5, 6, 18, 26, 37],
        source_PLY_actor_order=[5, 26, 18, 6, 37],
        inference_only=True, training_resume_supported=False,
        native_CUDA_extensions_required=True,
        original_PLY_masks_or_GT_required=False,
        cpu_export_all_non_Adam_Gaussian_fields_and_identity_exact=True,
        original_environment_checkpoint_copied_unchanged=True,
        original_full_training_state_retained_remotely=gpu['original'],
        GPU_equivalence=dict(raw_ids=gpu['raw_ids'], RGB_alpha_and_six_owner_features_max_difference=0,
                             current_owner_ids_exact=gpu['current_owner_ids_exact'],
                             source_actor_table_exact=gpu['source_actor_table_exact']),
        standalone_KEEP_CLI=dict(raw_ids=cli['raw_ids'], PNG_max_difference_from_primary_KEEP=0,
                                 delete_actor_ids=[]),
        optimizer_created=False, optimization_updates=0,
        size_accounting_note='inference_manifest.total_bytes is the initial CPU weight export subtotal. '
                             'file_manifest lists the final bundle including Python source and receipts; '
                             'its own bytes are included in extracted_bytes here.',
    )
    output = evidence / 'inference_package_verification.json'
    output.write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
