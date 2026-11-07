"""CPU 打包身份元数据及冻结源码；只引用既有高斯权重，不复制或修改它们。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

CAM = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT = CAM / 'scene008_v44_full5_integration_20261107_r1'
REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
OUT = ROOT / 'semantic_identity_package49'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(value)
    return digest.hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    source = REPO / 'scripts/worldsim_v82/drivingeditor_dynamic_edit'
    assert sha(source / 'v44_integration.py') == 'c177c97bea2c73e896fa925d3ecaa8ed490bb5b810e352ed77b2af23c5610c44'
    for suffix, folder in [('49', 'calibration49'), ('4', 'historical_calibration4')]:
        current = ROOT / ('semantic_actor5_calibration' + suffix)
        destination = OUT / folder
        destination.mkdir()
        for name in ['semantic_actor5_manifest.json', 'semantic_actor5_rows.npz',
                     'original_owner_and_lineage.npz', 'calibration_protocol.json', 'receipt.json']:
            shutil.copy2(current / name, destination / name)
        manifest = json.loads((destination / 'semantic_actor5_manifest.json').read_text())
        assert sha(destination / manifest['selection_file']) == manifest['selection_sha256']
        assert sha(destination / manifest['ancestry_file']) == manifest['ancestry_sha256']
        assert sha(destination / manifest['protocol_file']) == manifest['protocol_sha256']
    shutil.copy2(ROOT / 'contract/full5_transform_matrix.txt', OUT / 'full5_transform_matrix.txt')
    api = OUT / 'api_source/drivingeditor_dynamic_edit'
    api.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in ['.py', '.json', '.md']:
            shutil.copy2(path, api / path.name)
    v44 = CAM / 'scene008_v44_rgb_asset_20261107_r1'
    contract = json.loads((v44 / 'contract/input_contract.json').read_text())
    dependencies = OUT / 'frozen_v44_reference'
    dependencies.mkdir()
    for path in [v44 / 'source/v44_runtime.py', v44 / 'contract/v44_contract.py', v44 / 'contract/input_contract.json']:
        shutil.copy2(path, dependencies / path.name)
    for name, expected in contract['expected']['small_file_sha256'].items():
        path = Path(name)
        if path.suffix == '.py':
            assert sha(path) == expected, str(path)
            target = OUT / 'dependency_code' / path.relative_to(CAM)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    reference = dict(
        package_kind='semantic_identity_metadata_only', Gaussian_weights_included=False,
        semantic_default_not_activated=True, quality_acceptance_not_implied=True,
        source_model=contract['paths']['native_parent'], formal_v44_contract=str(v44 / 'contract/input_contract.json'),
        formal_v44_runtime_sha256=sha(v44 / 'source/v44_runtime.py'),
        Full5_model=str(CAM / 'scene008_full_dynamic_edit_20261107_r1/inference10k'),
        Full5_Gaussian_files_sha256=json.loads((OUT / 'calibration49/semantic_actor5_manifest.json').read_text())['gaussian_files_sha256'],
        Full5_existing_local_archive_sha256='6581bfc907f527e322aebdf674c66cd3b75b0cf7321a7f67fff22a0e67277a80',
        v44_frozen_inputs=contract['paths'], v44_frozen_input_expected=contract['expected'],
        repo_HEAD_at_pack=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        adapter_source_sha256=sha(source / 'v44_integration.py'),
        native_CUDA_extensions_and_compatible_environment_required=True,
        original_source_ancestry_is_separate_from_estimated_semantic_identity=True,
        no_original_BUILD_mask_required_to_select_rows_during_QUERY=True,
        no_Gaussian_training_updates_in_package_build=True)
    (OUT / 'dependency_references.json').write_text(json.dumps(reference, indent=2) + '\n')
    readme = '''# Semantic actor5 identity metadata package

This is a frozen engineering candidate, not an adoption or quality-pass claim.
The original v44 remains the baseline. The strict raw94 photometric failures remain in the reports.

calibration49/ is the current explicit BUILD-all49 identity estimate. It contains the full
inside/total visible-mass arrays and exact static/dynamic row IDs. The predicate is fixed:
total_mass > 0 and inside_mass / total_mass > 0.5. No Gaussian position, colour, opacity,
time, velocity, scale, original source actor or lineage was changed. Unknown/unobserved rows
are not guessed. historical_calibration4/ preserves the preceding four-view candidate.
Both folders retain original_owner_and_lineage.npz separately from semantic selections.

Weights are NOT copied here. Restore the referenced Full5 inference10k package and the
original formal v44 dependencies listed in dependency_references.json. Full5 weights must
match the two manifest SHA256 values. The original v44 static/env/road/gates/fine/C2 and its
native a37 path remain frozen. Original v44 data/code paths must satisfy its input contract;
this is not a standalone CUDA environment or a relocation of that baseline.

The full inside/total NPZ is deliberately retained: the current loader rechecks the exact
majority predicate. Do not replace it with only row IDs. Full5 transform metadata is copied
here so QUERY need not read its old source PLY tree. Original BUILD actor masks are used for
calibration, not for query-frame clipping. The original v44 runtime retains its existing
camera/pose/asset dependencies. Native CUDA extensions are still required.

In a fresh worker, add PACKAGE/api_source and the formal v44 source directory to PYTHONPATH:

    from pathlib import Path
    from v44_runtime import V44Runtime
    from drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
    package = Path('/path/to/semantic_identity_package49')
    base = V44Runtime('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
    runtime = SemanticActor5V44Integration(
        base, semantic_sidecar_path=package/'calibration49/semantic_actor5_manifest.json',
        full5_model='/path/to/restored/inference10k',
        full5_transform_path=package/'full5_transform_matrix.txt')
    keep = runtime.render_scene(94, delete=False)
    delete_a37 = runtime.render_scene(94, delete=True)
    receipt = runtime.receipt()

Only the selected Full5 static/dynamic rows are appended under semantic group5. Source37
rows selected as semantic5 keep their original source ancestry in the separate table.
This does not mean all source37 rows were deleted. The v44 target a37 follows its original
edit path; there is no static-source retirement in this mode. All components participate
in one native depth sort, with no pixel compositing or colour fitting. No optimizer exists.

Architecture: Full5 frozen weights + BUILD masks -> visible alphaT masses -> fixed semantic
row sidecar -> selected static/dynamic rows + original v44 -> joint native KEEP / DELETE37.
Full49 calibration statistics do not authorize full49 candidate image rendering.
'''
    (OUT / 'README.md').write_text(readme)
    files = {path.relative_to(OUT).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(OUT.rglob('*')) if path.is_file()}
    index = dict(kind='semantic_identity_metadata_package49', files=files,
                 payload_bytes=sum(value['bytes'] for value in files.values()), file_count=len(files),
                 weights_included=False, source_ancestry_kept_separate=True)
    (OUT / 'file_manifest.json').write_text(json.dumps(index, indent=2) + '\n')
    archive = ROOT / 'semantic_identity_package49.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=6) as output:
        for name in list(files) + ['file_manifest.json']:
            output.add(OUT / name, arcname=OUT.name + '/' + name, recursive=False)
    receipt = dict(path=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
                   file_count=len(files) + 1, payload_bytes=index['payload_bytes'],
                   manifest_sha256=sha(OUT / 'file_manifest.json'), weights_copied=False, GPU_used=False)
    (ROOT / 'semantic_identity_package49_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
