"""CPU-only C49 checkpoint -> independent canonical road material package."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import shutil

import numpy as np

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT = CAMPAIGN / 'scene008_v44_a5_c49_rgb_20261107_r1'
C_MANIFEST = CAMPAIGN / 'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/scene_manifest.json'
QUALIFICATION = ROOT / 'parent_visibility49/qualification_manifest.json'
CONTRACT = ROOT / 'contract/C49_parent_RGB_training_v1.json'
COLOUR_KEYS = {'parent_render_colours_srgb', 'render_colours_srgb'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')


def same_array_bytes(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.tobytes(order='C') == right.tobytes(order='C'))


def export_material(checkpoint, output, stage, source_manifest=C_MANIFEST,
                    qualification=QUALIFICATION, contract=CONTRACT):
    # torch is needed only to decode the CPU checkpoint; no scene/CUDA loader.
    import torch
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    source_manifest, qualification, contract = (
        Path(p).resolve() for p in (source_manifest, qualification, contract))
    if output.exists():
        raise FileExistsError('export never overwrites a previous package: ' + str(output))
    if stage not in ('engineering_smoke', 'formal'):
        raise ValueError('stage must distinguish smoke from formal updates')
    cm = json.loads(source_manifest.read_text(encoding='utf-8'))
    qm = json.loads(qualification.read_text(encoding='utf-8'))
    source_asset = source_manifest.parent / cm['new_road']['file']
    selection = qualification.parent / qm['selection_file']
    if (sha(source_asset) != cm['new_road']['sha256']
            or sha(selection) != qm['selection_sha256']
            or qm['source_C_asset_sha256'] != cm['new_road']['sha256']):
        raise ValueError('source C/qualification binding changed')
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    if state['schema'] != 'C49_PARENT_RGB_FULL_STATE_V1':
        raise ValueError('unsupported training checkpoint')
    config = state['config']
    if (config['qualification_manifest_sha256'] != sha(qualification)
            or config['source_C_asset_sha256'] != sha(source_asset)
            or config['training_contract_sha256'] != sha(contract)):
        raise ValueError('checkpoint does not bind these inputs')
    if (qm['raw_ids'] != list(range(89, 138))
            or qm['states'] != ['keep', 'delete']):
        raise ValueError('requires the registered all49 BUILD qualification')
    with np.load(source_asset, allow_pickle=False) as saved:
        before = {key: saved[key].copy() for key in saved.files}
    with np.load(selection, allow_pickle=False) as selected:
        ids = selected['eligible_parent_ids'].copy()
        mass = selected['parent_mass'].copy()
        initial = selected['initial_parent_RGB'].copy()
        mapping = selected['child_to_parent_ids'].copy()
    if (ids.dtype != np.int64 or not np.array_equal(ids, np.flatnonzero(mass > 0))
            or len(ids) != qm['eligible_parents'] or len(ids) != config['eligible_parents']
            or not same_array_bytes(initial, before['parent_render_colours_srgb'])
            or not same_array_bytes(mapping, before['child_to_parent_ids'])
            or sha_bytes(initial) != config['initial_parent_RGB_sha256']
            or sha_bytes(ids) != config['eligible_parent_ids_sha256']):
        raise ValueError('qualified parent rows/initial table changed')
    value = state['parent_RGB']
    if (not torch.is_tensor(value) or value.dtype != torch.float32
            or value.shape != (len(ids), 3) or not bool(torch.isfinite(value).all())
            or float(value.min()) < 0 or float(value.max()) > 1):
        raise ValueError('invalid qualified RGB checkpoint values')
    step = state['step']
    if not isinstance(step, int) or isinstance(step, bool) or step < 0 or step > 10000:
        raise ValueError('invalid checkpoint step')
    # The file must come from a complete update boundary, not an in-flight leaf.
    if len(state['history']) != step or (step and state['history'][-1]['step'] != step):
        raise ValueError('checkpoint history does not end at the saved step')
    new = {key: val.copy() for key, val in before.items()}
    parents = initial.copy()
    parents[ids] = value.detach().cpu().numpy()
    new['parent_render_colours_srgb'] = parents
    new['render_colours_srgb'] = parents[mapping]
    zero = mass == 0
    if not same_array_bytes(parents[zero], initial[zero]):
        raise AssertionError('zero-mass parents changed')
    if not np.array_equal(new['render_colours_srgb'].reshape(10098, 4, 3),
                          np.repeat(parents[:, None, :], 4, axis=1)):
        raise AssertionError('four child rows do not inherit exact parent RGB')

    output.mkdir(parents=True)
    asset_path = output / 'new_road_C49_parentRGB.npz'
    np.savez_compressed(asset_path, **new)
    with np.load(asset_path, allow_pickle=False) as actual:
        if set(actual.files) != set(before):
            raise AssertionError('asset fields changed')
        for key in before:
            reference = new[key] if key in COLOUR_KEYS else before[key]
            if not same_array_bytes(actual[key], reference):
                raise AssertionError('asset bytes differ: ' + key)
    manifest = copy.deepcopy(cm)
    for item in manifest['replay_sources'].values():
        src = source_manifest.parent / item['file']
        if sha(src) != item['sha256']:
            raise ValueError('frozen replay source changed: ' + str(src))
        dst = output / item['file']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    metadata = output / 'provenance'
    metadata.mkdir()
    for src, name in ((source_manifest, 'source_C_scene_manifest.json'),
                      (qualification, 'qualification_manifest.json'),
                      (selection, 'qualified_parent_rows.npz'),
                      (contract, 'C49_training_contract.json')):
        shutil.copyfile(src, metadata / name)
    source_files = []
    for path, identity in config['files'].items():
        src = Path(path)
        if src.suffix == '.py':
            if sha(src) != identity['sha256']:
                raise ValueError('executed source changed: ' + path)
            dst = metadata / src.name
            shutil.copyfile(src, dst)
            source_files.append(dict(file=str(dst.relative_to(output)), sha256=sha(dst)))
    exporter = metadata / 'export_C49_material.py'
    shutil.copyfile(Path(__file__).resolve(), exporter)
    write(metadata / 'executed_config.json', config)
    changed = [key for key in before if not same_array_bytes(new[key], before[key])]
    provenance = dict(stage=stage, checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), checkpoint_schema=state['schema'],
        checkpoint_step=step, formal_updates=step if stage == 'formal' else 0,
        engineering_updates_at_checkpoint=step if stage == 'engineering_smoke' else 0,
        new_optimizer_initialization='fresh Adam .01 from saved C warm RGB',
        source_C_manifest_sha256=sha(source_manifest), source_C_asset_sha256=sha(source_asset),
        qualification_file='provenance/qualification_manifest.json',
        qualification_sha256=sha(qualification), selection_file='provenance/qualified_parent_rows.npz',
        selection_sha256=sha(selection), eligible_parents=len(ids), zero_mass_frozen_parents=int(zero.sum()),
        actual_changed_array_keys=changed, only_allowed_array_keys=sorted(COLOUR_KEYS),
        all_other_array_dtype_shape_and_bytes_exact=True,
        fixed_zero_mass_parent_colours_exact=True, exact_four_child_inheritance=True,
        all49_current_role='BUILD', query_requires_Adam=False, query_requires_targets=False,
        exporter_sha256=sha(__file__), executed_source_snapshots=source_files,
        original_model_geometry_opacity_identity_updates=0, adoption_claim=False)
    manifest['new_road'].update(file=asset_path.name, sha256=sha(asset_path),
        material_variant='C49_shared_parentRGB_' + stage,
        qualified_parent_rows=len(ids), frozen_parent_rows=int(zero.sum()))
    manifest['lineage'] = dict(source_C_manifest_file='provenance/source_C_scene_manifest.json',
        source_C_manifest_sha256=sha(source_manifest), source_C_asset_path=str(source_asset),
        source_C_asset_sha256=sha(source_asset), C49_export=provenance)
    manifest['material_writeback'] = dict(shared_parent_RGB=True, child_inheritance=4,
        formal_steps=provenance['formal_updates'], engineering_checkpoint_steps=provenance['engineering_updates_at_checkpoint'],
        current_parent_colour_key='parent_render_colours_srgb', current_child_colour_key='render_colours_srgb',
        other_colour_tables='unchanged original-C provenance; not the current render colour',
        optimizer='fresh Adam .01 projected[0,1]', original_parameter_updates=0,
        geometry_updates=0, opacity_updates=0, identity_updates=0)
    manifest['known_unresolved'] = [
        'Material-only export does not repair newroad/C2 depth-ordering conflicts.',
        'All49 targets including former HELD9 are BUILD; no held-out claim.',
        'Constructed fusion/edgeblend targets are not independent hidden-road RGB truth.',
        'Quality requires independent photometric and vehicle/geometry guards.',
        'This package reuses the original frozen V44 and Semantic-a5 dependencies.']
    manifest.update(exporter_sha256=sha(__file__), GPU_parity_receipt='GPU_parity_not_run.json',
        GPU_parity_state_at_export='PENDING_NEW_ASSET_REPLAY', adoption_claim=False, human_verdict=None)
    write(output / 'scene_manifest.json', manifest)
    write(output / 'GPU_parity_not_run.json', dict(status='NOT_RUN_BY_CPU_EXPORTER',
        additional_native_forwards=0, model_updates=0,
        note='Compare this package with the saved checkpoint in a separately accounted replay.'))
    write(output / 'export_receipt.json', dict(status='PASS_CPU_ONLY_COLOUR_EXPORT_PENDING_GPU_REPLAY',
        scene_manifest_sha256=sha(output / 'scene_manifest.json'), new_asset_sha256=sha(asset_path),
        source_checkpoint_sha256=sha(checkpoint), parent_rows=10098, child_rows=40392,
        **provenance))
    (output / 'architecture.svg').write_text(ARCHITECTURE, encoding='utf-8')
    (output / 'README.md').write_text(README, encoding='utf-8')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU exporter unexpectedly initialized CUDA')
    return output / 'scene_manifest.json'


def sha_bytes(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


ARCHITECTURE = '''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="130" viewBox="0 0 900 130">
<rect width="900" height="130" fill="white"/><g fill="#eef4ff" stroke="#47658a">
<rect x="12" y="32" width="168" height="65" rx="7"/><rect x="218" y="32" width="180" height="65" rx="7"/>
<rect x="438" y="32" width="195" height="65" rx="7"/><rect x="675" y="32" width="210" height="65" rx="7"/></g>
<g fill="#152b45" font-family="sans-serif" font-size="15" text-anchor="middle">
<text x="96" y="58">C warm + checkpoint</text><text x="96" y="80">2119 parent RGB</text>
<text x="308" y="58">CPU colour-only export</text><text x="308" y="80">geometry / IDs exact</text>
<text x="535" y="58">New asset + manifest</text><text x="535" y="80">40392 child rows</text>
<text x="780" y="58">Frozen 21K joint renderer</text><text x="780" y="80">KEEP / DELETE37</text></g>
<g stroke="#47658a" stroke-width="2" fill="#47658a"><path d="M182 65H210M400 65H430M635 65H667"/>
<path d="M210 65l-7-4v8zM430 65l-7-4v8zM667 65l-7-4v8z"/></g></svg>'''

README = '''# C49 共享父格道路材质推理资产

![组件与数据流](architecture.svg)

本包只更新完整 C 支撑的父格、子行渲染颜色。其余数组的 dtype、shape 和字节逐项相同，原 21K 场景与 a5 语义侧车仍为独立冻结依赖。7979 个零质量父格保持源 C 颜色，40392 子行保持原四子共享映射。

推理使用现有 `RoadFusionV44Integration(semantic_runtime, 本包/scene_manifest.json)`，不读取训练 Adam 或目标图像。原 V44 loader 的既有输入依赖保持；本包不包含原场景大权重。资格和训练元数据位于 provenance。

本包是研究候选，默认未采纳。全部 49 帧属于 BUILD；烟检查点与正式步数分开记录。材质导出不代表轮胎遮挡、道路接缝或删除质量通过。CPU 导出没有额外 GPU 前向；新的资产重放须另行记录。
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['engineering_smoke', 'formal'], required=True)
    parser.add_argument('--source-manifest', type=Path, default=C_MANIFEST)
    parser.add_argument('--qualification', type=Path, default=QUALIFICATION)
    parser.add_argument('--contract', type=Path, default=CONTRACT)
    args = parser.parse_args()
    path = export_material(args.checkpoint, args.output, args.stage,
                           args.source_manifest, args.qualification, args.contract)
    print(json.dumps(dict(scene_manifest=str(path), sha256=sha(path)), indent=2))


if __name__ == '__main__':
    main()
