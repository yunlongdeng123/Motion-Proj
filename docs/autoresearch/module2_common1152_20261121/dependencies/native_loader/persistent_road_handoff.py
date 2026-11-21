"""Explicit persistent road/static-opacity handoff loader; factory defaults stay unchanged.

The final v44.scene manifest supplies a saved road and combined static gates.
An independent proof pins their contents, native row topology and model files.
Metric raster scales every row and camera together. Optional per-road-row
minimum_model_depth is applied in original model units before that scaling.
"""
from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .v44_metric_raster import metric_raster


PROOF_SCHEMA = 'persistent_road_handoff.v1'
CHECKPOINT_SCHEMA = 'persistent_static_opacity_handoff.v1'
HARD_NEAR_MODEL = .2  # Inherited native CUDA center-near ABI; not a scene parameter.


def save_persistent_road_handoff(source_manifest, source_proof, road_path, output_dir,
                                 *, training_record=None):
    """保存新道路并原样继承静态门控；不训练、不退役任何原生行。

    先验证父资产与绑定，再写入全新目录。几何/保持/画质验收仍由调用方
    完成；本函数只建立公共加载器可以严格重载的持久资产关系。
    """
    import torch

    source_manifest, source_proof, road_path = [Path(p).resolve()
        for p in (source_manifest, source_proof, road_path)]
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('持久资产须保存到全新目录；保留旧现场')
    source = json.loads(source_manifest.read_text(encoding='utf-8'))
    previous = json.loads(source_proof.read_text(encoding='utf-8'))
    if source.get('schema') != 'v44.scene.v1' or previous.get('schema') != PROOF_SCHEMA:
        raise ValueError('父场景必须为显式持久道路资产')
    _pin(previous['manifest'], source_manifest, source_proof.parent)
    old_road = _path(source['road'], source_manifest.parent)
    _pin(previous['road'], old_road, source_proof.parent)
    _road_metadata(old_road, previous['road'])
    model = _path(source['model'], source_manifest.parent)
    native = {item['role']: item for item in previous['native_sources']}
    if len(native) != len(previous['native_sources']):
        raise ValueError('父模型角色重复')
    for role, prefix in [('static', 'chkpnt'), ('dynamic', 'dynamic_chkpnt'), ('environment', 'env_light_chkpnt')]:
        _pin(native[role], model / f'{prefix}{source["iteration"]}.pth', source_proof.parent)
    if source.get('ownership') is not None:
        _pin(previous['ownership'], _path(source['ownership'], source_manifest.parent), source_proof.parent)
    index_path = _path(source['static_gates']['indices'], source_manifest.parent)
    gate_path = _path(source['static_gates']['checkpoint'], source_manifest.parent)
    _pin(previous['gates']['indices'], index_path, source_proof.parent)
    _pin(previous['gates']['checkpoint'], gate_path, source_proof.parent)
    saved = torch.load(gate_path, map_location='cpu', weights_only=False)
    with np.load(index_path, allow_pickle=False) as data:
        ids = data['ids'].copy()
    ns = _count(previous['topology']['static_rows'], 'static_rows')
    _count(previous['topology']['dynamic_rows'], 'dynamic_rows')
    gate, contributors = saved.get('gate'), saved.get('contributors')
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
            or (ids < 0).any() or (ids >= ns).any()
            or saved.get('schema') != CHECKPOINT_SCHEMA
            or not isinstance(gate, torch.Tensor) or not gate.is_floating_point()
            or gate.shape != ids.shape or not bool(torch.isfinite(gate).all())
            or not isinstance(contributors, torch.Tensor)
            or contributors.dtype not in (torch.int32, torch.int64)
            or not np.array_equal(contributors.numpy(), ids)):
        raise ValueError('父静态门控拓扑或数值无效')
    expected = dict(manifest_sha256=_sha(source_manifest), road_sha256=_sha(old_road), **previous['topology'])
    if (saved.get('binding') != previous['gates']['checkpoint_binding']
            or any(saved['binding'].get(k) != v for k, v in expected.items())):
        raise ValueError('父静态门控绑定不一致')
    with np.load(road_path, allow_pickle=False) as data:
        road_record = dict(path=str(road_path), sha256=_sha(road_path),
            total_rows=len(data['active']), active_rows=int(data['active'].sum()),
            meter_to_model=float(data['meter_to_model']))
        for name in ['reference_srgb', 'road_opacity', 'gaussian_sigma_m']:
            if name in data and not np.isfinite(data[name]).all():
                raise ValueError('新道路参数包含非有限值: ' + name)
        for name in ['reference_srgb', 'road_opacity']:
            if name in data and ((data[name] < 0).any() or (data[name] > 1).any()):
                raise ValueError('新道路颜色/透明度不在[0,1]: ' + name)
        if 'gaussian_sigma_m' in data and (data['gaussian_sigma_m'] <= 0).any():
            raise ValueError('新道路尺度必须为正')
    _road_metadata(road_path, road_record)
    scene = copy.deepcopy(source)
    scene['model'], scene['road'] = str(model), str(road_path)
    if scene.get('ownership') is not None:
        scene['ownership'] = str(_path(scene['ownership'], source_manifest.parent))
    scene['static_gates'] = dict(indices=str(output / 'static_indices.npz'), checkpoint=str(output / 'static_opacity.pt'))
    build = dict(schema='road_asset_persistent_build.v2', source_manifest=str(source_manifest),
        source_proof=str(source_proof), road=str(road_path), retired_static_rows=0,
        retained_old_gate_rows=len(ids), native_model_retrained=False,
        static_gate_values_exact=True, training_record=training_record, quality_pass_claimed=False)
    # 在创建目录前验证JSON可序列化；不留下貌似完整的NaN合同。
    scene_text = json.dumps(scene, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    build_text = json.dumps(build, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'scene.json').write_text(scene_text, encoding='utf-8')
    np.savez_compressed(output / 'static_indices.npz', ids=ids)
    binding = dict(manifest_sha256=_sha(output / 'scene.json'), road_sha256=_sha(road_path), **previous['topology'])
    torch.save(dict(schema=CHECKPOINT_SCHEMA, gate=gate.clone(), contributors=contributors.clone(),
        binding=binding), output / 'static_opacity.pt')
    def pin(path):
        return dict(path=str(path), sha256=_sha(path))
    proof = copy.deepcopy(previous)
    proof.update(manifest=pin(output / 'scene.json'), road=road_record)
    proof['gates'] = dict(indices=pin(output / 'static_indices.npz'), checkpoint=pin(output / 'static_opacity.pt'), checkpoint_binding=binding)
    (output / 'proof.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'build.json').write_text(build_text, encoding='utf-8')
    return output / 'scene.json', output / 'proof.json'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _path(value, base):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _pin(record, expected, base):
    path = _path(record['path'], base)
    if path != Path(expected).resolve() or not path.is_file() or _sha(path) != record['sha256']:
        raise ValueError(f'persistent handoff file binding differs: {expected}')
    return path


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _road_metadata(path, declared):
    with np.load(path, allow_pickle=False) as data:
        active = data['active'].copy()
        xyz = data['model_xyz'].copy()
        scale_value = np.asarray(data['meter_to_model'])
        minimum = data['minimum_model_depth'].copy() if 'minimum_model_depth' in data.files else None
        transform = data['model_from_world'].copy()
    if active.ndim != 1 or active.dtype.kind not in 'biu' or not np.isin(active, [0, 1]).all():
        raise ValueError('road active must be a complete bool/0-1 world-row array')
    active = active.astype(bool)
    if xyz.shape != (len(active), 3) or not np.isfinite(xyz).all():
        raise ValueError('road xyz and active topology differ')
    if scale_value.shape != ():
        raise ValueError('road meter_to_model must be scalar')
    scale = _finite(scale_value.item(), 'road meter_to_model')
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('road transform must be finite 4x4')
    rotation = transform[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=2e-5):
        raise ValueError('road transform is not a similarity matching meter_to_model')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError('road transform has an invalid homogeneous row')
    if _count(declared['total_rows'], 'road total_rows') != len(active) or _count(declared['active_rows'], 'road active_rows') != int(active.sum()):
        raise ValueError('road metadata topology differs from proof')
    if not math.isclose(_finite(declared['meter_to_model'], 'proof road scale'), scale, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('road metric scale differs from proof')
    if minimum is not None:
        if minimum.shape != active.shape or minimum.dtype.kind not in 'iuf' or not np.isfinite(minimum).all() or (minimum < 0).any():
            raise ValueError('minimum_model_depth must be finite nonnegative and match complete road active length')
    return active, xyz, scale, minimum


def _extend_snapshot(scene, added):
    """Append watched immutable state without hiding any old state mutation."""
    scene.assert_persistent_unchanged()
    old = dict(scene._versions)
    scene._watched = tuple(scene._watched) + tuple(added)
    current = scene._snapshot()
    if any(current.get(key) != value for key, value in old.items()):
        raise RuntimeError('old persistent state changed while installing the handoff')
    scene._versions = current
    scene.assert_persistent_unchanged()


def load_persistent_handoff(manifest_path, proof_path, *, factory=None):
    """Return ``(scene, manifest, receipt)`` from saved assets and strict proof.

    ``factory`` is injectable for CPU protocol tests. The normal path uses the
    existing from_manifest, including its original persistent-state snapshot.
    No actor IDs, scene IDs, frame counts or edit queries are encoded here.
    """
    import torch

    manifest_path = Path(manifest_path).resolve()
    proof_path = Path(proof_path).resolve()
    source = json.loads(manifest_path.read_text(encoding='utf-8'))
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    if source.get('schema') != 'v44.scene.v1' or proof.get('schema') != PROOF_SCHEMA:
        raise ValueError('explicit v44.scene and persistent handoff proof are required')
    if source.get('raster_units') is not None:
        raise ValueError('metric raster is supplied only by this proof; refuse double wrapping')
    _pin(proof['manifest'], manifest_path, proof_path.parent)
    road_path = _path(source['road'], manifest_path.parent)
    _pin(proof['road'], road_path, proof_path.parent)
    active, road_xyz, meter_to_model, minimum = _road_metadata(road_path, proof['road'])
    units = proof['raster_units']
    if set(units) != {'unit_scale', 'legacy_native_near'}:
        raise ValueError('raster_units requires exactly unit_scale and legacy_native_near')
    unit_scale = _finite(units['unit_scale'], 'unit_scale')
    legacy_near = _finite(units['legacy_native_near'], 'legacy_native_near')
    if unit_scale < 1 or not math.isclose(unit_scale * meter_to_model, 1., abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError('unit_scale must be >=1 and exactly convert the declared model similarity to meters')
    if legacy_near != HARD_NEAR_MODEL:
        raise ValueError('legacy_native_near does not match the inherited .2 CUDA ABI')
    topology = proof['topology']
    ns = _count(topology['static_rows'], 'static_rows')
    nd = _count(topology['dynamic_rows'], 'dynamic_rows')
    model_dir = _path(source['model'], manifest_path.parent)
    iteration = _count(source['iteration'], 'iteration')
    native = {record['role']: record for record in proof['native_sources']}
    if len(native) != len(proof['native_sources']):
        raise ValueError('native source roles must be unique')
    # These are the actual checkpoint names consumed by scene_io.load_native_models.
    for role, prefix in [('static', 'chkpnt'), ('dynamic', 'dynamic_chkpnt'), ('environment', 'env_light_chkpnt')]:
        if role not in native:
            raise ValueError(f'missing native source binding: {role}')
        _pin(native[role], model_dir / f'{prefix}{iteration}.pth', proof_path.parent)
    if source.get('ownership') is not None:
        _pin(proof['ownership'], _path(source['ownership'], manifest_path.parent), proof_path.parent)
    gate_paths = source['static_gates']
    index_path = _path(gate_paths['indices'], manifest_path.parent)
    checkpoint_path = _path(gate_paths['checkpoint'], manifest_path.parent)
    _pin(proof['gates']['indices'], index_path, proof_path.parent)
    _pin(proof['gates']['checkpoint'], checkpoint_path, proof_path.parent)
    with np.load(index_path, allow_pickle=False) as data:
        indices = data['ids'].copy()
    if indices.dtype.kind not in 'iu' or indices.ndim != 1 or len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= ns).any():
        raise ValueError('combined gate IDs are not unique valid static row indices')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if saved.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('combined gate checkpoint schema differs')
    gate = saved['gate']
    if not isinstance(gate, torch.Tensor) or not gate.is_floating_point() or gate.shape != indices.shape or not bool(torch.isfinite(gate).all()):
        raise ValueError('combined gate logits must be a finite floating vector matching IDs')
    contributors = saved['contributors']
    if not isinstance(contributors, torch.Tensor) or contributors.dtype not in (torch.int32, torch.int64) or not np.array_equal(contributors.cpu().numpy(), indices):
        raise ValueError('checkpoint contributor topology differs from combined indices')
    if json.dumps(saved.get('binding'), sort_keys=True, allow_nan=False) != json.dumps(proof['gates']['checkpoint_binding'], sort_keys=True, allow_nan=False):
        raise ValueError('checkpoint binding differs from independently pinned handoff proof')
    required_binding = dict(manifest_sha256=proof['manifest']['sha256'], road_sha256=proof['road']['sha256'], static_rows=ns, dynamic_rows=nd)
    if any(saved['binding'].get(key) != value for key, value in required_binding.items()):
        raise ValueError('combined gate binding does not match this manifest/road/native topology')
    if factory is None:
        from .v44_factory import from_manifest
        factory = from_manifest
    scene, manifest = factory(manifest_path)
    scene.assert_persistent_unchanged()
    if Path(manifest.model).resolve() != model_dir or manifest.iteration != iteration or Path(manifest.road).resolve() != road_path:
        raise ValueError('factory returned a different model, iteration or persistent road manifest')
    actual_ns = len(scene.static.get_xyz)
    actual_nd = 0 if scene.dynamic is None else len(scene.dynamic.get_xyz)
    if (actual_ns, actual_nd) != (ns, nd):
        raise ValueError('loaded native topology differs from saved gate binding')
    if getattr(scene, '_raster_units', None) is not None or hasattr(scene, '_persistent_road_handoff'):
        raise ValueError('the loaded scene already has a metric/handoff wrapper')
    expected_gates = scene.static.get_xyz.new_ones(ns)
    expected_gates[torch.as_tensor(indices.astype(np.int64), device=expected_gates.device)] = gate.sigmoid().to(expected_gates.device, expected_gates.dtype)
    if scene._gates is None or not torch.equal(scene._gates, expected_gates):
        raise ValueError('factory did not load the exact saved combined gate multipliers')
    road_providers = [i for i, provider in enumerate(scene._providers) if provider.kind == 'road']
    if len(road_providers) != 1:
        raise ValueError('persistent handoff requires exactly one saved road provider')
    road_index = road_providers[0]
    provider = scene._providers[road_index]
    if not provider.state_objects or not torch.equal(provider.state_objects[0].xyz,
            torch.as_tensor(road_xyz[active], dtype=scene.static.get_xyz.dtype, device=scene.static.get_xyz.device)):
        raise ValueError('loaded road row order differs from the pinned complete asset')
    minimum_tensor = None
    if minimum is not None:
        minimum_tensor = torch.as_tensor(minimum[active].copy(), dtype=scene.static.get_xyz.dtype,
                                         device=scene.static.get_xyz.device).clone()
        original_road_rows = provider.rows
        def bounded_road_rows(frame_key, camera, pipe):
            values = original_road_rows(frame_key, camera, pipe)
            if values is None:
                return None
            if len(values['means3D']) != len(minimum_tensor):
                raise RuntimeError('persistent road topology changed after handoff load')
            depth = values['means3D'] @ camera.world_view_transform[:3, 2] + camera.world_view_transform[3, 2]
            return dict(values, mask=values['mask'] & (depth > minimum_tensor))
        wrapped_provider = replace(provider, rows=bounded_road_rows)
        providers = list(scene._providers)
        providers[road_index] = wrapped_provider
        scene._providers = tuple(providers)
        _extend_snapshot(scene, (minimum_tensor, wrapped_provider))
    ops = scene._bindings()
    scene._ops = replace(ops, raster=metric_raster(ops.raster, unit_scale))
    original_rows = scene._rows
    native_prefix = ns + (nd if scene._include_dynamic else 0)
    def metric_rows(frame_key, focus, diagnostic_ids, branch_alpha):
        values, owners, camera, coverage = original_rows(frame_key, focus, diagnostic_ids, branch_alpha)
        means = values['means3D'][:native_prefix]
        depth = means @ camera.world_view_transform[:3, 2] + camera.world_view_transform[3, 2]
        mask = values['mask'].clone()
        mask[:native_prefix] &= depth > legacy_near
        return dict(values, mask=mask), owners, camera, coverage
    scene._rows = metric_rows
    receipt = dict(schema=PROOF_SCHEMA, manifest_sha256=proof['manifest']['sha256'],
        proof_sha256=_sha(proof_path), road_sha256=proof['road']['sha256'],
        combined_checkpoint_sha256=proof['gates']['checkpoint']['sha256'],
        static_rows=ns, dynamic_rows=nd, gated_static_rows=len(indices),
        road_total_rows=len(active), road_active_rows=int(active.sum()), meter_to_model=meter_to_model,
        unit_scale=unit_scale, physical_road_near_m=HARD_NEAR_MODEL,
        native_legacy_near_preserved=True, depth_features_in_original_model_units=True,
        optional_minimum_model_depth=minimum is not None,
        bounded_active_road_rows=int((minimum[active] > 0).sum()) if minimum is not None else 0,
        added_minimum_tensor_is_watched=minimum is not None,
        persistent_world_asset=True, image_space_final_paste=False, optimizer_updates=0,
        quality_pass_claimed=False)
    scene._persistent_road_handoff = dict(receipt=receipt, minimum_model_depth=minimum_tensor)
    scene.assert_persistent_unchanged()
    return scene, manifest, receipt
