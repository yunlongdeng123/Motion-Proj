"""持久道路资产与静态退役的统一保存、重载和四阶段成像导出。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def save_scene(source_manifest, source_proof, road, output, retirement=None):
    """保留原生模型与旧门控，只合入独立保存的道路和固定静态行退役。"""
    import torch

    source_manifest, source_proof, road = map(lambda p: Path(p).resolve(), (source_manifest, source_proof, road))
    output = Path(output).resolve()
    if (output / 'scene.json').exists():
        raise FileExistsError('场景已冻结；请使用新输出目录')
    output.mkdir(parents=True, exist_ok=True)
    source = read(source_manifest)
    previous = read(source_proof)
    if pin(source_manifest) != previous['manifest']:
        raise ValueError('父场景与proof不匹配')
    saved = torch.load(source['static_gates']['checkpoint'], map_location='cpu', weights_only=False)
    with np.load(source['static_gates']['indices'], allow_pickle=False) as data:
        old_ids = data['ids'].astype(np.int64)
    count = previous['topology']['static_rows']
    if saved['gate'].shape != old_ids.shape or not np.array_equal(saved['contributors'].numpy(), old_ids):
        raise ValueError('父门控拓扑不一致')
    if pin(source['static_gates']['checkpoint']) != previous['gates']['checkpoint']:
        raise ValueError('父门控checkpoint与proof不匹配')
    if pin(source['static_gates']['indices']) != previous['gates']['indices']:
        raise ValueError('父门控行号与proof不匹配')
    retired = np.empty(0, dtype=np.int64)
    if retirement:
        with np.load(retirement, allow_pickle=False) as data:
            retired = data['ids']
        if retired.ndim != 1 or retired.dtype.kind not in 'iu' or len(np.unique(retired)) != len(retired) or (retired < 0).any() or (retired >= count).any():
            raise ValueError('退役列表必须为互异的原始静态行号')
        retired = retired.astype(np.int64)
        with np.load(source['ownership'], allow_pickle=False) as data:
            static_owners = data['static_actor_ids']
        if len(static_owners) != count or (static_owners[retired] > 0).any():
            raise ValueError('禁止退役已有明确车辆归属的静态行')
    ids = np.union1d(old_ids, retired)
    # 极小但非零的旧门控仍逐值保留；新退役在float32 sigmoid中为精确0。
    gate = torch.full((len(ids),), 18., dtype=saved['gate'].dtype)
    gate[torch.from_numpy(np.searchsorted(ids, old_ids))] = saved['gate']
    gate[torch.from_numpy(np.searchsorted(ids, retired))] = -100.
    np.savez_compressed(output / 'static_indices.npz', ids=ids)
    scene = copy.deepcopy(source)
    scene['road'] = str(road)
    scene['static_gates'] = dict(indices=str(output / 'static_indices.npz'), checkpoint=str(output / 'static_opacity.pt'))
    write(output / 'scene.json', scene)
    binding = dict(manifest_sha256=pin(output / 'scene.json')['sha256'], road_sha256=pin(road)['sha256'], **previous['topology'])
    torch.save(dict(schema='persistent_static_opacity_handoff.v1', gate=gate, contributors=torch.from_numpy(ids), binding=binding), output / 'static_opacity.pt')
    proof = copy.deepcopy(previous)
    proof['manifest'] = pin(output / 'scene.json')
    with np.load(road, allow_pickle=False) as data:
        proof['road'] = dict(**pin(road), total_rows=len(data['active']), active_rows=int(data['active'].sum()), meter_to_model=float(data['meter_to_model']))
    proof['gates'] = dict(indices=pin(output / 'static_indices.npz'), checkpoint=pin(output / 'static_opacity.pt'), checkpoint_binding=binding)
    write(output / 'proof.json', proof)
    write(output / 'build.json', dict(schema='road_asset_persistent_build.v1', source_manifest=str(source_manifest), source_proof=str(source_proof), road=str(road), retired_static_rows=len(retired), retained_old_gate_rows=len(old_ids), native_model_retrained=False, training_updates=0, GPU='keep_on'))
    return output / 'scene.json', output / 'proof.json'


def _save_rgb(path, value):
    from PIL import Image
    value = np.asarray(value)
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    picture = Image.fromarray(np.uint8(np.clip(value, 0, 1) * 255 + .5))
    if Path(path).suffix.lower() == '.webp':
        picture.save(path, lossless=True, method=4)
    else:
        picture.save(path)


def render(manifest_path, proof_path, output, keys, domain_dir=None, baseline=None):
    """道路单层与联合层在同一物理相机计算，最终RGB由同排序光栅产生。"""
    import torch
    from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff

    torch.set_num_threads(2)
    scene, manifest, receipt = load_persistent_handoff(manifest_path, proof_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    road_provider = next(provider for provider in scene._providers if provider.kind == 'road')
    ns = len(scene.static.get_xyz)
    nd = len(scene.dynamic.get_xyz) if scene._include_dynamic else 0
    rows = []
    started = time.time()
    with torch.no_grad():
        for key in keys:
            dest = output / f'{key:03d}'
            dest.mkdir(exist_ok=True)
            values, owners, camera, coverage = scene._rows(key, (10,), (10, 8, 21, 20, 24), True)
            values = dict(values, mask=values['mask'] & ~scene._contains(owners, (10,)))
            pack = scene._bindings().raster(camera, scene.pipe, 0, values, scene._background)
            environment = (1 - pack['alpha']) * scene.environment(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
            rgb = pack['rgb'] + environment
            # 颜色分支标记只进入extra缓冲；所有层仍使用同一完整场景的深度排序。
            index = torch.arange(len(values['means3D']), device=values['means3D'].device)
            features = torch.zeros_like(values['features'])
            features[:, :2] = values['features'][:, :2]
            depth = values['means3D'] @ camera.world_view_transform[:3, 2] + camera.world_view_transform[3, 2]
            for j, flag in enumerate((index < ns, (index >= ns) & (index < ns + nd), index >= ns + nd)):
                features[:, 2 + 3*j:5 + 3*j] = values['colors_precomp'] * flag[:, None]
                features[:, 11+j] = flag
                features[:, 14+j] = depth * flag
            parts = scene._bindings().raster(camera, scene.pipe, 0, dict(values, features=features), scene._background)
            static_rgb, dynamic_rgb, road_rgb = [parts['extra'][3*j:3*j + 3] for j in range(3)]
            rgb_error = float((static_rgb + dynamic_rgb + road_rgb + environment - rgb).abs().max())
            road_rows = road_provider.rows(key, camera, scene.pipe)
            road_only = scene._bindings().raster(camera, scene.pipe, 0, road_rows, scene._background)
            array = lambda value: value.detach().cpu().numpy()
            data = dict(rgb=array(rgb), road_only_rgb=array(road_only['rgb']), road_only_alpha=array(road_only['alpha']).squeeze(), road_alphaT=array(pack['extra'][3]).squeeze(), static_alphaT=array(pack['extra'][20]).squeeze(), dynamic_alphaT=array(pack['extra'][21]).squeeze(), total_alpha=array(pack['alpha']).squeeze(), static_rgb=array(static_rgb), dynamic_rgb=array(dynamic_rgb), road_rgb=array(road_rgb), environment_rgb=array(environment))
            for j, name in enumerate(('static', 'dynamic', 'road')):
                data[name + '_depth_mass_model'] = array(parts['extra'][12+j])
            branch_error = float(np.max(abs(data['road_alphaT'] + data['static_alphaT'] + data['dynamic_alphaT'] - data['total_alpha'])))
            if domain_dir:
                with np.load(Path(domain_dir) / f'{key:03d}.npz', allow_pickle=False) as masks:
                    data.update(domain=masks['fit'] & ~masks['K'], target=masks['target'], protected=masks['K'])
            if baseline and 'protected' in data:
                with np.load(Path(baseline) / f'{key:03d}' / 'arrays.npz', allow_pickle=False) as old:
                    delta = np.max(abs(data['rgb'] - old['rgb']), axis=0)
                    k = data['protected']
                    protected = dict(pixels=int(k.sum()), changed_gt1byte=int((k & (delta > 1/255 + 1e-7)).sum()), max_abs=float(delta[k].max()) if k.any() else None)
            else:
                protected = None
            target_alpha = float(pack['extra'][6].max())
            finite = all(np.isfinite(value).all() for value in data.values())
            if not finite or rgb_error >= 1e-5 or branch_error >= 1e-5 or target_alpha != 0:
                raise RuntimeError(f'成像配平或删除约束失败: raw={key}, RGB={rgb_error}, alpha={branch_error}')
            np.savez_compressed(dest / 'arrays.npz', **data)
            for name, value in [('joint', data['rgb']), ('road_only', data['road_only_rgb']), ('static', data['static_rgb']), ('road_contribution', data['road_rgb'])]:
                _save_rgb(dest / f'{name}.webp', value)
            record = dict(raw=key, finite=True, rgb_conservation_error=rgb_error, alpha_conservation_error=branch_error, deleted_owner_alpha_max=target_alpha, protected=protected)
            rows.append(record)
            write(output / 'progress.json', dict(completed_frames=len(rows), total_frames=len(keys), raw=key, seconds=time.time()-started))
            if len(rows) % 25 == 0 or len(rows) == len(keys):
                print(json.dumps(dict(rendered=len(rows), raw=key, seconds=time.time()-started)), flush=True)
    scene.assert_persistent_unchanged()
    result = dict(schema='road_asset_stage_render.v1', completed=True, frames=len(rows), keys=keys, rows=rows, loader=receipt, seconds=time.time()-started, source_scene=str(manifest_path), float_arrays=True, same_depth_sort=True, final_image_paste=False, training_updates=0, GPU='keep_on')
    write(output / 'completed.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('pin')
    for name in ('source-manifest', 'source-proof', 'road', 'output'):
        build.add_argument('--' + name, required=True)
    build.add_argument('--retirement')
    draw = sub.add_parser('render')
    for name in ('manifest', 'proof', 'output'):
        draw.add_argument('--' + name, required=True)
    draw.add_argument('--frames', default='86,110,134,179,198')
    draw.add_argument('--domain-dir')
    draw.add_argument('--baseline')
    args = parser.parse_args()
    if args.command == 'pin':
        save_scene(args.source_manifest, args.source_proof, args.road, args.output, args.retirement)
    else:
        keys = list(range(199)) if args.frames == 'all' else [int(key) for key in args.frames.split(',')]
        render(args.manifest, args.proof, args.output, keys, args.domain_dir, args.baseline)


if __name__ == '__main__':
    main()
