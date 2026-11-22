"""只读 CPU 颜色链审计。标准输出 JSON；不导入 Torch/native，不写远端文件。

执行示例（远端输入路径）：
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python - < audit_color_chain_cpu.py
原生测量仅重读已经存在的 core_float.npz；相机计算明确标为 CPU 重建。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from pathlib import Path
import argparse
import io
import json
import sys
import time
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def read(path):
    return json.loads(Path(path).read_text())


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if keys is None else keys)}


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
        a, b, equal_nan=True) if a.dtype.kind in 'fc' else (
        a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))


def stats(a):
    a = np.asarray(a, np.float64).ravel()
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), min=float(a.min()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), max=float(a.max()), mean=float(a.mean()))


def delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    valid = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    d = a[valid].astype(np.float64) - b[valid].astype(np.float64)
    return dict(n=int(valid.sum()), equal_rows=int((d == 0).all(-1).sum()),
                MAE=float(abs(d).mean()) if d.size else None,
                max_abs=float(abs(d).max()) if d.size else None,
                mean_signed=d.mean(0).tolist() if d.size else None)


def counts(a):
    u, n = np.unique(a, return_counts=True)
    return [{'value': v, 'count': c} for v, c in zip(u.tolist(), n.tolist())]


def project(x, k, w):
    q = x @ w[:3, :3].T + w[:3, 3]
    h = q @ k.T
    return h[:, :2] / h[:, 2:] - .5, q


def brown(q, intrinsic):
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = np.asarray(intrinsic)
    x, y = (q[:, :2] / q[:, 2:]).T
    rr = x*x + y*y
    radial = 1 + k1*rr + k2*rr*rr + k3*rr*rr*rr
    xd = x*radial + 2*p1*x*y + p2*(rr+2*x*x)
    yd = y*radial + p1*(rr+2*y*y) + 2*p2*x*y
    return np.c_[fx*xd+cx-.5, fy*yd+cy-.5]


def sample(photo, uv):
    h, w = photo.shape[:2]
    valid = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] <= h-1)
    rgb = np.stack([map_coordinates(photo[:, :, c], [uv[:, 1], uv[:, 0]],
                                   order=1, mode='nearest', prefilter=False) for c in range(3)], 1)
    return rgb, valid


def linear(rgb):
    return np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path,
                        default=Path('/root/autodl-tmp/drivingeditor_repro/campaign'))
    parser.add_argument('--bank', type=Path,
                        default=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs'))
    parser.add_argument('--frames', nargs='+', type=int, default=[86, 98, 100, 108, 110, 116, 128, 137, 173])
    args = parser.parse_args()
    started = time.monotonic()
    campaign, bank = args.campaign, args.bank
    asset = campaign/'road_surface_residual_20261121_r2/appearance_fit/asset'
    quality = campaign/'road_gaussian_source_quality_20261122_r1'
    manifest = read(campaign/'road_gaussian_transfer_20261122_r2_isometric/source_selection/manifest.json')
    road = load(asset/'road.npz')
    provenance = load(asset/'provenance.npz', [
        'world_xyz', 'row_rgb', 'observed_reference_srgb', 'effective_row_rgb', 'base_row_rgb',
        'capacity_source_prior_reference_srgb', 'capacity_effective_rgb', 'source', 'known', 'unknown',
        'source_raw', 'source_uv_native', 'effective_material_source', 'effective_material_source_raw',
        'effective_material_source_uv_native', 'preserved_legacy_rgb'])
    active = np.flatnonzero(road['active'])
    scene = read(asset/'persistent/scene.json')
    frame_names = {int(e['key']): e['image_name'] for e in scene['frames']}
    cameras = {e['img_name']: e for e in read(Path(scene['model'])/'cameras.json')}
    bank_names = {int(e['raw']): e['id'] for e in read(bank/'receipt.json')['frames']}
    result = dict(schema='source_color_chain_readonly_cpu.v1', asset=str(asset),
                  CPU_only=True, GPU_calls=0, optimizer_updates=0, new_native_renders=0,
                  remote_asset_writes=0, frames=args.frames, sources=[], views=[],
                  archived_native_outputs=[], root_chain_probe_CPU_reread=[], refit_export={},
                  interpretation='数组/投影审计。CPU 相机重建不等于新原生渲染；行颜色与像素颜色的差异不能直接归因为着色错误。')
    packets = {}
    for entry in manifest['candidates']:
        patch = entry['candidate_id']
        if patch not in ('patch_00', 'patch_02'):
            continue
        packet = load(entry['packet'])
        packets[patch] = packet
        ids = packet['world_row_ids'][packet['core_mask']]
        pos = np.searchsorted(active, ids)
        colors = road['reference_srgb'][ids]
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **packet)
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as z:
            roundtrip = {k: exact(packet[k], z[k]) for k in packet}
        row = dict(patch=patch, packet=str(entry['packet']), core_rows=len(ids),
                   all_rows=len(packet['world_row_ids']),
                   active_order_exact=bool(np.array_equal(active[pos], ids)),
                   packet_RGB_parent_exact=exact(packet['reference_srgb'], road['reference_srgb'][packet['world_row_ids']]),
                   packet_world_parent_exact=exact(packet['world_xyz'], provenance['world_xyz'][packet['world_row_ids']]),
                   packet_sigma_parent_exact=exact(packet['gaussian_sigma_m'], road['gaussian_sigma_m'][packet['world_row_ids']]),
                   RGB_dtype=str(colors.dtype), RGB_mean=colors.mean(0).tolist(), RGB_std=colors.std(0).tolist(),
                   core_sigma_m_unique=np.unique(packet['gaussian_sigma_m'][packet['core_mask']],axis=0).tolist(),
                   nearest_core_center_distance_m=stats(cKDTree(packet['world_xyz'][packet['core_mask']]).query(
                       packet['world_xyz'][packet['core_mask']],k=2)[0][:,1]),
                   npz_roundtrip_all_fields_exact=all(roundtrip.values()), npz_roundtrip_fields=roundtrip,
                   original_source_ledger={}, RGB_ledger_differences={}, history=[])
        for key in ('source', 'known', 'unknown', 'source_raw', 'effective_material_source',
                    'effective_material_source_raw', 'preserved_legacy_rgb'):
            row['original_source_ledger'][key] = counts(provenance[key][ids])
        for key in ('source_uv_native', 'effective_material_source_uv_native'):
            row['original_source_ledger'][key+'_finite_rows'] = int(np.isfinite(provenance[key][ids]).all(1).sum())
        for key in ('row_rgb', 'observed_reference_srgb', 'effective_row_rgb', 'base_row_rgb',
                    'capacity_source_prior_reference_srgb', 'capacity_effective_rgb'):
            row['RGB_ledger_differences'][key] = delta(colors, provenance[key][ids])
        stages = [
            'road_asset_20261116_r1/phase2/formal_r1/asset',
            'road_common1152_20261121/formal/r3/G_NUMERIC/asset',
            'road_surface_adaptive_20261121_r1/coverage_fit/asset',
            'road_surface_adaptive_20261121_r1/appearance_fit/asset',
            'road_surface_residual_20261121_r2/coverage_fit/asset',
            'road_surface_residual_20261121_r2/appearance_fit/asset']
        previous = None
        for stage in stages:
            path = campaign/stage/'road.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as z:
                old = z['reference_srgb'][ids]
            row['history'].append(dict(asset=str(path.parent), current_minus_stage=delta(colors, old),
                                       stage_minus_previous=delta(old, previous) if previous is not None else None))
            previous = old
        result['sources'].append(row)

    for raw in args.frames:
        observation = read(bank/f'observations/{raw:03d}.json')
        with Image.open(bank/observation['image_path']) as im:
            photo = np.asarray(im.convert('RGB'), np.float32)/255
            info = dict(mode=im.mode, info_keys=list(im.info), has_ICC='icc_profile' in im.info,
                        gamma=im.info.get('gamma'), has_sRGB='srgb' in im.info)
        cal = {line.split(':')[0]: np.fromstring(line.split(':')[1], sep=' ')
               for line in (bank/'calib'/(bank_names[raw]+'.txt')).read_text().splitlines()}
        k = cal['P0'].reshape(3, 4)[:, :3]
        cv_from_ego = np.eye(4)
        cv_from_ego[:3] = cal['Tr_velo_to_cam_0'].reshape(3, 4)
        w = cv_from_ego @ np.linalg.inv(np.asarray(observation['ego_world']).reshape(4, 4))
        image_w = cv_from_ego @ np.linalg.inv(np.asarray(observation['camera_pose_world']).reshape(4, 4))
        saved = cameras[frame_names[raw]]
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = np.asarray(saved['rotation']), saved['position']
        native_w = np.linalg.inv(c2w)
        height, width = photo.shape[:2]
        native_k = np.array([[saved['fx']*width/saved['width'], 0, saved['cx']*width/saved['width']],
                             [0, saved['fy']*height/saved['height'], saved['cy']*height/saved['height']], [0, 0, 1]])
        qualified_path = campaign/'road_multisource_20261122_r1/sources/qualified_observations'/f'{raw:03d}.npz'
        qualified = load(qualified_path, ['query_id', 'uv_native', 'raw_linear']) if qualified_path.exists() else None
        for patch, packet in packets.items():
            ids = packet['world_row_ids'][packet['core_mask']]
            world = packet['world_xyz'][packet['core_mask']]
            color = road['reference_srgb'][ids]
            uv, q = project(world, k, w)
            native_uv, native_q = project(road['model_xyz'][ids], native_k, native_w)
            # CPU float32 仿真；不是 CUDA 矩阵执行结果。
            native32, _ = project(road['model_xyz'][ids].astype(np.float32),
                                  native_k.astype(np.float32), native_w.astype(np.float32))
            ndc = np.c_[2*native_k[0, 0]/width*native_q[:, 0]/native_q[:, 2]+(2*native_k[0, 2]-width)/width,
                         2*native_k[1, 1]/height*native_q[:, 1]/native_q[:, 2]+(2*native_k[1, 2]-height)/height]
            ndc_pixels = ((ndc+1)*np.array([width, height])-1)*.5
            distorted = brown(q, observation['camera_intrinsic_Brown'])
            image_pose_uv, image_q = project(world, k, image_w)
            distorted_image_pose = brown(image_q, observation['camera_intrinsic_Brown'])
            rgb, legal = sample(photo, uv)
            rgb_native, legal_native = sample(photo, native_uv)
            rgb_nohalf, legal_nohalf = sample(photo, uv+.5)
            rgb_brown, legal_brown = sample(photo, distorted)
            v = dict(patch=patch, raw=raw, photo=str(bank/observation['image_path']), PIL=info,
                     positive_depth_rows=int((q[:, 2] > 0).sum()), in_frame_rows=int(legal.sum()),
                     camera_K_max_abs=float(abs(k-native_k).max()),
                     CPU_native64_vs_calib_pixel_L2=stats(np.linalg.norm(native_uv-uv, axis=1)),
                     CPU_native32_vs_calib_pixel_L2=stats(np.linalg.norm(native32-uv, axis=1)),
                     CPU_CenterShift_ndc_formula_max_abs=float(abs(ndc_pixels-native_uv).max()),
                     Brown_vs_pinhole_pixel_L2=stats(np.linalg.norm(distorted-uv, axis=1)),
                     image_pose_vs_ego_pose_pixel_L2=stats(np.linalg.norm(image_pose_uv-uv, axis=1)),
                     Brown_image_pose_vs_ego_pinhole_pixel_L2=stats(np.linalg.norm(distorted_image_pose-uv, axis=1)),
                     point_RGB_vs_pinhole_photo=delta(color[legal], rgb[legal]),
                     point_RGB_vs_photo_note='仅逐行投影取样差异。原生输出是多高斯按深度和 alpha 混合，不能由此直接判断输出偏色。',
                     pinhole_vs_native_sample=delta(rgb[legal & legal_native], rgb_native[legal & legal_native]),
                     half_pixel_sampling_change=delta(rgb[legal & legal_nohalf], rgb_nohalf[legal & legal_nohalf]),
                     Brown_sampling_change=delta(rgb[legal & legal_brown], rgb_brown[legal & legal_brown]))
            witness_take = packet['witness_source_raw'] == raw
            witness_query = packet['witness_query_id'][witness_take]
            witness_uv = packet['witness_source_uv_native'][witness_take]
            witness = dict(rows=len(witness_query), stored_UV_is_per_query_not_per_Gaussian=True)
            if len(witness_query):
                fresh_pos = np.searchsorted(packet['fresh_query_ids'], witness_query)
                assert np.array_equal(packet['fresh_query_ids'][fresh_pos], witness_query)
                expected_uv, _ = project(packet['fresh_world_xyz'][fresh_pos], k, w)
                witness['pinhole_projection_vs_stored_UV_L2'] = stats(np.linalg.norm(expected_uv-witness_uv, axis=1))
                if qualified is not None:
                    qual_pos = np.searchsorted(qualified['query_id'], witness_query)
                    assert np.array_equal(qualified['query_id'][qual_pos], witness_query)
                    witness['qualified_UV_exact'] = bool(np.array_equal(qualified['uv_native'][qual_pos], witness_uv))
                    sample_at_qual, _ = sample(photo, qualified['uv_native'][qual_pos])
                    witness['bilinear_sRGB_then_linear_vs_stored_raw_linear'] = delta(
                        linear(sample_at_qual), qualified['raw_linear'][qual_pos])
            v['retrieval_witness'] = witness
            result['views'].append(v)

    # 只读历史原生浮点输出；本脚本不把 CPU 重建冒充原生实测。
    for patch in packets:
        for raw in (86, 110):
            folder = quality/'native1920'/patch/f'{raw:03d}'
            path = folder/'core_float.npz'
            if not path.exists():
                continue
            old = load(path)
            mask = old['score_mask']
            photo, rgb, alpha = old['observation'], old['isolated_rgb'], old['alpha']
            rgba = np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)
            actual_png = np.asarray(Image.open(folder/'isolated.png').convert('RGB'))
            result['archived_native_outputs'].append(dict(
                patch=patch, raw=raw, path=str(path), evidence_kind='existing_native_float_output_CPU_reread',
                score_pixels=int(mask.sum()), alpha_on_score=stats(alpha[mask]),
                effective_RGB_vs_photo=delta(rgb[mask], photo[mask]),
                alpha_normalization_RGB_change=delta(rgb[mask], (rgb*alpha[:, :, None])[mask]),
                PNG_equals_float_clamp_round=bool(np.array_equal(rgba, actual_png))))
    current_probe=campaign/'road_source_fidelity_20261122_r2/chain_probe_initial'
    for patch,source_packet in packets.items():
        for raw in (86,110):
            folder=current_probe/patch/f'{raw:03d}'
            if not (folder/'chain_float.npz').exists():continue
            f=load(folder/'chain_float.npz');saved=load(folder/'source_direct_rgb.npz')
            core=f['source_core_rows'];mask=f['score_mask'];color=np.asarray([.18,.32,.70],np.float32)
            result['root_chain_probe_CPU_reread'].append(dict(patch=patch,raw=raw,
                evidence_kind='root new native output, independently CPU reread',
                path=str(folder/'chain_float.npz'),
                selected_row_IDs_exact=exact(f['world_row_ids'],source_packet['world_row_ids']),
                original_row_RGB_exact=exact(f['original_row_rgb'],source_packet['reference_srgb']),
                sample_to_saved_core_RGB_exact=exact(f['sampled_rgb'][core],saved['reference_srgb'][core]),
                all_noncolor_fields_exact=all(exact(source_packet[k],saved[k]) for k in source_packet if k!='reference_srgb'),
                halo_RGB_exact=exact(saved['reference_srgb'][~core],source_packet['reference_srgb'][~core]),
                direct_vs_reloaded_render_max_abs=float(abs(f['direct_rgb']-f['reloaded_rgb']).max()),
                constant_effective_RGB_max_abs=float(abs(f['constant_rgb'][mask]-color).max())))
    refit = quality/'source_rgb_refit_qualified_r2/source_patch_00_refit.npz'
    if refit.exists():
        old, new = packets['patch_00'], load(refit)
        result['refit_export'] = dict(path=str(refit),
            noncolor_fields_exact={k: exact(old[k], new[k]) for k in old if k != 'reference_srgb'},
            halo_RGB_exact=exact(old['reference_srgb'][~old['core_mask']], new['reference_srgb'][~old['core_mask']]),
            core_RGB_change=delta(new['reference_srgb'][old['core_mask']], old['reference_srgb'][old['core_mask']]),
            actual_reloaded_native_render_measured_here=False,
            code_boundary='refit_source_rgb.evaluate 使用内存 param；文件保存发生在 final=evaluate 之后，旧 receipt 本身不证明 NPZ/reload 后的原生输出。')
    result.update(wall_seconds=time.monotonic()-started,
                  torch_imported='torch' in sys.modules, native_imported=any(k.startswith('diff_gaussian') for k in sys.modules))
    # ASCII JSON 避免 Windows shell/SSH 管线对中文字符串二次编码。
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
