"""独立核对投影双域候选与父高斯的三维相容性；不重算 adjoint 或修改 probe。

父高斯从同一冻结 NativeMotion 按原相机/时刻求值；距离与协方差计算用
NumPy CPU。NativeMotion 的加载/prepare 仍使用其现有 CUDA 后端，本脚本
不调用 raster、训练或 SSH。<=3 sigma 只是描述性计数，不改变候选/阈值。
观测质心是 alpha*T 加权的深度像素反投影点云质心，不是真实表面或 GT。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

from two_sided_split import covariance_from_quaternion_scale

CTX = '14705303724557273004_3105_000_3125_000'
TASK = 'WS-V43-EDITABLE-OWNERSHIP-20261030'
ANCHORS = [98, 113, 114]
SIGMA_DESCRIPTION = 3.0
TINY_TAIL_DESCRIPTION = 1e-8  # 与已运行 split 的默认 review 门限对应；不重调。


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf8')


def finite_value(value):
    value = float(value)
    return value if math.isfinite(value) else None


def distribution(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'finite_count': 0, 'quantiles': None}
    levels = [0, .1, .25, .5, .75, .9, 1]
    return dict(finite_count=len(values), mean=float(values.mean()),
                quantiles=dict(levels=levels, values=np.quantile(values, levels).tolist()))


def geometry_metrics(mean_model, quaternion, scales, model_from_world,
                     target_centroid_world, road_centroid_world):
    """父高斯世界协方差及两个质心距离；失败由调用者逐行保留为 invalid。"""
    transform = np.asarray(model_from_world, np.float64)
    inverse = np.linalg.inv(transform[:3, :3])
    mean_world = inverse @ (np.asarray(mean_model, np.float64) - transform[:3, 3])
    covariance_model = covariance_from_quaternion_scale(quaternion, scales)
    covariance_world = inverse @ covariance_model @ inverse.T
    covariance_world = .5 * (covariance_world + covariance_world.T)
    root = np.linalg.cholesky(covariance_world)
    target = np.asarray(target_centroid_world, np.float64)
    road = np.asarray(road_centroid_world, np.float64)
    require(np.isfinite(target).all() and np.isfinite(road).all(), 'Nonfinite observed centroid')
    delta_target, delta_road = target - mean_world, road - mean_world
    separation = float(np.linalg.norm(target - road))
    require(separation > 0 and math.isfinite(separation), 'Observed centroids coincide or are nonfinite')
    normal = (target - road) / separation
    midpoint = (target + road) * .5
    plane_sigma = math.sqrt(float(normal @ covariance_world @ normal))
    require(plane_sigma > 0 and math.isfinite(plane_sigma), 'Invalid projected parent sigma')
    a = float(normal @ (midpoint - mean_world)) / plane_sigma
    log_tail = float(log_ndtr(-abs(a)))
    target_mahalanobis = float(np.linalg.norm(np.linalg.solve(root, delta_target)))
    road_mahalanobis = float(np.linalg.norm(np.linalg.solve(root, delta_road)))
    return dict(parent_mean_world=mean_world, parent_covariance_world=covariance_world,
                parent_axis_sigma_m=np.sqrt(np.linalg.eigvalsh(covariance_world)),
                target_euclidean_m=float(np.linalg.norm(delta_target)),
                road_euclidean_m=float(np.linalg.norm(delta_road)),
                target_mahalanobis=target_mahalanobis, road_mahalanobis=road_mahalanobis,
                centroid_separation_m=separation, plane_normal_world=normal,
                plane_midpoint_world=midpoint, plane_sigma_m=plane_sigma,
                standardized_a=a, small_tail_log_mass=log_tail,
                small_tail_mass=float(math.exp(log_tail)),
                both_centroids_within_3sigma=target_mahalanobis <= SIGMA_DESCRIPTION and road_mahalanobis <= SIGMA_DESCRIPTION,
                tiny_tail_descriptive=log_tail < math.log(TINY_TAIL_DESCRIPTION))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-contract', type=Path, required=True)
    parser.add_argument('--code-dir', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--probe', '--probe-root', dest='probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raws', type=int, nargs='+', default=ANCHORS)
    args = parser.parse_args()
    require(len(args.raws) == len(set(args.raws)) and set(args.raws) <= set(ANCHORS), 'Only unique three-anchor raws are supported')
    require(not args.output.exists(), 'Choose fresh independent output; preserve previous attempts')
    require(not args.output.resolve().is_relative_to(args.probe.resolve()), 'Audit output must not be inside the frozen probe')
    auth = read(args.authorization)
    require(auth['task'] == TASK and auth['context'] == CTX and auth['allow_gpu_probe'], 'Task probe authorization differs')
    require(not auth['allow_classifier_training'] and not auth['resume_old_queue'], 'Training/old queue must stay disabled')
    contract = read(args.input_contract)
    require(contract['context'] == CTX and contract['branch_sizes'] == dict(static=918850, actor=6241), 'Input context/population differs')
    require(contract['static_namespace'] == contract['static_checkpoint_ref'] + '@30000:static', 'Static namespace/checkpoint differs')
    ownership_path = args.input_contract.parent / contract['source_assets']['ownership']['path']
    ownership_sha = sha(ownership_path)
    require(ownership_sha == contract['source_assets']['ownership']['sha256'], 'Frozen ownership hash differs')
    source = ownership_path.read_text(encoding='utf-8-sig')
    ownership = json.JSONDecoder().raw_decode(source[source.index('=') + 1:])[0]
    require(ownership['context'] == CTX and ownership['static_namespace'] == contract['static_namespace'] and ownership['branch_sizes'] == contract['branch_sizes'], 'Ownership source binding differs')
    actor_sha = sha(contract['actor_checkpoint_ref'])
    require(actor_sha == contract['actor_checkpoint_sha256'], 'Frozen actor checkpoint hash differs')
    probe_summary = read(args.probe / 'summary.json')
    require(probe_summary['task'] == TASK and probe_summary['context'] == CTX and probe_summary['status'] == 'complete' and probe_summary['training_updates'] == 0, 'Require completed frozen probe from this task')
    sys.path.insert(0, str(args.code_dir))
    module = importlib.import_module('train_actor')
    torch = module.torch
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    records = []

    def authorized():
        require(read(args.authorization)['allow_gpu_probe'] and not (module.ROOT / 'USER_STOP').exists(), 'Probe authorization withdrawn or user stop')

    def array(value):
        return value.detach().cpu().numpy()

    with importlib.import_module('gpu_slot').exclusive():
        authorized()
        native = module.NativeMotion(Path(contract['static_checkpoint_ref']), 30000)
        require(len(native.static._xyz) == 918850, 'Native static population differs')
        manifest = read(module.ROOT / 'datasets' / CTX / 'manifest.json')
        require(manifest['context'] == CTX and [f['raw'] for f in manifest['frames']] == list(range(77, 126)), 'Dataset frame binding differs')
        static_sha = sha(Path(contract['static_checkpoint_ref']) / 'chkpnt30000.pth')
        for raw in args.raws:
            authorized()
            index = raw - 77
            frame = manifest['frames'][index]
            expected = contract['groups']['v43_actor_10000']['frames'][str(raw)]
            require(expected['raw'] == raw and expected['camera_index'] == index, 'Contract camera index differs')
            require(str(frame['frame_id']) == str(expected['frame_id']) == str(native.records[index]['img_name']), 'Observed/native frame_id differs')
            require(np.isclose(float(frame['model_time']), float(expected['model_time']), rtol=0, atol=1e-12), 'Observed/native model_time differs')
            support_path = args.probe / f'{raw:03d}' / 'dual_support.npz'
            proposal_path = args.probe / f'{raw:03d}' / 'split_proposals.json'
            with np.load(support_path, allow_pickle=False) as data:
                support = {key: data[key].copy() for key in data.files}
            ids = support['ids'].astype(np.int64)
            require(ids.ndim == 1 and len(np.unique(ids)) == len(ids) and np.all((ids >= 0) & (ids < 918850)), 'Invalid static candidate IDs')
            count = len(ids)
            for key in ['target_centroid_world', 'road_centroid_world', 'parent_xyz_model']:
                require(support[key].shape == (count, 3) and np.isfinite(support[key]).all(), f'Invalid {key}')
            require(support['target_mass'].shape == support['road_mass'].shape == (count,), 'Mass shape differs')
            ranking_mass = np.minimum(support['target_mass'], support['road_mass'])
            require(np.isfinite(ranking_mass).all() and np.all(np.diff(ranking_mass) <= 0), 'Frozen candidate rank differs')
            proposals = read(proposal_path)
            outcome = {int(item['row']): dict(kind='retained', result=item['result']) for item in proposals['proposals']}
            outcome.update({int(item['row']): dict(kind='skipped', reason=item['reason'], details=item.get('details')) for item in proposals['skipped']})
            camera = native.camera(dict(camera_index=index, frame_id=native.records[index]['img_name'], camera_model_time=frame['model_time']), {})
            with torch.no_grad():
                prepared = module.prepare_rasterization(camera, native.static, native.args)
                rows = torch.as_tensor(ids, device='cuda', dtype=torch.long)
                means = array(prepared['means3D'][rows]).astype(np.float64)
                rotations = array(prepared['rotations'][rows]).astype(np.float64)
                scales = array(prepared['scales'][rows]).astype(np.float64)
            center_error = float(np.max(np.abs(means - support['parent_xyz_model']))) if count else 0.
            require(center_error <= 1e-6, 'Current static means differ from frozen probe parent rows')
            scalar_keys = ['target_euclidean_m', 'road_euclidean_m', 'target_mahalanobis', 'road_mahalanobis', 'centroid_separation_m', 'plane_sigma_m', 'standardized_a', 'small_tail_log_mass', 'small_tail_mass']
            arrays = {key: np.full(count, np.nan) for key in scalar_keys}
            arrays.update(parent_mean_world=np.full((count, 3), np.nan), parent_covariance_world=np.full((count, 3, 3), np.nan), parent_axis_sigma_m=np.full((count, 3), np.nan), plane_normal_world=np.full((count, 3), np.nan), plane_midpoint_world=np.full((count, 3), np.nan), valid_geometry=np.zeros(count, bool), both_centroids_within_3sigma=np.zeros(count, bool), tiny_tail_descriptive=np.zeros(count, bool))
            errors = [''] * count
            top8 = []
            for position, row in enumerate(ids):
                try:
                    metric = geometry_metrics(means[position], rotations[position], scales[position], native.T, support['target_centroid_world'][position], support['road_centroid_world'][position])
                    for key, value in metric.items():
                        arrays[key][position] = value
                    arrays['valid_geometry'][position] = True
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                    errors[position] = type(error).__name__ + ': ' + str(error)
                if position < 8:
                    previous = outcome.get(int(row), {'kind': 'not_evaluated_by_probe'})
                    saved_a = previous.get('result', {}).get('standardized_offset')
                    top8.append(dict(rank=position + 1, row_id=int(row), target_mass=float(support['target_mass'][position]), road_mass=float(support['road_mass'][position]), valid_geometry=bool(arrays['valid_geometry'][position]), error=errors[position] or None,
                        **{key: finite_value(arrays[key][position]) for key in scalar_keys}, both_centroids_within_3sigma=bool(arrays['both_centroids_within_3sigma'][position]), tiny_tail_descriptive=bool(arrays['tiny_tail_descriptive'][position]), probe_outcome=previous['kind'], probe_skip_reason=previous.get('reason'), probe_skip_details=previous.get('details'), probe_standardized_a=saved_a, standardized_a_difference=None if saved_a is None or not arrays['valid_geometry'][position] else float(arrays['standardized_a'][position] - saved_a)))
            np.savez_compressed(args.output / f'{raw:03d}_geometry.npz', ids=ids, candidate_rank=np.arange(1, count + 1), target_mass=support['target_mass'], road_mass=support['road_mass'], target_centroid_world=support['target_centroid_world'], road_centroid_world=support['road_centroid_world'], parent_xyz_model=means, geometry_error=np.asarray(errors, dtype=str), **arrays)
            valid = arrays['valid_geometry']
            record = dict(raw=raw, frame_id=str(frame['frame_id']), camera_index=index, model_time=frame['model_time'], row_namespace=contract['static_namespace'], source_support_file=str(support_path), source_support_sha256=sha(support_path), source_split_proposals_sha256=sha(proposal_path), native_parent_mean_max_abs_delta_model=center_error, candidates=count, valid_geometry_rows=int(valid.sum()), invalid_geometry_rows=int((~valid).sum()), target_centroid_within_3sigma_rows=int((valid & (arrays['target_mahalanobis'] <= SIGMA_DESCRIPTION)).sum()), road_centroid_within_3sigma_rows=int((valid & (arrays['road_mahalanobis'] <= SIGMA_DESCRIPTION)).sum()), both_centroids_within_3sigma_rows=int(arrays['both_centroids_within_3sigma'].sum()), tiny_tail_descriptive_rows=int(arrays['tiny_tail_descriptive'].sum()), all_candidate_statistics={key: distribution(arrays[key]) for key in scalar_keys}, top8=top8)
            records.append(record)
            write(args.output / f'{raw:03d}_summary.json', record)
            print(json.dumps(dict(raw=raw, candidates=count, both_centroids_within_3sigma_rows=record['both_centroids_within_3sigma_rows'], tiny_tail_top8=sum(item['tiny_tail_descriptive'] for item in top8))), flush=True)
            del prepared, rows, means, rotations, scales
        require(native.unchanged(), 'Frozen native checkpoint or tensors changed')
    summary = dict(schema='v43-support-geometry-diagnostic-v1', task=TASK, context=CTX, status='complete', records=records,
        input_contract_sha256=sha(args.input_contract), ownership_sha256=ownership_sha, actor_checkpoint_sha256=actor_sha, static_checkpoint_ref=contract['static_checkpoint_ref'], static_checkpoint_sha256=static_sha, source_probe_summary_sha256=sha(args.probe / 'summary.json'), source_checkpoint_unchanged=True,
        thresholds=dict(descriptive_mahalanobis_sigma=SIGMA_DESCRIPTION, descriptive_tiny_tail_mass=TINY_TAIL_DESCRIPTION, thresholds_adjusted=False),
        definitions=dict(mahalanobis='sqrt((observed_centroid-parent_mean)^T parent_covariance^-1 (observed_centroid-parent_mean))', standardized_a='n=(target-road)/norm(target-road); midpoint=(target+road)/2; a=n dot (midpoint-parent_mean) / sqrt(n^T parent_covariance n)', covariance_world='inverse(T[:3,:3]) @ covariance_model @ inverse(T[:3,:3]).T', ranking='frozen dual_support.npz order, descending min(target_mass, road_mass); top8 not reranked', support_context='intact static30K+Actor10K alpha*T weighted observed-depth pixel backprojections, inherited from probe'),
        limitations=['Point-cloud centroids are not true surfaces, surface GT, or Gaussian ownership labels.', 'Projected dual-domain support does not establish spatial mixed support or centroid coverage by the parent.', '<=3 sigma is a descriptive ellipsoid/centroid counter only, not a tuned acceptance rule.', 'Tiny tail flags describe the existing split initialization; no thresholds, candidates or split results are changed.', 'No native raster/adjoint is recomputed; native loader/prepare uses its existing CUDA-backed implementation, geometry calculations are NumPy CPU.'],
        execution=dict(native_raster_calls=0, adjoint_calls=0, training_updates=0, source_probe_modified=False), semantic_ground_truth=False, quality_accepted=False, human_verdict=None, wall_seconds=time.monotonic() - started)
    write(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
