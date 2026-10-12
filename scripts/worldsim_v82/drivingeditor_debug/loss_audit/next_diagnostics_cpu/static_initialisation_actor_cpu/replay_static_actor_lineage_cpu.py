"""Reconstruct scene0017 initialization lineage from the unchanged original inputs.

No CUDA/model instantiation, training, data writes or checkpoint copies. A GT
oriented box is independent audit evidence; box membership is never called a
per-return semantic label. Only already-selected actor return identity is exact.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
ROOT = BASE / 'campaign/vgpu_bugfix_20261012'
DATA = BASE / 'data/waymo_paper_scenes/scene0017_000_048'
OUT = ROOT / 'loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu'
ORIGINAL = BASE / 'campaign/resume_vgpu_baseline_20261011/frozen_repository'
S = ROOT / 'loss_audit/next_diagnostics_cpu/observed_static_rgb_S'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_ply(p):
    data = PlyData.read(str(p))['vertex'].data
    xyz = np.column_stack([data[k] for k in ('x', 'y', 'z')]).astype(np.float32)
    time = np.asarray(data['nx'], dtype=np.float32)
    require(all(np.array_equal(time, data[k]) for k in ('ny', 'nz', 'time')), 'PLY time fields disagree')
    return xyz, time


def box_coordinates(xyz, label):
    delta = xyz - np.asarray(label['center'])
    c, s = np.cos(label['heading']), np.sin(label['heading'])
    local = np.column_stack((delta[:, 0] * c + delta[:, 1] * s, -delta[:, 0] * s + delta[:, 1] * c, delta[:, 2]))
    half = np.asarray(label['size_lwh']) * 0.5
    return local, np.all(np.abs(local) <= half, axis=1)


def actor_selection(raw_xyz, row, metadata, selection):
    name = row['id']
    ids = selection['per_frame_instance_ids'].get(name, selection['default_instance_ids'])
    mask = np.isin(np.asarray(Image.open(DATA / 'mask/mask_0/Annotations' / (name + '.png'))), ids)
    require(np.array_equal(mask, np.asarray(Image.open(DATA / 'dynamic_masks/dynamic_mask_0' / (name + '.png'))) > 0), 'Stored actor-body mask differs')
    calib = {line.split(':', 1)[0]: np.fromstring(line.split(':', 1)[1], sep=' ') for line in (DATA / 'calib' / (name + '.txt')).read_text().splitlines() if ':' in line}
    xyz = raw_xyz.astype(np.float64)
    homo = np.column_stack((xyz, np.ones(len(xyz))))
    camera = homo @ calib['Tr_velo_to_cam_0'].reshape(3, 4).T
    positive = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.01)
    positive_ids = np.flatnonzero(positive)
    uvw = np.column_stack((camera[positive], np.ones(positive.sum()))) @ calib['P0'].reshape(3, 4).T
    uv = np.floor(uvw[:, :2] / uvw[:, 2:3] + 0.5).astype(np.int64)
    h, w = mask.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    indices = positive_ids[inside]; uv = uv[inside]
    z = camera[indices, 2]
    pixel_bin = metadata['occlusion_filter']['pixel_bin_size']
    bw = (w + pixel_bin - 1) // pixel_bin; bh = (h + pixel_bin - 1) // pixel_bin
    bins = uv[:, 1] // pixel_bin * bw + uv[:, 0] // pixel_bin
    nearest = np.full(bw * bh, np.inf)
    np.minimum.at(nearest, bins, z)
    target = mask[uv[:, 1], uv[:, 0]]
    visible = z <= nearest[bins] + metadata['occlusion_filter']['depth_tolerance_m']
    keep = target & visible
    chosen = indices[keep]
    if len(chosen):
        pairs = cKDTree(xyz[chosen]).query_pairs(metadata['component_filter']['radius_m'], output_type='ndarray')
        graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(len(chosen), len(chosen)))
        _, labels = connected_components(graph, directed=False)
        counts = np.bincount(labels)
        keep[np.flatnonzero(keep)[labels != counts.argmax()]] = False
    chosen = indices[keep]
    full_front = np.zeros(len(xyz), dtype=bool); full_front[indices] = True
    full_mask = np.zeros(len(xyz), dtype=bool); full_mask[indices[target]] = True
    full_visible = np.zeros(len(xyz), dtype=bool); full_visible[indices[target & visible]] = True
    full_uv = np.full((len(xyz), 2), -1, dtype=np.int32); full_uv[indices] = uv
    return chosen, full_front, full_mask, full_visible, full_uv


def main():
    prep = json.loads((DATA / 'preparation_manifest.json').read_text())
    actor = json.loads((DATA / 'dynamic_actor_preparation.json').read_text())
    selected_actor = json.loads((DATA / 'selected_actor_0.json').read_text())
    raw_proof = json.loads((OUT / 'raw49_labels_layout.json').read_text())
    prior = json.loads((ROOT / 'sgcm_data_audit/initial_geometry_and_time.json').read_text())
    binding = json.loads((S / 'materialized_plan/experiment_binding.json').read_text())
    expected_input = json.loads((ROOT / 'sgcm_data_audit/outside_opacity_review/common49_input_manifest_review_cpu.json').read_text())['file_sha256']
    relevant = ['preparation_manifest.json', 'dynamic_actor_preparation.json', 'selected_actor_0.json', 'points3d.ply', 'points3d_dynamic_0.ply', 'transform_matrix.txt']
    relevant += [f'{folder}/{r["id"]}{suffix}' for r in prep['frames'] for folder, suffix in [('velodyne', '.bin'), ('pose', '.txt'), ('calib', '.txt'), ('mask/mask_0/Annotations', '.png'), ('dynamic_masks/dynamic_mask_0', '.png')]]
    pins = {str(DATA / relative): sha(DATA / relative) for relative in relevant}
    for relative in relevant:
        require(pins[str(DATA / relative)] == expected_input[relative], 'Approved original input pin changed: ' + relative)
    source_paths = [ORIGINAL / 'scene/waymo_loader2.py', ORIGINAL / 'scene/gaussian_model.py', ORIGINAL / 'scene/__init__.py', BASE / 'scripts/prepare_waymo_pilot.py', BASE / 'scripts/build_paper_actor.py', S / 'frozen_repository/train.py', S / 'static_anchor_guard.py', S / 'materialized_plan/experiment_binding.json', S / 'materialized_plan/static_anchor_prepared_plan.json', S / 'materialized_plan/continuation_static_anchor_harness.py', S / 'materialized_plan/static_anchor_effective_config.yaml']
    pins.update({str(p): sha(p) for p in source_paths})
    require(sha(BASE / 'scripts/build_paper_actor.py') == actor['adapter_sha256'], 'Actual actor preparation source version changed')
    require(raw_proof['status'] == 'PASS_EXISTING_RAW49_RETURN_LAYOUT_AND_LABELS_CPU', 'Original raw row proof missing')
    car_type = raw_proof['segmentation_Type_enum']['TYPE_CAR']
    ply, ply_time = read_ply(DATA / 'points3d.ply')
    dynamic, dynamic_time = read_ply(DATA / 'points3d_dynamic_0.ply')
    require(len(ply) == prep['num_points'] == prior['static_PLY_points'], 'Initial sampled input size differs')
    require(sorted(p.name for p in DATA.glob('points3d_dynamic_*.ply')) == ['points3d_dynamic_0.ply'], 'Unbound extra actor cloud')
    distances, _ = cKDTree(dynamic.astype(np.float64)).query(ply.astype(np.float64), k=1, distance_upper_bound=0.1, workers=1)
    static_keep = distances >= 0.1
    require(int(static_keep.sum()) == prior['static_initialized'], 'Exact loader static exclusion differs')
    counts = np.asarray([r['lidar_points'] for r in prep['frames']], dtype=np.int64)
    offsets = np.r_[0, counts.cumsum()]
    require(offsets[-1] == prep['total_lidar_points_before_sampling'], 'Original raw concatenation total differs')
    drawn = np.random.RandomState(prep['seed']).choice(offsets[-1], prep['num_points'], replace=True)
    frame_index = np.searchsorted(offsets[1:], drawn, side='right').astype(np.uint8)
    bin_index = (drawn - offsets[frame_index]).astype(np.uint32)
    transform = np.loadtxt(DATA / 'transform_matrix.txt')
    lineage = {'sample_index': np.arange(len(ply), dtype=np.uint32), 'global_raw_row': drawn, 'frame_index': frame_index, 'converted_row_index': bin_index, 'static_keep': static_keep,
               'static_initialized_index': np.full(len(ply), -1, dtype=np.int32), 'world_xyz': ply,
               'return_index': np.zeros(len(ply), dtype=np.uint8), 'laser_name': np.zeros(len(ply), dtype=np.uint8), 'range_row': np.zeros(len(ply), dtype=np.uint16), 'range_col': np.zeros(len(ply), dtype=np.uint16),
               'selected_actor_origin': np.zeros(len(ply), dtype=bool), 'annotated_target_obb': np.zeros(len(ply), dtype=bool), 'front_in_frustum': np.zeros(len(ply), dtype=bool), 'front_actor_mask': np.zeros(len(ply), dtype=bool), 'front_visible_actor_mask': np.zeros(len(ply), dtype=bool), 'model_time': ply_time,
               'instance_label': np.full(len(ply), -1, dtype=np.int32), 'semantic_label': np.full(len(ply), -1, dtype=np.int32), 'linked_target_instance_label': np.full(len(ply), -1, dtype=np.int32), 'semantic_linked_target': np.zeros(len(ply), dtype=bool)}
    lineage['static_initialized_index'][static_keep] = np.arange(static_keep.sum(), dtype=np.int32)
    target_id, frames, witness = None, [], []
    for fi, row in enumerate(prep['frames']):
        name = row['id']; proof = raw_proof['frames'][fi]
        require(proof['frame'] == name and sha(DATA / 'velodyne' / (name + '.bin')) == proof['bin_SHA256'], 'Raw source row pin differs')
        raw = np.fromfile(DATA / 'velodyne' / (name + '.bin'), dtype=np.float32).reshape(-1, 6)
        pose = np.loadtxt(DATA / 'pose' / (name + '.txt'))
        world = (np.pad(raw[:, :3], ((0, 0), (0, 1)), constant_values=1) @ pose.T)[:, :3].astype(np.float32)
        idx = np.flatnonzero(frame_index == fi); bi = bin_index[idx]
        require(np.array_equal(world[bi], ply[idx]) and np.array_equal(np.full(len(idx), row['model_time'], dtype=np.float32), ply_time[idx]), 'Seed0 draw/pose/float32/time replay not byte-exact')
        chosen, front, body_mask, visible_body, uv = actor_selection(raw[:, :3], row, actor, selected_actor)
        actor_time = dynamic_time == np.float32(row['model_time'])
        require(np.array_equal(world[chosen], dynamic[actor_time]) and len(chosen) == actor['frames'][fi]['retained_visible_lidar_points'], 'Actor-return replay differs from actual PLY bytes/order')
        selected = np.zeros(len(raw), dtype=bool); selected[chosen] = True
        candidates = []
        for label in proof['labels']:
            if label['type'] == 1:
                _, inside = box_coordinates(raw[:, :3].astype(np.float64), label)
                candidates.append({'id': label['id'], 'selected_actor_return_overlap': int(inside[chosen].sum()), 'all_raw_rows_inside': int(inside.sum()), 'annotation_num_points': label['num_lidar_points_in_box']})
        candidates.sort(key=lambda x: (-x['selected_actor_return_overlap'], x['id']))
        require(len(candidates) > 1 and candidates[0]['selected_actor_return_overlap'] > candidates[1]['selected_actor_return_overlap'], 'Target identity not uniquely associated by independent object labels')
        if target_id is None:
            target_id = candidates[0]['id']
        require(candidates[0]['id'] == target_id, 'Waymo object ID association changes; preserve unresolved evidence rather than assign a target')
        label = next(l for l in proof['labels'] if l['id'] == target_id)
        local, within = box_coordinates(raw[:, :3].astype(np.float64), label)
        require(sha(proof['layout_path']) == proof['layout_SHA256'], 'Raw layout pin differs')
        with np.load(proof['layout_path']) as layout:
            for key in ('return_index', 'laser_name', 'range_row', 'range_col', 'instance_label', 'semantic_label'):
                lineage[key][idx] = layout[key][bi]
            semantic = layout['semantic_label'].copy(); instance = layout['instance_label'].copy()
        # Link frame-local native point-instance IDs through the independently
        # tracked target OBB and actual selected actor returns. No mask/box test
        # is then used to select all returns of the linked native instance.
        seed = chosen[(semantic[chosen] == car_type) & (instance[chosen] > 0) & within[chosen]]
        seed_ids, seed_counts = np.unique(instance[seed], return_counts=True)
        order = np.argsort(-seed_counts, kind='stable')
        seed_ids, seed_counts = seed_ids[order], seed_counts[order]
        link = int(seed_ids[0]) if len(seed_ids) and (len(seed_ids) == 1 or seed_counts[0] > seed_counts[1]) else -1
        semantic_target = (semantic == car_type) & (instance == link) if link > 0 else np.zeros(len(raw), dtype=bool)
        lineage['linked_target_instance_label'][idx] = link
        lineage['semantic_linked_target'][idx] = semantic_target[bi]
        for key, values in [('selected_actor_origin', selected), ('annotated_target_obb', within), ('front_in_frustum', front), ('front_actor_mask', body_mask), ('front_visible_actor_mask', visible_body)]:
            lineage[key][idx] = values[bi]
        retained = static_keep[idx]
        associated = retained & within[bi]
        frame_record = {'frame': name, 'source_index': row['source_index'], 'timestamp_micros': row['timestamp_micros'], 'model_time': row['model_time'], 'sampled_PLY_rows': len(idx), 'static_retained_rows': int(retained.sum()), 'static_unique_raw_returns': int(len(np.unique(bi[retained]))),
                        'selected_actor_raw_returns': len(chosen), 'selected_actor_sampled_rows': int(selected[bi].sum()), 'selected_actor_rows_remaining_static': int((retained & selected[bi]).sum()),
                        'target_GT_object_id': target_id, 'target_GT_box': label, 'object_identity_association_top2': candidates[:2], 'target_GT_box_retained_static_rows': int(associated.sum()),
                        'target_GT_box_retained_unique_raw_returns': int(len(np.unique(bi[associated]))), 'associated_rows_in_current_front_actor_mask': int((associated & body_mask[bi]).sum()), 'associated_rows_outside_front_frustum': int((associated & ~front[bi]).sum()),
                        'associated_rows_in_front_frustum_outside_actor_mask': int((associated & front[bi] & ~body_mask[bi]).sum()), 'associated_rows_visible_actor_mask': int((associated & visible_body[bi]).sum()),
                        'native_target_instance_link': {'instance_id': link if link > 0 else None, 'seed_return_counts': dict(zip(map(str, seed_ids.tolist()), seed_counts.tolist())), 'seed_rule': 'Native CAR+positiveinstance inside same-time independently tracked target OBB among exact original actor returns; unique mode, no tuned overlap threshold', 'all_native_linked_target_returns': int(semantic_target.sum()), 'linked_native_target_returns_inside_target_OBB': int((semantic_target & within).sum()), 'linked_native_target_retained_static_rows': int((retained & semantic_target[bi]).sum()), 'linked_native_target_retained_unique_returns': int(len(np.unique(bi[retained & semantic_target[bi]]))), 'linked_native_target_retained_static_outside_target_OBB': int((retained & semantic_target[bi] & ~within[bi]).sum()), 'linked_native_target_retained_static_outside_front_actor_mask': int((retained & semantic_target[bi] & ~body_mask[bi]).sum()), 'known_actor_native_semantic_counts': {str(int(k)): int(v) for k, v in zip(*np.unique(semantic[chosen][semantic[chosen] >= 0], return_counts=True))}},
                        'sample_XYZ_time_exact_PLY': True, 'actor_XYZ_time_order_exact_PLY': True}
        frames.append(frame_record)
        for j in idx[associated | (retained & semantic_target[bi])]:
            raw_row = int(bin_index[j]); coord = local[raw_row]; half = np.asarray(label['size_lwh']) / 2
            witness.append({'sample_index': int(j), 'static_initialized_index': int(lineage['static_initialized_index'][j]), 'frame': name, 'source_index': row['source_index'], 'timestamp_micros': row['timestamp_micros'], 'model_time': float(ply_time[j]), 'global_raw_row': int(drawn[j]), 'converted_row_index': raw_row, 'return_index': int(lineage['return_index'][j]), 'laser_name': int(lineage['laser_name'][j]), 'range_row': int(lineage['range_row'][j]), 'range_col': int(lineage['range_col'][j]),
                            'target_GT_object_id': target_id, 'box_local_x': float(coord[0]), 'box_local_y': float(coord[1]), 'box_local_z': float(coord[2]), 'minimum_box_face_clearance_m': float(np.min(half - np.abs(coord))),
                            'vehicle_x': float(raw[raw_row, 0]), 'vehicle_y': float(raw[raw_row, 1]), 'vehicle_z': float(raw[raw_row, 2]), 'world_x': float(ply[j, 0]), 'world_y': float(ply[j, 1]), 'world_z': float(ply[j, 2]), 'front_u': int(uv[raw_row, 0]), 'front_v': int(uv[raw_row, 1]), 'in_current_front_actor_mask': bool(body_mask[raw_row]), 'within_visible_actor_mask_before_component': bool(visible_body[raw_row]), 'native_semantic_class': int(semantic[raw_row]), 'native_instance_id': int(instance[raw_row]), 'linked_target_instance_id': link if link > 0 else None, 'classification': 'native CAR instance associated to target, independent per-return label' if semantic_target[raw_row] else 'same-time target OBB-associated only; semantic identity unresolved' if semantic[raw_row] <= 0 else 'native semantic label available; target-OBB geometry alone is not identity'})
        if fi % 10 == 0:
            print(json.dumps({'frame': fi, 'known_selected_actor_static': frame_record['selected_actor_rows_remaining_static'], 'GT_target_OBB_static': frame_record['target_GT_box_retained_static_rows']}), flush=True)
    require(not np.any(lineage['selected_actor_origin'] & static_keep), 'Unexpected exact selected-actor return leaked into static initializer')
    model_xyz = (np.pad(ply, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3].astype(np.float32)
    lineage['expected_initializer_model_xyz'] = model_xyz
    map_path = OUT / 'all600000_initialisation_lineage.npz'
    np.savez_compressed(map_path, **lineage)
    witness_path = OUT / 'retained_static_target_OBB_witnesses.csv'
    if witness:
        with witness_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(witness[0])); writer.writeheader(); writer.writerows(witness)
    associated = lineage['annotated_target_obb'] & static_keep
    semantic_known = static_keep & (lineage['semantic_label'] > 0)
    semantic_target_retained = static_keep & lineage['semantic_linked_target']
    semantic_unknown = static_keep & ((lineage['semantic_label'] <= 0) | ((lineage['semantic_label'] == car_type) & ((lineage['instance_label'] <= 0) | (lineage['linked_target_instance_label'] <= 0))))
    totals = {key: sum(r[key] for r in frames) for key in ['static_retained_rows', 'static_unique_raw_returns', 'selected_actor_sampled_rows', 'selected_actor_rows_remaining_static', 'target_GT_box_retained_static_rows', 'target_GT_box_retained_unique_raw_returns', 'associated_rows_in_current_front_actor_mask', 'associated_rows_outside_front_frustum', 'associated_rows_in_front_frustum_outside_actor_mask', 'associated_rows_visible_actor_mask']}
    totals.update(native_semantically_labeled_retained_rows=int(semantic_known.sum()), native_linked_target_retained_rows=int(semantic_target_retained.sum()), native_linked_target_retained_unique_raw_returns=int(len(np.unique(drawn[semantic_target_retained]))), direct_target_identity_unresolved_rows=int(semantic_unknown.sum()))
    speed = np.asarray([np.linalg.norm(r['target_GT_box']['speed_xy']) for r in frames])
    result = {'status': 'CPU_STATIC_INITIALISATION_LINEAGE_AND_TARGET_ASSOCIATION_AUDIT_COMPLETE', 'GPU_or_training_started': False, 'training_inputs_S_or_original_source_modified': False,
              'all600000_source_draw_pose_worldXYZ_modeltime_fields_exact': True, 'all36406_actor_source_return_replay_exact': True, 'raw_row_return_laser_range_pixel_layout_verified': True,
              'sampling': {'method': 'np.random.RandomState(seed0).choice(8667791,600000,replace=True), frame prefix sums, bin row, float64 ego pose then float32 world XYZ', 'draw_indices_originally_saved': False, 'lineage_status': 'The declared deterministic recipe reproduces every PLY coordinate/time byte; recovered indices are a full replay-consistent original-return lineage.', 'sample_count': len(ply), 'unique_sampled_original_returns': int(len(np.unique(drawn)))},
              'static_initialization': {'original_PLY_rows': len(ply), 'excluded_nearest_dynamic_union_distance_lt0_1m': int((~static_keep).sum()), 'retained_Gaussian_input_rows': int(static_keep.sum()), 'unique_original_returns_retained': int(len(np.unique(drawn[static_keep]))), 'loader_no_additional_static_sample_or_order_change': True, 'random_init_points': binding['training_config']['random_init_point'], 'PCA_transform_path': str(DATA / 'transform_matrix.txt'), 'expected_initializer_model_XYZ_dtype': 'float32', 'historical_actual_step0_Gaussian_snapshot_available': False},
              'target_identity': {'Waymo_track_id': target_id, 'all49_unique_top_overlap_with_actual_actor_returns': True, 'speed_mps_min_median_max': [float(speed.min()), float(np.median(speed)), float(speed.max())]},
              'counts': totals, 'target_box_associated_retained_static_fraction': float(associated.sum() / static_keep.sum()), 'target_box_associated_fraction_is_a_certified_contamination_rate': False,
              'definite_result': 'Zero exact returns selected into the actual36406-point actor cloud remain in the expected static initializer. Native CAR+instance labels on the five labeled TOP frames separately identify additional target-associated original returns retained in static; this is a witnessed subset, not extrapolation to the unlabeled frames.',
              'independent_annotation_result': 'Additional retained static rows inside the same-time oriented3D box of a uniquely associated moving Waymo object are candidate target-associated returns; these are not a swept AABB or another-frame projection.',
              'semantic_identification': {'raw49_per_return_semantic_instance_label_bytes_present': True, 'labeled_frames': [r['source_index'] for r in frames if r['native_target_instance_link']['instance_id'] is not None], 'semantic_labels_only_TOP_two_returns': True, 'retained_rows_with_native_semantic_class': int(semantic_known.sum()), 'fraction_without_native_semantic_class': float((static_keep.sum() - semantic_known.sum()) / static_keep.sum()), 'linked_target_retained_rows': int(semantic_target_retained.sum()), 'linked_target_retained_fraction_witnessed_lower_bound': float(semantic_target_retained.sum() / static_keep.sum()), 'not_extrapolated_full49_contamination_rate': True, 'direct_target_identity_unresolved_rows': int(semantic_unknown.sum()), 'direct_target_identity_unresolved_fraction': float(semantic_unknown.sum() / static_keep.sum()), 'annotated_target_box_candidates_without_native_semantic_class': int((associated & ~semantic_known).sum()), 'instance_to_Waymo_track_ID_mapping_provided_in_raw': False, 'instance_to_track_mapping_derived_here': 'Per labeled frame, unique native CAR instance overlap with exact selected actor returns inside its consistently associated same-time Waymo target OBB. Native instance labels define the rest of that instance without any RGB mask/box selection.', 'outside_box_interpretation': 'Outside the current annotated target box geometrically, not proof of static/moving class; native linked-instance labels may identify target points outside the box.'},
              'missing_evidence': ['Native target instance/motion labels for the44 unsegmented frames and non-TOP returns', 'An explicit direct frame-local pointinstance-to-stringTrackID mapping; current linkage is derived and recorded', 'Original stored RNG draw/point-ID manifest (recipe replay is exact but historical draw indices were not saved)', 'Per-return physical acquisition timestamps/object-motion compensation; Nx6 final field is TOP range index, not timestamp', 'An actual saved step0 Gaussian bundle to compare historical parameter bytes'],
              'next_evidence': 'Validate candidate return witnesses against genuine same-frame RGB/physical surfaces and official point labels if they exist elsewhere locally. A future M-detector replay must preserve each raw return index/scan time and compare assignments independently; do not change inputs or add a training intervention from box counts alone.',
              'limits': ['GT point/object labels are independent audit-only annotations, not used by training or original actor preparation.', 'Ego-compensated converted XYZ is the actual input; object scan-time motion and box boundary uncertainty affect box-only association.', 'Native instance-to-target linkage is derived from multiple recorded identity cues, not a supplied cross-frame point-track mapping.', 'S continues independently; no full init model, new training plan, data rewrite, detector installation or dataset download was performed.'],
              'pins': pins, 'source_script_SHA256': sha(__file__), 'lineage_map_path': str(map_path), 'lineage_map_SHA256': sha(map_path), 'witness_csv_path': str(witness_path) if witness else None, 'frames': frames}
    for p, expected in pins.items():
        require(sha(p) == expected, 'Audited input or frozen S source changed: ' + p)
    result['all_audited_input_and_S_source_SHA_unchanged_after'] = True
    output = OUT / 'static_initialisation_actor_audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'status': result['status'], 'counts': totals, 'target_track': target_id, 'box_associated_fraction_not_semantic_contamination_rate': result['target_box_associated_retained_static_fraction'], 'receipt_SHA256': sha(output)}))


if __name__ == '__main__':
    main()
