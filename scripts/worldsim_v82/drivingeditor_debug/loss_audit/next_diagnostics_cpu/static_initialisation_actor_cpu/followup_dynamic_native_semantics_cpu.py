"""Separate native point-label composition of the actual actor PLY; audit only."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
ROOT = BASE / 'campaign/vgpu_bugfix_20261012'
DATA = BASE / 'data/waymo_paper_scenes/scene0017_000_048'
OUT = ROOT / 'loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu'
CANONICAL_SHA = 'df97e5d48ee2be553b54e127dbeff2be12dafa8561c93d6bc87bc630059b8ecd'
RAW_SHA = 'fa0448b99abc28c22f40fc7722ecb1d68dbcc68dbe88ae44b5aed0357b894806'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def main():
    canonical_path = OUT / 'static_initialisation_actor_audit.json'
    raw_path = OUT / 'raw49_labels_layout.json'
    require(sha(canonical_path) == CANONICAL_SHA and sha(raw_path) == RAW_SHA, 'Frozen canonical evidence changed')
    canonical = json.loads(canonical_path.read_text()); raw_proof = json.loads(raw_path.read_text())
    helper_path = OUT / 'replay_static_actor_lineage_cpu.py'
    require(sha(helper_path) == canonical['source_script_SHA256'], 'Frozen replay math changed')
    spec = importlib.util.spec_from_file_location('immutable_actor_lineage_math', helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    prep = json.loads((DATA / 'preparation_manifest.json').read_text())
    actor = json.loads((DATA / 'dynamic_actor_preparation.json').read_text())
    selection = json.loads((DATA / 'selected_actor_0.json').read_text())
    dynamic, time = helper.read_ply(DATA / 'points3d_dynamic_0.ply')
    require(len(dynamic) == 36406, 'Actual actor point count differs')
    enum = raw_proof['segmentation_Type_enum']; names = {v: k for k, v in enum.items()}
    require(enum['TYPE_ROAD'] == 18 and enum['TYPE_CAR'] == 1, 'Actual imported Waymo segmentation proto enum differs')
    geometric = json.loads((ROOT / 'sgcm_data_audit/sgcm_runtime_geometry.json').read_text())
    plane = np.asarray(geometric['ground_fit']['plane_anchor'])
    first_pose_inverse = np.linalg.inv(np.loadtxt(DATA / 'pose/0017000.txt'))
    source_labels = np.full(len(dynamic), -1, dtype=np.int32)
    source_instance = np.full(len(dynamic), -1, dtype=np.int32)
    all_labeled, noncar, frames = [], [], []
    for fi in canonical['semantic_identification']['labeled_frames']:
        row = prep['frames'][fi]; frame = row['id']; proof = raw_proof['frames'][fi]
        original_record = canonical['frames'][fi]
        for p in [DATA / 'velodyne' / (frame + '.bin'), DATA / 'pose' / (frame + '.txt'), DATA / 'calib' / (frame + '.txt'), DATA / 'mask/mask_0/Annotations' / (frame + '.png')]:
            require(sha(p) == canonical['pins'][str(p)], 'Original source input changed')
        raw = np.fromfile(DATA / 'velodyne' / (frame + '.bin'), dtype=np.float32).reshape(-1, 6)
        chosen, front, mask, visible, uv = helper.actor_selection(raw[:, :3], row, actor, selection)
        pose = np.loadtxt(DATA / 'pose' / (frame + '.txt'))
        world = (np.pad(raw[chosen, :3], ((0, 0), (0, 1)), constant_values=1) @ pose.T)[:, :3].astype(np.float32)
        ply_indices = np.flatnonzero(time == np.float32(row['model_time']))
        require(np.array_equal(world, dynamic[ply_indices]), 'Actual actor PLY replay differs')
        require(sha(proof['layout_path']) == proof['layout_SHA256'], 'Original return-label sidecar changed')
        label = original_record['target_GT_box']
        local, inside = helper.box_coordinates(raw[chosen, :3].astype(np.float64), label)
        anchor = np.column_stack((world, np.ones(len(world)))) @ first_pose_inverse.T
        signed_ground = anchor[:, :3] @ plane[:3] + plane[3]
        with np.load(proof['layout_path']) as layout:
            semantic = layout['semantic_label'][chosen]; instances = layout['instance_label'][chosen]
            source_labels[ply_indices] = semantic; source_instance[ply_indices] = instances
            for j in np.flatnonzero(semantic >= 0):
                idx = int(chosen[j]); cls = int(semantic[j])
                record = {'actor_PLY_row': int(ply_indices[j]), 'frame': frame, 'source_index': fi, 'timestamp_micros': row['timestamp_micros'], 'model_time': float(time[ply_indices[j]]), 'converted_row_index': idx, 'return_index': int(layout['return_index'][idx]), 'laser_name': int(layout['laser_name'][idx]), 'range_row': int(layout['range_row'][idx]), 'range_col': int(layout['range_col'][idx]), 'native_semantic_class': cls, 'native_semantic_name': names.get(cls, 'not_labeled'), 'native_instance_id': int(instances[j]), 'target_linked_native_instance_id': original_record['native_target_instance_link']['instance_id'], 'inside_target_GT_oriented_box': bool(inside[j]), 'in_front_actor_body_mask': bool(mask[idx]), 'within_visible_actor_mask': bool(visible[idx]), 'front_u': int(uv[idx, 0]), 'front_v': int(uv[idx, 1]), 'vehicle_x': float(raw[idx, 0]), 'vehicle_y': float(raw[idx, 1]), 'vehicle_z': float(raw[idx, 2]), 'world_x': float(world[j, 0]), 'world_y': float(world[j, 1]), 'world_z': float(world[j, 2]), 'signed_fitted_ground_distance_m': float(signed_ground[j])}
                all_labeled.append(record)
                if cls != enum['TYPE_CAR']:
                    noncar.append(record)
        valid = semantic > 0
        frames.append({'frame': frame, 'actual_actor_PLY_rows': len(chosen), 'native_available_including_undefined': int((semantic >= 0).sum()), 'native_valid_semantic_rows': int(valid.sum()), 'native_semantic_counts': {names.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(semantic[semantic >= 0], return_counts=True))}, 'valid_nonCAR_rows': int((valid & (semantic != enum['TYPE_CAR'])).sum())})
    for name, records in [('dynamic_actor_all_native_label_witnesses.csv', all_labeled), ('dynamic_actor_noncar_or_undefined_witnesses.csv', noncar)]:
        path = OUT / name
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    roads = [r for r in noncar if r['native_semantic_class'] == enum['TYPE_ROAD']]
    valid = source_labels > 0
    result = {'status': 'CPU_ACTUAL_DYNAMIC_ACTOR_PLY_NATIVE_SEMANTIC_FOLLOWUP_COMPLETE', 'GPU_or_training_started': False, 'canonical_audit_modified': False,
              'canonical_audit_SHA256': CANONICAL_SHA, 'raw_labels_layout_SHA256': RAW_SHA,
              'actual_actor_PLY_total_rows': 36406, 'semantic_labeled_frames': [25, 30, 35, 40, 45], 'labels_only_TOP_two_returns': True,
              'actual_Waymo_proto_ENUM_TYPE_ROAD': enum['TYPE_ROAD'], 'actual_Waymo_proto_ENUM_TYPE_CAR': enum['TYPE_CAR'],
              'enum_source': {'path': '/root/autodl-tmp/envs/dggt_data_cpu/lib/python3.10/site-packages/waymo_open_dataset/protos/segmentation_pb2.py', 'SHA256': sha('/root/autodl-tmp/envs/dggt_data_cpu/lib/python3.10/site-packages/waymo_open_dataset/protos/segmentation_pb2.py'), 'verification': 'Canonical raw helper imported this exact SDK descriptor and exported the Type enum; no inferred classnumber or prediction.', 'official_schema_URL': 'https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/segmentation.proto'},
              'counts': {'native_valid_semantic_rows': int(valid.sum()), 'native_CAR_rows': int((source_labels == enum['TYPE_CAR']).sum()), 'native_ROAD_rows': len(roads), 'other_known_nonCAR_rows': int((valid & (source_labels != enum['TYPE_CAR']) & (source_labels != enum['TYPE_ROAD'])).sum()), 'native_UNDEFINED_rows': int((source_labels == 0).sum()), 'missing_native_labels': int((source_labels < 0).sum())},
              'ROAD_witnessed_fraction_of_all_actor_PLY_not_full_contamination_estimate': len(roads) / len(dynamic), 'native_valid_label_fraction': float(valid.sum() / len(dynamic)),
              'ROAD_geometry': {'inside_current_front_body_mask_rows': sum(r['in_front_actor_body_mask'] for r in roads), 'inside_current_target_GT_box_rows': sum(r['inside_target_GT_oriented_box'] for r in roads), 'ground_signed_min_median_max_m': [float(np.min([r['signed_fitted_ground_distance_m'] for r in roads])), float(np.median([r['signed_fitted_ground_distance_m'] for r in roads])), float(np.max([r['signed_fitted_ground_distance_m'] for r in roads]))]},
              'physical_vs_appearance': 'Native ROAD is a physical surface class, not an actor-car identity or a motion/shadow label. These returns were selected through the existing body mask and0.5m component, not an explicit shadow mask. Road can carry a moving shadow in RGB, so paper shadow appearance supervision requires its own domain/geometry design; the raw semantic class does not decide that design.',
              'causal_limit': 'The witnessed24 ROAD returns do not explain or quantify the broad trained dynamic-road image by themselves. Branch allocation/global gradients/topology and missing shadow supervision remain distinct factors.',
              'training_input_effect': 'None: native labels/object boxes are audit-only, no sparse-label cleaning or detector output substitution.',
              'ROAD_witnesses': roads, 'all_labeled_witnesses_csv': str(OUT / 'dynamic_actor_all_native_label_witnesses.csv'), 'noncar_or_undefined_witnesses_csv': str(OUT / 'dynamic_actor_noncar_or_undefined_witnesses.csv'), 'frames': frames, 'source_script_SHA256': sha(__file__)}
    require(sha(canonical_path) == CANONICAL_SHA and sha(raw_path) == RAW_SHA, 'Frozen audit changed during separate followup')
    path = OUT / 'dynamic_actor_native_semantics_followup_cpu.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'ROAD_geometry': result['ROAD_geometry'], 'receipt_SHA256': sha(path)}))


if __name__ == '__main__':
    main()
