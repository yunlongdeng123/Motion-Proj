"""Read the existing first49 Waymo records; no TensorFlow, conversion or GPU.

Reconstruct return/laser/range-pixel lineage of each existing converted row and
verify its intensity/elongation/TOP index against the original range images.
Write only independent audit evidence, never the scene or a training directory.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import segmentation_pb2

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
DATA = BASE / 'data/waymo_paper_scenes/scene0017_000_048'
OUT = BASE / 'campaign/vgpu_bugfix_20261012/loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'raw_layout').mkdir(exist_ok=True)
    prep = json.loads((DATA / 'preparation_manifest.json').read_text())
    raw_path = Path(prep['raw_path'])
    stat = raw_path.stat()
    require(stat.st_size == prep['raw_size_bytes'], 'Original existing TFRecord size differs')
    rows = []
    with raw_path.open('rb') as stream:
        for index, expected in enumerate(prep['frames']):
            require(index == expected['source_index'], 'Audit intentionally requires existing0..48 contiguous window')
            header = stream.read(12)
            require(len(header) == 12, 'Truncated TFRecord header')
            size = struct.unpack('<Q', header[:8])[0]
            frame = dataset_pb2.Frame.FromString(stream.read(size))
            require(len(stream.read(4)) == 4, 'Truncated TFRecord CRC field')
            require(frame.timestamp_micros == expected['timestamp_micros'] and frame.context.name == prep['segment_context_name'], 'Raw frame/time/context identity differs')
            converted = np.fromfile(DATA / 'velodyne' / (expected['id'] + '.bin'), dtype=np.float32).reshape(-1, 6)
            recorded_pose = np.loadtxt(DATA / 'pose' / (expected['id'] + '.txt'))
            raw_pose = np.asarray(frame.pose.transform).reshape(4, 4)
            require(np.allclose(recorded_pose, raw_pose, rtol=0, atol=1e-12), 'Converted pose differs from raw Frame pose')
            lasers = {laser.name: laser for laser in frame.lasers}
            calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
            parts, chunks, cursor, segmentation = [], [], 0, []
            for ret in (0, 1):
                for calibration in calibrations:
                    image = getattr(lasers[calibration.name], 'ri_return' + str(ret + 1))
                    matrix = dataset_pb2.MatrixFloat.FromString(zlib.decompress(image.range_image_compressed))
                    array = np.asarray(matrix.data, dtype=np.float32).reshape(matrix.shape.dims)
                    valid = (array[..., 0] > 0) & (array[..., 3] != 1.0)
                    rr, cc = np.nonzero(valid)
                    count = len(rr)
                    end = cursor + count
                    top = (ret * array.shape[0] + rr) * array.shape[1] + cc if calibration.name == 1 else np.full(count, -1)
                    require(np.array_equal(converted[cursor:end, 3], array[rr, cc, 1]) and np.array_equal(converted[cursor:end, 4], array[rr, cc, 2]) and np.array_equal(converted[cursor:end, 5], top.astype(np.float32)), 'Raw return/laser row order or saved fields differ')
                    if image.segmentation_label_compressed:
                        labels = dataset_pb2.MatrixInt32.FromString(zlib.decompress(image.segmentation_label_compressed))
                        label_array = np.asarray(labels.data, dtype=np.int32).reshape(labels.shape.dims)
                        require(label_array.shape == (array.shape[0], array.shape[1], 2), 'Per-return segmentation shape differs')
                        instance, semantic = label_array[rr, cc, 0], label_array[rr, cc, 1]
                    else:
                        instance = np.full(count, -1, dtype=np.int32)
                        semantic = np.full(count, -1, dtype=np.int32)
                    parts.append((np.full(count, ret, dtype=np.uint8), np.full(count, calibration.name, dtype=np.uint8), rr.astype(np.uint16), cc.astype(np.uint16), instance, semantic))
                    chunks.append({'return_index': ret, 'laser_name': calibration.name, 'range_shape': list(array.shape), 'row_begin': cursor, 'row_end': end, 'points': count})
                    segmentation.append({'return_index': ret, 'laser_name': calibration.name, 'segmentation_compressed_bytes': len(image.segmentation_label_compressed)})
                    cursor = end
            require(cursor == len(converted) == expected['lidar_points'], 'Raw all-return point count differs')
            layout_path = OUT / 'raw_layout' / (expected['id'] + '.npz')
            np.savez_compressed(layout_path, return_index=np.concatenate([x[0] for x in parts]), laser_name=np.concatenate([x[1] for x in parts]), range_row=np.concatenate([x[2] for x in parts]), range_col=np.concatenate([x[3] for x in parts]), instance_label=np.concatenate([x[4] for x in parts]), semantic_label=np.concatenate([x[5] for x in parts]))
            labels = []
            for item in frame.laser_labels:
                labels.append({'id': item.id, 'type': item.type, 'center': [item.box.center_x, item.box.center_y, item.box.center_z], 'size_lwh': [item.box.length, item.box.width, item.box.height], 'heading': item.box.heading, 'speed_xy': [item.metadata.speed_x, item.metadata.speed_y], 'num_lidar_points_in_box': item.num_lidar_points_in_box, 'num_top_lidar_points_in_box': item.num_top_lidar_points_in_box, 'detection_difficulty_level': item.detection_difficulty_level, 'tracking_difficulty_level': item.tracking_difficulty_level})
            rows.append({'frame': expected['id'], 'source_index': index, 'timestamp_micros': frame.timestamp_micros, 'model_time': expected['model_time'], 'raw_proto_pose_max_abs_error': float(np.max(np.abs(recorded_pose - raw_pose))), 'bin_SHA256': sha(DATA / 'velodyne' / (expected['id'] + '.bin')), 'layout_path': str(layout_path), 'layout_SHA256': sha(layout_path), 'all_intensity_elongation_TOP_indices_exact': True, 'chunks': chunks, 'segmentation_availability': segmentation, 'labels': labels})
            if index % 10 == 0:
                print(json.dumps({'raw_frame': index, 'rows_verified': cursor}), flush=True)
    result = {'status': 'PASS_EXISTING_RAW49_RETURN_LAYOUT_AND_LABELS_CPU', 'GPU_or_training_started': False, 'TensorFlow_or_Torch_imported': False, 'training_input_modified': False, 'source_raw': str(raw_path), 'raw_size_bytes': stat.st_size, 'historical_raw_SHA256': prep['raw_sha256'], 'full_raw_SHA_recomputed_here': False, 'raw_crc_verified_here': False, 'first49_context_timestamp_pose_and_all_saved_lidar_fields_verified': True, 'point_semantic_instance_labels_present': any(x['segmentation_compressed_bytes'] for row in rows for x in row['segmentation_availability']), 'segmentation_Type_enum': {v.name:v.number for v in segmentation_pb2.Segmentation.DESCRIPTOR.enum_types_by_name['Type'].values}, 'segmentation_channels': ['frame-local instance_id', 'semantic_class'], 'source_script_SHA256': sha(__file__), 'frames': rows, 'limit': 'Motion-compensated XYZ conversion itself is inherited; row lineage is independently verified from range validity/NLZ, per-return order, intensity, elongation and TOP range index. Box labels are object annotations, not per-return semantic identities; per-return instance IDs are frame-local and need independently recorded linkage to the selected actor.'}
    path = OUT / 'raw49_labels_layout.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    require(raw_path.stat().st_size == stat.st_size and raw_path.stat().st_mtime_ns == stat.st_mtime_ns, 'Raw file changed during audit')
    print(json.dumps({'status': result['status'], 'frames': len(rows), 'point_semantic_instance_labels_present': result['point_semantic_instance_labels_present'], 'receipt_SHA256': sha(path)}))


if __name__ == '__main__':
    main()
