"""199审计的固定分母、缺证据和口径边界；无模型/训练调用。"""

from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from audit_199_road_exposure import frame_inventory, rectangle, paired_summary, annotation_envelope, packed_matrix
from run_v3_arm import validate_main49_sources


class ExposureTests(unittest.TestCase):
    def test_main49_guard_rejects_audit_or_199_reference_promotion(self):
        frames = [f'0008{i:03d}' for i in range(89, 138)]
        validate_main49_sources(dict(frame_order=frames, frames=dict.fromkeys(frames, {})))
        for order in [frames[::-1], frames[:-1], [f'0008{i:03d}' for i in range(199)]]:
            with self.assertRaises(RuntimeError):
                validate_main49_sources(dict(frame_order=order, frames=dict.fromkeys(order, {})))
        with self.assertRaises(RuntimeError):
            validate_main49_sources(dict(status='CPU_AUDIT_COMPLETE_NOT_TRAINING_INPUT'))

    def test_packed_decoder_matches_native_protobuf(self):
        sys.path.append('/root/autodl-tmp/envs/dggt_data_cpu/lib/python3.10/site-packages')
        try:
            from waymo_open_dataset import dataset_pb2 as pb
        except ImportError:
            self.skipTest('Waymo protobuf installed only on audit host')
        value = pb.MatrixFloat()
        value.shape.dims.extend([2, 3])
        value.data.extend([1., -2., 0., .1, 1e-8, 100.])
        compressed = zlib.compress(value.SerializeToString())
        np.testing.assert_array_equal(packed_matrix(compressed, pb), np.asarray(value.data, np.float32).reshape(2, 3))
        with self.assertRaises(ValueError):
            packed_matrix(zlib.compress(value.SerializeToString()+b'\x1a\x00'), pb)
        value.data[0] = float('nan')
        with self.assertRaises(ValueError):
            packed_matrix(zlib.compress(value.SerializeToString()), pb)

    def test_inventory_reads_actual_file_boundary_not_assumed_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'raw.tfrecord'
            path.write_bytes(struct.pack('<QI', 3, 0)+b'abc'+b'0000')
            self.assertEqual(frame_inventory(path), [0])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                frame_inventory(path)

    def test_new_confirmation_requires_two_sources_and_fixed_surface_support(self):
        old = np.array([[0, 1, 1, 0], [0, 0, 1, 0]], bool)
        added = np.array([[1, 1, 1, 1], [0, 0, 0, 1]], bool)
        result = paired_summary(old, added, np.array([1, 1, 1, 0], bool), np.arange(4))
        self.assertEqual(result['world_units'], 4)
        self.assertEqual(result['newly_confirmed'], 1)
        self.assertEqual(result['zero_to_confirmed'], 0)
        self.assertEqual(result['zero_to_single'], 1)
        self.assertEqual(result['all199']['unsupported'], 1)
        self.assertEqual(result['already_confirmed_with_more_sources'], 1)

    def test_added_no_observation_does_not_create_confirmation(self):
        result = paired_summary(np.zeros((49, 3), bool), np.zeros((150, 3), bool),
                                np.ones(3, bool), np.arange(3))
        self.assertEqual(result['newly_confirmed'], 0)
        self.assertEqual(result['all199']['zero_source_supported'], 3)

    def test_rectangles_clip_and_do_not_wrap_negative_indices(self):
        mask = np.zeros((5, 5), bool)
        rectangle(mask, [-10, -10, 1, 1])
        self.assertTrue(mask[:2, :2].all())
        self.assertFalse(mask[-1].any())
        with self.assertRaises(ValueError):
            rectangle(mask, [0, 0, np.nan, 1])

    def test_front_raw_camera_boxes_are_conservative_occluders_not_donor_regions(self):
        b = NS(center_x=100., center_y=100., length=40., width=20.)
        group = NS(name=1, labels=[NS(box=b)])
        frame = NS(camera_labels=[group], projected_lidar_labels=[])
        mask = annotation_envelope(frame, [], np.eye(3), np.eye(4), None)
        self.assertTrue(mask[25, 25])
        self.assertFalse(mask[0, 0])

    def test_missing_front_annotation_fails_closed_not_empty_background(self):
        frame = NS(camera_labels=[], projected_lidar_labels=[])
        self.assertTrue(annotation_envelope(frame, [], np.eye(3), np.eye(4), None).all())

    def test_side_rear_box_crossing_near_plane_does_not_black_out_front_view(self):
        frame = NS(camera_labels=[NS(name=1, labels=[])], projected_lidar_labels=[])
        box = dict(center=np.array([-10., 0, 0]), half=np.array([1., 1, 1]), axes=np.eye(3))
        intrinsic = np.array([[100., 0, 240], [0, 100, 160], [0, 0, 1.]])
        transform = lambda points, mat: points @ mat[:3, :3].T+mat[:3, 3]
        mask = annotation_envelope(frame, [box], intrinsic, np.eye(4), transform)
        self.assertFalse(mask.any())


if __name__ == '__main__':
    unittest.main()
