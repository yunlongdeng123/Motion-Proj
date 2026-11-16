"""CPU guards for receipt-based native PCL time mapping and unchanged normalization."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from v44_vehicle_prior import model_time_frame_indices, normalization_contract, frame_sources, frame_mask_path


class VehiclePriorTests(unittest.TestCase):
    def test_native_float32_times_map_to_irregular_receipt_not_raw_offset(self):
        frames = [{"raw": 91, "model_time": 1.75}, {"raw": 5, "model_time": -.4}, {"raw": 44, "model_time": .23}]
        index = model_time_frame_indices(np.array([-.4, .23, 1.75], np.float32), frames)
        self.assertEqual([frames[i]["raw"] for i in index], [5, 44, 91])

    def test_far_unrecorded_or_ambiguous_time_cannot_invent_frame(self):
        frames = [{"model_time": 0.}, {"model_time": 1.}]
        for values in ([.5], [10.], [np.nan]):
            with self.assertRaises(ValueError): model_time_frame_indices(values, frames)

    def test_time_tolerance_is_explicit_and_does_not_set_frame_budget(self):
        frames = [{"model_time": 0.}, {"model_time": 1.}]
        np.testing.assert_array_equal(model_time_frame_indices([.01], frames, tolerance=.02), [0])
        with self.assertRaises(ValueError): model_time_frame_indices([.01], frames)

    def test_duplicate_or_float32_colliding_frame_times_fail(self):
        for times in ([0., 0.], [1e9, 1e9+.01]):
            with self.assertRaises(ValueError):
                model_time_frame_indices([times[0]], [{"model_time": t} for t in times])

    def test_wholebox_normalization_matches_inherited_formula(self):
        template = dict(native_to_training_raw_rotation=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                        training_mean_full_bbox_span_xyz=[2., 3., 4.], training_mean_xyz=[.2, .3, .4],
                        training_mean_full_bbox_center_xyz=[.1, .2, .3], training_scalar_std=2.,
                        front_sign_assumption="caller-frozen")
        contract = normalization_contract([4., 2., 1.5], template)
        np.testing.assert_allclose(contract["pvd_to_actor_native_4x4"],
                                   [[0, 1, 0, .05], [0, 0, 2, .1], [2, 0, 0, .1], [0, 0, 0, 1]])
        self.assertFalse(contract["partial_bounds_scaled"])
        self.assertFalse(contract["partial_bounds_centered"])
        self.assertEqual(contract["front_sign_assumption"], "caller-frozen")

    def test_observation_and_mask_binding_follow_metadata_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/"observations").mkdir()
            rows = [{"raw": 7, "model_time": -.4, "timestamp_micros": 10, "id": "native-opaque"}]
            (root/"receipt.json").write_text(json.dumps({"frames": rows}))
            (root/"observations/arbitrary-capture.json").write_text(json.dumps({"raw": 7, "timestamp_micros": 10}))
            _, bound = frame_sources(root)
            self.assertEqual(bound[0]["observation_path"].name, "arbitrary-capture.json")
            self.assertEqual(frame_mask_path(root/"mask", bound[0]), root/"mask/arbitrary-capture/labels.png")

    def test_timestamp_binding_rejects_source_time_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/"observations").mkdir()
            (root/"receipt.json").write_text(json.dumps({"frames": [{"raw": 7, "model_time": 0., "timestamp_micros": 10}]}))
            (root/"observations/key.json").write_text(json.dumps({"raw": 7, "timestamp_micros": 11}))
            with self.assertRaises(ValueError): frame_sources(root)


if __name__ == "__main__": unittest.main()
