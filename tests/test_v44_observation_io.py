"""CPU source/calibration/provenance regressions using a real on-disk mini bank."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from v44_observation_io import load_waymo_evidence, own_volume_depth


class ObservationIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.bank = self.root / "inputs"
        for folder in ("observations", "prompts", "image/native", "velodyne", "timestamp"):
            (self.bank / folder).mkdir(parents=True, exist_ok=True)
        self.key = "capture-key"; self.track = "arbitrary-target"; self.timestamp = 1234567
        self.pose = np.eye(4); self.pose[0, 3] = 10.
        self.target = {"track_id": self.track, "dimensions": [4., 4., 4.],
                       "world_from_actor": self.pose.ravel().tolist()}
        self.other = {"track_id": "unmeasured-other", "dimensions": [100., 100., 100.],
                      "world_from_actor": np.eye(4).ravel().tolist()}
        self.obs = {"raw": 17, "timestamp_micros": self.timestamp,
                    "ego_world": np.eye(4).ravel().tolist(), "camera_pose_world": np.eye(4).ravel().tolist(),
                    "camera_extrinsic": np.eye(4).ravel().tolist(),
                    "camera_intrinsic_Brown": [4., 4., 4., 3., 0., 0., 0., 0., 0.],
                    "camera_size": [8, 6], "image_path": "image/native/unique_capture.png",
                    "lidar_path": "velodyne/source_returns.bin", "boxes": [self.target, self.other]}
        self.rows = np.array([[10, 0, 0, .2, .3, 101], [4, 0, 0, .2, .3, 102],
                              [-2, 0, 0, .2, .3, 103], [10, -20, 0, .2, .3, 104],
                              [np.nan, 0, 0, .2, .3, 105], [10, 1, 1, .2, .3, 106],
                              [10, 0, -1, .2, .3, -1], [0, 0, 0, .2, .3, 107],
                              [10, 0, 0, .8, .9, 108]], dtype="<f4")
        self.rows.tofile(self.bank / self.obs["lidar_path"])
        self.rgb = np.arange(6*8*3, dtype=np.uint8).reshape(6, 8, 3)
        Image.fromarray(self.rgb).save(self.bank / self.obs["image_path"])
        self.write_observation(); self.write_prompts([self.target, self.other])
        (self.bank / "timestamp/unique_capture.txt").write_text(str(self.timestamp))
        self.receipt = {"frames": [{"raw": 17, "id": "unique_capture", "timestamp_micros": self.timestamp,
                                    "lidar_points": len(self.rows)}]}
        self.write_json("receipt.json", self.receipt)
        self.write_json("source_records.json", [{"raw": 17, "timestamp_micros": self.timestamp}])

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, relative, data):
        (self.bank / relative).write_text(json.dumps(data), encoding="utf8")

    def write_observation(self):
        self.write_json(f"observations/{self.key}.json", self.obs)

    def write_prompts(self, rows):
        self.write_json(f"prompts/{self.key}.json", rows)

    def load(self, **kwargs):
        return load_waymo_evidence(self.bank, self.key, self.target, **kwargs)

    def test_native_rgb_and_calibration_are_retained(self):
        e = self.load()
        self.assertEqual(e["image_shape"], (6, 8))
        np.testing.assert_array_equal(e["rgb"], self.rgb)
        np.testing.assert_array_equal(e["intrinsics"], [[4, 0, 4], [0, 4, 3], [0, 0, 1]])
        self.assertEqual(e["source"]["depth_convention"], "camera_forward_X_metres")

    def test_brown_projection_uses_camera_forward_left_up_axes(self):
        e = self.load()
        at = np.flatnonzero(e["source_indices"] == 5)[0]
        np.testing.assert_allclose(e["projected_uv"][at], [3.6, 2.6])
        self.assertEqual(e["depth"][at], 10.)

    def test_nonzero_brown_distortion_is_applied(self):
        self.obs["camera_intrinsic_Brown"][4:] = [.1, .05, .01, -.02, .001]
        self.write_observation(); e = self.load(); at = np.flatnonzero(e["source_indices"] == 5)[0]
        np.testing.assert_allclose(e["projected_uv"][at], [3.5967919968, 2.5991919968], atol=1e-10)

    def test_source_row_and_encoded_source_pixel_provenance_are_exact(self):
        e = self.load(); expected = np.array([0, 1, 5, 6, 8])
        np.testing.assert_array_equal(e["source_indices"], expected)
        np.testing.assert_array_equal(e["lidar_rows"], self.rows[expected])
        np.testing.assert_array_equal(e["source_pixel_index"], [101, 102, 106, -1, 108])
        np.testing.assert_array_equal(e["rejected_source_indices"], [2, 3, 4, 7])

    def test_multiple_depth_surfaces_in_same_native_pixel_are_not_zbuffered(self):
        e = self.load()
        same = (e["native_pixel_xy"] == [4, 3]).all(1)
        self.assertIn(4., e["depth"][same])
        self.assertIn(10., e["depth"][same])
        self.assertGreaterEqual(len(e["depth"][same]), 3)

    def test_actual_own_and_foreground_indices_address_returned_rows(self):
        e = self.load()
        np.testing.assert_array_equal(e["source_indices"][e["own_indices"]], [0, 5, 6, 8])
        np.testing.assert_array_equal(e["source_indices"][e["foreground_indices"]], [1])
        np.testing.assert_array_equal(e["foreground_uv"], e["projected_uv"][e["foreground_indices"]])
        np.testing.assert_array_equal(e["own_supported_uv"], e["projected_uv"][e["own_indices"]])

    def test_exact_per_return_ray_entry_and_exit(self):
        e = self.load()
        np.testing.assert_allclose(e["ray_near"][:2], [8., 8.])
        np.testing.assert_allclose(e["ray_far"][:2], [12., 12.])
        inside = e["own_source"]
        self.assertTrue((e["depth"][inside] >= e["ray_near"][inside] - 1e-6).all())
        self.assertTrue((e["depth"][inside] <= e["ray_far"][inside] + 1e-6).all())

    def test_other_obbs_never_supply_opaque_occlusion(self):
        first = self.load(); self.obs["boxes"] = [self.target]; self.write_observation()
        self.write_prompts([self.target]); second = self.load()
        for key in ("projected_uv", "depth", "own_source", "foreground_indices", "ray_near", "ray_far"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_outside_query_ray_stays_unknown_not_foreground(self):
        rows = np.vstack([self.rows, [4., -2., 0., .2, .3, 109.]])
        rows.astype("<f4").tofile(self.bank / self.obs["lidar_path"])
        self.receipt["frames"][0]["lidar_points"] = len(rows); self.write_json("receipt.json", self.receipt)
        e = self.load(); at = np.flatnonzero(e["source_indices"] == len(rows)-1)[0]
        self.assertTrue(np.isnan(e["ray_near"][at]))
        self.assertNotIn(at, e["foreground_indices"])

    def test_frame_key_lookup_uses_metadata_without_suffix_guessing(self):
        direct = self.load()
        for key in (17, "17", "unique_capture"):
            e = load_waymo_evidence(self.bank, key, self.target)
            np.testing.assert_array_equal(e["source_indices"], direct["source_indices"])
            self.assertEqual(e["raw"], 17)

    def test_prep_root_and_explicit_observation_path_are_supported(self):
        e = load_waymo_evidence(self.root, "external-key", self.target,
                               observation_path=f"observations/{self.key}.json")
        self.assertEqual(Path(e["source"]["bank"]), self.bank.resolve())

    def test_opaque_track_name_resolves_only_same_frame_geometry(self):
        e = load_waymo_evidence(self.bank, self.key, self.track)
        self.assertEqual(e["target_geometry_origin"], "same_frame")
        with self.assertRaises(KeyError):
            load_waymo_evidence(self.bank, self.key, "absent-track")

    def test_observation_box_fallback_does_not_depend_on_front_prompt_gate(self):
        self.write_prompts([])
        e = load_waymo_evidence(self.bank, self.key, self.track)
        self.assertEqual(e["target_track"]["track_id"], self.track)

    def test_explicit_inferred_geometry_retains_metadata_without_claiming_gt(self):
        target = {**self.target, "track_id": "explicit-missing-track", "inferred": True,
                  "evidence": "caller supplied short-gap geometry", "raw": 17}
        e = load_waymo_evidence(self.bank, self.key, target)
        self.assertTrue(e["target_track"]["inferred"])
        self.assertEqual(e["target_geometry_origin"], "explicit")
        self.assertEqual(e["target_track"]["track_id"], "explicit-missing-track")

    def test_capture_pose_and_extrinsic_are_not_replaced_with_ego_pose(self):
        ego, image, extrinsic = np.eye(4), np.eye(4), np.eye(4)
        ego[0, 3], image[0, 3], extrinsic[0, 3] = 2., 3., 1.
        self.obs.update(ego_world=ego.ravel().tolist(), camera_pose_world=image.ravel().tolist(),
                        camera_extrinsic=extrinsic.ravel().tolist()); self.write_observation()
        e = self.load(); at = np.flatnonzero(e["source_indices"] == 0)[0]
        self.assertEqual(e["depth"][at], 8.)
        self.assertEqual(e["camera_to_world"][0, 3], 4.)

    def test_optional_dense_grid_is_geometry_only_and_visibility_ready(self):
        from v44_visibility import Evidence, visibility_evidence
        e = self.load(output_shape=(6, 8))
        self.assertEqual(e["own_near"].shape, (6, 8))
        self.assertEqual(e["own_near"][3, 4], 8.)
        self.assertEqual(e["own_far"][3, 4], 12.)
        result = visibility_evidence(**e["visibility_kwargs"])
        self.assertEqual(result["state"][3, 4], Evidence.MIXED)
        self.assertFalse(result["conflict"][3, 4])

    def test_dense_maps_are_not_built_by_default(self):
        e = self.load()
        self.assertNotIn("own_near", e)
        near, far = own_volume_depth(e, (3, 4))
        self.assertEqual(near.shape, (3, 4)); self.assertEqual(far.shape, (3, 4))

    def test_timestamp_and_source_count_mismatches_fail(self):
        (self.bank / "timestamp/unique_capture.txt").write_text(str(self.timestamp+1))
        with self.assertRaises(ValueError): self.load()
        (self.bank / "timestamp/unique_capture.txt").write_text(str(self.timestamp))
        self.receipt["frames"][0]["timestamp_micros"] += 1; self.write_json("receipt.json", self.receipt)
        with self.assertRaises(ValueError): self.load()
        self.receipt["frames"][0]["timestamp_micros"] -= 1
        self.receipt["frames"][0]["lidar_points"] += 1; self.write_json("receipt.json", self.receipt)
        with self.assertRaises(ValueError): self.load()

    def test_wrong_native_rgb_size_or_binary_row_width_fails(self):
        Image.fromarray(np.zeros((6, 9, 3), np.uint8)).save(self.bank / self.obs["image_path"])
        with self.assertRaises(ValueError): self.load()
        Image.fromarray(self.rgb).save(self.bank / self.obs["image_path"])
        with (self.bank / self.obs["lidar_path"]).open("ab") as stream: stream.write(np.float32(1.).tobytes())
        with self.assertRaises(ValueError): self.load()

    def test_declared_alternative_lidar_row_width_does_not_fabricate_provenance(self):
        self.rows[:, :3].tofile(self.bank / self.obs["lidar_path"])
        e = self.load(lidar_columns=3, source_pixel_column=None)
        self.assertIsNone(e["source_pixel_index"])
        self.assertEqual(e["lidar_rows"].shape[1], 3)

    def test_invalid_target_geometry_or_wrong_time_is_rejected(self):
        for update in ({"dimensions": [4., -1., 4.]}, {"raw": 18},
                       {"world_from_actor": (np.eye(4)*2).ravel().tolist()}):
            with self.assertRaises(ValueError):
                load_waymo_evidence(self.bank, self.key, {**self.target, **update})

    def test_actor_renaming_never_changes_projection_or_point_ownership(self):
        first = self.load(); target = {**self.target, "track_id": "renamed-actor", "alias": "also-renamed"}
        second = load_waymo_evidence(self.bank, self.key, target)
        for key in ("projected_uv", "own_source", "ray_near", "source_pixel_index"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_input_files_and_explicit_target_are_unchanged(self):
        files = sorted(p for p in self.bank.rglob("*") if p.is_file())
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        target = deepcopy(self.target); e = self.load(output_shape=(3, 4))
        self.assertEqual(self.target, target)
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.bank.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(e["source"]["projection_policy"], "all_actual_returns_no_zbuffer_no_neighbor_min")


if __name__ == "__main__":
    unittest.main()
