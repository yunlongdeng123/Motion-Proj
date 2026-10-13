"""CPU 几何/输入合同测试；小型合成 fixture 仅供测试，不能作为真实准备输出。"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

_SOURCE = Path(__file__).resolve().parents[1] / "data.py"
_SPEC = importlib.util.spec_from_file_location("waymo_instance_data_test_target", _SOURCE)
data = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = data
_SPEC.loader.exec_module(data)


def label(track="stationary", center=(0, 0, 0), size=(4, 2, 2), heading=0, kind="TYPE_VEHICLE"):
    return {"id": track, "type": kind, "box": dict(zip(
        ("center_x", "center_y", "center_z", "length", "width", "height", "heading"), (*center, *size, heading))),
        "metadata": {"speed_x": 0, "speed_y": 0}}


def pose(yaw=0, translation=(0, 0, 0)):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    value = np.eye(4)
    value[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    value[:3, 3] = translation
    return value


def frame(index, labels, ego=None):
    row = {"id": f"0017{index:03d}", "source_index": index, "timestamp_micros": 1500000000000000 + index * 100000,
           "model_time": index * .02 - .01}
    audit = {"frame_id": row["id"], "timestamp_micros": row["timestamp_micros"], "laser_labels": labels}
    return data.FrameAnnotations.from_audit(audit, row, np.eye(4) if ego is None else ego, "unit-test fixture")


def save_test_ply(path, xyz, times):
    dtype = np.dtype([(key, "<f4") for key in ("x", "y", "z", "nx", "ny", "nz")]
                     + [(key, "u1") for key in ("red", "green", "blue")] + [("time", "<f4")])
    vertices = np.zeros(len(xyz), dtype=dtype)
    for axis, key in enumerate(("x", "y", "z")):
        vertices[key] = np.asarray(xyz)[:, axis]
    for key in ("nx", "ny", "nz", "time"):
        vertices[key] = times
    vertices["red"] = np.arange(len(xyz)) + 1
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
    header += "".join("property " + ("uchar" if key in ("red", "green", "blue") else "float") + " " + key + "\n" for key in dtype.names)
    with Path(path).open("wb") as stream:
        stream.write((header + "end_header\n").encode("ascii"))
        vertices.tofile(stream)


def prepare_isolated(scene, output, **kwargs):
    """CPU 准备在独立进程验证，避免模型测试在收集时已导入 Torch。"""
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in kwargs.items()}
    script = """import json,sys
sys.path.insert(0,sys.argv[1])
from data import prepare_scene
try:
 report=prepare_scene(sys.argv[2],sys.argv[3],**json.loads(sys.argv[4]))
 print(json.dumps({'report':report},ensure_ascii=True))
except ValueError as error:
 print(json.dumps({'error':str(error)},ensure_ascii=True))
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", script, str(_SOURCE.parent), str(scene), str(output), json.dumps(arguments)],
                            capture_output=True, text=True, env=environment, check=True, timeout=45)
    value = json.loads(result.stdout)
    if "error" in value:
        raise ValueError(value["error"])
    return value["report"]


class GeometryContracts(unittest.TestCase):
    def test_waymo_lwh_heading_and_full_ego_world_normalization_scale_once(self):
        box = data.WaymoBox.from_label(label(center=(5, 3, .5), heading=np.pi / 2))
        # length 沿 box x，heading 旋转到车辆 y；禁止交换 length/width。
        np.testing.assert_array_equal(box.contains_vehicle([[5, 4.5, .5], [6.5, 3, .5]]), [True, False])
        ego = pose(.7, (100, -30, 2))
        pitch = np.array([[np.cos(.2), 0, np.sin(.2)], [0, 1, 0], [-np.sin(.2), 0, np.cos(.2)]])
        ego[:3, :3] = ego[:3, :3] @ pitch
        normalization = pose(-.4, (-3, 1, -.2))
        normalization[:3, :3] *= .1
        local_m = np.array([[1, .4, .2], [-.2, -.3, -.1]])
        vehicle = data.transform_xyz(local_m, box.vehicle_from_local_m)
        expected = data.transform_xyz(data.transform_xyz(vehicle, ego), normalization)
        normalized_pose = box.normalized_pose(ego, normalization)
        np.testing.assert_allclose(data.transform_xyz(local_m * .1, normalized_pose), expected, atol=1e-12)
        np.testing.assert_allclose(normalized_pose[:3, :3].T @ normalized_pose[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(box.local_xyz_m(vehicle), local_m, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "uniform proper similarity"):
            wrong = normalization.copy()
            wrong[:3, 0] *= 2
            data.similarity_parts(wrong)

    def test_stationary_track_included_and_raw_physical_time_not_model_time(self):
        first = frame(0, [label("parked", center=(5, 0, 0)), label("moving", center=(-5, 0, 0))])
        second = frame(1, [label("parked", center=(4, 0, 0)), label("moving", center=(-3, 0, 0))], pose(translation=(1, 0, 0)))
        clouds = [np.array([[5.2, 0, 0], [4.8, 0, 0], [-5, 0, 0], [30, 0, 0]]),
                  np.array([[4.2, 0, 0], [3.8, 0, 0], [-3, 0, 0], [30, 0, 0]])]
        normalization = np.diag([.1, .1, .1, 1])
        bundle, report = data.make_instance_bundle([first, second], clouds, normalization)
        parked = bundle["instance_ids"].tolist().index("parked")
        self.assertTrue(bundle["valid"][:, parked].all())
        np.testing.assert_allclose(bundle["poses"][0, parked], bundle["poses"][1, parked])
        local = bundle["local_xyz"][bundle["point_ids"] == parked]
        np.testing.assert_allclose(local[:2], local[2:], atol=1e-7)
        np.testing.assert_allclose(bundle["box_size"][:, parked], [[.4, .2, .2], [.4, .2, .2]])
        self.assertAlmostEqual(float(np.diff(bundle["physical_timestamps"])[0]), .1, places=6)
        self.assertAlmostEqual(float(np.diff(bundle["timestamps"])[0]), .02)
        self.assertEqual(report["motion_filter"], "none")

    def test_frame_valid_independent_of_front_visibility_and_mask_known(self):
        frames = [frame(0, [label("off_front")]), frame(1, [])]
        cloud = np.array([[0, 0, 0], [.2, 0, 0], [20, 0, 0]])
        bundle, report = data.make_instance_bundle(frames, [cloud, cloud], np.eye(4),
                                                  mask_known=[[True], [False]], front_visible=[[0], [-1]])
        np.testing.assert_array_equal(bundle["valid"], [[True], [False]])
        np.testing.assert_array_equal(bundle["mask_known"], [[True], [False]])
        np.testing.assert_array_equal(bundle["front_visible"], [[0], [-1]])
        self.assertIsNone(report["trajectories"][0]["frames"][1]["pose_normalized_world_from_local"])
        self.assertEqual(len(bundle["local_xyz"]), 2)
        self.assertEqual(report["mask_complete_frames"], 1)  # 缺 3D 标注不等于已知空 2D mask。

    def test_oriented_overlaps_reject_ambiguity_and_background_removes_union(self):
        boxes = [data.WaymoBox.from_label(label("a", center=(0, 0, 0))), data.WaymoBox.from_label(label("b", center=(2, 0, 0)))]
        xyz = np.array([[-1, 0, 0], [1, 0, 0], [3, 0, 0], [20, 0, 0]])
        owner, union = data.assign_lidar_to_boxes(xyz, boxes)
        np.testing.assert_array_equal(owner, [0, -2, 1, -1])
        np.testing.assert_array_equal(union, [True, True, True, False])
        reversed_owner, reversed_union = data.assign_lidar_to_boxes(xyz, boxes[::-1])
        np.testing.assert_array_equal(reversed_owner, [1, -2, 0, -1])
        np.testing.assert_array_equal(reversed_union, union)
        frames = [frame(0, [label("a", center=(0, 0, 0)), label("b", center=(2, 0, 0))])]
        keep, report = data.filter_static_seed(xyz, [-.01] * len(xyz), frames, ["a", "b"])
        np.testing.assert_array_equal(keep, [False, False, False, True])
        self.assertEqual(report["excluded_points"], 3)

    def test_background_seed_time_specific_and_original_order_reused(self):
        frames = [frame(0, [label("a", center=(0, 0, 0))]), frame(1, [label("a", center=(10, 0, 0))])]
        xyz = np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0], [10, 0, 0]])
        keep, report = data.filter_static_seed(xyz, [-.01, .01, -.01, .01], frames, ["a"])
        np.testing.assert_array_equal(np.flatnonzero(keep), [1, 2])
        self.assertTrue(report["sampling_and_order_reused"])
        with self.assertRaisesRegex(ValueError, "no exact source frame"):
            data.filter_static_seed(xyz, [0, 0, 0, 0], frames, ["a"])

    def test_missing_real_fields_unknown_track_and_empty_points_refused(self):
        bad_label = label()
        del bad_label["box"]["length"]
        with self.assertRaisesRegex(ValueError, "Incomplete genuine"):
            data.WaymoBox.from_label(bad_label)
        with self.assertRaisesRegex(ValueError, "raw track ID"):
            data.WaymoBox.from_label({**label(), "id": ""})
        with self.assertRaisesRegex(ValueError, "laser_labels absent"):
            row = {"id": "a", "source_index": 0, "model_time": 0}
            data.FrameAnnotations.from_audit({"timestamp_micros": 1}, row, np.eye(4), "fixture")
        frames = [frame(0, [label("real")])]
        cloud = np.array([[0, 0, 0], [20, 0, 0]])
        with self.assertRaisesRegex(ValueError, "unknown IDs"):
            data.make_instance_bundle(frames, [cloud], np.eye(4), ["made-up"])
        with self.assertRaisesRegex(ValueError, "Missing/invalid real LiDAR"):
            data.make_instance_bundle(frames, [np.empty((0, 6))], np.eye(4))
        with self.assertRaisesRegex(ValueError, "no unambiguous LiDAR"):
            data.make_instance_bundle(frames, [np.array([[20, 0, 0]])], np.eye(4), ["real"])
        with self.assertRaisesRegex(ValueError, "No genuine unambiguous"):
            data.make_instance_bundle(frames, [np.array([[20, 0, 0]])], np.eye(4))

    def test_zero_lidar_track_has_identity_no_template_points(self):
        frames = [frame(0, [label("observed"), label("unobserved", center=(20, 0, 0))])]
        bundle, report = data.make_instance_bundle(frames, [np.array([[0, 0, 0], [.1, 0, 0]])], np.eye(4))
        self.assertEqual(bundle["instance_ids"].tolist(), ["observed", "unobserved"])
        np.testing.assert_array_equal(bundle["instance_initialized"], [True, False])
        self.assertEqual(report["zero_lidar_instances"], ["unobserved"])
        self.assertEqual(len(bundle["local_xyz"]), 2)


class PreparedIOContracts(unittest.TestCase):
    def fixture(self, root):
        scene, audits = root / "source", root / "audit"
        scene.mkdir()
        audits.mkdir()
        for directory in ("pose", "timestamp", "velodyne", "calib", "image/image_0", "mask/mask_0/Annotations"):
            (scene / directory).mkdir(parents=True)
        rows = []
        for index in range(2):
            current = frame(index, [label("parked"), label("off_front", center=(8, 0, 0))])
            row = {"id": current.frame_id, "source_index": index, "timestamp_micros": current.timestamp_micros, "model_time": current.model_time}
            rows.append(row)
            data.write_json_new(audits / (current.frame_id + ".json"), {"frame_id": current.frame_id, "timestamp_micros": current.timestamp_micros,
                                                                       "laser_labels": [label("parked"), label("off_front", center=(8, 0, 0))]})
            np.savetxt(scene / "pose" / (current.frame_id + ".txt"), np.eye(4))
            (scene / "timestamp" / (current.frame_id + ".txt")).write_text(str(current.timestamp_micros))
            cloud = np.array([[0, 0, .2, 1, 0, -1], [.1, 0, .2, 1, 0, -1], [8, 0, 0, 1, 0, -1], [20, 0, 0, 1, 0, -1]], dtype=np.float32)
            cloud.tofile(scene / "velodyne" / (current.frame_id + ".bin"))
            calibration = "P0: 2 0 2 0 0 2 2 0 0 0 1 0\nTr_velo_to_cam_0: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            (scene / "calib" / (current.frame_id + ".txt")).write_text(calibration)
            Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8)).save(scene / "image/image_0" / (current.frame_id + ".png"))
            labels = np.zeros((4, 4), dtype=np.uint8)
            labels[2, 2] = 7
            Image.fromarray(labels).save(scene / "mask/mask_0/Annotations" / (current.frame_id + ".png"))
        data.write_json_new(scene / "preparation_manifest.json", {"frames": rows, "model_frame_interval": .02, "camera": "FRONT", "num_points": 4})
        np.savetxt(scene / "transform_matrix.txt", np.diag([.1, .1, .1, 1]))
        save_test_ply(scene / "points3d.ply", [[0, 0, .2], [20, 0, 0], [8, 0, 0], [20, 0, 0]], [-.01, -.01, .01, .01])
        return scene, audits, rows

    def test_end_to_end_cpu_reuses_seed_and_marks_missing_masks_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, audits, rows = self.fixture(root)
            original_ply = (scene / "points3d.ply").read_bytes()
            original_lidar = [(scene / "velodyne" / (row["id"] + ".bin")).read_bytes() for row in rows]
            selection = root / "selection.json"
            data.write_json_new(selection, {"status": "visually_reviewed", "scene": str(scene), "actors": [{"raw_track_id": "parked", "frames": [
                {"frame_id": row["id"], "reviewed": True, "review_note": "Unit-test actual supplied PNG ID", "visibility": "visible", "instance_ids": [7]} for row in rows]}]})
            output = root / "new_instances"
            report = prepare_isolated(scene, output, annotations=audits, selection_path=selection, strict_existing_scene=False)
            self.assertTrue(report["geometry_ready"])
            self.assertFalse(report["full_mask_supervision_ready"])
            self.assertFalse(report["Torch_imported"] or report["TF_imported"])
            self.assertEqual(report["masks"]["known_pairs"], 2)
            with np.load(output / "instances.npz", allow_pickle=False) as bundle:
                self.assertEqual(bundle["instance_ids"].tolist(), ["off_front", "parked"])
                np.testing.assert_array_equal(bundle["mask_known"], [[False, True], [False, True]])
                self.assertTrue(bundle["valid"].all())
                np.testing.assert_allclose(bundle["poses"][:, 1, :3, 3], 0)
            with np.load(output / "background.npz", allow_pickle=False) as background:
                np.testing.assert_array_equal(background["seed_indices"], [1, 3])
                np.testing.assert_allclose(background["xyz"], [[2, 0, 0], [2, 0, 0]])
                np.testing.assert_allclose(background["rgb"][:, 0], [2 / 255, 4 / 255])
            self.assertEqual((scene / "points3d.ply").read_bytes(), original_ply)
            self.assertEqual([(scene / "velodyne" / (row["id"] + ".bin")).read_bytes() for row in rows], original_lidar)
            trajectory = data.read_json(output / "trajectories.json")
            self.assertEqual(trajectory["trajectories"][1]["raw_track_id"], "parked")
            with self.assertRaisesRegex(ValueError, "No-clobber"):
                prepare_isolated(scene, output, annotations=audits, strict_existing_scene=False)

    def test_missing_real_sources_refuse_and_unreviewed_or_fake_masks_refuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, audits, rows = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "exactly one real annotation"):
                prepare_isolated(scene, root / "missing_annotations", strict_existing_scene=False)
            (scene / "transform_matrix.txt").unlink()
            with self.assertRaisesRegex(ValueError, "normalization transform missing"):
                prepare_isolated(scene, root / "missing_normalization", annotations=audits, strict_existing_scene=False)
            np.savetxt(scene / "transform_matrix.txt", np.eye(4))
            (audits / (rows[0]["id"] + ".json")).unlink()
            with self.assertRaisesRegex(ValueError, "track/box audit missing"):
                prepare_isolated(scene, root / "missing_audit_frame", annotations=audits, strict_existing_scene=False)
            labels = np.ones((4, 4), dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "mask review required"):
                data._actual_body_from_selection(scene, rows[0]["id"], labels, {"reviewed": False, "instance_ids": [1], "visibility": "visible"})
            with self.assertRaisesRegex(ValueError, "DEVA ID absent"):
                data._actual_body_from_selection(scene, rows[0]["id"], labels,
                                                   {"reviewed": True, "review_note": "fixture", "instance_ids": [7], "visibility": "visible"})


if __name__ == "__main__":
    unittest.main()
