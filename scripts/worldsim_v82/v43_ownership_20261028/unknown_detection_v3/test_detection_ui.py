"""独立合成fixture的接线测试；不写入共享实验目录。"""

import copy
import importlib.util
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("install_detection_ui", HERE / "install_detection_ui.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def synthetic_png(path):
    def chunk(name, value):
        return struct.pack(">I", len(value)) + name + value + struct.pack(">I", zlib.crc32(name + value))

    pixels = (b"\x00" + b"\x18\x28\x38" * 480) * 320
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 480, 320, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic_target_detection_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scene = self.root / "synthetic_scene"
        self.scene.mkdir()
        self.context, self.track = "synthetic_scene", "synthetic_target"
        data = dict(context=self.context, target_track=self.track, frames={"109": {}, "110": {}}, groups=[dict(id="historical_group", frames={})])
        (self.scene / "scene.js").write_text("window.SCENE_DATA=" + json.dumps(data) + ";", encoding="utf-8")
        corners = [[n & 1, (n >> 1) & 1, (n >> 2) & 1] for n in range(8)]
        self.state = dict(supported=True, static_visible=True, world_time=0, camera=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], objects=[dict(key="target", cloud="v43:synthetic_actor", corners_local=corners), dict(key="another", cloud="dynamic", corners_local=corners)])
        geometry = {"motion_no_edit": {"109": self.state, "110": self.state}, "reconstruction_30000": {"109": self.state, "110": self.state}}
        (self.scene / "edit_geometry.js").write_text("window.EDIT_GEOMETRY=" + json.dumps(dict(groups=geometry)) + ";", encoding="utf-8")
        dump(self.scene / "v43_groups_data.json", dict(groups=[dict(id="historical_v43", frames={})], geometry={"v43_actor_10000": {"109": self.state, "110": self.state}}, clouds={"v43:synthetic_actor": {"synthetic": True}}))
        (self.scene / "index.html").write_text('<html><body><div id="historical">原历史面板</div><script src="../viewer.js?v=old"></script></body></html>', encoding="utf-8")
        self.shared = self.root / "index.html"
        self.shared.write_text('<html><main id="workspace"><p>原场景入口</p></main></html>', encoding="utf-8")
        synthetic_png(self.root / "109.png")
        self.delivery = dict(schema="target-detection-ui-v1", context=self.context, target_track=self.track, static_total=918850, source_checkpoint="synthetic_frozen.pt", checkpoint_frozen=True, geometry_base_group="v43_actor_10000", actor=dict(cloud="v43:synthetic_actor", role_default=1, row_namespace="synthetic_actor_namespace"), globalsource_frames=[1, 2, 3], frames={"109": dict(source_raw=109, evidence=dict(ids=[10, 20, 30, 40, 50], static_role=[0, 1, 2, 3, 4], bg_protected=[1, 0, 0, 0, 1], shadow_effect=[0, 0, 0, 1, 1]), images=dict(baseline="109.png", remove_car="109.png", remove_u="109.png"), render_receipts={key: dict(status="complete", renderer="native_gaussian", source_raw=109) for key in ("baseline", "remove_car")})})
        self.path = self.root / "delivery.json"
        dump(self.path, self.delivery)

    def install(self):
        return installer.build(self.path, self.scene, self.shared)

    def manifest(self):
        return installer.read_assignment(self.scene / "target_detection_data.js", "TARGET_DETECTION_DELIVERY")

    def test_native_receipts_pending_and_role_specific_geometry(self):
        receipt = self.install()
        manifest = self.manifest()
        self.assertEqual(receipt["rendered"]["remove_car"], dict(complete=1, pending=1))
        self.assertEqual(receipt["rendered"]["remove_u"], dict(complete=0, pending=2))
        self.assertNotIn("remove_u", manifest["frames"]["109"]["images"])
        self.assertNotIn("baseline", manifest["frames"]["110"]["images"])
        geometry = manifest["geometry"]
        car = geometry[installer.GROUP_PREFIX + "remove_car"]["109"]
        u = geometry[installer.GROUP_PREFIX + "remove_u"]["109"]
        both = geometry[installer.GROUP_PREFIX + "remove_car_u"]["109"]
        self.assertEqual(car["static_suppressed_rows"], [20])
        self.assertEqual(u["static_suppressed_rows"], [30, 40, 50])
        self.assertEqual(both["static_suppressed_rows"], [20, 30, 40, 50])
        self.assertEqual([o["cloud"] for o in car["objects"]], ["dynamic"])
        self.assertEqual([o["cloud"] for o in u["objects"]], ["v43:synthetic_actor", "dynamic"])
        self.assertEqual(len(car["reference_boxes"]), 1)
        self.assertFalse(geometry[installer.GROUP_PREFIX + "baseline"]["110"]["supported"])
        self.assertIn("原历史面板", (self.scene / "index.html").read_text(encoding="utf-8"))
        self.assertEqual(installer.read_json(self.scene / "v43_groups_data.json")["groups"][0]["id"], "historical_v43")

    def test_reinstallation_is_idempotent_and_preserves_history(self):
        self.install()
        self.install()
        html = (self.scene / "index.html").read_text(encoding="utf-8")
        self.assertEqual(html.count('src="target_detection_data.js?'), 1)
        self.assertEqual(html.count('src="../scene_target_detection.js?'), 1)
        self.assertEqual(html.count('src="../viewer.js?v=old"'), 1)
        self.assertEqual(self.shared.read_text(encoding="utf-8").count("target-detection-link-start"), 1)

    def test_wrong_raw_is_rejected(self):
        self.delivery["frames"]["109"]["render_receipts"]["baseline"]["source_raw"] = 110
        dump(self.path, self.delivery)
        with self.assertRaisesRegex(ValueError, "源raw不匹配"):
            self.install()

    def test_static_namespace_and_role0_cannot_be_deleted(self):
        bad = copy.deepcopy(self.delivery)
        bad["frames"]["109"]["evidence"]["ids"] = [10, 20, 918850, 40, 50]
        dump(self.path, bad)
        with self.assertRaisesRegex(ValueError, "原static行号"):
            self.install()
        bad = copy.deepcopy(self.delivery)
        bad["removal_roles"] = dict(remove_u=[0, 2])
        dump(self.path, bad)
        with self.assertRaisesRegex(ValueError, "不能删除本目标外role0"):
            self.install()

    def test_shadow_effect_is_an_independent_boolean_array(self):
        self.delivery["frames"]["109"]["evidence"]["shadow_effect"] = [0, 0, 0, 1, 2]
        dump(self.path, self.delivery)
        with self.assertRaisesRegex(ValueError, "shadow_effect 必须是独立布尔量"):
            self.install()
        self.delivery["frames"]["109"]["evidence"]["shadow_effect"] = [0, 0, 1]
        dump(self.path, self.delivery)
        with self.assertRaisesRegex(ValueError, "shadow_effect 与 ids 长度不符"):
            self.install()

    def test_pixel_shadow_effect_preserves_body_role(self):
        evidence = copy.deepcopy(self.delivery["frames"]["109"]["evidence"])
        evidence["pixel_role"] = [0] * (480 * 320)
        evidence["pixel_role"][10] = 1
        evidence["pixel_shadow_effect"] = [0] * (480 * 320)
        evidence["pixel_shadow_effect"][10] = True
        result = installer.validate_evidence(evidence, 918850, 109)
        self.assertEqual(result["pixel_role"][10], 1)
        self.assertIs(result["pixel_shadow_effect"][10], True)
        evidence["pixel_shadow_effect"][10] = 2
        with self.assertRaisesRegex(ValueError, "pixel_shadow_effect 必须为480×320独立布尔/u8"):
            installer.validate_evidence(evidence, 918850, 109)
        evidence["pixel_shadow_effect"] = [0]
        with self.assertRaisesRegex(ValueError, "pixel_shadow_effect 必须为480×320独立布尔/u8"):
            installer.validate_evidence(evidence, 918850, 109)


if __name__ == "__main__":
    unittest.main()
