"""view-aware二维监督的支持、坐标及源级带宽合同。"""
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

from teacher_supervision_cpu import build_supervision
from teacher_supervision_view_cpu import (build_view_supervision, fixed_view_spec,
                                         mapped_resample, validate_view)


def domains(shape):
    out = {k: np.zeros(shape, bool) for k in ("A", "C", "S", "K", "protected")}
    out["C"][:] = True
    out["teacher_write"] = np.ones(shape, bool)
    return out


def build(raw, observed, masks, window, output_wh, valid=None, view_name=None):
    h, w = observed.shape[:2]
    x0, y0, x1, y1 = window
    matrix = [[(x1-x0)/raw.shape[1], 0, x0], [0, (y1-y0)/raw.shape[0], y0], [0, 0, 1]]
    return build_view_supervision(raw, observed, masks,
        source_model_valid=np.ones(raw.shape[:2], bool) if valid is None else valid,
        native_window_xyxy=window, native_from_model=matrix, output_wh=output_wh,
        intrinsics=[[w, 0, w/2], [0, h, h/2], [0, 0, 1]], camera_wh=(w, h),
        camera_convention="half_pixel", source_kind="lossless_raw_model", view_name=view_name)


class ViewSupervisionContractTest(unittest.TestCase):
    def test_window_offset_and_crop_outside_never_gets_C(self):
        raw = np.repeat(np.array([100, 150, 200], np.uint8)[:, None, None], 4, axis=1)
        raw = np.repeat(raw, 3, axis=2)
        observed = np.full((8, 8, 3), 17, np.uint8)
        masks = domains((8, 8))
        masks["C"][:2] = False
        masks["A"][:2] = True
        packet, report = build(raw, observed, masks, (0, 2, 8, 8), (4, 4))
        self.assertFalse(packet["source_RGB_valid"][0].any())
        self.assertFalse(packet["C"][0].any())
        self.assertTrue(packet["A"][0].all())
        self.assertTrue(packet["C"][1:].all())
        np.testing.assert_array_equal(packet["target_C_srgb"][:, 0], 0)
        np.testing.assert_array_equal(packet["target_C_srgb"][:, 1:], raw.transpose(2, 0, 1).astype(np.float32) / 255)
        np.testing.assert_allclose(packet["target_A_srgb"], 17/255, atol=1e-8, rtol=0)
        self.assertFalse(report["training_ready"])
        self.assertTrue(report["no_new_3d_high_frequency_authorization"])

    def test_partial_crop_coverage_and_invalid_source_reject_full_filter(self):
        raw = np.full((3, 4, 3), 200, np.uint8)
        observed = np.full((8, 8, 3), 90, np.uint8)
        masks = domains((8, 8))
        masks["C"][:2] = False
        valid = np.ones((3, 4), bool)
        valid[2, 2] = False
        packet, _ = build(raw, observed, masks, (0, 2, 8, 8), (2, 2), valid)
        # 上排虽与crop有一半面积重叠，仍不得获得目标RGB或监督。
        self.assertFalse(packet["source_RGB_valid"][0].any())
        self.assertFalse(packet["C"][0].any())
        # 下排右目标涉及四个非零源支持，其中一个invalid即可整像素拒绝。
        self.assertFalse(packet["source_RGB_valid"][1, 1])
        self.assertFalse(packet["C"][1, 1])
        np.testing.assert_array_equal(packet["target_C_srgb"][:, 1, 1], 0)
        self.assertTrue(packet["source_RGB_valid"][1, 0])
        self.assertTrue(packet["C"][1, 0])

    def test_native_footprint_only_is_insufficient_and_K_any_excludes(self):
        raw = np.full((8, 8, 3), 120, np.uint8)
        observed = np.full((12, 12, 3), 80, np.uint8)
        masks = domains((12, 12))
        masks["K"][1, 2] = True
        masks["C"][1, 2] = False
        masks["teacher_write"][1, 2] = False
        packet, _ = build(raw, observed, masks, (0, 0, 12, 12), (5, 5))
        self.assertTrue(packet["native_footprint_C"][1, 1])
        self.assertFalse(packet["K"][1, 1])
        self.assertFalse(packet["filter_support_C"][1, 1])
        self.assertFalse(packet["C"][1, 1])
        self.assertFalse((packet["C"] & packet["K"]).any())
        self.assertGreater(int(packet["K"].sum()), 0)

    def test_noninteger_mapped_areas_have_expected_support_and_mean(self):
        row = np.array([20, 40, 80], np.float32)
        source = np.broadcast_to(row[None, :, None], (2, 3, 1))
        # 源单元位于x=[2,4],[4,6],[6,8]，输出五格各宽1.6。
        out = mapped_resample(source, (5, 5), (2, 2, 8, 8), (0, 0, 8, 8))
        valid = mapped_resample(np.ones((2, 3), bool), (5, 5), (2, 2, 8, 8), (0, 0, 8, 8), mask_mode="all")
        self.assertFalse(valid[:, 0].any())
        self.assertFalse(valid[:, 1].any())
        self.assertTrue(valid[2:, 2:].all())
        np.testing.assert_allclose(out[3, :, 0], [0, 15, 30, 50, 80], rtol=0, atol=1e-6)

    def test_bad_mapping_fixed_crop_size_and_missing_valid_reject(self):
        spec = fixed_view_spec("crop768")
        validate_view((768, 272), (1920, 1280), spec["native_window_xyxy"], spec["native_from_model"], view_name="crop768")
        bad = np.array(spec["native_from_model"], float)
        bad[1, 2] = 0
        with self.assertRaisesRegex(ValueError, "native_from_model"):
            validate_view((768, 272), (1920, 1280), spec["native_window_xyxy"], bad, view_name="crop768")
        with self.assertRaisesRegex(ValueError, "大小"):
            validate_view((768, 512), (1920, 1280), spec["native_window_xyxy"],
                          [[2.5, 0, 0], [0, 680/512, 600], [0, 0, 1]], view_name="crop768")
        raw = np.full((3, 4, 3), 120, np.uint8)
        with self.assertRaisesRegex(ValueError, "source_model_valid"):
            build(raw, np.zeros((8, 8, 3), np.uint8), domains((8, 8)), (0, 2, 8, 8), (4, 4), np.ones((8, 8), bool))
        with self.assertRaisesRegex(ValueError, "放大"):
            build(raw, np.zeros((8, 8, 3), np.uint8), domains((8, 8)), (0, 2, 8, 8), (8, 8))

    def test_full_frame_old_interface_is_exact_without_old_changes(self):
        rng = np.random.default_rng(101)
        raw = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        observed = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
        masks = domains((12, 12))
        first, _ = build(raw, observed, masks, (0, 0, 12, 12), (5, 5))
        second, _ = build_supervision(raw, observed, masks, output_wh=(5, 5),
            intrinsics=[[12, 0, 6], [0, 12, 6], [0, 0, 1]], camera_wh=(12, 12),
            camera_convention="half_pixel", source_kind="lossless_raw_model")
        for key in second:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

    def test_1152_native_teacher_identity_and_crop_cannot_upsample_to1152(self):
        rng = np.random.default_rng(202)
        raw = rng.integers(0, 256, (768, 1152, 3), dtype=np.uint8)
        observed = np.full((1280, 1920, 3), 60, np.uint8)
        masks = domains(observed.shape[:2])
        packet, report = build(raw, observed, masks, (0, 0, 1920, 1280), (1152, 768), view_name="full1152")
        np.testing.assert_array_equal(packet["target_C_srgb"], raw.transpose(2, 0, 1).astype(np.float32) / 255)
        self.assertTrue(packet["source_RGB_valid"].all())
        self.assertEqual(report["output_wh"], [1152, 768])
        crop = np.zeros((272, 768, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "放大"):
            build(crop, observed, masks, (0, 600, 1920, 1280), (1152, 768), view_name="crop768")

    def test_cli_preserves_exact_source_PNG_and_mapping(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            raw = np.full((3, 4, 3), 177, np.uint8)
            observed = np.full((8, 8, 3), 80, np.uint8)
            Image.fromarray(raw).save(root / "raw.png")
            Image.fromarray(observed).save(root / "observed.png")
            Image.fromarray(np.full((3, 4), 255, np.uint8)).save(root / "valid.png")
            masks = domains((8, 8))
            masks["C"][:2] = False
            np.savez_compressed(root / "domains.npz", **masks)
            config = dict(raw_model=str(root / "raw.png"), observation=str(root / "observed.png"),
                          source_model_valid=str(root / "valid.png"), native_domains=str(root / "domains.npz"),
                          source_kind="lossless_raw_model", output_wh=[4, 4],
                          camera=dict(intrinsics=[[8, 0, 4], [0, 8, 4], [0, 0, 1]], camera_wh=[8, 8], convention="half_pixel"),
                          view=dict(name=None, native_window_xyxy=[0, 2, 8, 8], native_from_model=[[2, 0, 0], [0, 2, 2], [0, 0, 1]]))
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            process = subprocess.run([sys.executable, str(Path(__file__).with_name("teacher_supervision_view_cpu.py")),
                                      "--config", str(root / "config.json"), "--output", str(root / "output")],
                                     capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual((root / "raw.png").read_bytes(), (root / "output/source/raw_model.png").read_bytes())
            mapping = json.loads((root / "output/source/view_mapping.json").read_text(encoding="utf-8"))
            self.assertEqual(mapping["view"]["native_window_xyxy"], [0, 2, 8, 8])
            with np.load(root / "output/supervision.npz") as z:
                self.assertFalse(z["C"][0].any())
                self.assertFalse(z["source_RGB_valid"][0].any())


if __name__ == "__main__":
    unittest.main()
