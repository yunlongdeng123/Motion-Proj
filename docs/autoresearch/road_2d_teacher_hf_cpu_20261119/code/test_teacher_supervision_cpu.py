"""有效支持、相机坐标与预乘损失的 CPU 合同测试。"""
import unittest

import numpy as np

from teacher_supervision_cpu import (area_rgb, build_supervision, footprint_mask,
                                     map_pixel_indices, premult_l1_numpy, scale_intrinsics)


def domains(shape, role="C"):
    out = {k: np.zeros(shape, bool) for k in ("A", "C", "S", "K", "protected")}
    out[role][:] = True
    out["teacher_write"] = np.ones(shape, bool)
    return out


def build(raw, observed, masks, output_wh):
    return build_supervision(raw, observed, masks, output_wh=output_wh,
                             intrinsics=[[8, 0, 6], [0, 8, 6], [0, 0, 1]],
                             camera_wh=(observed.shape[1], observed.shape[0]),
                             camera_convention="half_pixel", source_kind="lossless_raw_model")


class SupervisionContractTest(unittest.TestCase):
    def test_identity_and_exact_area_mean(self):
        rng = np.random.default_rng(101)
        raw = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        np.testing.assert_array_equal(area_rgb(raw, (8, 8)), raw.astype(np.float32) / 255)
        np.testing.assert_allclose(area_rgb(raw, (4, 4)),
                                   (raw.astype(np.float32) / 255).reshape(4, 2, 4, 2, 3).mean((1, 3)),
                                   rtol=0, atol=6e-8)

    def test_noninteger_area_footprint(self):
        raw = np.broadcast_to(np.arange(8, dtype=np.uint8)[None, :, None], (8, 8, 3)).copy()
        expected = np.array([0.375, 1.875, 3.5, 5.125, 6.625]) / 255
        np.testing.assert_allclose(area_rgb(raw, (5, 5))[0, :, 0], expected, rtol=0, atol=2e-9)

    def test_filter_support_rejects_native_footprint_only_permission(self):
        raw = np.full((8, 8, 3), 120, np.uint8)
        observed = np.full((12, 12, 3), 80, np.uint8)
        masks = domains((12, 12))
        masks["K"][1, 2] = True
        masks["C"][1, 2] = False
        masks["teacher_write"][1, 2] = False
        packet, report = build(raw, observed, masks, (5, 5))
        self.assertTrue(packet["native_footprint_C"][1, 1])
        self.assertFalse(packet["K"][1, 1])
        self.assertFalse(packet["filter_support_C"][1, 1])
        self.assertFalse(packet["C"][1, 1])
        self.assertGreater(report["counts"]["C"]["filter_support_excluded_pixels"], 0)
        self.assertFalse(report["training_ready"])

    def test_protected_any_and_native_all_have_no_fractional_rounding_leak(self):
        mask = np.zeros((12, 12), bool)
        mask[1, 2] = True
        any_mask = footprint_mask(mask, (5, 5), "any")
        all_inverse = footprint_mask(~mask, (5, 5), "all")
        np.testing.assert_array_equal(any_mask, ~all_inverse)
        self.assertEqual(int(any_mask.sum()), 2)

    def test_A_uses_real_observation_and_permission_violations_raise(self):
        raw = np.full((8, 8, 3), 10, np.uint8)
        observed = np.full((12, 12, 3), 220, np.uint8)
        masks = domains((12, 12), "A")
        masks["teacher_write"][:] = False
        packet, _ = build(raw, observed, masks, (5, 5))
        self.assertTrue(packet["A"].all())
        np.testing.assert_allclose(packet["target_A_srgb"], 220 / 255, rtol=0, atol=1e-7)
        np.testing.assert_allclose(packet["target_C_srgb"], 10 / 255, rtol=0, atol=1e-7)
        masks["A"][:] = False
        masks["C"][:] = True
        with self.assertRaisesRegex(ValueError, "写许可"):
            build(raw, observed, masks, (5, 5))
        with self.assertRaisesRegex(ValueError, "放大"):
            build(raw, observed, domains((12, 12)), (12, 12))

    def test_camera_scaling_preserves_rays_in_both_conventions(self):
        k = np.array([[1000, 0.2, 961], [0, 1010, 641], [0, 0, 1.]])
        xy = np.array([[0, 0], [192, 771], [1919, 1279]], dtype=float)
        dest_xy = map_pixel_indices(xy, (1920, 1280), (768, 512))
        for convention, offset in (("half_pixel", .5), ("integer_center", 0.0)):
            out_k = scale_intrinsics(k, (1920, 1280), (768, 512), convention)
            source_rays = np.c_[xy + offset, np.ones(len(xy))] @ np.linalg.inv(k).T
            out_rays = np.c_[dest_xy + offset, np.ones(len(xy))] @ np.linalg.inv(out_k).T
            np.testing.assert_allclose(source_rays, out_rays, rtol=0, atol=3e-16)
        integer = scale_intrinsics(k, (1920, 1280), (480, 320), "integer_center")
        self.assertAlmostEqual(integer[0, 2], .25 * (961 + .5) - .5)
        with self.assertRaises(ValueError):
            scale_intrinsics(k, (1920, 1280), (480, 320), "guessed")

    def test_premult_loss_does_not_depremultiply_and_empty_domain_is_explicit(self):
        target = np.ones((3, 2, 2), np.float32)
        alpha = np.full((2, 2), .01, np.float32)
        mask = np.ones((2, 2), bool)
        self.assertAlmostEqual(premult_l1_numpy(np.zeros_like(target), alpha, target, mask), .01)
        self.assertEqual(premult_l1_numpy(alpha[None] * target, alpha, target, mask), 0)
        with self.assertRaisesRegex(ValueError, "为空"):
            premult_l1_numpy(target, alpha, target, np.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
