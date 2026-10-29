"""CPU 数学验证；运行本文件可同时输出 split_math_validation.json。

python test_two_sided_split.py --validation-json split_math_validation.json
只核验矩、参数转换、异常与显式风险标记；不验证 native 渲染或语义归属。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

from two_sided_split import (
    METHOD, NumericalSplitError, cov_to_quat_wxyz_scale,
    covariance_from_quaternion_scale, covariance_to_quaternion_scale,
    quat_wxyz_scale_to_cov, quat_wxyz_to_rotation,
    raw_mixture_moments, split_gaussian, split_gaussian_two_sided,
)


_METRICS: dict[str, float] = {}


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return np.r_[math.cos(angle/2), axis*math.sin(angle/2)]


def _fixture(a=0.8, opacity=0.7):
    mean = np.array([2.3, -0.7, 1.1])
    quaternion = _axis_angle([1, 2, -0.5], 0.93)
    covariance = quat_wxyz_scale_to_cov(quaternion, [2.1, 0.7, 0.23])
    normal = np.array([0.4, -1.2, 0.7])
    offset = float(normal @ mean + a*math.sqrt(normal @ covariance @ normal))
    return split_gaussian_two_sided(mean, covariance, normal, offset,
                                    parent_id="spatialmixed_example", split_id="checked_plane",
                                    parent_opacity=opacity)


def _moment_errors(result):
    mass, first, second = raw_mixture_moments(result.children)
    expected_second = result.parent_covariance + np.outer(result.parent_mean, result.parent_mean)
    return {
        "zeroth_abs_error": abs(mass-1.0),
        "first_scaled_max_error": float(np.max(np.abs(first-result.parent_mean))) /
                                  max(1.0, float(np.max(np.abs(result.parent_mean)))),
        "second_scaled_max_error": float(np.max(np.abs(second-expected_second))) /
                                   max(1.0, float(np.max(np.abs(expected_second)))),
    }


class TwoSidedMomentTests(unittest.TestCase):
    def assertMoments(self, result, tolerance=3e-12):
        for name, error in _moment_errors(result).items():
            _METRICS[name] = max(_METRICS.get(name, 0.0), error)
            self.assertLess(error, tolerance, name)
        self.assertEqual(len(result.children), 2)
        for child in result.children:
            self.assertGreater(np.linalg.eigvalsh(child.covariance)[0], 0)

    def test_center_cut_has_known_half_normal_moments(self):
        result = split_gaussian_two_sided([0, 0, 0], np.eye(3), [1, 0, 0], 0,
                                         parent_id="center", parent_opacity=0.6)
        expected_shift = math.sqrt(2/math.pi)
        expected_covariance = np.diag([1-2/math.pi, 1, 1])
        np.testing.assert_allclose(result.mass_weights, [0.5, 0.5], atol=0, rtol=2e-15)
        np.testing.assert_allclose(result.children[0].mean, [-expected_shift, 0, 0], atol=2e-15)
        np.testing.assert_allclose(result.children[1].mean, [expected_shift, 0, 0], atol=2e-15)
        for child in result.children:
            np.testing.assert_allclose(child.covariance, expected_covariance, atol=2e-15)
        self.assertMoments(result)

    def test_rotated_anisotropic_offcenter_child_moments_independent_integrals(self):
        # 利用高斯积分的边界积分恒等式直接构造非归一化 0/1/2 阶矩，
        # 再从 X = mu + v Z + independent_residual 推到每一侧；不调用实现的 tail 代码。
        for a in (-3.5, -1.75, 0.0, 0.8, 3.5):
            with self.subTest(a=a):
                result = _fixture(a)
                phi = math.exp(-a*a/2)/math.sqrt(2*math.pi)
                normal = result.plane_normal
                v = result.parent_covariance @ normal / math.sqrt(normal @ result.parent_covariance @ normal)
                residual = result.parent_covariance - np.outer(v, v)
                for child, left in zip(result.children, (True, False)):
                    p = 0.5*math.erfc((-a if left else a)/math.sqrt(2))
                    first_z = -phi if left else phi
                    second_z = p + (-a*phi if left else a*phi)
                    first = p*result.parent_mean + v*first_z
                    mu = first/p
                    second = p*(residual + np.outer(result.parent_mean, result.parent_mean))
                    second += first_z*(np.outer(result.parent_mean, v)+np.outer(v, result.parent_mean))
                    second += second_z*np.outer(v, v)
                    np.testing.assert_allclose(child.mass_weight, p, rtol=2e-14)
                    np.testing.assert_allclose(child.mean, mu, rtol=3e-13, atol=3e-13)
                    np.testing.assert_allclose(child.covariance, second/p-np.outer(mu, mu),
                                               rtol=3e-11, atol=3e-12)
                self.assertLessEqual(normal @ result.children[0].mean, result.plane_offset)
                self.assertGreater(normal @ result.children[1].mean, result.plane_offset)
                self.assertMoments(result)

    def test_random_rotated_anisotropic_reconstructs_orders_zero_one_two(self):
        rng = np.random.default_rng(20261030)
        for index in range(100):
            quat = rng.normal(size=4)
            covariance = quat_wxyz_scale_to_cov(quat, np.exp(rng.uniform(-2.3, 2.3, 3)))
            mean, normal = rng.normal(size=(2, 3))
            a = rng.uniform(-5, 5)
            offset = normal @ mean + a*math.sqrt(normal @ covariance @ normal)
            result = split_gaussian_two_sided(mean, covariance, normal, offset,
                                             parent_id=f"random_{index}", parent_opacity=0.73)
            self.assertMoments(result)

    def test_plane_positive_rescaling_and_orientation_reversal(self):
        result = _fixture(-0.8)
        scaled = split_gaussian_two_sided(result.parent_mean, result.parent_covariance,
                                         result.plane_normal*23, result.plane_offset*23,
                                         parent_id="scaled", parent_opacity=result.parent_opacity)
        reversed_result = split_gaussian_two_sided(result.parent_mean, result.parent_covariance,
                                                  -result.plane_normal, -result.plane_offset,
                                                  parent_id="reversed", parent_opacity=result.parent_opacity)
        for first, second, reverse in zip(result.children, scaled.children, reversed_result.children[::-1]):
            for other in (second, reverse):
                np.testing.assert_allclose(first.mean, other.mean, atol=3e-14)
                np.testing.assert_allclose(first.covariance, other.covariance, atol=3e-14)
                self.assertAlmostEqual(first.mass_weight, other.mass_weight, places=14)

    def test_peak_opacity_and_normalized_mass_are_separate(self):
        result = _fixture(0.8, opacity=0.73)
        parent_volume = math.sqrt(np.linalg.det(result.parent_covariance))
        self.assertAlmostEqual(result.parent_reduced_mass, 0.73*parent_volume, places=14)
        sum_reduced = 0.0
        for child in result.children:
            child_volume = math.sqrt(np.linalg.det(child.covariance))
            candidate = 0.73*child.mass_weight*parent_volume/child_volume
            self.assertAlmostEqual(child.opacity_candidate, candidate, places=13)
            self.assertAlmostEqual(child.reduced_mass, result.parent_reduced_mass*child.mass_weight, places=14)
            self.assertAlmostEqual(child.opacity_candidate*child_volume, child.reduced_mass, places=14)
            sum_reduced += child.reduced_mass
        self.assertAlmostEqual(sum_reduced, result.parent_reduced_mass, places=14)

    def test_unsafe_peak_is_reported_without_clipping_or_covariance_changes(self):
        result = _fixture(1.0, opacity=0.99)
        safe_parent = _fixture(1.0, opacity=0.3)
        bulk = result.children[0]
        self.assertGreater(bulk.opacity_candidate, 1.0)
        self.assertFalse(bulk.opacity_candidate_safe)
        self.assertTrue(result.needs_review)
        self.assertIn("unsafe_peak_opacity_candidate_not_clipped", bulk.review_reasons)
        for high, low in zip(result.children, safe_parent.children):
            np.testing.assert_array_equal(high.mean, low.mean)
            np.testing.assert_array_equal(high.covariance, low.covariance)
            self.assertEqual(high.mass_weight, low.mass_weight)
        self.assertMoments(result)

    def test_tiny_tail_keeps_both_children_and_stable_variance(self):
        result = split_gaussian_two_sided([0, 0, 0], np.eye(3), [1, 0, 0], 12,
                                         parent_id="tiny_tail", parent_opacity=0.5)
        tail = result.children[1]
        p_reference = 0.5*math.erfc(12/math.sqrt(2))
        np.testing.assert_allclose(tail.mass_weight, p_reference, rtol=3e-14)
        self.assertGreater(tail.mass_weight, 0)
        self.assertTrue(tail.needs_review)
        self.assertIn("tiny_tail_mass", tail.review_reasons)
        self.assertGreater(tail.covariance[0, 0], 0)
        self.assertLess(tail.covariance[0, 0], 0.01)
        self.assertMoments(result)

    def test_continued_fraction_tail_matches_independent_numerical_quadrature(self):
        # 对 w=t*(Z-t) 做 Gauss-Legendre 数值积分，直接归一化 exp(-w-w^2/(2t^2))。
        # 不使用 CDF、Mills ratio、实现中的连分式或方差公式。
        nodes, weights = np.polynomial.legendre.leggauss(128)
        w = (nodes+1)*24
        weights = weights*24
        for t in (5.0, 8.0, 12.0, 40.0):
            with self.subTest(t=t):
                density = np.exp(-w-w*w/(2*t*t))
                integrals = [float(np.sum(weights*density*(w/t)**order))/t for order in range(3)]
                reference_mean = t+integrals[1]/integrals[0]
                reference_variance = integrals[2]/integrals[0]-(integrals[1]/integrals[0])**2
                reference_log_mass = -t*t/2-math.log(2*math.pi)/2+math.log(integrals[0])
                result = split_gaussian_two_sided([0, 0, 0], np.eye(3), [1, 0, 0], t,
                                                 parent_id=f"quadrature_{t}")
                tail = result.children[1]
                variance_error = abs(tail.covariance[0, 0]/reference_variance-1)
                _METRICS["tail_variance_quadrature_relative_error"] = max(
                    _METRICS.get("tail_variance_quadrature_relative_error", 0), variance_error)
                self.assertLess(variance_error, 3e-12)
                self.assertAlmostEqual(tail.mean[0], reference_mean, places=11)
                self.assertLess(abs(tail.log_mass_weight-reference_log_mass), 5e-12)

    def test_extreme_tail_underflow_keeps_child_and_log_mass_explicit(self):
        result = split_gaussian_two_sided([0, 0, 0], np.eye(3), [1, 0, 0], 40,
                                         parent_id="underflow", parent_opacity=0.5)
        tail = result.children[1]
        self.assertEqual(len(result.children), 2)
        self.assertEqual(tail.mass_weight, 0.0)
        self.assertLess(tail.log_mass_weight, -800)
        self.assertTrue(math.isfinite(tail.log_mass_weight))
        self.assertGreater(tail.mean[0], 40)
        self.assertGreater(tail.covariance[0, 0], 0)
        self.assertIn("mass_weight_underflow_log_mass_retained", tail.review_reasons)
        self.assertIn("opacity_candidate_underflow_log_candidate_retained", tail.review_reasons)
        self.assertMoments(result)

    def test_zero_opacity_preserves_geometric_split(self):
        result = _fixture(-0.8, opacity=0.0)
        self.assertEqual(result.parent_reduced_mass, 0)
        for child in result.children:
            self.assertGreater(child.mass_weight, 0)
            self.assertEqual(child.reduced_mass, 0)
            self.assertEqual(child.opacity_candidate, 0)
            self.assertEqual(child.log_reduced_mass, -math.inf)
        self.assertMoments(result)

    def test_lineage_and_scope_flags(self):
        result = _fixture()
        payload = result.to_dict()
        self.assertEqual(payload["method"], METHOD)
        self.assertTrue(payload["initialization_only"])
        self.assertFalse(payload["exact_pixel_alpha_conservation"])
        self.assertFalse(payload["semantic_boundary_discovery"])
        self.assertFalse(payload["complete_evsplitting_implementation"])
        self.assertEqual(payload["lineage"]["child_ids"],
                         ["spatialmixed_example/checked_plane/leq", "spatialmixed_example/checked_plane/gt"])
        for child in payload["children"]:
            self.assertEqual(child["parent_id"], "spatialmixed_example")
            self.assertIsNone(child["semantic_owner"])

    def test_native_probe_adapter_has_numeric_gate_and_never_clips(self):
        covariance = covariance_from_quaternion_scale([1, 0, 0, 0], [1, 1, 1])
        safe = split_gaussian([0, 0, 0], covariance, 0.7, [1, 0, 0], 0.8, parent_id=42)
        self.assertTrue(safe["native_safe"])
        self.assertEqual(safe["parent_id"], "42")
        self.assertEqual(safe["warnings"], [])
        unsafe = split_gaussian([0, 0, 0], covariance, 0.99, [1, 0, 0], 1, parent_id=42)
        self.assertFalse(unsafe["native_safe"])
        self.assertGreater(unsafe["children"][0]["opacity_candidate"], 1)
        self.assertEqual(len(unsafe["children"]), 2)
        self.assertTrue(unsafe["warnings"])
        q, scales = covariance_to_quaternion_scale(np.asarray(safe["children"][0]["covariance"]))
        np.testing.assert_allclose(covariance_from_quaternion_scale(q, scales),
                                   safe["children"][0]["covariance"], atol=3e-14)

    def test_json_payload_is_strict_even_for_log_zero_opacity(self):
        payload = _fixture(opacity=0).to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertIsNone(payload["log_parent_reduced_mass"])
        self.assertEqual(payload["nonfinite_values"]["log_parent_reduced_mass"], "-inf")
        for child in payload["children"]:
            self.assertIsNone(child["log_opacity_candidate"])
            self.assertEqual(child["nonfinite_values"]["log_opacity_candidate"], "-inf")


class QuaternionCovarianceTests(unittest.TestCase):
    def test_wxyz_known_quarter_turn(self):
        q = np.array([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
        expected_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(quat_wxyz_to_rotation(q), expected_rotation, atol=5e-16)
        np.testing.assert_allclose(quat_wxyz_scale_to_cov(q, [3, 2, 1]), np.diag([4, 9, 1]), atol=4e-15)

    def test_parent_and_children_quaternion_covariance_roundtrip(self):
        result = _fixture(-1.4)
        for covariance in [result.parent_covariance, *(child.covariance for child in result.children)]:
            q, scales = cov_to_quat_wxyz_scale(covariance)
            recovered = quat_wxyz_scale_to_cov(q, scales)
            error = float(np.max(np.abs(recovered-covariance)))/float(np.max(np.abs(covariance)))
            _METRICS["quat_cov_roundtrip_scaled_max_error"] = max(
                _METRICS.get("quat_cov_roundtrip_scaled_max_error", 0), error)
            np.testing.assert_allclose(recovered, covariance, rtol=3e-13, atol=3e-14)
            self.assertAlmostEqual(float(q@q), 1.0, places=14)
            self.assertGreater(np.linalg.det(quat_wxyz_to_rotation(q)), 0.999999999999)
            self.assertTrue(np.all(scales[:-1] >= scales[1:]))
            q_again, scales_again = cov_to_quat_wxyz_scale(covariance)
            np.testing.assert_array_equal(q, q_again)
            np.testing.assert_array_equal(scales, scales_again)

    def test_reference_preserves_equivalent_axis_order_and_quaternion_sign(self):
        q = -_axis_angle([0.1, 1, -0.2], math.pi+0.02)
        original_scales = np.array([0.2, 1.4, 0.6])
        covariance = quat_wxyz_scale_to_cov(q, original_scales)
        converted_q, converted_scales = cov_to_quat_wxyz_scale(covariance,
                                                            reference_quaternion_wxyz=q)
        self.assertGreater(float(converted_q@q), 1-3e-14)
        np.testing.assert_allclose(converted_scales, original_scales, atol=3e-14)
        np.testing.assert_allclose(quat_wxyz_scale_to_cov(converted_q, converted_scales), covariance, atol=3e-14)

    def test_repeated_eigenvalues_have_deterministic_basis(self):
        q, scales = cov_to_quat_wxyz_scale(np.eye(3)*4)
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-15)
        np.testing.assert_allclose(scales, [2, 2, 2], atol=1e-15)
        reference = _axis_angle([1, 2, -3], 1.1)
        q_ref, scales_ref = cov_to_quat_wxyz_scale(np.eye(3)*4,
                                                  reference_quaternion_wxyz=reference)
        self.assertGreater(float(q_ref@reference), 1-2e-14)
        np.testing.assert_allclose(quat_wxyz_scale_to_cov(q_ref, scales_ref), np.eye(3)*4, atol=4e-14)
        covariance = quat_wxyz_scale_to_cov(reference, [1, 2, 2])
        q_repeated, s_repeated = cov_to_quat_wxyz_scale(covariance)
        np.testing.assert_allclose(quat_wxyz_scale_to_cov(q_repeated, s_repeated), covariance, atol=3e-14)

    def test_reference_avoids_antipodal_jump_across_half_turn(self):
        reference = _axis_angle([0.3, 1, 0.2], math.pi-1e-6)
        next_q = _axis_angle([0.3, 1, 0.2], math.pi+1e-6)
        covariance = quat_wxyz_scale_to_cov(next_q, [1.5, 0.5, 0.2])
        q, _ = cov_to_quat_wxyz_scale(covariance, reference_quaternion_wxyz=reference)
        self.assertGreater(float(q@reference), 1-1e-10)

    def test_scaled_and_negated_quaternion_give_same_covariance(self):
        q = _axis_angle([0.3, 1, 0.2], 0.4)
        covariance = quat_wxyz_scale_to_cov(q, [2, 1, 0.5])
        np.testing.assert_allclose(quat_wxyz_scale_to_cov(-7*q, [2, 1, 0.5]), covariance, atol=2e-15)


class InvalidInputTests(unittest.TestCase):
    def split(self, **kwargs):
        arguments = dict(mean=[0, 0, 0], covariance=np.eye(3), plane_normal=[1, 0, 0],
                         plane_offset=0, parent_id="invalid")
        arguments.update(kwargs)
        return split_gaussian_two_sided(**arguments)

    def test_rejects_zero_nearzero_or_nonfinite_plane(self):
        for normal in ([0, 0, 0], [1e-15, 0, 0], [math.inf, 0, 0], [1, 0], [math.nan, 0, 0]):
            with self.subTest(normal=normal), self.assertRaises(ValueError):
                self.split(plane_normal=normal)
        for offset in (math.inf, math.nan):
            with self.assertRaises(ValueError):
                self.split(plane_offset=offset)

    def test_rejects_invalid_covariances_without_jitter(self):
        invalid = [np.zeros((3, 3)), np.diag([1, 1, -1]), np.diag([1, 1, 0]),
                   np.ones((2, 2)), np.diag([1, 1, math.nan]),
                   np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])]
        for covariance in invalid:
            with self.subTest(covariance=covariance.tolist()), self.assertRaises(ValueError):
                self.split(covariance=covariance)
        with self.assertRaises(NumericalSplitError):
            self.split(covariance=np.diag([1, 1, 1e-16]))

    def test_rejects_bad_mean_opacity_threshold_and_lineage(self):
        for mean in ([0, 0], [0, math.nan, 0]):
            with self.assertRaises(ValueError):
                self.split(mean=mean)
        for opacity in (-0.01, 1.01, math.nan):
            with self.assertRaises(ValueError):
                self.split(parent_opacity=opacity)
        for threshold in (0, 0.5, math.inf):
            with self.assertRaises(ValueError):
                self.split(tail_review_threshold=threshold)
        for identifier in ("", " ", None):
            with self.assertRaises(ValueError):
                self.split(parent_id=identifier)
        with self.assertRaises(ValueError):
            self.split(split_id="")
        for floor in (0, -1, math.nan):
            with self.assertRaises(ValueError):
                self.split(normal_min_norm=floor)

    def test_rejects_unrepresentable_split_instead_of_dropping_tail(self):
        with self.assertRaises(NumericalSplitError):
            self.split(plane_offset=1e155)
        with self.assertRaises(NumericalSplitError):
            self.split(plane_offset=1e8)

    def test_rejects_invalid_quaternion_or_scale(self):
        for quat in ([0, 0, 0, 0], [1e-15, 0, 0, 0], [1, 0, 0], [math.nan, 0, 0, 1]):
            with self.assertRaises(ValueError):
                quat_wxyz_scale_to_cov(quat, [1, 1, 1])
        for scales in ([1, 0, 1], [1, -1, 1], [1, math.inf, 1], [1, 1]):
            with self.assertRaises(ValueError):
                quat_wxyz_scale_to_cov([1, 0, 0, 0], scales)
        with self.assertRaises(ValueError):
            cov_to_quat_wxyz_scale(np.eye(3), reference_quaternion_wxyz=[0, 0, 0, 0])


def _summary(test_result):
    representative = {}
    for name, a, opacity in (("rotated_offcenter", 0.8, 0.7),
                             ("unsafe_opacity", 1.0, 0.99),
                             ("tiny_tail", 12.0, 0.5),
                             ("underflow_tail", 40.0, 0.5)):
        case = _fixture(a, opacity)
        representative[name] = {
            "standardized_offset": case.standardized_offset,
            "mass_weights": case.mass_weights.tolist(),
            "log_mass_weights": [child.log_mass_weight for child in case.children],
            "opacity_candidates": case.opacity_candidates.tolist(),
            "review_reasons": [list(child.review_reasons) for child in case.children],
            "reconstruction": _moment_errors(case),
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "status": "passed" if test_result.wasSuccessful() else "failed",
        "tests_run": test_result.testsRun,
        "failures": len(test_result.failures), "errors": len(test_result.errors),
        "skipped": len(test_result.skipped),
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "device": "CPU"},
        "scope": {
            "normalized_moment_orders_validated": [0, 1, 2],
            "rotated_anisotropic_random_cases": 100,
            "both_children_retained": True,
            "peak_opacity_formula": "parent_opacity * p_child * sqrt(det_parent / det_child)",
            "peak_opacity_clipping": False,
            "native_quaternion_order": "wxyz",
            "scale_convention": "positive_linear_standard_deviation",
            "complete_evsplitting_implementation": False,
            "native_renderer_validated": False,
            "exact_pixel_alpha_conservation": False,
            "semantic_boundary_discovery": False,
            "verified_real_3d_cut_plane_available": False,
            "r2_bulk_split_performed": False,
            "ssh_gpu_training_used": False,
        },
        "max_numerical_errors": _METRICS,
        "representative_cases": representative,
        "limitations": [
            "Moment-matched initialization only; child Gaussians retain cross-plane tails.",
            "Tiny tails are retained; float64 mass underflow is explicit and finite log-mass is retained.",
            "Condition number >= 1e14 is rejected rather than silently regularized.",
            "Unsafe peak opacity candidates are reported and never clipped.",
            "No semantic owner or real 3D plane is inferred; no automatic application to the R2 set.",
            "Native pixel alpha and compositing must be separately validated.",
        ],
        "background_reference": "https://orca.cardiff.ac.uk/174775/1/3680528.3687592.pdf",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-json", type=Path)
    arguments = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if arguments.validation_json is not None:
        payload = _summary(result)
        arguments.validation_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
                                             encoding="utf-8")
        print(f"Validation JSON: {arguments.validation_json.resolve()}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
