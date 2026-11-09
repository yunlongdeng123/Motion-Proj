"""CPU协议测试：toy compositor不代表真实CUDA/native RGB验收。"""
from __future__ import annotations
from types import SimpleNamespace
import unittest
import torch
from v44_scene import V44Scene, RowProvider, NativeOps, GROUPS, FEATURE_CHANNELS, BRANCH_ALPHA_NAMES


class Model:
    def __init__(self, depths, colors, precomputed=False):
        self._xyz = torch.tensor([[0., 0., z] for z in depths])
        self._rgb = torch.tensor(colors, dtype=torch.float32).reshape(-1, 3)
        self._sh = ((self._rgb - .5) / .28209479177387814)[:, None, :]
        self._opacity = torch.full((len(depths), 1), .25)
        self._mask = torch.ones(len(depths), dtype=torch.bool)
        self.active_sh_degree, self.precomputed = 0, precomputed

    @property
    def get_xyz(self):
        return self._xyz


class Camera:
    camera_center = torch.zeros(3)

    def get_world_directions(self, training):
        assert training is False
        return torch.tensor([[[0.]], [[0.]], [[1.]]])


class Environment:
    def __init__(self):
        self.base = torch.tensor([.1, .2, .3])

    def __call__(self, directions):
        return directions * 0 + self.base


class ToyNative:
    """只有字段/调用顺序和线性alpha协议；没有Gaussian/CUDA实现。"""
    def __init__(self):
        self.prepare_calls, self.raster_calls, self.seen = [], 0, []
        self.ops = NativeOps(self.prepare, self.sh, self.merge, self.raster)

    def prepare(self, camera, model, pipe, dynamic=False):
        self.prepare_calls.append((model, dynamic))
        n = len(model.get_xyz)
        return dict(means3D=model._xyz, shs=None if model.precomputed else model._sh,
            colors_precomp=model._rgb if model.precomputed else None,
            features=torch.stack((model._xyz[:, 2], torch.ones(n)), 1),
            opacities=model._opacity, scales=torch.ones(n, 3), rotations=torch.ones(n, 4),
            cov3D_precomp=None, mask=model._mask)

    @staticmethod
    def sh(degree, coefficients, direction):
        assert degree == 0 and direction.shape[-1] == 3
        return coefficients[:, :, 0] * .28209479177387814

    @staticmethod
    def merge(left, right):
        assert set(left) == set(right)
        return {k: None if left[k] is None else torch.cat((left[k], right[k]), 0) for k in left}

    def raster(self, camera, pipe, degree, values, background):
        self.raster_calls += 1
        self.seen.append({k: v.clone() if torch.is_tensor(v) else v for k, v in values.items()})
        rgb, feature = torch.zeros(3), torch.zeros(values["features"].shape[1])
        transmittance = torch.tensor(1.)
        for row in torch.argsort(values["means3D"][:, 2], stable=True):
            if values["mask"][row]:
                weight = transmittance * values["opacities"][row, 0]
                rgb = rgb + weight * values["colors_precomp"][row]
                feature = feature + weight * values["features"][row]
                transmittance = transmittance * (1 - values["opacities"][row, 0])
        rgb = rgb + transmittance * background
        return dict(rgb=rgb[:, None, None], depth_sum=feature[:1, None, None],
            alpha=feature[1:2, None, None], extra=feature[2:, None, None], radii=values["mask"].float())


class Surface:
    def __init__(self, backend, model, missing_keys=()):
        self.backend, self.model, self.missing_keys, self.calls = backend, model, set(missing_keys), []

    def rows(self, frame_key, camera, pipe):
        self.calls.append((frame_key, camera, pipe))
        return None if frame_key in self.missing_keys else self.backend.prepare(camera, self.model, pipe)


class SceneProtocols(unittest.TestCase):
    def setUp(self):
        self.backend = ToyNative()
        self.static = Model([1., 3., 5.], [[.8, .1, .2], [.1, .8, .3], [.2, .3, .8]])
        self.dynamic = Model([2., 4.], [[.2, .7, .2], [.7, .2, .6]], precomputed=True)
        self.env, self.pipe = Environment(), SimpleNamespace(including_dynamic=True, adding=False)
        self.source = {"static": torch.tensor([0, 0, 0]), "dynamic": torch.tensor([91, 91])}
        self.semantic = {"static": torch.tensor([47, 91, -1]), "dynamic": torch.tensor([91, -1])}
        self.camera_keys = []

    def camera(self, frame_key):
        self.camera_keys.append(frame_key)
        return Camera()

    def scene(self, **kwargs):
        return V44Scene(self.static, self.dynamic, self.env, self.pipe, self.camera,
            source_owner_ids=self.source, semantic_owner_ids=self.semantic,
            native_ops=self.backend.ops, **kwargs)

    def test_cpu_keep_protocol_matches_unmodified_toy_native_rgb_and_env(self):
        scene, cam = self.scene(), Camera()
        values = []
        for model, dynamic in ((self.static, False), (self.dynamic, True)):
            rows = self.backend.prepare(cam, model, self.pipe, dynamic=dynamic)
            if rows["colors_precomp"] is None:
                direction = torch.nn.functional.normalize(rows["means3D"] - cam.camera_center, dim=1)
                rows = dict(rows, colors_precomp=(self.backend.sh(0, rows["shs"].transpose(1, 2), direction) + .5).clamp_min(0))
            rows = dict(rows, shs=None, means2D=torch.zeros_like(rows["means3D"]))
            values.append(rows)
        reference = self.backend.raster(cam, self.pipe, 0, self.backend.merge(*values), torch.zeros(3))
        expected = reference["rgb"] + (1 - reference["alpha"]) * self.env.base[:, None, None]
        before = self.backend.raster_calls
        result = scene.render(("arbitrary_frame", 4321))
        self.assertEqual(self.backend.raster_calls - before, 1)
        torch.testing.assert_close(result["rgb"], expected, rtol=0, atol=0)
        self.assertEqual(self.camera_keys[-1], ("arbitrary_frame", 4321))
        scene.assert_unchanged()

    def test_delete_filters_static_and_dynamic_semantic_owners_and_keeps_unknown(self):
        scene = self.scene()
        result = scene.render("frame-x", delete_actor_ids=(91,))
        self.assertEqual(self.backend.seen[-1]["mask"].tolist(), [True, False, True, False, True])
        self.assertEqual(result["row_owners"].tolist(), [47, 91, -1, 91, -1])
        self.assertEqual(self.dynamic._mask.tolist(), [True, True])
        with self.assertRaises(ValueError):
            scene.render("frame-x", delete_actor_ids=(-1,))

    def test_focus_changes_diagnostics_only_and_groups_conserve_alpha(self):
        scene = self.scene()
        keep = scene.render("same", focus_actor_ids=(91,))
        other = scene.render("same", focus_actor_ids=(47,))
        torch.testing.assert_close(keep["rgb"], other["rgb"], rtol=0, atol=0)
        torch.testing.assert_close(keep["row_owners"], other["row_owners"])
        self.assertNotEqual(float(keep["diagnostics"]["target"]), float(other["diagnostics"]["target"]))
        self.assertEqual(tuple(keep["diagnostics"]), GROUPS)
        torch.testing.assert_close(sum(keep["diagnostics"].values()), keep["alpha"])

    def test_semantic_unknown_never_automatically_falls_back_to_source_owner(self):
        scene = self.scene()
        scene.render("k", delete_actor_ids=(91,))
        self.assertTrue(self.backend.seen[-1]["mask"][-1])  # source91、semantic-1
        self.semantic["static"].fill_(1234)                 # 外部输入不改变已冻结数组
        self.assertEqual(scene.render("k")["row_owners"].tolist(), [47, 91, -1, 91, -1])

    def test_row_providers_merge_once_and_handoff_is_query_independent(self):
        asset = Model([6.], [[.4, .5, .6]], precomputed=True)
        surface = Surface(self.backend, asset)
        provider = RowProvider(surface.rows, 91, retire_static=torch.tensor([False, True, False]),
            retire_dynamic=torch.tensor([True, False]), state_objects=(asset,))
        scene = self.scene(row_providers=(provider,))
        before = self.backend.raster_calls
        keep = scene.render(901)
        expected_rows = torch.cat((self.static._xyz, self.dynamic._xyz, asset._xyz))
        torch.testing.assert_close(self.backend.seen[-1]["means3D"], expected_rows)
        self.assertEqual(self.backend.seen[-1]["mask"].tolist(), [True, False, True, False, True, True])
        deleted = scene.render(901, delete_actor_ids=(91,))
        self.assertEqual(self.backend.seen[-1]["mask"].tolist(), [True, False, True, False, True, False])
        self.assertEqual(keep["coverage"], deleted["coverage"])
        self.assertEqual(self.backend.raster_calls - before, 2)
        self.assertEqual([call[0] for call in surface.calls], [901, 901])

    def test_missing_provider_falls_back_native_for_keep_and_delete(self):
        asset = Model([6.], [[.4, .5, .6]], precomputed=True)
        surface = Surface(self.backend, asset, missing_keys=("gap",))
        provider = RowProvider(surface.rows, 91, retire_static=torch.tensor([False, True, False]))
        scene = self.scene(row_providers=(provider,))
        keep = scene.render("gap")
        self.assertEqual(self.backend.seen[-1]["mask"].tolist(), [True] * 5)
        deleted = scene.render("gap", delete_actor_ids=(91,))
        self.assertEqual(self.backend.seen[-1]["mask"].tolist(), [True, False, True, False, True])
        self.assertTrue(keep["coverage"][0]["native_fallback"])
        self.assertEqual(keep["coverage"], deleted["coverage"])
        strict = self.scene(row_providers=(provider,), require_complete_coverage=True)
        with self.assertRaisesRegex(ValueError, "coverage gap"):
            strict.render("gap")

    def test_same_frame_provider_availability_change_is_not_silent(self):
        surface = Surface(self.backend, Model([6.], [[.4, .5, .6]], precomputed=True), missing_keys=("key",))
        scene = self.scene(row_providers=(RowProvider(surface.rows, 77),))
        scene.render("key")
        surface.missing_keys.clear()
        with self.assertRaisesRegex(RuntimeError, "availability"):
            scene.render("key")

    def test_static_gates_are_temporary_and_model_mutation_is_detected(self):
        original = self.static._opacity.clone()
        scene = self.scene(static_gates=torch.tensor([1., .5, 0.]))
        scene.render(1)
        torch.testing.assert_close(self.static._opacity, original)
        torch.testing.assert_close(self.backend.seen[-1]["opacities"][:3], original * torch.tensor([[1.], [.5], [0.]]))
        self.static._xyz.add_(1.)
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            scene.render(2)

    def test_provider_assets_and_frozen_owners_have_version_guards(self):
        asset = Model([6.], [[.4, .5, .6]], precomputed=True)
        provider = RowProvider(Surface(self.backend, asset).rows, 77, state_objects=(asset,))
        scene = self.scene(row_providers=(provider,))
        asset._opacity.mul_(.5)
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            scene.render(2)
        other = self.scene()
        other._owners_s.fill_(77)
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            other.render(3)

    def test_alpha_conservation_violation_is_rejected(self):
        native = self.backend.raster
        def broken(*args):
            result = native(*args)
            result["extra"] = result["extra"] * .5
            return result
        scene = self.scene()
        scene._ops = NativeOps(self.backend.prepare, self.backend.sh, self.backend.merge, broken)
        with self.assertRaisesRegex(RuntimeError, "sumalpha"):
            scene.render(1)

    def test_static_only_pipe_keeps_native_branch_selection(self):
        self.pipe.including_dynamic = False
        scene = self.scene()
        result = scene.render("static-only")
        self.assertEqual(result["row_owners"].tolist(), [47, 91, -1])
        self.assertFalse(any(dynamic for _, dynamic in self.backend.prepare_calls))

    def test_owner_protocol_and_manifest_background_dtype(self):
        scene = self.scene(background=[0, 0, 0])
        self.assertEqual(scene._background.dtype, self.static.get_xyz.dtype)
        scene.render("background")
        with self.assertRaises(ValueError):
            scene.render("bad-query", delete_actor_ids=(-2,))
        self.semantic["static"][0] = -2
        with self.assertRaises(ValueError):
            self.scene()
        self.semantic["static"][0] = 47
        with self.assertRaises(ValueError):
            self.scene(background=[float("nan"), 0, 0])

    def test_persistent_callback_runs_at_clip_boundary_not_per_frame(self):
        asset = Model([6.], [[.4, .5, .6]], precomputed=True)
        surface, checks = Surface(self.backend, asset), []
        provider = RowProvider(surface.rows, 77, state_objects=(asset,),
            assert_persistent_unchanged=lambda: checks.append("checked"))
        scene = self.scene(row_providers=(provider,))
        scene.render(1)
        surface._render_xyz = torch.ones(2, 3)  # 合法LoD缓存不属于持久state_objects
        scene.render(2)
        self.assertEqual(checks, [])
        scene.assert_persistent_unchanged()
        self.assertEqual(checks, ["checked"])

    def test_road_and_surface_groups_share_the_same_raster(self):
        road = Model([8.], [[.4, .4, .4]], precomputed=True)
        surface = Model([9.], [[.8, .4, .2]], precomputed=True)
        providers = (RowProvider(Surface(self.backend, road).rows, 0, kind="road", state_objects=(road,)),
                     RowProvider(Surface(self.backend, surface).rows, 77, state_objects=(surface,)))
        scene, before = self.scene(row_providers=providers), self.backend.raster_calls
        result = scene.render("road-surface", focus_actor_ids=(91,))
        self.assertEqual(self.backend.raster_calls - before, 1)
        self.assertGreater(float(result["diagnostics"]["road"]), 0.)
        self.assertGreater(float(result["diagnostics"]["other_surface"]), 0.)
        torch.testing.assert_close(sum(result["diagnostics"].values()), result["alpha"])

    def test_registered_owner_alpha_is_correct_and_does_not_change_rgb(self):
        scene = self.scene()
        baseline = scene.render("labelled", focus_actor_ids=(91,))
        self.assertEqual(tuple(baseline["owner_alpha_T"].shape), (0, 1, 1))
        self.assertEqual(self.backend.seen[-1]["features"].shape[1], FEATURE_CHANNELS)
        self.assertTrue((self.backend.seen[-1]["features"][:, 8:] == 0).all())
        labelled = scene.render("labelled", focus_actor_ids=(91,), diagnostic_actor_ids=(91, 47))
        self.assertEqual(labelled["owner_alpha_actor_ids"], (91, 47))
        # 深度次序47,91,91,unknown,unknown；每行opacity=.25。未知不强配给source91。
        torch.testing.assert_close(labelled["owner_alpha_T"][:, 0, 0], torch.tensor([.328125, .25]), rtol=0, atol=0)
        torch.testing.assert_close(labelled["rgb"], baseline["rgb"], rtol=0, atol=0)
        torch.testing.assert_close(labelled["row_owners"], baseline["row_owners"])
        torch.testing.assert_close(sum(labelled["diagnostics"].values()), labelled["alpha"])
        self.assertGreater(float(labelled["extra"].sum()), float(labelled["alpha"]))
        refocus = scene.render("labelled", focus_actor_ids=(47,), diagnostic_actor_ids=(91, 47))
        torch.testing.assert_close(refocus["owner_alpha_T"], labelled["owner_alpha_T"], rtol=0, atol=0)
        torch.testing.assert_close(refocus["rgb"], labelled["rgb"], rtol=0, atol=0)
        deleted_plain = scene.render("labelled", delete_actor_ids=(91,))
        deleted_tagged = scene.render("labelled", delete_actor_ids=(91,), diagnostic_actor_ids=(91, 47))
        torch.testing.assert_close(deleted_plain["rgb"], deleted_tagged["rgb"], rtol=0, atol=0)
        self.assertEqual(float(deleted_tagged["owner_alpha_T"][0]), 0.)
        self.assertEqual(float(deleted_tagged["owner_alpha_T"][1]), .25)
        scene.assert_unchanged()

    def test_owner_diagnostic_capacity_and_duplicate_ids_are_explicit(self):
        scene = self.scene()
        full = scene.render("capacity", diagnostic_actor_ids=tuple(range(1, 17)))
        self.assertEqual(tuple(full["owner_alpha_T"].shape), (16, 1, 1))
        with self.assertRaisesRegex(ValueError, "At most 16"):
            scene.render("too-many", diagnostic_actor_ids=tuple(range(1, 18)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            scene.render("duplicate", diagnostic_actor_ids=(91, 91))
        for invalid in (0, -1, -2):
            with self.assertRaises(ValueError):
                scene.render("nonpositive", diagnostic_actor_ids=(invalid,))

    def test_nonfocus_native_branch_labels_include_unknown_and_preserve_rgb(self):
        scene = self.scene()
        plain = scene.render("branches", focus_actor_ids=(91,), diagnostic_actor_ids=(91, 47))
        result = scene.render("branches", focus_actor_ids=(91,), diagnostic_actor_ids=(91, 47), diagnostic_branch_alpha=True)
        self.assertEqual(result["branch_alpha_names"], BRANCH_ALPHA_NAMES)
        self.assertEqual(tuple(plain["branch_alpha_T"].shape), (0, 1, 1))
        # unknown静态/动态分别处在深度5/4，不能丢进静态或动态已知owner组。
        torch.testing.assert_close(result["branch_alpha_T"][:, 0, 0], torch.tensor([.3291015625, .10546875]), rtol=0, atol=0)
        self.assertGreater(float(result["branch_alpha_T"][0]), float(result["diagnostics"]["other_static"]))
        self.assertGreater(float(result["branch_alpha_T"][1]), float(result["diagnostics"]["other_dynamic"]))
        torch.testing.assert_close(result["rgb"], plain["rgb"], rtol=0, atol=0)
        torch.testing.assert_close(result["row_owners"], plain["row_owners"])
        torch.testing.assert_close(result["owner_alpha_T"], plain["owner_alpha_T"], rtol=0, atol=0)
        torch.testing.assert_close(sum(result["diagnostics"].values()), result["alpha"])
        deleted_plain = scene.render("branches", delete_actor_ids=(91,), focus_actor_ids=(91,))
        deleted = scene.render("branches", delete_actor_ids=(91,), focus_actor_ids=(91,), diagnostic_branch_alpha=True)
        torch.testing.assert_close(deleted["rgb"], deleted_plain["rgb"], rtol=0, atol=0)
        scene.assert_unchanged()

    def test_branch_mode_has_fourteen_owner_slots_and_no_surface_branch_tags(self):
        asset = Model([8.], [[.4, .4, .4]], precomputed=True)
        provider = RowProvider(Surface(self.backend, asset).rows, 77, state_objects=(asset,))
        scene = self.scene(row_providers=(provider,))
        ids = tuple(range(1, 14)) + (91,)
        result = scene.render("capacity14", focus_actor_ids=(91,), diagnostic_actor_ids=ids, diagnostic_branch_alpha=True)
        self.assertEqual(tuple(result["owner_alpha_T"].shape), (14, 1, 1))
        self.assertEqual(float(result["owner_alpha_T"][-1]), .328125)
        torch.testing.assert_close(result["branch_alpha_T"][:, 0, 0], torch.tensor([.3291015625, .10546875]), rtol=0, atol=0)
        self.assertTrue((self.backend.seen[-1]["features"][-1, 22:24] == 0).all())
        with self.assertRaisesRegex(ValueError, "At most 14"):
            scene.render("overflow14", diagnostic_actor_ids=tuple(range(1, 16)), diagnostic_branch_alpha=True)
        with self.assertRaises(ValueError):
            scene.render("bad-bool", diagnostic_branch_alpha=1)


if __name__ == "__main__":
    unittest.main()
