import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
import contextlib
import io
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
import torch
from shadow_rebuild.tracing import gaussian_ray_integral, geometry_transmittance
from shadow_rebuild.model import SceneModel
from shadow_rebuild.data import Dataset, clean_reference
from shadow_rebuild.losses import mask_terms
from shadow_rebuild.train import run, objective
from fixture import create

torch.set_num_threads(2)


class CoreMath(unittest.TestCase):
    def test_rotated_ray_integral_against_scipy_quadrature(self):
        rotation = Rotation.from_euler("xyz", [.3, .6, -.8]).as_matrix()
        cov = rotation@np.diag([.1, .3, .7])**2@rotation.T
        mu = np.array([.1, -.2, .7])
        o = np.array([.2, -.5, -.4])
        d = np.array([.1, .4, 1.]); d /= np.linalg.norm(d)
        precision = np.linalg.inv(cov)
        density = lambda t: 2.3*np.exp(-.5*(o+t*d-mu)@precision@(o+t*d-mu))
        for upper in (1.2, None):
            expected = quad(density, 0, upper or np.inf, epsabs=1e-11)[0]
            tensor = lambda x: torch.tensor(x, dtype=torch.float64)
            actual = gaussian_ray_integral(tensor(o[None]), tensor(d[None]), tensor(mu[None]), tensor(cov[None]), tensor([2.3]), upper=upper)
            self.assertAlmostEqual(actual.item(), expected, places=9)

    def test_ray_integral_gradcheck(self):
        o = torch.tensor([[.2, -.2, -.3]], dtype=torch.float64)
        d = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
        mu = torch.tensor([[.1, 0., .8]], dtype=torch.float64, requires_grad=True)
        scale = torch.tensor([.2, .3, .4], dtype=torch.float64, requires_grad=True)
        amplitude = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
        function = lambda m, s, a: gaussian_ray_integral(o, d, m, torch.diag(s.square())[None], a).sum()
        self.assertTrue(torch.autograd.gradcheck(function, (mu, scale, amplitude), eps=1e-6, atol=1e-5))

    def test_reference_requires_repeated_bright_observations(self):
        observations = np.array([[[.1, .1, .1], [.1, .1, .1], [.1, .1, .1], [.8, .8, .8]]], np.float32)
        _, confidence = clean_reference(observations, np.ones((1, 4), bool))
        self.assertEqual(confidence[0], 0)
        observations = np.array([[[.1, .1, .1], [.1, .1, .1], [.4, .4, .4], [.4, .4, .4], [.4, .4, .4]]], np.float32)
        ref, confidence = clean_reference(observations, np.ones((1, 5), bool))
        np.testing.assert_allclose(ref, .4)
        self.assertGreater(confidence[0], 0)
        _, confidence = clean_reference(observations, np.array([[1, 0, 1, 0, 0]], bool))
        self.assertEqual(confidence[0], 0)


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = create(cls.root/"data")
        cls.dataset = Dataset(cls.root/"data", synthetic=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def model(self):
        return SceneModel(self.dataset.seed, self.config)

    def test_masks_outside_and_unknown_have_distinct_gradients(self):
        frame = self.dataset.frame(0)
        raw = torch.full(frame.masks.shape, .5, requires_grad=True)
        inside, outside, identity = mask_terms(dict(raw_actor_alpha=raw, visible_labels=raw), frame)
        outside.backward()
        self.assertGreater(float(raw.grad[~frame.masks].abs().sum()), 0)
        self.assertEqual(float(raw.grad[frame.masks].abs().sum()), 0)
        frame.mask_known[:] = False
        raw.grad.zero_()
        terms = mask_terms(dict(raw_actor_alpha=raw, visible_labels=raw), frame)
        sum(terms).backward()
        self.assertEqual(float(raw.grad.abs().sum()), 0)

    def test_geometry_and_shadow_share_edited_actor(self):
        model, frame = self.model(), self.dataset.frame(0)
        before = model.field.evaluate(frame.poses, frame.valid, frame.camera.center)
        after = model.field.evaluate(frame.poses, frame.valid, frame.camera.center, {0: {"translation": [1., .2, 0], "yaw": .5}})
        static = before["owner"] == -1
        self.assertTrue(torch.equal(before["xyz"][static], after["xyz"][static]))
        local = model.field.xyz[model.field.owner == 0]
        from shadow_rebuild.math3d import rotation_z
        r = rotation_z(.5, device="cpu", dtype=torch.float32)@frame.poses[0, :3, :3]
        expected = local@r.T+frame.poses[0, :3, 3]+torch.tensor([1., .2, 0])
        torch.testing.assert_close(after["xyz"][~static], expected)
        o = torch.tensor([[0., 0., 0.], [.3, -.2, 0.]])
        d = torch.tensor([[.2, 0., 1.], [.2, 0., 1.]])
        d = d/d.norm(dim=-1, keepdim=True)
        direct = geometry_transmittance(o, d, before, cull=False)
        shifted = {k: v.clone() for k, v in before.items()}
        delta = torch.tensor([.4, -.3, .1])
        shifted["xyz"] = shifted["xyz"]+delta
        torch.testing.assert_close(direct, geometry_transmittance(o+delta, d, shifted, cull=False))
        deleted = model.field.evaluate(frame.poses, frame.valid, frame.camera.center, {0: {"delete": True}})
        self.assertTrue((deleted["owner"] == -1).all())
        self.assertTrue(torch.equal(geometry_transmittance(o, d, deleted), torch.ones(2)))

    def test_broad_phase_matches_full_integral(self):
        model, frame = self.model(), self.dataset.frame(0)
        g = model.field.evaluate(frame.poses, frame.valid, frame.camera.center)
        o = torch.tensor([[-20., 10., 0.], [0., -.2, 0.], [.2, -.2, 0.]])
        d = torch.tensor([[0., 0., 1.]]).expand_as(o)
        torch.testing.assert_close(geometry_transmittance(o, d, g, cull=True), geometry_transmittance(o, d, g, cull=False), atol=1e-6, rtol=1e-6)

    def test_stage2_freezes_geometry_but_trains_material_and_light(self):
        model, frame = self.model(), self.dataset.frame(0)
        model.set_stage("material")
        loss, _, _ = objective(model, frame, 1, "reference", "material", torch.Generator().manual_seed(0))
        loss.backward()
        for name in ("xyz", "quaternion", "log_scales", "opacity_logit"):
            self.assertIsNone(getattr(model.field, name).grad)
        self.assertGreater(model.field.albedo_logit.grad.abs().sum().item(), 0)
        self.assertGreater(model.light.ambient_logit.grad.abs().sum().item(), 0)

    def test_shadow_sun_direction_and_shape_gradients_finite(self):
        model, frame = self.model(), self.dataset.frame(0)
        model.set_stage("joint")
        g = model.field.evaluate(frame.poses, frame.valid, frame.camera.center)
        position = torch.tensor([[-.7, -.25, 0.], [-.8, -.25, 0.], [-.6, -.2, 0.]])
        normal = torch.tensor([[0., 0., 1.]]).expand_as(position)
        value = model.light.visibility(position, normal, g).sum()
        value.backward()
        for p in (model.light.azimuth, model.field.xyz, model.field.log_scales):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_stage3_position_bound_and_static_no_time(self):
        model = self.model()
        model.field.anchor()
        with torch.no_grad():
            model.field.xyz.add_(100)
        model.field.project_constraints("joint")
        self.assertLessEqual((model.field.xyz-model.field.anchor_xyz).norm(dim=-1).max().item(), self.config.joint_bound_m+1e-5)
        a, b = self.dataset.frame(0), self.dataset.frame(4)
        ea = model.field.evaluate(a.poses, a.valid, a.camera.center)
        eb = model.field.evaluate(b.poses, b.valid, b.camera.center)
        mask = ea["owner"] == -1
        self.assertTrue(torch.equal(ea["xyz"][mask], eb["xyz"][mask]))
        self.assertTrue(torch.equal(ea["opacity"][mask], eb["opacity"][mask]))

    def test_full_three_stage_and_resume_after_densification(self):
        with contextlib.redirect_stdout(io.StringIO()):
            full, done = run(self.dataset, self.config, self.root/"full")
            _, stopped = run(self.dataset, self.config, self.root/"resume", max_updates=2)
            resumed, resumed_done = run(self.dataset, self.config, self.root/"resume", resume=self.root/"resume/latest.pt")
        self.assertEqual(done["global_step"], 10)
        self.assertEqual(resumed_done["global_step"], 10)
        self.assertGreater(len(full.field.xyz), len(self.dataset.seed["xyz"]))
        for key, value in full.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
        checkpoint = torch.load(self.root/"full/latest.pt", weights_only=False)
        other = torch.load(self.root/"resume/latest.pt", weights_only=False)
        for key, state in checkpoint["optimizer"]["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, other["optimizer"]["state"][key][name]))

    def test_no_cuda_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_clean_road_ablation_does_not_inherit_bright_reference_initialization(self):
        ablated = SceneModel(self.dataset.seed, replace(self.config, use_clean_road=False))
        from shadow_rebuild.math3d import srgb_to_linear
        expected = srgb_to_linear(torch.tensor(self.dataset.seed["observed_rgb"]))
        torch.testing.assert_close(ablated.field.appearance_logit.sigmoid(), expected.clamp(.01, .99))
        full = self.model()
        self.assertGreater((ablated.field.appearance_logit-full.field.appearance_logit).abs().sum().item(), 0)

    def test_road_receiver_uses_road_conditioned_geometry_and_preserves_actor_radiance(self):
        model, frame = self.model(), self.dataset.frame(0)
        g, p = model.buffers(frame, 1, "reference")
        ids = torch.arange(frame.rgb.shape[0]*frame.rgb.shape[1])
        rgb, _ = model.shade_pixels(p, g, ids, shadows=False)
        road_rgb, _ = model.light.shade(p["road_position"].reshape(-1, 3), p["road_normal"].reshape(-1, 3),
                                       p["road_albedo"].reshape(-1, 3), g, (p["road_alpha"].flatten() > 1e-5).float(), enabled=False)
        reconstructed_nonroad = rgb-road_rgb*p["road_alpha"].flatten()[:, None]-(1-p["alpha"].flatten()[:, None])*p["sky"].reshape(-1, 3)
        expected = (p["appearance"]-p["road_appearance"]).reshape(-1, 3)
        torch.testing.assert_close(reconstructed_nonroad, expected, atol=1e-7, rtol=1e-5)
        supported = p["road_alpha"] > .01
        # 合成地面在z≈0，不能混进z≈.5的车体接收面。
        self.assertLess(p["road_position"][supported][:, 2].abs().max().item(), .06)


if __name__ == "__main__":
    unittest.main(verbosity=2)
