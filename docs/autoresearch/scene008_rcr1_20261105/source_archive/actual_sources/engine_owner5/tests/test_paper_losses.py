"""Independent equation values, gradients and empty-mask cases; CPU-only."""
import math
import unittest
import torch
from utils.paper_losses import (sky_loss, dynamic_opacity_loss, boundary_distance,
                               nps_loss, sparse_inverse_depth_loss, compose_paper_loss)


class PaperLossTests(unittest.TestCase):
    def test_sky_equation_and_gradient(self):
        alpha = torch.tensor([[[0.5, 0.3]]], dtype=torch.double, requires_grad=True)
        mask = torch.tensor([[[1., 0.]]], dtype=torch.double)
        loss = sky_loss(alpha, mask)
        self.assertAlmostEqual(loss.item(), -0.5 * math.log(0.5) / 2, places=12)
        loss.backward()
        self.assertAlmostEqual(alpha.grad[0, 0, 0].item(), (-math.log(0.5) + 1.) / 2, places=12)
        self.assertEqual(alpha.grad[0, 0, 1].item(), 0.)

    def test_dynamic_equation_mask_domain_and_gradient(self):
        alpha = torch.tensor([[[0.5, 0.2]]], dtype=torch.double, requires_grad=True)
        loss = dynamic_opacity_loss(alpha, torch.tensor([[[1., 0.]]], dtype=torch.double))
        self.assertAlmostEqual(loss.item(), -0.5 * math.log(0.5) / 2, places=12)
        loss.backward()
        self.assertAlmostEqual(alpha.grad[0, 0, 0].item(), (math.log(0.5) - 1.) / 2, places=12)
        self.assertEqual(alpha.grad[0, 0, 1].item(), 0.)

    def test_opacity_clamp_extremes(self):
        for fn in [sky_loss, dynamic_opacity_loss]:
            alpha = torch.tensor([[[0., 1., -0.2, 1.2]]], requires_grad=True)
            value = fn(alpha, torch.ones_like(alpha))
            value.backward()
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.isfinite(alpha.grad).all())

    def test_nps_hand_computed_center(self):
        feature = torch.arange(9, dtype=torch.double).reshape(1, 3, 3).requires_grad_()
        mask = torch.zeros(1, 3, 3, dtype=torch.double)
        mask[0, 1, 1] = 1
        # Center4 has four differences3,3,1,1 and nearest static distance1.
        self.assertEqual(nps_loss(feature, mask).item(), 8.)
        nps_loss(feature, mask).backward()
        self.assertTrue(torch.isfinite(feature.grad).all())

    def test_nps_against_exhaustive_pixel_reference(self):
        torch.manual_seed(41)
        feature = torch.randn(3, 5, 6, dtype=torch.double, requires_grad=True)
        mask = torch.zeros(1, 5, 6, dtype=torch.double)
        mask[:, 1:4, 1:5] = 1
        mask[:, 0, 0] = 1  # an image-edge foreground source
        static_pixels = [(y, x) for y in range(5) for x in range(6) if not mask[0, y, x]]
        expected = 0.
        for y in range(5):
            for x in range(6):
                if not mask[0, y, x]:
                    continue
                distance = min(math.hypot(y - sy, x - sx) for sy, sx in static_pixels)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 5 and 0 <= nx < 6:
                        vector = feature.detach()[:, y, x] - feature.detach()[:, ny, nx]
                        expected += math.sqrt(sum(float(v) ** 2 for v in vector)) / distance
        actual = nps_loss(feature, mask)
        self.assertAlmostEqual(actual.item(), expected, places=10)
        distance = boundary_distance(mask)
        self.assertEqual(distance[0, 2, 2].item(), 2.)
        self.assertTrue(torch.autograd.gradcheck(lambda f: nps_loss(f, mask, distance), (feature,)))

    def test_nps_empty_and_full_masks_have_finite_zero_gradient(self):
        for fill in [0., 1.]:
            feature = torch.ones(3, 3, 4, requires_grad=True)
            value = nps_loss(feature, torch.full((1, 3, 4), fill))
            value.backward()
            self.assertEqual(value.item(), 0.)
            self.assertTrue(torch.equal(feature.grad, torch.zeros_like(feature)))

    def test_sparse_depth_uses_image_area_not_valid_count(self):
        rendered = torch.tensor([[[2., 9.], [7., 3.]]], dtype=torch.double, requires_grad=True)
        lidar = torch.tensor([[[1., 0.], [0., 0.]]], dtype=torch.double)
        value = sparse_inverse_depth_loss(rendered, lidar, eps=0.)
        self.assertEqual(value.item(), 0.125)  # |1-1/2| / four pixels
        value.backward()
        self.assertEqual(rendered.grad[0, 0, 0].item(), 0.0625)
        self.assertEqual(torch.count_nonzero(rendered.grad).item(), 1)

    def test_depth_all_invalid_returns_zero(self):
        rendered = torch.zeros(1, 2, 2, requires_grad=True)
        value = sparse_inverse_depth_loss(rendered, torch.zeros_like(rendered))
        value.backward()
        self.assertEqual(value.item(), 0.)
        self.assertTrue(torch.isfinite(rendered.grad).all())

    def test_total_composition_has_no_hidden_terms(self):
        terms = [torch.tensor(float(i), requires_grad=True) for i in range(1, 7)]
        value = compose_paper_loss(*terms, lambda_sky=.05, lambda_nps=.01,
                                   lambda_opacity=.05, lambda_depth=.1)
        self.assertAlmostEqual(value.item(), 1+2+.05*3+.01*4+.05*5+.1*6, places=6)
        value.backward()
        for term, expected in zip(terms, [1, 1, .05, .01, .05, .1]):
            self.assertAlmostEqual(term.grad.item(), expected, places=6)

    def test_dynamic_rgb_ablation_removes_its_gradient(self):
        terms = [torch.tensor(float(i), requires_grad=True) for i in range(1, 7)]
        value = compose_paper_loss(*terms, lambda_sky=.05, lambda_nps=.01,
                                   lambda_opacity=.05, lambda_depth=.1,
                                   lambda_dynamic_rgb=0.)
        value.backward()
        self.assertEqual(terms[1].grad.item(), 0.)
        self.assertEqual(terms[0].grad.item(), 1.)


if __name__ == "__main__":
    unittest.main()
