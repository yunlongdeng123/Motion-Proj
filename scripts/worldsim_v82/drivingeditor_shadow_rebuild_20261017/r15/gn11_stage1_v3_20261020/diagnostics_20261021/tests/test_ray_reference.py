import sys
from pathlib import Path
import unittest
import numpy as np
test_root=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(test_root/'src' if (test_root/'src').exists() else test_root))
from ray_reference import composite, project, covariance, sh_color, candidates


class RayTests(unittest.TestCase):
    def test_weights_and_T(self):
        v = composite(np.array([.5, .5], np.float32), np.eye(3, dtype=np.float32)[:2], np.array([1, 2]))
        np.testing.assert_allclose(v['weights'], [.5, .25])
        self.assertAlmostEqual(v['T'], .25)
        self.assertAlmostEqual(v['depth_sum'], 1)

    def test_removal_recomposites(self):
        a = np.array([.5, .5], np.float32); c = np.eye(3, dtype=np.float32)[:2]
        out = composite(a, c, np.ones(2), [0])
        self.assertAlmostEqual(float(out['rgb'][1]), .5)
        self.assertNotEqual(float(out['rgb'][1]), .25)

    def test_threshold_crossing_not_added(self):
        v = composite(np.array([.99, .99, .99], np.float32), np.ones((3, 3)), np.ones(3))
        self.assertLess(len(v['indices']), 3)
        self.assertGreaterEqual(v['T'], .0001)

    def test_covariance_axis_rotation(self):
        q = np.array([[1, 0, 0, 0]], np.float32)
        np.testing.assert_allclose(covariance([[1, 2, 3]], q)[0], np.diag([1, 4, 9]))

    def test_dc_color_not_clipped_above_one(self):
        sh = np.zeros((1, 16, 3), np.float32); sh[:, 0] = 10
        self.assertGreater(float(sh_color(sh, np.array([[0, 0, 1]]), 3)[0, 0]), 1)

    def test_projection_half_pixel_and_large_ellipse(self):
        p = project(np.array([[0, 0, 5]], np.float32), np.eye(3)[None]*.01,
                    np.array([.5]), np.array([True]), np.eye(4), [[500, 0, 240], [0, 500, 160], [0, 0, 1]])
        np.testing.assert_allclose(p['uv'][0], [239.5, 159.5])
        self.assertGreater(p['radius'][0], 0)

    def test_center_not_at_pixel_can_contribute(self):
        p=project(np.array([[.2,0,5]],np.float32),np.eye(3)[None]*.1,
                  np.array([.8]),np.array([True]),np.eye(4),[[500,0,240],[0,500,160],[0,0,1]])
        idx,a=candidates(p,240,160)
        self.assertGreater(abs(p['uv'][0,0]-240),10)
        self.assertEqual(len(idx),1)
        self.assertGreater(a[0],1/255)

    def test_low_alpha_high_radiance_not_ignored(self):
        c=composite(np.array([.01,.5],np.float32),np.array([[100,100,100],[1,1,1]],np.float32),np.array([1,2],np.float32))
        self.assertLess(c['weights'][0],c['weights'][1])
        self.assertGreater(c['weights'][0]*100,c['weights'][1])


if __name__ == '__main__':
    unittest.main()
