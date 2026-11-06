"""检查 CPU 导入隔离、世界场迁移和特征归因。

不构造原生场景，不能替代待执行的 CUDA 重放。
运行：python -B test_v44_runtime_cpu.py
"""
from pathlib import Path
import subprocess
import sys
import unittest

import v44_runtime as runtime


class RuntimeCPUChecks(unittest.TestCase):
    def test_import_does_not_load_cuda_execution_modules(self):
        code = ("import sys; import v44_runtime; "
                "assert not any(x in sys.modules for x in "
                "('torch','numpy','final_runtime','scene_runtime','repair37','native_binding'))")
        subprocess.run([sys.executable, "-B", "-c", code],
                       cwd=Path(__file__).resolve().parent, check=True)

    def test_world_transfer_preserves_known_bilinear_field(self):
        import numpy as np
        knots = dict(low_m=[2., -3.], high_m=[7., 5.], grid=[11, 9])
        u, v = np.meshgrid(np.linspace(2, 7, 11), np.linspace(-3, 5, 9), indexing="ij")
        coefficients = (2*u-3*v+u*v).reshape(-1)
        xy = np.asarray([[2., -3.], [7., 5.], [3.5, .25], [-10., 8.]])
        basis = runtime.world_basis(xy, knots)
        clipped = np.clip(xy, knots["low_m"], knots["high_m"])
        oracle = 2*clipped[:, 0]-3*clipped[:, 1]+clipped[:, 0]*clipped[:, 1]
        np.testing.assert_allclose(basis.sum(1), 1., atol=1e-7, rtol=0)
        self.assertGreaterEqual(float(basis.min()), 0.)
        np.testing.assert_allclose(basis@coefficients, oracle, atol=2e-6, rtol=0)
        with self.assertRaises(ValueError):
            runtime.world_basis(xy, dict(knots, grid=[6, 5]))

    def test_one_forward_group_attribution_keeps_rgb_and_background(self):
        import torch
        rt = runtime.V44Runtime.__new__(runtime.V44Runtime)
        rt.torch, rt.pipe, rt.render_calls = torch, None, 0
        colours = torch.tensor([[1.2, .2, .1], [.1, .8, .3], [.4, .5, .6], [.9, .2, .7]])
        weights = torch.tensor([.1, .2, .3, .1])
        tags = [rt._tag(dict(features=torch.tensor([[float(i+1), 1.]]),
                            colors_precomp=colours[i:i+1]), group)
                for i, group in enumerate(runtime.GROUPS)]
        rows = {key: torch.cat([row[key] for row in tags]) for key in tags[0]}
        background, env = torch.tensor([.2, .3, .1]), torch.tensor([.4, .2, .5])[:, None, None]
        calls = []

        def raster(camera, pipe, degree, values, bg):
            calls.append(1)
            accumulated = (values["features"]*weights[:, None]).sum(0)[:, None, None]
            rgb = (values["colors_precomp"]*weights[:, None]).sum(0)[:, None, None]
            alpha = weights.sum().reshape(1, 1, 1)
            return dict(rgb=rgb+(1-alpha)*bg[:, None, None], alpha=alpha,
                        depth_sum=accumulated[:1], extra=accumulated[2:])

        rt.raster = raster
        scene = dict(rows=rows, cam=None, bg=background, env=env, meta=dict(raw=113))
        out = rt.render(113, scene)
        oracle = ((colours*weights[:, None]).sum(0) +
                  (1-weights.sum())*(background+env[:, 0, 0]))[:, None, None]
        torch.testing.assert_close(out["rgb"], oracle)
        torch.testing.assert_close(out["group_rgb_T"][:, :, 0, 0], colours*weights[:, None])
        torch.testing.assert_close(out["group_alpha_T"][:, 0, 0], weights)
        self.assertEqual(len(calls), 1)
        self.assertEqual(rt.render_calls, 1)
        self.assertLess(out["color_reconstruction_maxabs"], 1e-6)
        self.assertIs(out["component_alpha_T"], out["group_alpha_T"])
        # 即使 RGB 有限，错误的特征/分组 ABI 也必须触发守恒检查。
        broken = dict(scene, rows=dict(rows, features=rows["features"].clone()))
        broken["rows"]["features"][0, 2] = 0
        with self.assertRaisesRegex(RuntimeError, "conservation"):
            rt.render(113, broken)

    def test_config_rejects_implicit_or_unknown_inputs(self):
        with self.assertRaises(ValueError):
            runtime.RuntimeConfig(Path("missing.json"), {})
        explicit = {name: "/explicit/"+name for name in runtime.REQUIRED_PATHS}
        with self.assertRaises(ValueError):
            runtime.RuntimeConfig(Path("contract.json"), explicit, sampler="legacy_m2")
        config = runtime.RuntimeConfig(Path("contract.json"), explicit, sampler="canonical")
        self.assertEqual(config.sampler, "canonical")


if __name__ == "__main__":
    unittest.main(verbosity=2)
