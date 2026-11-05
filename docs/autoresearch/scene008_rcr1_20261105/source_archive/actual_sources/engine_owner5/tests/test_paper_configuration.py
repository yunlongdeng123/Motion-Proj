"""Verify the actual profile and existing dynamic LR construction on CPU."""
import ast
import math
from pathlib import Path
import unittest
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


class PaperConfigurationTests(unittest.TestCase):
    def test_explicit_paper_settings_and_actual_dynamic_schedule(self):
        args = OmegaConf.merge(OmegaConf.load(ROOT / "configs/base.yaml"),
                               OmegaConf.load(ROOT / "configs/paper_waymo_reconstruction.yaml"))
        self.assertEqual(args.iterations, 50000)
        self.assertEqual(args.random_init_point, 0)
        self.assertEqual(args.dynamic_random_init_point, 0)
        self.assertEqual(args.lambda_dssim, .2)
        self.assertTrue(args.paper_method)
        tree = ast.parse((ROOT / "scene/gaussian_model.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Dynamic_GaussianModel")
        setup = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "training_setup")
        assign = next(n for n in ast.walk(setup) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Attribute) and t.attr == "xyz_scheduler_args" for t in n.targets))
        # Evaluate the actual scheduler keyword expressions, not a duplicate
        # implementation of the dynamic multiplier or schedule horizon.
        kwargs = {kw.arg: eval(compile(ast.Expression(kw.value), "<actual-dynamic-lr>", "eval"),
                               {"training_args": args}) for kw in assign.value.keywords}
        utils_tree = ast.parse((ROOT / "utils/general_utils.py").read_text())
        function = next(n for n in utils_tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_expon_lr_func")
        scope = {"np": np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<actual-lr-function>", "exec"), scope)
        schedule = scope["get_expon_lr_func"](**kwargs)
        self.assertAlmostEqual(schedule(0), .0016, places=12)
        self.assertAlmostEqual(schedule(25000), math.sqrt(.0016 * .0000016), places=12)
        self.assertAlmostEqual(schedule(50000), .0000016, places=12)
        self.assertEqual(kwargs["max_steps"], 50000)


if __name__ == "__main__":
    unittest.main()
