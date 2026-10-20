"""实际 V1 源码与 v3 的三项损失、归一化及梯度配对；不加载 CUDA。"""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine_overlay" if (ROOT / "engine_overlay").is_dir() else ROOT / "engine"
PLAN = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
V1 = Path(PLAN["basis"]["five_identity_v1"])
sys.path.insert(0, str(ENGINE))
from utils.v3_regularizer_contract import V1_REGULARIZER_CONFIG, validate_v1_regularizers


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mapped_state(source, feature, alpha, static_alpha, config, mask):
    """执行 train.py 中实际特征赋值和三个 state 表达式，不另写其分母。"""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wanted = {"static_feature", "static_t_map", "static_v_map", "legacy_static_feature",
              "legacy_static_t_map", "legacy_static_v_map"}
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in wanted
             # 排除 dynamic_gaussians is None 的另一路径。
             and not (isinstance(node.value, ast.Name) and node.value.id in {"t_map", "v_map"})]
    nodes.sort(key=lambda node: node.lineno)
    namespace = dict(render_pkg={"static_feature": feature}, alpha=alpha,
                     static_alpha=static_alpha, paper_method=True, EPS=1e-5)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    state_node = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "hybrid_state"
                              for target in node.targets) and isinstance(node.value, ast.Dict))
    zero = feature.sum() * 0
    state = {name: zero for name in (
        "global_rgb", "dynamic_rgb", "sky_rgb", "static_rgb", "sky", "depth",
        "inverse_depth_smoothness", "velocity_smoothness")}
    for key, value in zip(state_node.keys, state_node.values):
        if isinstance(key, ast.Constant) and key.value in {"static_t_map", "static_v_map", "nps_features"}:
            state[key.value] = eval(compile(ast.Expression(value), str(source), "eval"), namespace)
    state.update(alpha=alpha, dynamic_alpha=torch.zeros_like(alpha), dynamic_mask=mask,
                 iteration=1, nps_distances=None)
    # 实际调用必须直接使用 args；拒绝中间改写配置。
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "compose_hybrid_loss"]
    assert len(calls) == 1 and isinstance(calls[0].args[1], ast.Name)
    assert calls[0].args[1].id == "args", "regularizer config was overridden inside training"
    validate_v1_regularizers(config)
    return state


class V1RegularizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = load_file("v1_loss_reference", V1 / "utils/hybrid30k_losses.py")
        cls.before_nps = load_file("v1_nps_reference", V1 / "utils/paper_losses.py")
        cls.after = load_file("v3_loss_actual", ENGINE / "utils/hybrid30k_losses.py")
        cls.after_nps = load_file("v3_nps_actual", ENGINE / "utils/paper_losses.py")
        cls.config = json.loads(Path(PLAN["baseline_config"]).read_text(encoding="utf-8"))
        validate_v1_regularizers(cls.config)

    def test_three_terms_match_v1_values_and_gradients_from_actual_train_maps(self):
        generator = torch.Generator().manual_seed(19)
        inputs = [torch.rand(shape, generator=generator, dtype=torch.float64) + offset
                  for shape, offset in [((4, 5, 6), .2), ((1, 5, 6), .4), ((1, 5, 6), .1)]]
        mask = torch.zeros(1, 5, 6, dtype=torch.float64)
        mask[:, 1:4, 1:5] = 1
        results = []
        for source, loss_module, nps_module in (
            (V1, self.before, self.before_nps), (ENGINE, self.after, self.after_nps)
        ):
            leaves = [value.clone().requires_grad_() for value in inputs]
            state = mapped_state(source / "train.py", *leaves, self.config, mask)
            state["nps_distances"] = nps_module.boundary_distance(mask)
            result = loss_module.compose_hybrid_loss(state, self.config, nps_fn=nps_module.nps_loss)
            values = {}
            for term in ("static_duration", "static_velocity", "nps"):
                values[term] = (result.raw[term].detach(), result.weighted[term].detach(),
                                torch.autograd.grad(result.weighted[term], leaves, retain_graph=True,
                                                    allow_unused=True))
            results.append(values)
        for term in results[0]:
            with self.subTest(term=term):
                first, second = results[0][term], results[1][term]
                torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
                torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
                for before, after in zip(first[2], second[2]):
                    if before is None:
                        self.assertIsNone(after)
                    else:
                        torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertGreater(float(first[0]), 0)
                self.assertGreater(float(first[2][0].abs().sum()), 0)
                # 前两项只依赖联合alpha；NPS只依赖独立背景alpha。
                self.assertIsNone(first[2][1] if term == "nps" else first[2][2])

    def test_nps_keeps_full_mask_four_neighbor_sum_not_foreground_mean(self):
        feature = torch.arange(3 * 5 * 6, dtype=torch.float64).reshape(3, 5, 6)
        mask = torch.zeros(1, 5, 6, dtype=torch.float64)
        mask[:, 1:4, 1:5] = 1
        distances = self.after_nps.boundary_distance(mask)
        expected = feature.sum() * 0
        for y in range(5):
            for x in range(6):
                if mask[0, y, x] == 0:
                    continue
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < 5 and 0 <= xx < 6:
                        expected += torch.linalg.vector_norm(feature[:, y, x] - feature[:, yy, xx]) / distances[0, y, x]
        torch.testing.assert_close(self.after_nps.nps_loss(feature, mask, distances), expected)
        self.assertGreater(float(expected), float(expected / mask.sum()))

    def test_every_regularizer_setting_is_enforced(self):
        validate_v1_regularizers(self.config)
        for key in V1_REGULARIZER_CONFIG:
            altered = copy.deepcopy(self.config)
            altered[key] = False if key == "hybrid_nps_enabled" else 0.0
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "V1 regularizer mismatch"):
                validate_v1_regularizers(altered)

    def test_selective_objective_is_not_connected_to_active_runtime(self):
        tree = ast.parse((ENGINE / "utils/v3_runtime_context.py").read_text(encoding="utf-8"))
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module == "utils.v3_temporal"
                             for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.Name) and node.id in {
            "selective_temporal_loss", "TemporalEvidence"} for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
