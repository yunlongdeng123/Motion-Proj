"""实际 torch CPU 检查；抽取历史 V 的独立函数 AST，不导入 renderer 或 train。"""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
import prepare_candidate_cpu as prepare


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def load_helper(path):
    spec = importlib.util.spec_from_file_location("checked_resume_sv_losses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_functions(path):
    names = {"_mask_like", "dynamic_opacity_loss", "diagnostic_dynamic_background_opacity_loss"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names or len(selected) != len(names):
        raise ValueError("历史 V 的三项实际函数缺失或重复")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def must_raise(error, call):
    try:
        call()
    except error:
        return
    raise AssertionError("预期拒绝未发生: " + error.__name__)


def source_contract(original, candidate):
    assert candidate == prepare.patch_source(original), "候选必须等于对实际 S 的局部补丁"
    assert candidate.count(prepare.S_BLOCK) == 1, "S 静态 RGB 原块必须保留一次"
    assert candidate.count(prepare.V_BLOCK) == 1, "V 只能插入一次"
    must_raise(ValueError, lambda: prepare.patch_source(candidate))
    must_raise(ValueError, lambda: prepare.patch_source(original + prepare.S_BLOCK))
    tree = ast.parse(candidate)
    training = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "training")
    assert isinstance(training.body[0], ast.Expr) and training.body[0].value.func.id == "reject_unbound_gpu_training"
    main = next(node for node in tree.body if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test))
    assert isinstance(main.body[0], ast.Expr) and main.body[0].value.func.id == "reject_unbound_gpu_training"
    calls = [node for node in ast.walk(training) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    v = [node for node in calls if node.func.id == "diagnostic_dynamic_background_opacity_loss"]
    foreground = [node for node in calls if node.func.id == "dynamic_opacity_loss"]
    assert len(v) == len(foreground) == 1
    assert ast.unparse(v[0].args[0]) == "raw_dynamic_alpha"
    raw = [node for node in ast.walk(training) if isinstance(node, ast.Assign)
           and any(isinstance(target, ast.Name) and target.id == "raw_dynamic_alpha" for target in node.targets)]
    assert len(raw) == 1 and ast.unparse(raw[0].value) == "render_pkg['dynamic_alpha']"
    premask = [node for node in ast.walk(training) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "dynamic_alpha" for target in node.targets)
               and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult)]
    assert foreground[0].lineno < raw[0].lineno < v[0].lineno < min(node.lineno for node in premask)


def run_checks(args):
    started = time.monotonic()
    original = args.source_train.read_text(encoding="utf-8")
    candidate = ((args.candidate_dir / "train.py").read_text(encoding="utf-8")
                 if args.candidate_dir else prepare.patch_source(original))
    helper_path = ((args.candidate_dir / "utils/resume_sv_losses.py")
                   if args.candidate_dir else prepare.HERE / "utils/resume_sv_losses.py")
    helper = load_helper(helper_path)
    reference = historical_functions(args.historical_v_losses)
    checks = []
    source_contract(original, candidate)
    if args.candidate_dir:
        assert (args.candidate_dir / "utils/paper_losses.py").read_bytes() == (
            args.source_train.parent / "utils/paper_losses.py").read_bytes(), "原 Eq13 文件不得修改"
    checks.append("actual_S_local_patch_once_raw_before_premask_foreground_unchanged")

    mask = torch.tensor([[[1., 0., 1., 0., 0., 1., 0., 0.]]], dtype=torch.float64)
    values = torch.tensor([[[-0.1, 0., 0.1, 0.3, 0.8, 0.99, 1., 1.1]]], dtype=torch.float64)
    for used_mask in (mask, mask[0]):
        alpha = values.clone().requires_grad_()
        expected_alpha = values.clone().requires_grad_()
        actual = helper.diagnostic_dynamic_background_opacity_loss(alpha, used_mask)
        expected = reference["diagnostic_dynamic_background_opacity_loss"](expected_alpha, used_mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(actual, alpha)[0]
        expected_grad = torch.autograd.grad(expected, expected_alpha)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        assert torch.isfinite(actual) and torch.isfinite(actual_grad).all()
    checks.append("historical_V_exact_value_and_gradient_including_clamp_endpoints")

    mask = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float64)
    alpha = torch.tensor([[[0.3, 0.4], [0.7, 0.6]]], dtype=torch.float64, requires_grad=True)
    outside = helper.diagnostic_dynamic_background_opacity_loss(alpha, mask)
    outside_grad = torch.autograd.grad(outside, alpha)[0]
    assert (outside_grad[mask == 0] > 0).all() and (outside_grad[mask == 1] == 0).all()
    stepped = alpha.detach() - 0.01 * outside_grad
    assert (stepped[mask == 0] < alpha.detach()[mask == 0]).all()
    torch.testing.assert_close(stepped[mask == 1], alpha.detach()[mask == 1], rtol=0, atol=0)
    inside = reference["dynamic_opacity_loss"](alpha, mask)
    inside_grad = torch.autograd.grad(inside, alpha)[0]
    assert (inside_grad[mask == 1] < 0).all() and (inside_grad[mask == 0] == 0).all()
    total = 0.05 * reference["dynamic_opacity_loss"](alpha, mask) + 0.05 * helper.diagnostic_dynamic_background_opacity_loss(alpha, mask)
    total_grad = torch.autograd.grad(total, alpha)[0]
    assert (total_grad[mask == 1] < 0).all() and (total_grad[mask == 0] > 0).all()
    expected_sum = sum(-float(a) * __import__("math").log1p(-float(a) / 1.0001)
                       for a in alpha.detach()[mask == 0]) / 4
    torch.testing.assert_close(outside.detach(), torch.tensor(expected_sum, dtype=torch.float64))
    checks.append("two_domain_gradient_direction_and_full_HW_normalization")

    for invalid_alpha in (alpha.detach()[0], torch.zeros(2, 2, 2), torch.zeros(1, 0, 2),
                          torch.zeros(1, 2, 2, dtype=torch.int64), torch.full((1, 2, 2), float("nan")),
                          torch.full((1, 2, 2), float("inf"))):
        must_raise(ValueError, lambda a=invalid_alpha: helper.diagnostic_dynamic_background_opacity_loss(a, mask))
    for invalid_mask in (torch.zeros(2, 3), torch.full((1, 2, 2), .5), torch.full((1, 2, 2), float("nan"))):
        must_raise(ValueError, lambda m=invalid_mask: helper.diagnostic_dynamic_background_opacity_loss(alpha, m))
    checks.append("shape_finite_and_strict_binary_input_contract")

    stats = helper.raw_alpha_domain_stats(alpha, mask)
    assert stats["raw_dynamic_alpha_inside_pixels"] == stats["raw_dynamic_alpha_outside_pixels"] == 2
    assert abs(stats["raw_dynamic_alpha_inside_mean"] - .45) < 1e-12
    assert abs(stats["raw_dynamic_alpha_outside_mean"] - .55) < 1e-12
    empty = helper.raw_alpha_domain_stats(alpha, torch.ones_like(mask))
    assert empty["raw_dynamic_alpha_outside_pixels"] == empty["raw_dynamic_alpha_outside_mean"] == 0
    env = {"dynamic_mask": mask, "render_pkg": {"dynamic_alpha": alpha}, "observed_static_rgb": torch.tensor(.4),
           "loss": torch.tensor(2.4), "log_dict": {"loss_original_objective_before_diagnostic": 2.},
           "diagnostic_dynamic_background_opacity_loss": helper.diagnostic_dynamic_background_opacity_loss,
           "raw_alpha_domain_stats": helper.raw_alpha_domain_stats}
    exec(compile(__import__("textwrap").dedent(prepare.V_BLOCK), "candidate_V_integration", "exec"), env)
    torch.testing.assert_close(env["loss"], torch.tensor(2.4) + .05 * outside)
    assert env["log_dict"]["loss_resume_sv_original"] == 2.
    assert env["log_dict"]["raw_dynamic_alpha_outside_mean"] > 0
    checks.append("original_S_V_loss_accounting_and_unmasked_domain_logging")

    expected_config = {"paper_method": True, "including_dynamic": True, "diagnostic_observed_static_rgb_enabled": True,
                       "diagnostic_lambda_observed_static_rgb": 1., "paper_lambda_dynamic_opacity": .05,
                       "lambda_dssim": .2, "iterations": 50000, "position_lr_max_steps": 50000, "seed": 0}
    helper.validate_sv_config(expected_config)
    must_raise(ValueError, lambda: helper.validate_sv_config(dict(expected_config, diagnostic_lambda_observed_static_rgb=.5)))
    must_raise(ValueError, lambda: helper.validate_sv_config(dict(expected_config, paper_method=1)))
    must_raise(ValueError, lambda: helper.validate_sv_config(dict(expected_config, iterations=50000.)))
    must_raise(RuntimeError, helper.reject_unbound_gpu_training)
    plan = (json.loads((args.candidate_dir.parent / "candidate_plan.json").read_text(encoding="utf-8"))
            if args.candidate_dir else prepare.candidate_plan())
    continuation = plan["continuation"]
    assert [continuation[k] for k in ("parent_iteration", "first_iteration", "stop_iteration", "steps", "args_iterations", "position_lr_max_steps", "seed", "train_frames")] == [30000, 30001, 32000, 2000, 50000, 50000, 0, 49]
    assert plan["execution"]["allow_training"] is False and plan["execution"]["new_runtime_binding"] is None
    assert plan["execution"]["reuse_historical_binding_or_approval"] is False
    checks.append("locked_S_weights_and_fail_closed_unbound_GPU_entrypoints")

    with tempfile.TemporaryDirectory(prefix="_cpu_test_", dir=prepare.HERE) as temp:
        source, destination = Path(temp) / "source", Path(temp) / "candidate"
        (source / "utils").mkdir(parents=True)
        (source / "train.py").write_text(original, encoding="utf-8")
        (source / "utils/paper_losses.py").write_text("# 测试复制；实际数学另从历史源 AST 检查\n", encoding="utf-8")
        (source / "data").mkdir()
        (source / "data/tiny.py").write_text("# 数据目录即使是 py 也排除\n", encoding="utf-8")
        (source / "checkpoint.pth").write_bytes(b"not_loaded")
        (source / "old_approval.json").write_text("{}", encoding="utf-8")
        (source / "__pycache__").mkdir()
        prepare.materialize_source_copy(source, destination, helper_path)
        assert not any((destination / name).exists() for name in ("data", "checkpoint.pth", "old_approval.json", "__pycache__"))
        must_raise(FileExistsError, lambda: prepare.materialize_source_copy(source, destination, helper_path))
    checks.append("fresh_source_copy_rejects_overwrite_excludes_assets_cache_and_old_approval")
    return {"status": "PASS_CPU_SOURCE_CHECKS", "checks": checks, "check_count": len(checks),
            "scope": "actual_materialized_candidate" if args.candidate_dir else "actual_S_patched_in_memory_and_temporary_copy",
            "source_train": str(args.source_train), "source_train_bytes": args.source_train.stat().st_size,
            "historical_V_losses": str(args.historical_v_losses), "historical_V_losses_bytes": args.historical_v_losses.stat().st_size,
            "torch_version": torch.__version__, "device": "cpu", "threads": 1, "GPU_used": False,
            "checkpoint_loaded": False, "gpu_harness_connected": False, "training_started": False,
            "elapsed_seconds": round(time.monotonic() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--source-train", type=Path, default=prepare.SOURCE / "train.py")
    parser.add_argument("--historical-v-losses", type=Path, default=prepare.HISTORICAL_V)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_checks(args)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
