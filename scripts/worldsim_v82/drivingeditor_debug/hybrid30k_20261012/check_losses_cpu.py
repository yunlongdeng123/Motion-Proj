"""独立 CPU 数值/梯度核验；不导入训练入口，不加载模型，不初始化 CUDA。"""

import argparse
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F

from losses import (HybridLossResult, combined_opacity_entropy, compare_objective_gradients,
                    compose_hybrid_loss, diagnose_gradients, full_dynamic_opacity_mse,
                    lidar_decay_factor, static_complement_rgb_loss, static_duration_loss,
                    static_velocity_loss)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
DTYPE = torch.float64
CHECKS = []


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def record(name, fn):
    fn()
    CHECKS.append({"name": name, "passed": True})


def assignment_expression(path, target):
    # 只读取作者赋值表达式；不执行整个 train，不复制第三方训练源码。
    module = ast.parse(path.read_text(encoding="utf-8-sig"))
    expressions = [node.value for node in ast.walk(module)
                   if isinstance(node, ast.Assign)
                   and any(isinstance(lhs, ast.Name) and lhs.id == target for lhs in node.targets)]
    if len(expressions) != 1:
        raise AssertionError(str(path) + ": 无法唯一定位 " + target)
    return compile(ast.Expression(expressions[0]), str(path), "eval")


def assert_legacy_value_gradient(implementation, expression, source_value, environment):
    actual_input = source_value.clone().requires_grad_(True)
    legacy_input = source_value.clone().requires_grad_(True)
    actual = implementation(actual_input)
    legacy = expression(legacy_input, environment)
    close(actual, legacy)
    close(torch.autograd.grad(actual, actual_input)[0],
          torch.autograd.grad(legacy, legacy_input)[0])


def check_legacy(legacy_dir):
    for filename in ("train_author.py", "train_compatible.py"):
        path = legacy_dir / filename
        formulas = {name: assignment_expression(path, name) for name in (
            "loss_t_reg", "loss_v_reg", "loss_opacity_entropy", "loss_convergence_to_one")}
        environment = {"torch": torch, "nn": torch.nn}
        t_map = torch.tensor([0.3, 0.8, 1.2, 2.0], dtype=DTYPE).reshape(1, 2, 2)
        v_map = torch.linspace(-0.8, 0.7, 12, dtype=DTYPE).reshape(3, 2, 2)
        alpha = torch.tensor([-0.1, 0.25, 0.85, 1.1], dtype=DTYPE).reshape(1, 2, 2)
        mask = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
        assert_legacy_value_gradient(static_duration_loss,
            lambda x, env: eval(formulas["loss_t_reg"], dict(env, static_t_map=x)), t_map, environment)
        assert_legacy_value_gradient(static_velocity_loss,
            lambda x, env: eval(formulas["loss_v_reg"], dict(env, static_v_map=x)), v_map, environment)
        assert_legacy_value_gradient(combined_opacity_entropy,
            lambda x, env: eval(formulas["loss_opacity_entropy"],
                                dict(env, o=x.clamp(1e-6, 1-1e-6))), alpha, environment)
        assert_legacy_value_gradient(lambda x: full_dynamic_opacity_mse(x, mask),
            lambda x, env: eval(formulas["loss_convergence_to_one"],
                                dict(env, dynamic_alpha=x, target=torch.ones_like(x)*mask)),
            alpha, environment)


def check_full_dynamic_domain():
    mask = torch.tensor([[1, 0], [1, 0]], dtype=DTYPE)
    alpha = torch.tensor([0.2, 0.7, -0.1, 1.1], dtype=DTYPE).reshape(1, 2, 2).requires_grad_(True)
    loss = full_dynamic_opacity_mse(alpha, mask)
    gradient = torch.autograd.grad(loss, alpha)[0]
    close(gradient, 2.0 * (alpha.detach()-mask) / mask.numel())
    assert bool((gradient[0, :, 0] < 0).all()) and bool((gradient[0, :, 1] > 0).all())
    # 覆盖无 foreground / 全 foreground；不使用任何空域分母。
    for constant_mask in (torch.zeros_like(mask), torch.ones_like(mask)):
        close(full_dynamic_opacity_mse(alpha, constant_mask), F.mse_loss(alpha, constant_mask.unsqueeze(0)))
    try:
        full_dynamic_opacity_mse(alpha, mask + 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("软 mask 必须被拒绝")


def ssim_callback(image, target):
    # 此替身仅检验调用/权重/全图 mask 运算；SSIM 本体继续使用未改动的项目函数。
    return 1.0 - (image-target).square().mean()


def check_static_rgb():
    image = torch.linspace(0.1, 0.9, 18, dtype=DTYPE).reshape(3, 2, 3).requires_grad_(True)
    target = torch.linspace(0.8, 0.2, 18, dtype=DTYPE).reshape(3, 2, 3)
    mask = torch.tensor([[1, 0, 0], [1, 0, 1]], dtype=DTYPE)
    actual = static_complement_rgb_loss(image, target, mask, ssim_callback)
    masked_image, masked_gt = image*(1-mask), target*(1-mask)
    reference = .8*F.l1_loss(masked_image, masked_gt) + .2*(1-ssim_callback(masked_image, masked_gt))
    close(actual, reference)
    actual_gradient = torch.autograd.grad(actual, image, retain_graph=True)[0]
    close(actual_gradient, torch.autograd.grad(reference, image)[0])
    close(actual_gradient*mask, torch.zeros_like(actual_gradient))
    assert actual_gradient.shape == image.shape


def check_guard_and_decay():
    zero = torch.zeros(1, 2, 2, dtype=DTYPE, requires_grad=True)
    guarded = static_duration_loss(zero, 1e-8)
    close(guarded, torch.tensor(1e8, dtype=DTYPE))
    close(torch.autograd.grad(guarded, zero)[0], torch.zeros_like(zero))
    assert zero.requires_grad
    for step in (0, 10000, 20000, 30000):
        assert lidar_decay_factor(step, 1.0) == math.exp(-step/8000)
        assert lidar_decay_factor(step, 0.0) == 1.0
        assert lidar_decay_factor(step, -1.0) == 1.0
    try:
        static_duration_loss(zero, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("零 guard 必须被拒绝")


def build_state(paper):
    h, w = 4, 5
    def leaf(start, stop, channels):
        return torch.linspace(start, stop, channels*h*w, dtype=DTYPE).reshape(channels,h,w).requires_grad_(True)
    inputs = {
        "image": leaf(.1, .8, 3), "static_image": leaf(.15, .85, 3),
        "dynamic_image": leaf(.2, .7, 3), "alpha": leaf(.3, .8, 1),
        "static_alpha": leaf(.15, .6, 1), "dynamic_alpha": leaf(.2, .9, 1),
        "static_feature": leaf(.1, .9, 4), "velocity_feature": leaf(-.4, .5, 3),
        "rendered_depth": leaf(3, 15, 1),
    }
    gt = torch.linspace(.6, .2, 3*h*w, dtype=DTYPE).reshape(3,h,w)
    mask = torch.zeros(1,h,w,dtype=DTYPE)
    mask[:,1:3,1:4] = 1
    sky_mask = torch.zeros_like(mask)
    sky_mask[:,0,:] = 1
    normalized_legacy = inputs["static_feature"]/inputs["alpha"].clamp_min(1e-5)
    normalized_paper_v = inputs["static_feature"][1:]/inputs["static_alpha"].clamp_min(1e-5)
    lidar = torch.zeros_like(mask)
    lidar[:,1,2] = 6
    lidar[:,3,4] = 10
    state = {
        "global_rgb": .8*F.l1_loss(inputs["image"],gt)+.2*(1-ssim_callback(inputs["image"],gt)),
        "dynamic_rgb": .8*F.l1_loss(inputs["dynamic_image"]*mask,gt*mask)
                       +.2*(1-ssim_callback(inputs["dynamic_image"]*mask,gt*mask)),
        "sky_rgb": .8*F.l1_loss(inputs["static_image"]*sky_mask,gt*sky_mask)
                   +.2*(1-ssim_callback(inputs["static_image"]*sky_mask,gt*sky_mask)),
        "static_rgb": static_complement_rgb_loss(inputs["static_image"],gt,mask,ssim_callback),
        "sky": paper.sky_loss(inputs["static_alpha"],sky_mask),
        "depth": paper.sparse_inverse_depth_loss(inputs["rendered_depth"],lidar),
        # 这里只核验外部 smoothness 标量按 .1 传递，不重写或替代 Kornia。
        "inverse_depth_smoothness": inputs["rendered_depth"].reciprocal().diff(dim=-1).abs().mean(),
        "velocity_smoothness": (inputs["velocity_feature"]/inputs["alpha"]).diff(dim=-1).abs().mean(),
        "dynamic_alpha": inputs["dynamic_alpha"], "dynamic_mask": mask, "alpha": inputs["alpha"],
        "static_t_map": normalized_legacy[:1], "static_v_map": normalized_legacy[1:],
        "nps_features": normalized_paper_v, "nps_distances": paper.boundary_distance(mask),
        "iteration": 20000,
    }
    return state, inputs


def legacy_reference_sum(state, paper, enabled):
    return (state["global_rgb"]+5*state["dynamic_rgb"]+5*state["sky_rgb"]+state["static_rgb"]
            +.05*state["sky"]+.1*math.exp(-state["iteration"]/8000)*state["depth"]
            +100*F.mse_loss(state["dynamic_alpha"],state["dynamic_mask"])
            +.1/state["static_t_map"].abs().mean()+.01*state["static_v_map"].abs().mean()
            +.1*state["inverse_depth_smoothness"]+.1*state["velocity_smoothness"]
            +.05*combined_opacity_entropy(state["alpha"])
            +(.01*paper.nps_loss(state["nps_features"],state["dynamic_mask"],state["nps_distances"])
              if enabled else 0.0))


def check_compose(paper):
    for enabled in (False, True):
        state, inputs = build_state(paper)
        calls = []
        def original_nps(features, mask, distances):
            calls.append((features,mask,distances))
            return paper.nps_loss(features,mask,distances)
        result = compose_hybrid_loss(state,{"hybrid_nps_enabled":enabled},nps_fn=original_nps)
        expected = legacy_reference_sum(state,paper,enabled)
        close(result.total,expected)
        close(sum(result.weighted.values()),result.total)
        actual_gradients = torch.autograd.grad(result.total,tuple(inputs.values()),retain_graph=True)
        reference_gradients = torch.autograd.grad(expected,tuple(inputs.values()),retain_graph=True)
        for actual, reference in zip(actual_gradients,reference_gradients):
            close(actual,reference)
        assert len(calls) == int(enabled)
        if enabled:
            assert calls[0][0] is state["nps_features"]
            assert calls[0][2] is state["nps_distances"]
            close(result.raw["nps"],paper.nps_loss(*calls[0]))
        assert not any(isinstance(value,torch.Tensor) for value in result.log_values().values())
        json.dumps(result.log_values(),allow_nan=False)
    state["global_rgb"] = state["global_rgb"] * float("nan")
    try:
        compose_hybrid_loss(state,{})
    except FloatingPointError:
        pass
    else:
        raise AssertionError("总损失非 finite 必须停止")


def rng_snapshot():
    return torch.get_rng_state().clone(),random.getstate(),np.random.get_state()


def assert_rng_unchanged(snapshot):
    close(torch.get_rng_state(),snapshot[0])
    assert random.getstate() == snapshot[1]
    current = np.random.get_state()
    assert current[0] == snapshot[2][0] and np.array_equal(current[1],snapshot[2][1])
    assert current[2:] == snapshot[2][2:]


def check_diagnostics():
    x = torch.tensor([.4,.8],dtype=DTYPE,requires_grad=True)
    retained = x.square()
    retained.retain_grad()
    deeper_retained = retained*3
    deeper_retained.retain_grad()
    for tensor in (x,retained,deeper_retained):
        tensor.grad = torch.full_like(tensor,.123)
    protected = [(tensor,tensor.grad,tensor.grad.clone()) for tensor in (x,retained,deeper_retained)]
    raw = {"static_rgb": (.2*deeper_retained).sum(), "nps": (.5*deeper_retained).sum()}
    weights = {"static_rgb":1.0,"nps":.01}
    weighted = {name:value*weights[name] for name,value in raw.items()}
    result = HybridLossResult(sum(weighted.values()),raw,weighted,weights,{})
    before_rng = rng_snapshot()
    diagnostics = diagnose_gradients(result,{"leaf":x,"retained":retained},
                                    preserve_grad_tensors=[deeper_retained])
    rewrite = compare_objective_gradients(result.total,result.total,
                                         {"leaf":x,"retained":retained},
                                         preserve_grad_tensors=[deeper_retained])
    assert_rng_unchanged(before_rng)
    for tensor,original,saved in protected:
        assert tensor.grad is original
        close(tensor.grad,saved)
    assert math.isclose(diagnostics["grad_hybrid/nps/leaf/weighted_l2_over_static_rgb"], .025,
                        rel_tol=1e-12, abs_tol=1e-15)
    assert rewrite["rewrite/leaf/delta_l2"] == 0.0
    assert not any(isinstance(value,torch.Tensor) for value in diagnostics.values())
    result.total.backward()
    close(x.grad,torch.full_like(x,.123)+1.23*x.detach())
    json.dumps(diagnostics,allow_nan=False)


def nps_summary(paper):
    state,inputs = build_state(paper)
    result = compose_hybrid_loss(state,{"hybrid_nps_enabled":True},nps_fn=paper.nps_loss)
    values = diagnose_gradients(result,{"static_feature":inputs["static_feature"],
                                        "raw_static_alpha":inputs["static_alpha"],
                                        "combined_alpha":inputs["alpha"]},
                               term_names=["nps","static_velocity","static_duration"])
    return {"scope":"确定性 4x5 CPU 合成图，仅验证日志入口；不能替代真实首步梯度量级",
            "raw":float(result.raw["nps"].detach().item()),
            "weighted":float(result.weighted["nps"].detach().item()),"gradients":values}


def main():
    parser = argparse.ArgumentParser()
    staging = Path(__file__).resolve().parents[6]/"drivingeditor_hybrid30k"
    parser.add_argument("--legacy-dir",type=Path,default=staging/"legacy")
    parser.add_argument("--paper-losses",type=Path,default=staging/"source"/"utils"/"paper_losses.py")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("unchanged_paper_losses_cpu",args.paper_losses)
    paper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper)
    record("actual_author_and_compatible_AST_value_gradient_equivalence",lambda:check_legacy(args.legacy_dir))
    record("raw_full_MSE_inside_outside_direction_and_extreme_masks",check_full_dynamic_domain)
    record("static_RGB_full_complement_08_02_callback_value_gradient",check_static_rgb)
    record("duration_denominator_guard_and_legacy_LiDAR_decay",check_guard_and_decay)
    record("whole_composition_value_gradient_original_NPS_on_off_and_finite_guard",lambda:check_compose(paper))
    record("diagnostics_preserve_leaf_retained_grad_RNG_and_final_backward",check_diagnostics)
    assert not torch.cuda.is_initialized()
    print(json.dumps({"passed":True,"cpu_only":True,"threads":torch.get_num_threads(),
                      "cuda_initialized":False,"checks":CHECKS,
                      "nps_synthetic_summary":nps_summary(paper)},ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
