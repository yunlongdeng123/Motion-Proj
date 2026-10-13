"""训练入口的 CPU 预检/优先级合同；所有外部 GPU 查询均为测试 stub。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


TRAIN_PATH = Path(__file__).resolve().parents[1] / "train.py"


def spec_fixture(tmp_path, *, complete=False):
    """只用于入口合同的单元 fixture，不可作为真实场景或实验输入。"""
    import numpy as np

    spec = {"scene_id": "test_fixture_0017"}
    for key in ("source_repository", "source_scene", "prepared", "matrix_tools"):
        directory = tmp_path / key
        directory.mkdir()
        spec[key] = str(directory)
    for key in ("reference_config", "background_checkpoint", "environment_checkpoint"):
        file = tmp_path / (key + ".json")
        file.write_text("{}", encoding="utf-8")
        spec[key] = str(file)
    known = np.full((49, 2), complete, dtype=bool)
    if not complete:
        known[:, 0] = True
    prepared = Path(spec["prepared"])
    frames = [f"fixture_{i:04d}" for i in range(49)]
    np.savez_compressed(
        prepared / "instances.npz", frame_ids=np.asarray(frames),
        instance_ids=np.asarray(["fixture_A", "fixture_B"]),
        timestamps=np.arange(49)*.02-.48,
        local_xyz=np.zeros((2, 3), dtype=np.float32), rgb=np.full((2, 3), .5, dtype=np.float32),
        scale=np.ones((2, 3), dtype=np.float32), point_ids=np.arange(2),
        poses=np.broadcast_to(np.eye(4), (49, 2, 4, 4)), valid=np.ones((49, 2), dtype=bool),
        box_size=np.ones((49, 2, 3), dtype=np.float32), mask_known=known,
        normalization=np.diag([.1, .1, .1, 1.]),
    )
    report = {
        "geometry_ready": True, "frames": [{"frame_id": name} for name in frames],
        "instance_count": 2, "real_instance_points": 2,
        "masks": {"unit_test_only": True, "known_pairs": int(known.sum())},
        "full_mask_supervision_ready": complete, "mask_known_pairs": int(known.sum()),
    }
    (prepared / "preparation_report.json").write_text(json.dumps(report), encoding="utf-8")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def fresh_entry(code, tmp_path):
    # 独立进程可以证明入口没有借用其他 pytest 用例已经导入的 Torch。
    prefix = f"""
import builtins, json, pathlib, runpy, sys
sys.path.insert(0, {str(TRAIN_PATH.parent)!r})
original_import = builtins.__import__
def deny_gpu_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'tensorflow', 'diff_gaussian_rasterization', 'kornia', 'scene'):
        raise AssertionError('CPU admission imported GPU/model module: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = deny_gpu_import
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-I", "-c", prefix + code], cwd=tmp_path, env=environment,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return json.loads(result.stdout)


def test_default_train_validation_imports_no_torch_or_models_and_creates_no_output(tmp_path):
    spec_path, _ = spec_fixture(tmp_path)
    output = tmp_path / "no_training_output"
    result = fresh_entry(f"""
import priority
def unexpected_admission(*args, **kwargs):
    raise AssertionError('default CPU validation requested a GPU slot')
priority.require_gpu_slot = unexpected_admission
sys.argv = [{str(TRAIN_PATH)!r}, '--spec', {str(spec_path)!r}, '--output', {str(output)!r}]
runpy.run_path({str(TRAIN_PATH)!r}, run_name='__main__')
assert 'torch' not in sys.modules
assert not pathlib.Path({str(output)!r}).exists()
""", tmp_path)
    assert result["status"] == "CPU_SPEC_VALIDATED_GPU_NOT_STARTED"
    assert result["default_background_frozen"] is True
    assert result["full_mask_supervision_ready"] is False


def test_run_rejects_streetcrafter_preparation_before_torch_or_output_creation(tmp_path):
    spec_path, _ = spec_fixture(tmp_path)
    proc_root = tmp_path / "proc_fixture"
    pid = proc_root / "123"
    pid.mkdir(parents=True)
    # 活跃 shell 准备脚本也必须拦截，即使假定 nvidia-smi 当前没有 GPU worker。
    (pid / "cmdline").write_bytes(b"/bin/bash\0/root/StreetCrafter/prepare_data.sh\0")
    output, lock = tmp_path / "blocked_training_output", tmp_path / "gpu_slot_fixture.lock"
    result = fresh_entry(f"""
import priority, subprocess, types
actual_inspect = priority.inspect_priority
def empty_gpu_query(command, **kwargs):
    assert command[0] == 'nvidia-smi'
    return subprocess.CompletedProcess(command, 0, '', '')
priority.inspect_priority = lambda proc_root='/proc': actual_inspect({str(proc_root)!r}, empty_gpu_query, own_pid=456)
# Windows 测试仅 stub Linux flock；真实 require_gpu_slot 的检查、拒绝、关闭路径仍执行。
sys.modules['fcntl'] = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
sys.argv = [{str(TRAIN_PATH)!r}, '--spec', {str(spec_path)!r}, '--run', '--output', {str(output)!r}, '--gpu-lock', {str(lock)!r}]
try:
    runpy.run_path({str(TRAIN_PATH)!r}, run_name='__main__')
except RuntimeError as error:
    receipt = json.loads(str(error))
    assert receipt['allowed'] is False
    assert receipt['streetcrafter_processes'][0]['pid'] == 123
    assert 'torch' not in sys.modules
    assert not pathlib.Path({str(output)!r}).exists()
    print(json.dumps({{'rejected': True, 'receipt': receipt, 'torch_imported': False, 'output_created': False}}))
else:
    raise AssertionError('priority competitor did not block --run')
""", tmp_path)
    assert result["rejected"] is True
    assert result["torch_imported"] is False and result["output_created"] is False
    assert result["receipt"]["other_compute_processes"] == []


def test_joint_background_validation_rejects_partial_real_mask_table_without_torch(tmp_path):
    spec_path, _ = spec_fixture(tmp_path)
    result = fresh_entry(f"""
sys.argv = [{str(TRAIN_PATH)!r}, '--spec', {str(spec_path)!r}, '--joint-background']
try:
    runpy.run_path({str(TRAIN_PATH)!r}, run_name='__main__')
except ValueError as error:
    assert 'torch' not in sys.modules
    print(json.dumps({{'rejected': True, 'error': str(error), 'torch_imported': False}}))
else:
    raise AssertionError('partial masks passed joint background validation')
""", tmp_path)
    assert result["rejected"] and result["torch_imported"] is False


def test_cpu_validation_rejects_stale_complete_mask_receipt_against_actual_bundle(tmp_path):
    spec_path, spec = spec_fixture(tmp_path)
    report_path = Path(spec["prepared"]) / "preparation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["full_mask_supervision_ready"] = True  # 实际 bundle 仍有未知身份。
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = fresh_entry(f"""
sys.argv = [{str(TRAIN_PATH)!r}, '--spec', {str(spec_path)!r}, '--joint-background']
try:
    runpy.run_path({str(TRAIN_PATH)!r}, run_name='__main__')
except ValueError as error:
    assert 'torch' not in sys.modules
    print(json.dumps({{'rejected': True, 'error': str(error), 'torch_imported': False}}))
else:
    raise AssertionError('stale complete-mask receipt overrode actual unknown masks')
""", tmp_path)
    assert result["rejected"] and result["torch_imported"] is False


def tiny_training_package():
    """真实 renderer/core 接口 + 两像素透明度参考，不模拟 CUDA 投影质量。"""
    import torch
    from scripts.worldsim_v82.drivingeditor_instances.renderer import render
    from scripts.worldsim_v82.drivingeditor_instances.rigid_model import RigidInstanceGaussianModel
    from scripts.worldsim_v82.drivingeditor_instances.tests.test_renderer_contract import StaticPVG, camera, pipe

    class TwoPixelRasterizer:
        def __init__(self, raster_settings):
            self.settings = raster_settings

        def __call__(self, **kwargs):
            xyz, payload = kwargs["means3D"], kwargs["features"]
            transmission = xyz.new_ones(2)
            rgb, feature = xyz.new_zeros((3, 2)), payload.new_zeros((payload.shape[1], 2))
            count = torch.zeros(2, dtype=torch.int32)
            for point in torch.argsort(xyz[:, 2], stable=True):
                # 前车 B 只覆盖左像素；后车 A 的右像素尾部属于 mask 外泄漏。
                coverage = xyz.new_tensor([1., 0.]) if float(xyz[point, 2].detach()) == 2. else xyz.new_ones(2)
                alpha = (kwargs["opacities"][point, 0]*coverage).clamp(max=0.99)
                next_t = transmission*(1-alpha)
                accepted = kwargs["mask"][point] & (alpha >= 1/255) & (next_t >= 0.0001)
                weight = torch.where(accepted, alpha*transmission, torch.zeros_like(alpha))
                rgb = rgb + kwargs["colors_precomp"][point, :, None]*weight[None]
                feature = feature + payload[point, :, None]*weight[None]
                transmission = torch.where(accepted, next_t, transmission)
                count = count + accepted.to(torch.int32)
            rgb = rgb + self.settings.bg[:, None]*transmission[None]
            return count[None, None], rgb[:, None], feature[:, None], kwargs["mask"].to(torch.int32)

    poses = torch.eye(4).expand(1, 2, 4, 4).clone()
    poses[0, :, 2, 3] = torch.tensor([4., 2.])
    model = RigidInstanceGaussianModel(
        torch.zeros(2, 3), torch.full((2, 3), .5), torch.ones(2, 3), [0, 1],
        ["A", "B"], [0.], poses, [[True, True]], sh_degree=0, initial_opacity=.8,
    )
    with torch.no_grad():
        model._opacity[1] = torch.logit(torch.tensor(.9))
    static = StaticPVG(depth=1., opacity=.6)
    static.get_opacity = static.get_opacity.detach()  # 与首阶段固定原静态分支一致。
    cam = camera()
    cam.image_width = 2
    package = render(cam, static, model, pipe(), torch.zeros(3), rasterizer_factory=TwoPixelRasterizer)
    masks = torch.tensor([[[0., 0.]], [[1., 0.]]])
    return package, masks, model, static


def test_actual_training_mask_objective_uses_joint_contribution_inside_and_outside_gradients():
    import torch
    from scripts.worldsim_v82.drivingeditor_instances.train import instance_mask_objective

    package, masks, model, static = tiny_training_package()
    torch.testing.assert_close(package["instance_visible_contribution"], torch.tensor([[[.032, .32]], [[.36, 0.]]]))
    objective = instance_mask_objective(package, masks, [True, True], ["A", "B"])
    torch.testing.assert_close(objective, torch.tensor(.128256))
    # 训练主循环明确使用 100；既包含 B 内部趋向1，也包含 A/背景域趋向0。
    (100*objective).backward()
    torch.testing.assert_close(model._opacity.grad, torch.tensor([[1.03424], [-1.19808]]), atol=2e-5, rtol=2e-5)
    assert static.get_opacity.grad is None
    before = model._opacity.detach().clone()
    optimizer = model.make_optimizer(lr=.001)
    optimizer.step()
    assert model._opacity[0] < before[0] and model._opacity[1] > before[1]


def test_actual_training_mask_objective_ignores_isolated_alpha_and_unknown_targets():
    import torch
    from scripts.worldsim_v82.drivingeditor_instances.train import instance_mask_objective

    package, masks, model, _ = tiny_training_package()
    original = instance_mask_objective(package, masks, [True, True], ["A", "B"])
    package["dynamic_alpha"] = torch.full_like(package["dynamic_alpha"], .123)
    altered = instance_mask_objective(package, masks, [True, True], ["A", "B"])
    torch.testing.assert_close(altered, original)
    partial = instance_mask_objective(package, masks, [False, True], ["A", "B"])
    partial.backward()
    assert model._opacity.grad[0] == 0  # 后车身份未知，不得到虚假的空mask直接梯度。
    assert model._opacity.grad[1] < 0


def test_actual_training_mask_objective_rejects_isolated_semantics_and_wrong_identity_order():
    from scripts.worldsim_v82.drivingeditor_instances.train import instance_mask_objective

    package, masks, _, _ = tiny_training_package()
    with pytest.raises(ValueError):
        instance_mask_objective(package, masks, [True, True], ["B", "A"])
    package["instance_visibility_semantics"] = "isolated_instance_alpha"
    with pytest.raises(ValueError):
        instance_mask_objective(package, masks, [True, True], ["A", "B"])
