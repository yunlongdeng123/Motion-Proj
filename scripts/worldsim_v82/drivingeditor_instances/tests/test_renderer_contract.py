"""CPU 合成合同；CUDA 合同必须经明确开关在实际安装版本上另行验收。"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


MODULE_PATH = Path(__file__).resolve().parents[1] / "renderer.py"
SPEC = importlib.util.spec_from_file_location("v82_instance_renderer_contract", MODULE_PATH)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def camera(timestamp=0.0, device="cpu"):
    return SimpleNamespace(
        timestamp=timestamp, FoVx=math.pi/2, FoVy=math.pi/2,
        image_height=1, image_width=1,
        world_view_transform=torch.eye(4, device=device),
        full_proj_transform=torch.eye(4, device=device),
        camera_center=torch.zeros(3, device=device),
        get_world_directions=lambda training: torch.tensor([[[0.]], [[0.]], [[1.]]], device=device),
    )


def pipe(compute_covariance=False):
    return SimpleNamespace(compute_cov3D_python=compute_covariance, convert_SHs_python=False, neg_fov=False, debug=False)


class StaticPVG:
    def __init__(self, depth=1.0, opacity=0.6, marginal=1.0):
        self.get_xyz = torch.tensor([[0., 0., depth]])
        self.get_opacity = torch.tensor([[opacity]], requires_grad=True)
        self.get_scaling = torch.ones(1, 3)
        self.get_rotation = torch.tensor([[1., 0., 0., 0.]])
        self.get_features = torch.zeros(1, 1, 3)
        self.active_sh_degree = 0
        self.marginal = torch.tensor([[marginal]])
        self.queried_times = []
        self.covariance_modifiers = []

    def get_xyz_SHM(self, timestamp):
        self.queried_times.append(timestamp)
        return self.get_xyz + torch.tensor([timestamp, 0., 0.])

    def get_marginal_t(self, timestamp):
        return self.marginal

    def get_covariance(self, scaling_modifier=1.0):
        self.covariance_modifiers.append(scaling_modifier)
        return torch.tensor([[1., 0., 0., 1., 0., 1.]]) * scaling_modifier**2


class WorldState:
    def __init__(self, xyz, opacity, ids, row_ids=None, features=None, transforms=None, rotation=None, scaling=None):
        self.xyz, self.opacity, self.point_ids = xyz, opacity, ids
        self.row_ids = torch.arange(len(xyz), device=xyz.device) if row_ids is None else row_ids
        self.features = torch.zeros(len(xyz), 1, 3, device=xyz.device) if features is None else features
        self.active_sh_degree = int(math.sqrt(self.features.shape[1])) - 1
        self.scaling = torch.ones_like(xyz) if scaling is None else scaling
        self.rotation = torch.tensor([[1., 0., 0., 0.]], device=xyz.device).expand(len(xyz), 4) if rotation is None else rotation
        self.world_from_local = torch.eye(4, device=xyz.device).expand(len(xyz), 4, 4).clone() if transforms is None else transforms
        self.direction_queries = []
        self.covariance_modifiers = []

    def local_view_directions(self, center):
        self.direction_queries.append(center)
        # 行向量 world direction @ R == 列向量 R^T @ world direction。
        return F.normalize(torch.einsum("ni,nij->nj", self.xyz-center, self.world_from_local[:, :3, :3]), dim=-1)

    def covariance(self, scaling_modifier=1., packed=True):
        assert packed
        self.covariance_modifiers.append(scaling_modifier)
        return torch.tensor([[1., 0., 0., 4., 0., 9.]], device=self.xyz.device).expand(len(self.xyz), 6) * scaling_modifier**2


class Instances:
    def __init__(self, depths=(4., 2.), opacities=(0.8, 0.9)):
        self.instance_ids = tuple(f"track_{i}" for i in range(len(depths)))
        xyz = torch.tensor([[0., 0., depth] for depth in depths], dtype=torch.float32).reshape(-1, 3)
        self.state = WorldState(xyz, torch.tensor(opacities).reshape(-1, 1).float().requires_grad_(), torch.arange(len(depths)))
        self.evaluations = []

    def evaluate(self, timestamp, edits=None):
        self.evaluations.append((timestamp, edits))
        return self.state


class ReferenceRasterizer:
    """单像素、完全重叠的深度排序参考；不冒充 CUDA 投影或真实 GPU 实测。"""

    def __init__(self, raster_settings, calls):
        self.settings, self.calls = raster_settings, calls

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        xyz, opacity, payload = kwargs["means3D"], kwargs["opacities"], kwargs["features"]
        order = torch.argsort(xyz[:, 2], stable=True)
        weights, accepted = [], []
        transmittance = xyz.new_tensor(1.)
        for idx in order:
            if not kwargs["mask"][idx]:
                continue
            alpha = opacity[idx, 0].clamp(max=0.99)
            if float(alpha.detach()) < 1/255:
                continue
            next_t = transmittance * (1-alpha)
            # 与已下载 forward.cu 一致：触发阈值的点不再加入该像素。
            if float(next_t.detach()) < 0.0001:
                break
            weights.append(alpha*transmittance)
            accepted.append(idx)
            transmittance = next_t
        if weights:
            weight = torch.stack(weights)
            indices = torch.stack(accepted)
            color = (kwargs["colors_precomp"][indices] * weight[:, None]).sum(0)
            feature = (payload[indices] * weight[:, None]).sum(0)
        else:
            color = xyz.new_zeros(3)
            feature = payload.new_zeros(payload.shape[1])
        color = color + transmittance*self.settings.bg
        return (
            torch.tensor([[[len(accepted)]]], dtype=torch.int32), color[:, None, None],
            feature[:, None, None], kwargs["mask"].to(torch.int32),
        )


def reference_factory(calls):
    return lambda raster_settings: ReferenceRasterizer(raster_settings, calls)


def run_render(static=None, instances=None, **kwargs):
    calls = []
    static = StaticPVG() if static is None else static
    instances = Instances() if instances is None else instances
    result = renderer.render(
        camera(), static, instances, pipe(), torch.zeros(3),
        rasterizer_factory=reference_factory(calls), **kwargs,
    )
    return result, calls


def test_front_vehicle_occludes_back_vehicle_in_joint_contributions():
    result, calls = run_render(static=StaticPVG(depth=10., opacity=0.5))
    # static point 输入在最前面，但 depth-sort 后是 B(.9), A(.8), static(.5)。
    assert result["instance_ids"] == ("track_0", "track_1")
    torch.testing.assert_close(result["instance_visible_contribution"][:, 0, 0], torch.tensor([0.08, 0.9]))
    torch.testing.assert_close(result["dynamic_visible_alpha"], torch.tensor([[[0.98]]]))
    torch.testing.assert_close(result["static_visible_alpha"], torch.tensor([[[0.01]]]))
    torch.testing.assert_close(result["alpha"], torch.tensor([[[0.99]]]))
    isolated_a = Instances(depths=(4.,), opacities=(0.8,))
    isolated, _ = run_render(static=StaticPVG(depth=10., opacity=0.), instances=isolated_a)
    torch.testing.assert_close(isolated["instance_visible_contribution"], torch.tensor([[[0.8]]]))
    assert result["instance_visible_contribution"][0, 0, 0] < isolated["dynamic_alpha"][0, 0, 0]
    assert calls[0]["means3D"].shape[0] == 3
    assert result["instance_visibility_semantics"] == renderer.VISIBILITY_SEMANTICS


def test_static_occluder_changes_visible_labels_but_not_raw_dynamic_body_union():
    static, model = StaticPVG(depth=1., opacity=0.6), Instances()
    result, _ = run_render(static=static, instances=model)
    torch.testing.assert_close(result["instance_visible_contribution"][:, 0, 0], torch.tensor([0.032, 0.36]))
    torch.testing.assert_close(result["dynamic_visible_alpha"], torch.tensor([[[0.392]]]))
    torch.testing.assert_close(result["dynamic_alpha"], torch.tensor([[[0.98]]]))
    assert result["dynamic_alpha"] is result["dynamic_alpha_raw"]
    assert result["visible_instance_alpha"] is result["instance_visible_contribution"]
    assert "raw_union_body_opacity" in result["dynamic_alpha_semantics"]
    # 原训练全部字段可用；没有 GT mask 参数，更不预乘监督 mask。
    assert {"render", "render_s", "render_d", "static_alpha", "dynamic_alpha", "static_feature", "static_depth", "env_map_s", "env_map"}.issubset(result)
    result["instance_visible_contribution"].sum().backward()
    torch.testing.assert_close(model.state.opacity.grad, torch.tensor([[0.04], [0.08]]))
    torch.testing.assert_close(static.get_opacity.grad, torch.tensor([[-0.98]]))


def test_inherited_dynamic_environment_fill_preserves_raw_render_and_alpha():
    calls, static, model = [], StaticPVG(depth=10.), Instances()
    env = lambda directions: directions.new_tensor([0.1, 0.2, 0.3]).expand_as(directions)
    options = dict(env_map=env, rasterizer_factory=reference_factory(calls))
    training = renderer.render(camera(), static, model, pipe(), torch.zeros(3), is_training=True, **options)
    evaluation = renderer.render(camera(), static, model, pipe(), torch.zeros(3), is_training=False, **options)
    expected_env = torch.tensor([0.1, 0.2, 0.3])[:, None, None]
    torch.testing.assert_close(training["render_d"], training["raw_render_d"] + (1-training["dynamic_alpha"])*expected_env)
    torch.testing.assert_close(evaluation["render_d"], evaluation["raw_render_d"])
    torch.testing.assert_close(training["dynamic_alpha"], evaluation["dynamic_alpha"])


def test_evaluation_under_no_grad_keeps_render_and_visibility_contract():
    with torch.no_grad():
        result, _ = run_render()
    assert not result["render"].requires_grad
    assert not result["instance_visible_contribution"].requires_grad
    torch.testing.assert_close(result["dynamic_visible_alpha"], torch.tensor([[[0.392]]]))


def test_capacity_batches_keep_every_occluder_and_identical_global_geometry():
    instances = Instances(depths=tuple(range(2, 15)), opacities=(0.07,)*13)
    static = StaticPVG(depth=1., opacity=0.3)
    result, calls = run_render(static=static, instances=instances, other=[torch.ones(1, 1), torch.zeros(1, 3)], max_features=10)
    assert result["instance_feature_passes"] == 5  # 4 other + depth/alpha/union = 7。
    full_calls = calls[:5]
    assert all(len(call["means3D"]) == 14 and call["features"].shape[1] <= 10 for call in full_calls)
    for call in full_calls[1:]:
        for key in ("means3D", "means2D", "colors_precomp", "opacities", "scales", "rotations", "mask"):
            assert call[key] is full_calls[0][key]
    # 身份通道 static=0；跨 chunk 的其他车仍保有 opacity，不可孤立 chunk。
    assert all(torch.count_nonzero(call["features"][0, 7:]) == 0 for call in full_calls)
    expected = 0.7*0.07*torch.tensor([0.93**i for i in range(13)])
    torch.testing.assert_close(result["instance_visible_contribution"][:, 0, 0], expected)
    torch.testing.assert_close(result["dynamic_visible_alpha"], result["instance_visible_contribution"].sum(0, keepdim=True))


def test_selecting_label_channels_does_not_remove_other_instances():
    result, calls = run_render(static=StaticPVG(depth=10., opacity=0.5), instance_ids=["track_0"], render_separate=False)
    assert len(calls) == 1 and len(calls[0]["means3D"]) == 3
    torch.testing.assert_close(result["instance_visible_contribution"], torch.tensor([[[0.08]]]))
    torch.testing.assert_close(result["dynamic_visible_alpha"], torch.tensor([[[0.98]]]))
    assert "dynamic_alpha" not in result


def test_object_local_sh_direction_used_even_when_pipe_requests_cuda_sh():
    model = Instances(depths=(3.,), opacities=(0.8,))
    model.state.xyz[0] = torch.tensor([1., 0., 3.])
    model.state.world_from_local[0, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    model.state.features = torch.zeros(1, 4, 3)
    model.state.features[:, 1, :] = 0.4  # y SH 对 90deg rotation 敏感。
    model.state.active_sh_degree = 1
    result, calls = run_render(instances=model)
    expected = renderer.evaluate_sh_rgb(model.state.features, model.state.local_view_directions(torch.zeros(3)), 1)
    wrong = renderer.evaluate_sh_rgb(model.state.features, F.normalize(model.state.xyz, dim=-1), 1)
    assert not torch.allclose(expected, wrong)
    torch.testing.assert_close(calls[0]["colors_precomp"][1:], expected)
    assert calls[0]["shs"] is None and model.state.direction_queries
    assert result["render"].shape == (3, 1, 1)


@pytest.mark.parametrize("python_sh", [False, True])
def test_static_sh_direction_gradient_respects_original_pipe_switch(python_sh):
    static = StaticPVG()
    static.get_xyz = torch.tensor([[1., 0., 2.]], requires_grad=True)
    static.get_features = torch.zeros(1, 4, 3, requires_grad=True)
    static.active_sh_degree = 1
    with torch.no_grad():
        static.get_features[:, 3] = 0.4  # RGB 对方向 x 有真实依赖。
    options = pipe()
    options.convert_SHs_python = python_sh
    prepared = renderer.prepare_static(camera(), static, options)
    prepared["colors_precomp"].sum().backward()
    if python_sh:
        assert static.get_xyz.grad is None
    else:
        # 当前 CUDA SH backward 含 dir 对 xyz 梯度；前向预计算不能静默丢掉。
        assert static.get_xyz.grad[0, 0] < 0 and static.get_xyz.grad[0, 2] > 0


def test_covariance_path_and_world_quaternion_path_are_joint_and_exclusive():
    static, model, cam, calls = StaticPVG(), Instances(), camera(), []
    modifier = 2.
    result = renderer.render(cam, static, model, pipe(True), torch.zeros(3), scaling_modifier=modifier, rasterizer_factory=reference_factory(calls))
    assert calls[0]["scales"] is None and calls[0]["rotations"] is None
    torch.testing.assert_close(calls[0]["cov3D_precomp"][1:], torch.tensor([[4., 0., 0., 16., 0., 36.]]).expand(2, 6))
    assert static.covariance_modifiers == [modifier] and model.state.covariance_modifiers == [modifier]
    _, quaternion_calls = run_render(instances=model)
    assert quaternion_calls[0]["cov3D_precomp"] is None
    torch.testing.assert_close(quaternion_calls[0]["rotations"][1:], model.state.rotation)
    torch.testing.assert_close(quaternion_calls[0]["scales"][1:], model.state.scaling)
    assert result["instance_row_ids"].tolist() == [0, 1]


def test_static_pvg_time_weights_mask_and_regularizer_features_remain_active():
    static, model, cam, calls = StaticPVG(opacity=0.8, marginal=0.5), Instances(), camera(0.25), []
    result = renderer.render(cam, static, model, pipe(), torch.zeros(3), other=[torch.tensor([[2., 1., 2., 3.]])], time_shift=0.1, rasterizer_factory=reference_factory(calls))
    assert static.queried_times == [0.25] and model.evaluations == [(0.25, None)]
    torch.testing.assert_close(calls[0]["means3D"][0], torch.tensor([0.25, 0., 1.]))
    torch.testing.assert_close(calls[0]["opacities"][0], torch.tensor([0.4]))
    torch.testing.assert_close(result["static_feature"][:, 0, 0], torch.tensor([0.8, 0.4, 0.8, 1.2]))
    assert torch.count_nonzero(calls[0]["features"][1:, :4]) == 0
    static.marginal[:] = 0.01
    masked, masked_calls = run_render(static=static, mask=torch.tensor([True]))
    assert not masked_calls[0]["mask"][0] and float(masked["static_alpha"][0, 0, 0]) == 0.


def test_capacity_and_id_errors_fail_before_calling_kernel():
    with pytest.raises(ValueError, match="MAX_FEATURES"):
        run_render(other=[torch.ones(1, 8)], max_features=10)
    with pytest.raises(ValueError, match="instance_ids"):
        run_render(instance_ids=["missing_track"])
    with pytest.raises(ValueError, match="instance_ids"):
        run_render(instance_ids=["track_0", "track_0"])


def test_deleted_or_inactive_tracks_keep_zero_visible_channels():
    model = Instances()
    model.state = WorldState(torch.empty(0, 3), torch.empty(0, 1), torch.empty(0, dtype=torch.long))
    result, calls = run_render(instances=model)
    assert result["instance_ids"] == ("track_0", "track_1")
    assert result["instance_visible_contribution"].shape == (2, 1, 1)
    assert torch.count_nonzero(result["instance_visible_contribution"]) == 0
    assert torch.count_nonzero(result["dynamic_alpha"]) == 0
    assert len(calls[0]["means3D"]) == 1


def test_named_edits_forward_to_the_same_joint_scene_path():
    model, calls = Instances(), []
    edits = {"track_0": {"delete": True}, "track_1": {"translation": [1., 0., 0.]}}
    renderer.render_with_edits(camera(), StaticPVG(), model, pipe(), torch.zeros(3), edits=edits, rasterizer_factory=reference_factory(calls))
    assert model.evaluations == [(0., edits)]
    assert len(calls[0]["means3D"]) == 3  # stub 只验证转发；实际 edit 几何由 rigid model 合同覆盖。


def test_actual_rigid_model_world_transforms_and_named_edits_enter_joint_renderer():
    from scripts.worldsim_v82.drivingeditor_instances.rigid_model import RigidInstanceGaussianModel

    poses = torch.eye(4).expand(1, 2, 4, 4).clone()
    poses[0, 0, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    poses[0, :, 2, 3] = torch.tensor([4., 2.])
    model = RigidInstanceGaussianModel(
        [[1., 0., 0.], [0., 0., 0.]], [[0.5]*3]*2, [[1., 2., 3.], [1., 1., 1.]],
        [0, 1], ["parked_A", "parked_B"], [0.], poses, [[True, True]],
        sh_degree=1, initial_opacity=0.8,
    )
    model.active_sh_degree = 1
    with torch.no_grad():
        model._features_rest[0, 0] = 0.4
    static, calls = StaticPVG(opacity=0.6), []
    options = dict(rasterizer_factory=reference_factory(calls))
    baseline = renderer.render(camera(), static, model, pipe(), torch.zeros(3), **options)
    torch.testing.assert_close(calls[0]["means3D"][1:], torch.tensor([[0., 1., 4.], [0., 0., 2.]]), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(calls[0]["rotations"][1], torch.tensor([2**-0.5, 0., 0., 2**-0.5]))
    torch.testing.assert_close(calls[0]["colors_precomp"][1], torch.full((3,), 0.5), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(baseline["instance_visible_contribution"][:, 0, 0], torch.tensor([0.064, 0.32]))

    calls.clear()
    rotated = renderer.render_with_edits(camera(), static, model, pipe(True), torch.zeros(3), edits={"parked_A": {"yaw_radians": math.pi/2}}, **options)
    torch.testing.assert_close(calls[0]["means3D"][1], torch.tensor([-1., 0., 4.]), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(calls[0]["cov3D_precomp"][1], torch.tensor([1., 0., 0., 4., 0., 9.]), atol=2e-6, rtol=1e-5)
    assert rotated["instance_point_ids"].tolist() == [0, 1]

    calls.clear()
    moved = renderer.render_with_edits(camera(), static, model, pipe(), torch.zeros(3), edits={"parked_B": {"translation": [0., 0., 4.]}}, **options)
    torch.testing.assert_close(calls[0]["means3D"][2], torch.tensor([0., 0., 6.]))
    torch.testing.assert_close(moved["instance_visible_contribution"][:, 0, 0], torch.tensor([0.32, 0.064]))

    calls.clear()
    deleted = renderer.render_with_edits(camera(), static, model, pipe(), torch.zeros(3), edits={"parked_B": {"delete": True}}, **options)
    assert len(calls[0]["means3D"]) == 2 and deleted["instance_row_ids"].tolist() == [0]
    torch.testing.assert_close(deleted["instance_visible_contribution"][:, 0, 0], torch.tensor([0.32, 0.]))
    # transient edits 不改写下一次正常渲染。
    normal = renderer.render(camera(), static, model, pipe(), torch.zeros(3), **options)
    torch.testing.assert_close(normal["instance_visible_contribution"], baseline["instance_visible_contribution"])


def test_actual_named_replacement_keeps_target_pose_owner_and_source_feature_rows():
    from scripts.worldsim_v82.drivingeditor_instances.rigid_model import RigidInstanceGaussianModel

    poses = torch.eye(4).expand(1, 2, 4, 4).clone()
    poses[0, 0, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    poses[0, :, 2, 3] = torch.tensor([4., 2.])
    model = RigidInstanceGaussianModel(
        [[2., 0., 0.], [1., 0., 0.]], [[0.8]*3, [0.2]*3], [[1.]*3, [1., 3., 2.]],
        [0, 1], ["target", "source"], [0.], poses, [[True, True]], sh_degree=0, initial_opacity=0.8,
    )
    with torch.no_grad():
        model._rotation[1] = torch.tensor([2**-0.5, 0., 0., 2**-0.5])
    calls = []
    result = renderer.render_with_edits(
        camera(), StaticPVG(), model, pipe(True), torch.zeros(3),
        edits={"target": {"replacement": "source"}, "source": {"translation": [10., 0., 0.]}},
        dynamic_other=[torch.tensor([[7.], [9.]])], rasterizer_factory=reference_factory(calls),
    )
    assert set(result["instance_point_ids"].tolist()) == {0, 1}
    assert result["instance_row_ids"].tolist() == [1, 1]
    target_row = 1 + int(torch.nonzero(result["instance_point_ids"] == 0)[0, 0])
    source_row = 1 + int(torch.nonzero(result["instance_point_ids"] == 1)[0, 0])
    torch.testing.assert_close(calls[0]["means3D"][target_row], torch.tensor([0., 1., 4.]), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(calls[0]["means3D"][source_row], torch.tensor([11., 0., 2.]), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(calls[0]["colors_precomp"][1:], torch.full((2, 3), 0.2))
    torch.testing.assert_close(calls[0]["features"][1:, 0], torch.tensor([9., 9.]))
    torch.testing.assert_close(calls[0]["cov3D_precomp"][target_row], torch.tensor([1., 0., 0., 9., 0., 4.]), atol=3e-6, rtol=1e-5)
    # 每条 track 的 label 通道依然按目标 owner 写 onehot，而不是 source row。
    torch.testing.assert_close(calls[0]["features"][1:, 4:], torch.eye(2)[result["instance_point_ids"]])
    torch.testing.assert_close(result["instance_visible_contribution"][:, 0, 0], torch.tensor([0.064, 0.32]))


def test_downloaded_forward_source_uses_identical_sorted_alpha_t_for_rgb_and_features():
    source_root = Path(os.environ.get("V82_RASTERIZER_SOURCE_ROOT", Path(__file__).resolve().parents[6] / "drivingeditor_hybrid30k" / "source"))
    source_path = source_root / "rasterizer_forward.cu"
    if not source_path.is_file():
        pytest.skip("实际下载的 DrivingEditor rasterizer source 未放在该 CPU workspace")
    source = source_path.read_text(encoding="utf-8")
    assert "int coll_id = point_list[range.x + progress]" in source
    assert "float weight = alpha * T" in source
    assert "colors[collected_id[j] * CHANNELS + ch] * weight" in source
    assert "features[collected_id[j] * S + ch] * weight" in source
    assert "T = test_T" in source


@pytest.mark.skipif(os.environ.get("V82_RUN_RASTERIZER_CUDA_CONTRACT") != "1", reason="实际 CUDA feature rasterizer 合同待 GPU 验收；默认 CPU 工作不启动 GPU")
def test_actual_cuda_rasterizer_joint_occlusion_contract():
    # 仅明确授权的 GPU 验收可设置开关；参考 CPU 测试不能代替这项。
    if not torch.cuda.is_available():
        pytest.skip("当前 PyTorch 没有可用 CUDA")
    pytest.importorskip("diff_gaussian_rasterization")
    static, model, cam = StaticPVG(depth=1., opacity=0.6), Instances(), camera(device="cuda")
    for obj in (static, model.state):
        for name, value in vars(obj).copy().items():
            if torch.is_tensor(value):
                setattr(obj, name, value.detach().cuda())
    # StaticPVG 的 SHM stub 创建 CPU tensor；真实 GPU 合同直接使用这组静态位置。
    static.get_xyz_SHM = lambda timestamp: static.get_xyz
    result = renderer.render(cam, static, model, pipe(), torch.zeros(3, device="cuda"))
    assert result["instance_visible_contribution"].is_cuda
    assert result["dynamic_visible_alpha"].max() < result["dynamic_alpha"].max()
    torch.testing.assert_close(result["dynamic_visible_alpha"], result["instance_visible_contribution"].sum(0, keepdim=True), atol=2e-6, rtol=2e-5)
    result["instance_visible_contribution"].sum().backward()
