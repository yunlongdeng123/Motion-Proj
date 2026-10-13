"""CPU 合同测试：实例编辑、观测有效性、姿态梯度与完整优化状态恢复。"""

import copy
import math

import numpy as np
import pytest
import torch

from scripts.worldsim_v82.drivingeditor_instances.rigid_model import (
    RigidInstanceGaussianModel,
    SH_C0,
    quaternion_to_matrix,
)


def bundle():
    poses = np.broadcast_to(np.eye(4, dtype=np.float64), (3, 3, 4, 4)).copy()
    poses[:, 0, :3, 3] = [[10, 0, 0], [12, 0, 0], [14, 0, 0]]
    poses[:, 1, :3, 3] = [[20, 5, 0], [23, 5, 0], [26, 5, 0]]
    poses[:, 2, :3, 3] = [[30, 0, 0]] * 3
    return dict(
        local_xyz=np.array([[1, 0, 0], [0, 2, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        rgb=np.array([[0.9, 0.2, 0.1], [0.7, 0.2, 0.1], [0.1, 0.3, 0.9], [0.1, 0.5, 0.9]], dtype=np.float64),
        scale=np.array([[1, 2, 3], [1.5, 1, 0.5], [0.5, 0.8, 1], [0.7, 0.4, 0.6]], dtype=np.float64),
        point_ids=np.array([0, 0, 1, 1]), instance_ids=np.array(["silver", "black", "zero-lidar"]),
        timestamps=np.array([-0.02, 0, 0.02], dtype=np.float64), poses=poses,
        valid=np.ones((3, 3), dtype=bool), color_known=np.array([True, True, True, False]),
        box_size=np.ones((3, 3, 3)), physical_timestamps=np.array([1.5e9, 1.5e9 + 0.1, 1.5e9 + 0.2]),
        raw_timestamp_micros=np.array([1500000000000000, 1500000000100000, 1500000000200000]),
    )


def model(**kwargs):
    return RigidInstanceGaussianModel.from_bundle(bundle(), dtype=torch.float64, sh_degree=1, **kwargs)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def training_loss(instance_model):
    state = instance_model.evaluate(-0.01)
    return (state.xyz.square().mean() + state.features.square().mean() + state.scaling.square().mean()
            + state.opacity.square().mean() + state.covariance(packed=False)[:, 0, 1].sum()
            + instance_model.pose_regularization(smoothness_weight=0.03)["total"])


def test_npz_initializes_only_measured_local_geometry(tmp_path):
    path = tmp_path / "prepared.npz"
    np.savez(path, **bundle())
    instances = RigidInstanceGaussianModel.from_npz(path, dtype=torch.float64, sh_degree=2)
    source = bundle()
    assert torch.equal(instances.local_xyz, torch.from_numpy(source["local_xyz"]))
    assert torch.equal(instances.point_ids, torch.from_numpy(source["point_ids"]))
    assert instances.track_ids == ("silver", "black", "zero-lidar")
    assert instances.num_points == 4 and instances.num_instances == 3
    snapshot = instances.evaluate(-0.02)
    torch.testing.assert_close(snapshot.features[:, 0] * SH_C0 + 0.5, torch.from_numpy(source["rgb"]))
    assert snapshot.features.shape == (4, 9, 3)
    assert torch.equal(snapshot.features[:, 1:], torch.zeros_like(snapshot.features[:, 1:]))
    torch.testing.assert_close(snapshot.scaling, torch.from_numpy(source["scale"]))
    assert instances.pose_translation_raw.requires_grad is False
    assert instances.pose_rotation_raw.requires_grad is False
    assert not hasattr(instances, "_velocity") and not hasattr(instances, "_t")


def test_delete_translate_and_rotate_are_independent():
    instances = model()
    original_parameters = {name: parameter.detach().clone() for name, parameter in instances.named_parameters()}
    baseline = instances.evaluate(-0.02)
    instances.translate("silver", [3, -1, 2])
    translated = instances.evaluate(-0.02)
    torch.testing.assert_close(translated.xyz[:2], baseline.xyz[:2] + torch.tensor([3, -1, 2]))
    assert torch.equal(translated.xyz[2:], baseline.xyz[2:])
    instances.rotate("silver", yaw_radians=math.pi / 2)
    rotated = instances.evaluate(-0.02)
    expected = torch.tensor([[13, 0, 2], [11, -1, 2]], dtype=torch.float64)
    torch.testing.assert_close(rotated.xyz[:2], expected, atol=1e-12, rtol=0)
    assert torch.equal(rotated.xyz[2:], baseline.xyz[2:])
    instances.set_deleted("silver")
    deleted = instances.evaluate(-0.02)
    assert deleted.point_ids.tolist() == [1, 1]
    assert deleted.row_ids.tolist() == [2, 3]
    assert torch.equal(deleted.xyz, baseline.xyz[2:])
    for name, parameter in instances.named_parameters():
        assert torch.equal(parameter, original_parameters[name])
    instances.reset_edits()
    assert torch.equal(instances.evaluate(-0.02).xyz, baseline.xyz)


def test_covariance_and_sh_directions_rotate_with_object():
    instances = model()
    baseline = instances.evaluate(-0.02)
    edited = instances.evaluate(-0.02, {"silver": {"yaw_radians": math.pi / 2}})
    ninety_degrees = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
    covariance = baseline.covariance(packed=False)
    expected = ninety_degrees @ covariance[:2] @ ninety_degrees.T
    torch.testing.assert_close(edited.covariance(packed=False)[:2], expected, atol=1e-12, rtol=0)
    assert torch.equal(edited.covariance(packed=False)[2:], covariance[2:])
    camera_at_object_center = [10, 0, 0]
    torch.testing.assert_close(edited.local_view_directions(camera_at_object_center)[:2],
                               baseline.local_view_directions(camera_at_object_center)[:2], atol=1e-12, rtol=0)
    assert edited.covariance().shape == (4, 6)
    assert torch.equal(edited.get_xyz_SHM(900, times=4), edited.xyz)
    assert torch.equal(edited.get_marginal_t(900), torch.ones_like(edited.opacity))
    # 临时编辑不得污染下一次正常查询。
    assert torch.equal(instances.evaluate(-0.02).xyz, baseline.xyz)


def test_pose_and_local_orientation_compose_for_anisotropic_gaussians():
    data = bundle()
    data["poses"][:, 0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    instances = RigidInstanceGaussianModel.from_bundle(data, dtype=torch.float64)
    with torch.no_grad():
        instances._rotation[0] = torch.tensor([math.sqrt(0.5), math.sqrt(0.5), 0, 0])
    state = instances.evaluate(-0.02)
    local = quaternion_to_matrix(instances._rotation[0])
    object_rotation = torch.tensor(data["poses"][0, 0, :3, :3])
    world_rotation = quaternion_to_matrix(state.rotation[0])
    torch.testing.assert_close(world_rotation, object_rotation @ local, atol=1e-12, rtol=0)
    expected = world_rotation @ torch.diag(torch.tensor(data["scale"][0]) ** 2) @ world_rotation.T
    torch.testing.assert_close(state.covariance(packed=False)[0], expected, atol=1e-12, rtol=0)


def test_missing_observations_are_not_filled_or_extrapolated():
    data = bundle()
    data["valid"][1, 1] = False
    data["poses"][1, 1] = np.nan
    instances = RigidInstanceGaussianModel.from_bundle(data)
    for timestamp in [-0.01, 0, 0.01]:
        assert instances.evaluate(timestamp).point_ids.tolist() == [0, 0]
    assert instances.evaluate(-0.02).point_ids.tolist() == [0, 0, 1, 1]
    assert instances.evaluate(0.02).point_ids.tolist() == [0, 0, 1, 1]
    for timestamp in [-0.0201, 0.0201, 4.0]:
        empty = instances.evaluate(timestamp)
        assert empty.xyz.shape == (0, 3)
        assert empty.covariance().shape == (0, 6)
        assert empty.local_view_directions([0, 0, 0]).shape == (0, 3)
    torch.testing.assert_close(instances.evaluate(-0.01).xyz[0], torch.tensor([12.0, 0, 0]))
    exact = RigidInstanceGaussianModel.from_bundle(data, interpolation="exact")
    assert exact.evaluate(-0.01).xyz.shape[0] == 0
    with pytest.raises(ValueError, match="finite scalar"):
        instances.evaluate(float("nan"))


def test_slerp_rotates_halfway_without_pvg_motion():
    data = bundle()
    data["poses"][1:, 0, :3, :3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    instances = RigidInstanceGaussianModel.from_bundle(data, dtype=torch.float64)
    halfway = instances.evaluate(-0.01)
    torch.testing.assert_close(halfway.xyz[0], torch.tensor([11, 1, 0], dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(halfway.xyz[1], torch.tensor([9, 0, 0], dtype=torch.float64), atol=1e-12, rtol=0)


def test_replacement_uses_source_asset_and_target_pose_and_identity():
    instances = model()
    baseline = instances.evaluate(-0.02)
    replaced = instances.evaluate(-0.02, {"silver": {"replacement": "black"}})
    owner = replaced.point_ids == 0
    original_source = replaced.point_ids == 1
    assert replaced.row_ids[owner].tolist() == [2, 3]
    torch.testing.assert_close(replaced.xyz[owner], torch.tensor([[11, 0, 0], [10, 1, 0]], dtype=torch.float64))
    assert torch.equal(replaced.features[owner], baseline.features[2:])
    assert torch.equal(replaced.scaling[owner], baseline.scaling[2:])
    assert torch.equal(replaced.xyz[original_source], baseline.xyz[2:])
    assert torch.equal(instances.evaluate(-0.02).xyz, baseline.xyz)
    instances.replace("silver", "black")
    assert torch.equal(instances.evaluate(-0.02).xyz, replaced.xyz)
    instances.set_deleted("black")
    target_only = instances.evaluate(-0.02)
    assert target_only.point_ids.tolist() == [0, 0]
    assert target_only.row_ids.tolist() == [2, 3]
    target_only.features.sum().backward()
    assert torch.equal(instances._features_dc.grad[:2], torch.zeros_like(instances._features_dc.grad[:2]))
    assert torch.all(instances._features_dc.grad[2:] == 1)
    with pytest.raises(ValueError, match="no initialized"):
        instances.replace("silver", "zero-lidar")


def test_pose_refinement_has_finite_zero_residual_gradients_and_strict_bounds():
    instances = model(refine_poses=True)
    loss = training_loss(instances)
    loss.backward()
    for parameter in (instances.pose_translation_raw, instances.pose_rotation_raw):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0
    with torch.no_grad():
        instances.pose_translation_raw[:] = 20
        instances.pose_rotation_raw[:] = -30
    translation, rotation = instances.bounded_pose_residuals()
    assert torch.all(torch.linalg.vector_norm(translation, dim=-1) < instances.pose_translation_limit)
    assert torch.all(torch.linalg.vector_norm(rotation, dim=-1) < instances.pose_rotation_limit)
    assert torch.isfinite(instances.pose_regularization()["total"])


def test_pose_regularization_ignores_invalid_frames_and_penalizes_variation():
    data = bundle()
    data["valid"][:, 1] = False
    instances = RigidInstanceGaussianModel.from_bundle(data, refine_poses=True, dtype=torch.float64)
    with torch.no_grad():
        instances.pose_translation_raw[0, 0, 0] = 0.1
        instances.pose_translation_raw[:, 1] = 200
        instances.pose_rotation_raw[:, 1] = 200
    terms = instances.pose_regularization()
    assert terms["translation_prior"] > 0 and terms["translation_smoothness"] > 0
    assert terms["rotation_prior"] == 0 and terms["rotation_smoothness"] == 0
    terms["total"].backward()
    assert torch.equal(instances.pose_translation_raw.grad[:, 1], torch.zeros((3, 3), dtype=torch.float64))
    assert torch.equal(instances.pose_rotation_raw.grad[:, 1], torch.zeros((3, 3), dtype=torch.float64))


def test_clone_prune_preserves_ids_and_adam_moments_and_keeps_training():
    instances = model(refine_poses=True)
    optimizer = instances.make_optimizer(lr=0.003, pose_lr=0.002, amsgrad=True)
    training_loss(instances).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    original_moments = {name: optimizer.state[parameter]["exp_avg"].clone()
                        for name, parameter in instances.named_parameters() if name.startswith("_")}
    old_xyz = instances.local_xyz.detach().clone()
    appended = instances.clone_points([0, 2], offsets=[[0.1, 0, 0], [0, 0.1, 0]])
    assert appended.tolist() == [4, 5]
    assert instances.point_ids.tolist() == [0, 0, 1, 1, 0, 1]
    assert instances.gaussian_ids.tolist() == [0, 1, 2, 3, 4, 5]
    assert instances.parent_gaussian_ids.tolist() == [0, 1, 2, 3, 0, 2]
    torch.testing.assert_close(instances.local_xyz[4], old_xyz[0] + torch.tensor([0.1, 0, 0]))
    for name, parameter in instances.named_parameters():
        if name in original_moments:
            moment = optimizer.state[parameter]["exp_avg"]
            assert torch.equal(moment[:4], original_moments[name])
            assert torch.equal(moment[4:], torch.zeros_like(moment[4:]))
            assert optimizer.state[parameter]["max_exp_avg_sq"].shape == parameter.shape
    retained = instances.prune_points([False, True, False, False, False, True])
    assert retained.tolist() == [0, 2, 3, 4]
    assert instances.point_ids.tolist() == [0, 1, 1, 0]
    assert instances.gaussian_ids.tolist() == [0, 2, 3, 4]
    for name, parameter in instances.named_parameters():
        if name in original_moments:
            assert torch.equal(optimizer.state[parameter]["exp_avg"][:3], original_moments[name][[0, 2, 3]])
    training_loss(instances).backward()
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in instances.parameters())
    instances.set_deleted("silver")
    assert instances.evaluate(-0.02).point_ids.tolist() == [1, 1]


def test_checkpoint_restores_every_parameter_adam_and_edit_then_continues_exactly(tmp_path):
    instances = model(refine_poses=True)
    optimizer = instances.make_optimizer({"f_rest": 0.0003}, lr=0.002, pose_lr=0.001, amsgrad=True)
    training_loss(instances).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    instances.clone_points([0])
    instances.translate("silver", [0.2, 0.1, 0])
    instances.rotate("silver", yaw_radians=0.1)
    instances.replace("silver", "black")
    instances.set_deleted("zero-lidar")
    instances.oneupSHdegree()
    instances.eval()
    captured = instances.capture()
    path = tmp_path / "checkpoint.pt"
    torch.save(captured, path)
    restored = RigidInstanceGaussianModel.from_capture(torch.load(path, weights_only=True))
    assert_tree_equal(captured, restored.capture())
    assert restored.training is False and restored.active_sh_degree == 1
    assert restored.track_ids == instances.track_ids
    assert torch.equal(instances.evaluate(-0.01).xyz, restored.evaluate(-0.01).xyz)
    for candidate in [instances, restored]:
        candidate.optimizer.zero_grad(set_to_none=True)
        training_loss(candidate).backward()
        candidate.optimizer.step()
    assert_tree_equal(instances.capture(), restored.capture())
    # capture 是稳定快照，后续训练不应反向改写待保存的旧状态。
    assert not torch.equal(captured["state_dict"]["_xyz"], instances.local_xyz)
    original_identity = id(restored)
    restored.restore(captured)
    assert id(restored) == original_identity
    assert_tree_equal(captured, restored.capture())


def test_restored_optimizer_update_cannot_modify_the_supplied_checkpoint():
    original = model(refine_poses=True)
    optimizer = original.make_optimizer()
    training_loss(original).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    captured = original.capture()
    immutable_reference = copy.deepcopy(captured)
    # 同一个CPU checkpoint直接恢复，专门检查load_state_dict同设备tensor别名。
    restored = RigidInstanceGaussianModel.from_capture(captured)
    training_loss(restored).backward()
    restored.optimizer.step()
    assert_tree_equal(captured, immutable_reference)
    assert not torch.equal(restored.local_xyz, captured['state_dict']['_xyz'])


def test_densification_accumulates_duplicate_source_rows_and_prunes_low_opacity():
    instances = model()
    instances.accumulate_densification_stats(
        [2, 2, 0, 1], [[3, 4, 0], [0, 2, 0], [1, 0, 0], [100, 100, 0]],
        visible=[True, True, True, False], radii=[1, 4, 2, 30],
    )
    assert instances.xyz_gradient_accum.flatten().tolist() == [1, 0, 7, 0]
    assert instances.denom.flatten().tolist() == [1, 0, 2, 0]
    assert instances.max_radii2D.tolist() == [2, 0, 4, 0]
    with torch.no_grad():
        instances._opacity[3] = -20
    result = instances.densify_and_prune(2.0, 0.001, max_new_points=1)
    assert result == {"cloned": 1, "pruned": 1, "points": 4}
    assert instances.point_ids.tolist() == [0, 0, 1, 1]
    assert instances.gaussian_ids.tolist() == [0, 1, 2, 4]
    assert instances.parent_gaussian_ids[-1] == 2
    assert torch.equal(instances.local_xyz[-1], instances.local_xyz[2])
    assert not instances.xyz_gradient_accum.any() and not instances.denom.any()
    assert not instances.max_radii2D.any()


def test_rejects_corrupt_geometry_and_allows_pruning_to_empty():
    data = bundle()
    data["poses"][0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="orthonormal"):
        RigidInstanceGaussianModel.from_bundle(data)
    instances = model()
    with pytest.raises(ValueError, match="bool vector"):
        instances.prune_points([0, 1, 0, 0])
    with pytest.raises(KeyError, match="unknown instance"):
        instances.set_deleted("unseen-track")
    instances.prune_points(torch.ones(4, dtype=torch.bool))
    snapshot = instances.evaluate(-0.02)
    assert snapshot.xyz.shape == (0, 3)
    assert snapshot.world_from_local.shape == (0, 4, 4)
    restored = RigidInstanceGaussianModel.from_capture(instances.capture())
    assert restored.num_points == 0
