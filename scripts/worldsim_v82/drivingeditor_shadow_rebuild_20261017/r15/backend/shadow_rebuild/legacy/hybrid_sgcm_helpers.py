"""SGCM 新增点准入及 clone 谱系；不修改 Eq8 或渲染器。

完整训练视图的中心投影与时间 alpha 是保守准入代理，并非遮挡后的
逐高斯渲染贡献。renderer.contrib 是每像素序号，不用于逐点准入。
本模块不消耗随机数。Torch 在运行期按需导入，CPU 测试无需 CUDA。
"""
import math

import numpy as np


LINEAGE_KEY = "sgcm_lineage_v1"
ALPHA_CUTOFF = 1.0 / 255.0
TEMPORAL_PREFILTER = 0.05
MIN_VIEW_DEPTH = 0.2


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def freeze_training_views(cameras):
    """复制实际 Camera 的小矩阵和 timestamp，不访问 RGB、mask 或真值。"""
    records = []
    for camera in cameras or ():
        world_view = _numpy(camera.world_view_transform)
        full_proj = _numpy(camera.full_proj_transform)
        timestamp = float(camera.timestamp)
        width, height = int(camera.image_width), int(camera.image_height)
        if (world_view.shape != (4, 4) or full_proj.shape != (4, 4)
                or not np.isfinite(world_view).all() or not np.isfinite(full_proj).all()
                or not math.isfinite(timestamp) or width <= 0 or height <= 0):
            raise ValueError("SGCM needs finite actual training-camera projections/timestamps")
        if not np.allclose(world_view[:, 3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError("SGCM camera transform must use Camera row-vector convention")
        name = str(getattr(camera, "image_path", getattr(camera, "image_name", len(records))))
        records.append({"world_view": world_view, "full_proj": full_proj,
                        "timestamp": timestamp, "width": width, "height": height, "id": name})
    if not records:
        raise ValueError("Guarded SGCM requires the complete actual training-camera collection")
    return tuple(records)


def enable_lineage(model, resuming=False):
    """首次训练初始化；恢复时必须已有 checkpoint 谱系以保留 clone 额度。"""
    import torch
    if getattr(model, "_sgcm_lineage_ids", None) is None:
        if resuming:
            raise ValueError("Guarded SGCM resume requires saved sgcm_lineage_v1 metadata")
        model._sgcm_lineage_ids = torch.arange(len(model.get_xyz), device=model.get_xyz.device,
                                               dtype=torch.int64)
        model._sgcm_clone_blocked = torch.zeros(len(model.get_xyz), device=model.get_xyz.device,
                                                dtype=torch.bool)
    validate_lineage(model)


def validate_lineage(model, expected_length=None):
    import torch
    ids = getattr(model, "_sgcm_lineage_ids", None)
    blocked = getattr(model, "_sgcm_clone_blocked", None)
    count = len(model.get_xyz) if expected_length is None else expected_length
    if (not isinstance(ids, torch.Tensor) or not isinstance(blocked, torch.Tensor)
            or ids.dtype != torch.int64 or blocked.dtype != torch.bool
            or ids.shape != (count,) or blocked.shape != (count,)
            or ids.device != model.get_xyz.device or blocked.device != model.get_xyz.device
            or (ids < 0).any()):
        raise RuntimeError("SGCM lineage does not match Gaussian topology")


def bind_sgcm_guard(controller, cameras, model, resuming=False):
    if not controller.config.guard_enabled:
        return
    if model is None:
        raise ValueError("Guarded SGCM must bind its static GaussianModel before densification")
    controller.training_views = freeze_training_views(cameras)
    enable_lineage(model, resuming=resuming)
    timestamps = [view["timestamp"] for view in controller.training_views]
    controller.metadata["addition_guard"] = {
        "enabled": True, "training_view_count": len(timestamps),
        "training_time_range": [min(timestamps), max(timestamps)],
        "training_view_ids": [view["id"] for view in controller.training_views],
        "temporal_prefilter_strict_gt": TEMPORAL_PREFILTER,
        "effective_alpha_min": ALPHA_CUTOFF, "view_depth_strict_gt_model": MIN_VIEW_DEPTH,
        "projection": "Camera row-vector world_view/full_proj; center NDC x,y in [-1,1]",
        "semantics": "training-view support proxy; no occlusion or exact per-Gaussian contribution",
        "clone_limit": "one SGCM clone per initial source lineage; all descendants inherit spent state",
        "split_children": "both proposed children need support in at least one training view",
        "checkpoint": "legacy 18-field capture tuple; optional optimizer.sgcm_lineage_v1 payload",
    }


def lineage_after_append(model, parent_indices, child_count):
    """由 optimizer-aware postfix 调用；传入的是 append 前的 parent 行。"""
    import torch
    if getattr(model, "_sgcm_lineage_ids", None) is None:
        return
    old_count = len(model.get_xyz) - child_count
    validate_lineage(model, expected_length=old_count)
    if child_count == 0:
        return
    if (not isinstance(parent_indices, torch.Tensor) or parent_indices.dtype != torch.int64
            or parent_indices.shape != (child_count,) or parent_indices.device != model.get_xyz.device
            or (parent_indices < 0).any() or (parent_indices >= old_count).any()):
        raise RuntimeError("SGCM append requires aligned source-parent indices")
    model._sgcm_lineage_ids = torch.cat((model._sgcm_lineage_ids,
                                         model._sgcm_lineage_ids[parent_indices]))
    model._sgcm_clone_blocked = torch.cat((model._sgcm_clone_blocked,
                                           model._sgcm_clone_blocked[parent_indices]))
    validate_lineage(model)


def lineage_after_prune(model, valid_mask):
    if getattr(model, "_sgcm_lineage_ids", None) is None:
        return
    validate_lineage(model, expected_length=len(valid_mask))
    model._sgcm_lineage_ids = model._sgcm_lineage_ids[valid_mask]
    model._sgcm_clone_blocked = model._sgcm_clone_blocked[valid_mask]
    validate_lineage(model)


def capture_sgcm_optimizer(model):
    """附加键只在 SGCM guard 被激活时出现；普通分支 state_dict 原样返回。"""
    state = model.optimizer.state_dict()
    if getattr(model, "_sgcm_lineage_ids", None) is None:
        return state
    validate_lineage(model)
    state = dict(state)
    state[LINEAGE_KEY] = {"schema_version": 1,
                          "lineage_ids": model._sgcm_lineage_ids.detach().cpu().clone(),
                          "clone_blocked": model._sgcm_clone_blocked.detach().cpu().clone()}
    return state


def restore_sgcm_optimizer(model, state):
    """先还原行 metadata，再将纯 Adam state 交给 load_state_dict。"""
    import torch
    state = dict(state)
    payload = state.pop(LINEAGE_KEY, None)
    model._sgcm_lineage_ids = None
    model._sgcm_clone_blocked = None
    if payload is not None:
        if (not isinstance(payload, dict)
                or set(payload) != {"schema_version", "lineage_ids", "clone_blocked"}
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or not isinstance(payload["lineage_ids"], torch.Tensor)
                or not isinstance(payload["clone_blocked"], torch.Tensor)
                or payload["lineage_ids"].dtype != torch.int64
                or payload["clone_blocked"].dtype != torch.bool):
            raise ValueError("Invalid SGCM checkpoint lineage payload")
        model._sgcm_lineage_ids = payload["lineage_ids"].to(model.get_xyz.device).clone()
        model._sgcm_clone_blocked = payload["clone_blocked"].to(model.get_xyz.device).clone()
        validate_lineage(model)
    return state


def training_view_support(controller, model, parent_indices, xyz_override=None):
    """实际离散 training timestamp 的静态 SHM、prefilter 和中心视锥代理。

    alpha 为峰值上界，忽略像素离中心的衰减和遮挡。没有通过意味着
    未获本代理准入，不能直接宣称整个训练窗的 exact RGB 贡献为零。
    """
    import torch
    views = getattr(controller, "training_views", ())
    if not views:
        raise ValueError("Guarded SGCM has no actual training views")
    xyz = model.get_xyz[parent_indices].detach() if xyz_override is None else xyz_override.detach()
    center = model.get_t[parent_indices, 0].detach()
    sigma = model.get_scaling_t[parent_indices, 0].detach()
    opacity = model.get_opacity[parent_indices, 0].detach()
    velocity = model._velocity[parent_indices].detach()
    cycle = float(model.T)
    if (xyz.shape != (len(parent_indices), 3) or velocity.shape != xyz.shape
            or not math.isfinite(cycle) or cycle <= 0
            or not all(torch.isfinite(value).all() for value in (xyz, center, sigma, opacity, velocity))
            or (sigma <= 0).any()):
        raise ValueError("Invalid static temporal/geometry inputs for SGCM training-view support")
    supported = torch.zeros(len(parent_indices), dtype=torch.bool, device=xyz.device)
    angular = 2.0 * math.pi / cycle
    for view in views:
        delta = center - view["timestamp"]
        marginal = torch.exp(-0.5 * (delta / sigma) ** 2)
        temporal = (marginal > TEMPORAL_PREFILTER) & (opacity * marginal >= ALPHA_CUTOFF)
        active = torch.nonzero(temporal & ~supported, as_tuple=False).flatten()
        if not len(active):
            continue
        # 等价于 static prepare_rasterization 中 get_xyz_SHM(timestamp)。
        means = xyz[active] + velocity[active] * (torch.sin(-delta[active] * angular) / angular)[:, None]
        homogeneous = torch.cat((means, torch.ones_like(means[:, :1])), dim=1)
        world_view = torch.as_tensor(view["world_view"], dtype=xyz.dtype, device=xyz.device)
        full_proj = torch.as_tensor(view["full_proj"], dtype=xyz.dtype, device=xyz.device)
        depth = (homogeneous @ world_view)[:, 2]
        clip = homogeneous @ full_proj
        valid_w = clip[:, 3] > 1e-7
        ndc_xy = clip[:, :2] / clip[:, 3:4].clamp_min(1e-7)
        in_view = ((depth > MIN_VIEW_DEPTH) & valid_w
                   & torch.isfinite(ndc_xy).all(dim=1) & (ndc_xy.abs() <= 1.0).all(dim=1))
        supported[active] |= in_view
        if supported.all():
            break
    return supported


def split_child_geometry(model, parent_indices):
    """与原 SGCM 的 largest-axis 两子点构造逐项一致。"""
    import torch
    xyz = model.get_xyz[parent_indices].detach()
    scales = model.get_scaling[parent_indices].detach()
    largest = scales.argmax(dim=1)
    quaternion = model._rotation[parent_indices].detach()
    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(dim=1)
    rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                            2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                            2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=1).reshape(-1, 3, 3)
    direction = rotation[torch.arange(len(parent_indices), device=xyz.device), :, largest]
    offset = direction * scales.gather(1, largest[:, None]) / math.sqrt(2.0)
    return torch.cat((xyz-offset, xyz+offset))


def guard_additions(controller, model, clone, split):
    import torch
    validate_lineage(model)
    clone_ids = torch.nonzero(clone, as_tuple=False).flatten()
    split_ids = torch.nonzero(split, as_tuple=False).flatten()
    clone_supported = training_view_support(controller, model, clone_ids)
    split_supported = training_view_support(controller, model, split_ids)
    child_supported = torch.ones(len(split_ids), dtype=torch.bool, device=clone.device)
    if len(split_ids):
        children = split_child_geometry(model, split_ids)
        supported = training_view_support(controller, model, split_ids.repeat(2), xyz_override=children)
        child_supported = supported[:len(split_ids)] & supported[len(split_ids):]
    blocked = model._sgcm_clone_blocked[clone_ids]
    result_clone, result_split = torch.zeros_like(clone), torch.zeros_like(split)
    result_clone[clone_ids] = clone_supported & ~blocked
    result_split[split_ids] = split_supported & child_supported
    stats = {"addition_guard_enabled": True,
             "geometric_candidate_clone": len(clone_ids), "geometric_candidate_split": len(split_ids),
             "excluded_clone_training_view_support": int((~clone_supported).sum().item()),
             "excluded_split_training_view_support": int((~split_supported).sum().item()),
             "excluded_split_child_training_view_support": int((split_supported & ~child_supported).sum().item()),
             "excluded_cloned_lineage": int((clone_supported & blocked).sum().item()),
             "training_view_count": len(controller.training_views)}
    return result_clone, result_split, stats


def select_clone_once(model, clone_ids, radii, budget):
    """同谱系至多选择一个；以原 radii 优先级稳定选择。"""
    import torch
    if not len(clone_ids) or budget <= 0:
        return clone_ids[:0], 0
    ordered = clone_ids[torch.argsort(radii[clone_ids], descending=True, stable=True)]
    _, inverse = torch.unique(model._sgcm_lineage_ids[ordered], sorted=True, return_inverse=True)
    first = torch.full((int(inverse.max().item())+1,), len(ordered), dtype=torch.int64, device=ordered.device)
    rank = torch.arange(len(ordered), dtype=torch.int64, device=ordered.device)
    first.scatter_reduce_(0, inverse, rank, reduce="amin", include_self=True)
    unique = ordered[torch.sort(first).values]
    return unique[:budget], len(ordered)-len(unique)


def mark_clone_lineages_spent(model, clone_ids):
    import torch
    if not len(clone_ids):
        return
    validate_lineage(model)
    selected = model._sgcm_lineage_ids[clone_ids]
    model._sgcm_clone_blocked |= torch.isin(model._sgcm_lineage_ids, selected)
