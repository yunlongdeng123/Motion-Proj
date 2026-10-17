"""实例监督使用全场景 alpha*T 贡献；未知实例标签不伪造为空掩码。"""
import torch
import torch.nn.functional as F


def visible_instance_loss(contribution, masks, known):
    if contribution.shape != masks.shape or contribution.ndim != 3:
        raise ValueError('可见贡献和真实实例 mask 必须同为 KxHxW')
    known = torch.as_tensor(known, device=contribution.device, dtype=torch.bool)
    masks = masks.to(contribution.device)
    if known.shape != (len(contribution),):
        raise ValueError('每个实例必须有明确 mask_known')
    if not bool(((masks == 0) | (masks == 1)).all()):
        raise ValueError('可见实例 mask 必须是严格二值')
    if bool((masks.sum(0) > 1).any()):
        raise ValueError('同像素的可见身份标签不可重叠')
    # 每个已知实例整图 H*W，内外两域同分母，实例之间等权。
    if known.any():
        return F.mse_loss(contribution[known], masks[known].to(contribution))
    return contribution.sum() * 0


def supervision_contract(mask_known, *, freeze_background):
    """固定背景阶段允许部分身份监督；联合背景阶段需要完整真实二维归属。"""
    complete = bool(torch.as_tensor(mask_known, dtype=torch.bool).all())
    if not freeze_background and not complete:
        raise ValueError('联合背景优化前需要完整实例可见 mask；未知归属不能默认为背景')
    return {'mask_complete': complete, 'background_frozen': bool(freeze_background),
            'unknown_masks_are_zero_targets': False,
            'opacity_domain': 'known_instance_full_scene_visible_contribution',
            'instance_normalization': 'equal_known_instances_then_full_HxW'}


def local_box_boundary(model, margin=0.0):
    """局部车体限制在真实尺寸内；框约束不是像素标签，也不是重建通过证明。"""
    sizes = model.box_size
    valid = model.pose_valid
    # 各实际有效观测尺寸取中位；缺观测不能用占位零框。
    if sizes.ndim == 2:
        half = sizes / 2
    else:
        if any(not bool(valid[:, i].any()) for i in range(sizes.shape[1])):
            raise ValueError('实例缺少任何真实有效盒观测')
        half = torch.stack([sizes[valid[:, i], i].median(dim=0).values / 2 for i in range(sizes.shape[1])])
    if not bool(torch.isfinite(half).all()) or bool((half <= 0).any()):
        raise ValueError('需要真实positive finite实例尺寸')
    if not len(model.local_xyz):
        return model.local_xyz.sum() * 0
    excess = (model.local_xyz.abs() - half[model.point_ids] - float(margin)).clamp_min(0)
    return excess.square().mean()
