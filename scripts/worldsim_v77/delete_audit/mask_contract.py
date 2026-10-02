"""实例覆盖合同：GT只作诊断，不能裁掉SAM里的可见车身。"""
import cv2
import numpy as np


def prepare_masks(sam, gt_target=None, neighbor_envelopes=None, visible_protected=None):
    core = np.asarray(sam, dtype=bool).copy()
    if core.ndim != 2:
        raise ValueError('实例mask必须是二维')
    height, width = core.shape
    model = np.zeros_like(core)
    write = cv2.dilate(core.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    if write.any():
        yy, xx = np.where(write)
        model[max(0, int(yy.min())-8):min(height, int(yy.max())+26),
              max(0, int(xx.min())-8):min(width, int(xx.max())+10)] = True
    neighbor = np.zeros_like(core) if neighbor_envelopes is None else np.asarray(neighbor_envelopes, bool)
    observed = None if visible_protected is None else np.asarray(visible_protected, bool)
    conflict = np.zeros_like(core) if observed is None else core & observed
    if conflict.any():
        raise ValueError('目标实例与可见保护实例重叠，须解决身份冲突后再构造条件')
    protect = (neighbor if observed is None else observed) & model & ~write
    d = cv2.distanceTransform(model.astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    fade = np.clip(d/8, 0, 1)
    alpha = fade*fade*(3-2*fade)
    alpha[protect] = 0
    alpha[write] = 1
    alpha[~model] = 0
    if observed is not None:
        # 可见保护车比外围膨胀优先；目标与保护实例冲突已在上面拒绝。
        write &= ~observed
        alpha[observed] = 0
    assert np.all(model[core]) and np.all(alpha[core] == 1)
    assert np.all(alpha[~model] == 0)
    gt = np.zeros_like(core) if gt_target is None else np.asarray(gt_target, bool)
    n, _, stats, _ = cv2.connectedComponentsWithStats(core.astype('uint8'), 8)
    details = {'mask_policy': 'sam_full_v2', 'empty_target': not bool(core.any()),
               'target_pixels': int(core.sum()), 'target_outside_model': int((core & ~model).sum()),
               'target_alpha_not_one': int((core & (alpha != 1)).sum()),
               'SAM_outside_GT_diagnostic_only': int((core & ~gt).sum()),
               'neighbor_envelope_overlap_diagnostic_only': int((core & neighbor).sum()),
               'component_areas': stats[1:, cv2.CC_STAT_AREA].tolist() if n > 1 else [],
               'protected_instance_masks_available': observed is not None,
               'instance_identity_approved': None}
    return {'core': core, 'write_mask': write, 'model_mask': model,
            'protect': protect, 'alpha': alpha}, details


def verify():
    sam = np.zeros((64, 96), bool); sam[20:42, 20:65] = True
    gt = np.zeros_like(sam); gt[20:42, 30:60] = True
    arrays, info = prepare_masks(sam, gt)
    assert info['SAM_outside_GT_diagnostic_only'] > 0
    assert np.all(arrays['model_mask'][sam]) and np.all(arrays['alpha'][sam] == 1)
    empty, _ = prepare_masks(np.zeros_like(sam), gt)
    assert all(not v.any() for v in empty.values())
    protected = np.zeros_like(sam); protected[15:48, 66:75] = True
    arrays, _ = prepare_masks(sam, gt, visible_protected=protected)
    assert not arrays['alpha'][protected].any()
    try:
        prepare_masks(sam, gt, visible_protected=sam)
    except ValueError:
        pass
    else:
        raise AssertionError('目标/保护身份冲突未拒绝')
    return {'GT_clip_removed': True, 'empty_frame_safe': True,
            'visible_protected_preserved': True, 'identity_conflict_rejected': True}
