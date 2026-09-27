"""v77 hybrid 的 CPU mask / 写回合同；不运行神经网络。"""
import cv2
import numpy as np


def mask_contract(delete, protect, observed, dilation_px):
    """observed 是通过来源/遮挡检查的布尔支持，不是原始浮点 confidence。"""
    masks = [np.asarray(x) for x in (delete, protect, observed)]
    if any(x.dtype != np.bool_ or x.ndim != 2 for x in masks):
        raise ValueError('mask 必须是二维 bool；confidence 必须先显式准入')
    if len({x.shape for x in masks}) != 1:
        raise ValueError('mask 分辨率不一致')
    if not isinstance(dilation_px, int) or dilation_px < 0:
        raise ValueError('dilation_px 必须是非负整数')
    delete, protect, observed = masks
    if np.any(delete & protect):
        raise ValueError('目标可见区域与保护区域冲突：须检查实例归属，不能静默删减目标')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_px+1,)*2)
    generate = (cv2.dilate(delete.astype(np.uint8), kernel) > 0) & ~protect
    accepted = observed & generate
    residual_delete = delete & ~accepted
    residual_generate = generate & ~accepted
    return dict(delete=delete.copy(), protect=protect.copy(), generate=generate,
                observed=accepted, residual_delete=residual_delete,
                residual_generate=residual_generate)


def compose_background(original, factual_evidence, generated, masks):
    """扩展区内允许生成；真实证据锁定；保护区和扩展区外逐像素保持。"""
    arrays = [np.asarray(x) for x in (original, factual_evidence, generated)]
    if any(x.dtype != np.uint8 or x.shape != (*masks['delete'].shape, 3) for x in arrays):
        raise ValueError('RGB 输入必须为同尺寸 uint8 HWC')
    result = original.copy()
    result[masks['residual_generate']] = generated[masks['residual_generate']]
    result[masks['observed']] = factual_evidence[masks['observed']]
    result[masks['protect']] = original[masks['protect']]
    return result
