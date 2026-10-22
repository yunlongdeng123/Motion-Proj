"""V1 兼容的显示域指标、原始浮点诊断与严格49帧汇总。"""

import math

import torch
import torch.nn.functional as F


METRIC_CONTRACT = dict(
    version='V4_EVAL_V1_CLAMP_FULL_PERCEPTUAL_49_R2_20261022',
    regions='prediction and GT clamped to [0,1] as native V1; no PNG quantization',
    raw_regions='unclamped prediction and GT; diagnostic only, not historical headline metrics',
    reliable_world_sources='unclamped road/source diagnostics; original source loss definition retained',
    psnr='native utils.loss_utils.psnr; no epsilon or upper cap',
    ssim='native utils.loss_utils.ssim; full image, default 11px window',
    lpips='native lpipsPyTorch.lpips(net_type=vgg, version=0.1); direct [0,1] inputs as V1',
    aggregation='arithmetic mean of per-frame metrics, as V1; not PSNR of aggregate MSE',
    perfect_psnr='psnr=null plus psnr_is_positive_infinity=true means +Infinity, included in mean')


def region_metrics(image, gt, mask, *, clamp=True):
    from utils.loss_utils import psnr
    if image.shape != gt.shape or not torch.isfinite(image).all() or not torch.isfinite(gt).all():
        raise ValueError('RGB 形状不匹配或原始浮点包含非有限值')
    mask = mask.bool()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape != image.shape[-2:]:
        raise ValueError('评价 mask 与图像尺寸不匹配')
    pixels = int(mask.sum())
    if not pixels:
        return dict(pixels=0, mae=None, mse=None, psnr=None, psnr_is_positive_infinity=False)
    prediction = image.clamp(0, 1) if clamp else image
    target = gt.clamp(0, 1) if clamp else gt
    if not mask.all():
        prediction, target = prediction[:, mask], target[:, mask]
    mse = float(F.mse_loss(prediction, target))
    score = float(psnr(prediction, target))
    infinite = score == math.inf
    if not math.isfinite(mse) or (not math.isfinite(score) and not infinite):
        raise ValueError('非有限 RGB 误差指标')
    return dict(pixels=pixels, mae=float(F.l1_loss(prediction, target)), mse=mse,
                psnr=None if infinite else score, psnr_is_positive_infinity=infinite)


def full_perceptual_metrics(image, gt, *, lpips_fn=None):
    from utils.loss_utils import ssim
    if lpips_fn is None:
        from lpipsPyTorch import lpips as lpips_fn
    prediction, target = image.clamp(0, 1), gt.clamp(0, 1)
    # 与 V1 调用相同实现、VGG/0.1 和输入范围，不另做 [-1,1] 转换。
    values = dict(ssim=float(ssim(prediction, target)),
                  lpips=float(lpips_fn(prediction, target, net_type='vgg', version='0.1')))
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('SSIM/LPIPS 非有限值')
    return values


def aggregate49(records, frame_order):
    if len(frame_order) != 49 or len(set(frame_order)) != 49 or [r['frame'] for r in records] != list(frame_order):
        raise ValueError('49帧汇总必须完整、无重复且按冻结帧序排列')
    aggregate = dict(frame_count=49, frame_order=list(frame_order),
                     aggregation=METRIC_CONTRACT['aggregation'])
    for domain in ('regions', 'raw_regions'):
        aggregate[domain] = {}
        for output in ('v1', 'composite'):
            regions = set(records[0][domain][output])
            if any(set(record[domain][output]) != regions for record in records):
                raise ValueError('各帧评价区域集合不一致')
            aggregate[domain][output] = {}
            for region in sorted(regions):
                values = [record[domain][output][region] for record in records]
                valid = [value for value in values if value['pixels'] > 0]
                if region == 'full' and len(valid) != 49:
                    raise ValueError('完整图指标不允许缺帧')
                metrics = ('mae', 'mse', 'psnr', 'ssim', 'lpips') if domain == 'regions' and region == 'full' else ('mae', 'mse', 'psnr')
                result = dict(frame_count=49, nonempty_frame_count=len(valid),
                              pixels=sum(value['pixels'] for value in valid))
                for metric in metrics:
                    if metric == 'psnr':
                        if any(value['psnr'] is None and not value['psnr_is_positive_infinity']
                               for value in valid):
                            raise ValueError('缺失 PSNR 不能作为完美重建跳过')
                        infinite = sum(value['psnr_is_positive_infinity'] for value in valid)
                        result['psnr_is_positive_infinity'] = bool(infinite)
                        result['psnr_positive_infinity_frames'] = infinite
                        if infinite:
                            result[metric] = None
                            continue
                    samples = [value[metric] for value in valid]
                    if any(value is None or not math.isfinite(value) for value in samples):
                        raise ValueError('不能静默跳过缺失或非有限的 '+metric)
                    result[metric] = math.fsum(samples)/len(samples) if samples else None
                aggregate[domain][output][region] = result
    return aggregate
