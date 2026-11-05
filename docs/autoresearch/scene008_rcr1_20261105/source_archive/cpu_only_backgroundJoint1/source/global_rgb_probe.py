"""One joint RGB term; backgroundJoint1 uses the original Ownership background."""
import torch.nn.functional as F
from native_ownership import Ownership, guarded


def ownership_background(evidence, union):
    collar = F.max_pool2d(evidence['visible_reviewed'].any(0)[None, None].float(), 5, 1, 2)[0].bool()
    background = ~collar & (union == 0)
    return background


class LossProbeOwnership(Ownership):
    def __init__(self, mode):
        if mode not in ('control', 'global', 'global_background'):
            raise ValueError(mode)
        super().__init__()
        self.loss_probe_mode = mode

    def terms(self, result, evidence, target, native_depth, sky_mask, union, args, iteration):
        terms = super().terms(result, evidence, target, native_depth, sky_mask, union, args, iteration)
        if self.loss_probe_mode in ('global', 'global_background'):
            from utils.hybrid30k_losses import photometric_rgb_loss
            from utils.loss_utils import ssim
            if self.loss_probe_mode == 'global':
                terms['global_rgb'] = photometric_rgb_loss(result['render'], target, ssim, args.lambda_dssim)
            else:
                background = ownership_background(evidence, union)
                terms['joint_background_rgb'] = photometric_rgb_loss(
                    guarded(result['render'], background), guarded(target, background), ssim, args.lambda_dssim)
        return terms
