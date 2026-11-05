"""Single-term probe: add the omitted unit-weight joint full-image RGB loss."""
from native_ownership import Ownership


class LossProbeOwnership(Ownership):
    def __init__(self, mode):
        if mode not in ('control', 'global'):
            raise ValueError(mode)
        super().__init__()
        self.loss_probe_mode = mode

    def terms(self, result, evidence, target, native_depth, sky_mask, union, args, iteration):
        terms = super().terms(result, evidence, target, native_depth, sky_mask, union, args, iteration)
        if self.loss_probe_mode == 'global':
            from utils.hybrid30k_losses import photometric_rgb_loss
            from utils.loss_utils import ssim
            terms['global_rgb'] = photometric_rgb_loss(result['render'], target, ssim, args.lambda_dssim)
        return terms
