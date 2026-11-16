"""Reusable optional road-inpainting adapters; no scene, actor or frame IDs.

The caller supplies a pretrained inpainting UNet, checkpoint and visible-only
structure condition.  This module does not change the default v44 scene path.
"""
import torch
from torch import nn
from torch.nn import functional as F

ATTENTION_TARGETS = ('to_q', 'to_k', 'to_v', 'to_out.0')
LOCAL_CONV_TARGETS = ('conv_in', 'conv_out') + tuple(
    f'{block}.resnets.{i}.{conv}'
    for block, count in [('down_blocks.0', 2), ('up_blocks.3', 3)]
    for i in range(count) for conv in ['conv1', 'conv2'])


class LaneAdapter(nn.Module):
    """A zero-initialized spatial residual with checkpoint-compatible keys."""
    def __init__(self, input_channels=8, hidden_channels=32, output_channels=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden_channels, output_channels, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition, size):
        return self.net(F.interpolate(condition, size=size, mode='area'))


def lora_configuration(local_convolutions=False, rank=4):
    """Configuration only: caller retains control of base weights and training."""
    from peft import LoraConfig
    targets = ATTENTION_TARGETS + (LOCAL_CONV_TARGETS if local_convolutions else ())
    return LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.,
                      target_modules=list(targets), init_lora_weights=True)


def register_lane_injection(unet, adapter, condition_getter):
    """Add a live structure residual at conv_in; return a removable hook.

    `condition_getter()` supplies B×8×H×W on the UNet device. Classifier-free
    guidance repeats the same spatial condition for conditional/unconditional
    batches. None is an error, rather than a silently disconnected branch.
    """
    def hook(module, inputs, output):
        condition = condition_getter()
        if condition is None or condition.ndim != 4:
            raise ValueError('an explicit batched visible-only lane condition is required')
        if output.shape[0] % condition.shape[0]:
            raise ValueError('condition batch is incompatible with UNet/CFG batch')
        if condition.device != output.device:
            raise ValueError('condition and UNet must share one device')
        if output.shape[0] != condition.shape[0]:
            condition = condition.repeat(output.shape[0]//condition.shape[0], 1, 1, 1)
        residual = adapter(condition, output.shape[-2:])
        if residual.shape != output.shape:
            raise ValueError('lane residual and UNet input features differ')
        return output + residual.to(output.dtype)
    return unet.conv_in.register_forward_hook(hook)


def validate_road_edit_masks(fill_mask, write_mask, protected_mask):
    """The generator's fill region and the final authorized write region differ."""
    import numpy as np
    fill, write, protected = [np.asarray(x, bool) for x in (fill_mask, write_mask, protected_mask)]
    if not (fill.shape == write.shape == protected.shape):
        raise ValueError('edit masks must share one image grid')
    if (write & ~fill).any():
        raise ValueError('final write region contains unfilled pixels')
    if (write & protected).any():
        raise ValueError('final write region intersects a retained instance')
    return dict(fill_pixels=int(fill.sum()), write_pixels=int(write.sum()),
                protected_pixels=int(protected.sum()), retained_write_overlap=0)
