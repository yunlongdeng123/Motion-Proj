"""Same fixed seen quadratic; only square/mean/scalar reductions become FP64.

Inherited residual E, mask-safe Gaussian, Sobel, coefficients and the ORIGINAL
common F0 normalizers stay unchanged. Precision qualification precedes use.
"""
import torch
from native_capacity_loss import NativeCapacityLoss,RegionQuadratic,LossSpec,tensor_sha


class FP64RegionQuadratic(RegionQuadratic):
    @staticmethod
    def square_mean(x,mask):
        if not int(mask.sum()):return x.double().sum()*0.
        return x[...,mask].double().square().mean()


class FP64ReductionLoss(NativeCapacityLoss):
    def __init__(self,frames,*,spec=LossSpec(),device='cpu'):
        self.spec=spec;self.device=torch.device(device);self.regions={}
        for raw,frame in frames.items():
            self.regions[int(raw)]={name:FP64RegionQuadratic(frame['targets'][name],frame['alpha'],
                frame['contexts'][name],frame['scores'][name],spec,self.device)for name,_ in spec.source_weights}
        if not self.regions:raise ValueError('empty objective frames')
        self.normalizers=None
        self.signature={str(raw)+':'+name:dict(context=tensor_sha(r.context),score=tensor_sha(r.score),
            teacherP=tensor_sha(r.teacher_P),inv_alpha=tensor_sha(r.inv_alpha),counts=r.counts)
            for raw,regions in self.regions.items()for name,r in regions.items()}

    def calibrate(self,initial_predictions):
        raise RuntimeError('restore ORIGINAL frozen commonF0 normalizers; never recalibrate for the completion')
