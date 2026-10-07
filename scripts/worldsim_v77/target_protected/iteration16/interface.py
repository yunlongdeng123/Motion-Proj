"""保留r48同指令CFG合同，增加可选空间对应sidecar。"""
from dataclasses import dataclass
import numpy as np
import torch
from iteration14.interface import DeletionRequest as ExistingRequest, encode_references, check_request


@dataclass
class DeletionRequest(ExistingRequest):
    routing_metadata: dict | None = None

    def validate(self):
        super().validate()
        if self.routing_metadata is None:return
        t = len(self.target_rgb);r = len(self.priors['references']);h,w = self.priors['geometry'].shape[-2:]
        expected = dict(routing_query_owner=(t,1,h,w),routing_query_q=(t,1,h,w),
            routing_reference_owner=(r,8,8),routing_reference_q=(r,8,8),
            routing_projected_uv=(t,r*64,2),routing_projected_radius=(t,r*64,2),routing_projected_q=(t,r*64))
        if set(expected)!=set(self.routing_metadata):raise ValueError('空间对应字段不完整')
        for k,shape in expected.items():
            a = self.routing_metadata[k]
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('对应shape/数值错误: '+k)
            if k.endswith('_q') and (a.min()<0 or a.max()>1):raise ValueError('可信度范围错误')

    def branch_condition(self, reference_latents, device):
        result = super().branch_condition(reference_latents,device)
        if self.routing_metadata is not None:
            result.update({k:torch.from_numpy(v.copy()).to(device) for k,v in self.routing_metadata.items()})
        return result


def drop_priors(condition, drop_rgb=False, drop_geometry=False, *, drop_parameters=False):
    from iteration15.interface import drop_priors as existing
    result = existing(condition,drop_rgb,drop_geometry,drop_parameters=drop_parameters)
    # 移除任一来源时不得通过sidecar偷带RGB或几何；C指令仍保持一致。
    if drop_rgb or drop_geometry:
        for key in result:
            if key.startswith('routing_'):result[key].zero_()
    return result
