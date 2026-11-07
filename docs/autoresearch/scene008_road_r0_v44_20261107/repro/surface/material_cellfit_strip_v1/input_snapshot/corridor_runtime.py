"""Conservative world-support control: extend old road strip longitudinally.

The boundary comes from the frozen original asset, never a pixel ROI or current
render. This is a support-authority ablation, not road semantic ground truth.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from r0_runtime import R0Runtime


class CorridorRuntime(R0Runtime):
    def __init__(self, asset_path, **kwargs):
        super().__init__(asset_path, **kwargs)
        lower=float(self.rt.road_xy[:, 1].min())
        upper=float(self.rt.road_xy[:, 1].max())
        with np.load(asset_path, allow_pickle=False) as z:
            xy=z['local_xyz'][self.extension.ids, :2]
        strip=(xy[:, 1]>=lower)&(xy[:, 1]<=upper)
        self.extension.active=torch.as_tensor(strip, dtype=torch.bool, device=self.extension.xyz.device)
        self._strip_version=self.extension.active._version
        self._strip_hash=hashlib.sha256(strip.tobytes()).hexdigest()
        self.extension.record.update(
            support_policy='Forward extension restricted to frozen original road lateral envelope; one persistent world mask',
            lateral_envelope_local_m=[lower,upper],
            effective_active_rows=int(strip.sum()), inactive_outside_strip_rows=int((~strip).sum()),
            support_mask_sha256=self._strip_hash,
            support_semantics='conservative local-repair domain, not a claimed road semantic segmentation',
            scope_change='registered after wide-domain preservation failure; no shape/colour/opacity refitting')

    def receipt(self):
        assert self.extension.active._version==self._strip_version
        assert hashlib.sha256(self.extension.active.cpu().numpy().tobytes()).hexdigest()==self._strip_hash
        value=super().receipt()
        value['persistent_world_support_mask']=self.extension.record
        return value
