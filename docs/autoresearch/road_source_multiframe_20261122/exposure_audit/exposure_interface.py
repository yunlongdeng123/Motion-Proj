"""Torch / NumPy 共用的已冻结标量曝光接口；不优化、不裁剪原图。"""
from pathlib import Path
import json


class SharedExposure:
    def __init__(self,path,required_frames=()):
        self.path=Path(path)
        self.receipt=json.loads(self.path.read_text())
        assert self.receipt['schema']=='source_shared_external_scalar_exposure.v1'
        assert self.receipt['calibration_frozen'] and self.receipt['anchor_raw']==115
        self.gain={int(r['raw']):float(r['gain']) for r in self.receipt['records'] if r['calibrated']}
        assert all(self.gain[r]>0 for r in self.gain)
        for raw in required_frames:
            if int(raw) not in self.gain:raise ValueError('Frame is not externally calibrated: '+str(raw))

    def canonical(self,raw,rgb):
        return rgb*self.gain[int(raw)]

    def original_exposure(self,raw,canonical_rgb):
        return canonical_rgb/self.gain[int(raw)]
