"""r47结构和state_dict不变；仅在指定采样调用记录模块响应。"""
import torch
import math
from iteration14.multi_prior import MultiPriorBranch as R47Branch


class MultiPriorBranch(R47Branch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture = None
        self.response_rows = []
        self._components = {}
        # 只保存几个标量，不保存全视频激活或完整注意力矩阵。
        for family in ('rgb', 'geometry'):
            for index, head in enumerate(getattr(self, family+'_heads')):
                head.register_forward_hook(
                    lambda module, args, output, f=family, i=index: self._capture_head(f, i, output))

    def _capture_head(self, family, index, output):
        if self.capture is not None:
            rms = float(output.detach().float().square().mean().sqrt())
            if not math.isfinite(rms):raise ValueError('模块响应非有限，停止推理')
            self._components[(family, index)] = rms

    def residual(self, index, activation):
        result = super().residual(index, activation)
        if self.capture is not None and self.enabled:
            delta = (result.detach().float()-activation.detach().float())
            base = float(activation.detach().float().square().mean().sqrt())
            drms = float(delta.square().mean().sqrt())
            if not math.isfinite(base) or not math.isfinite(drms):raise ValueError('融合响应非有限，停止推理')
            self.response_rows.append(dict(self.capture, block=self.blocks[index],
                activation_rms=base, gated_delta_rms=drms, delta_to_activation=drms/max(base, 1e-12),
                rgb_head_rms=self._components.get(('rgb', index)),
                geometry_head_rms=self._components.get(('geometry', index))))
        return result

    def start_response(self):
        self.response_rows = []
        self._components = {}
        self.capture = None
