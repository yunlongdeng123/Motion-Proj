"""Projected state的第一层空间接口。保留原9通道；零初始化多尺度残差。

这里只实现B臂的D/V/O/N/U/Q，不偷偷把RGB身份或Y塞进空间条件。
几何/身份增益必须后续同预算对照；关闭或零初始化时严格保持原模型。
"""
import torch
from torch import nn
from torch.nn import functional as F


def pack_geometry(state):
    required = {'D', 'V', 'O', 'N', 'U', 'Q'}
    if not required <= state.keys(): raise ValueError('缺失合法状态字段')
    d, v, o, n, u, q = (state[k] for k in ['D', 'V', 'O', 'N', 'U', 'Q'])
    if not all(x.shape == d.shape for x in [v, o, n, u, q]): raise ValueError('状态尺寸不一致')
    if not (torch.all(~(o.bool() & n.bool())) and torch.all(~(u.bool() & (o.bool() | n.bool())))
            and torch.all(o.bool() | n.bool() | u.bool())):
        raise ValueError('O/N/U状态不满足排他覆盖')
    if torch.any(v.bool() != (o.bool() | n.bool())): raise ValueError('V应与正证据有效范围相符')
    known = v.bool()
    depth = torch.where(known, torch.log1p(d.clamp(0, 80))/torch.log(torch.tensor(81., device=d.device)), 0.)
    confidence = torch.where(known, q.clamp(0, 1), 0.)
    return torch.stack([depth, v.float(), o.float(), n.float(), u.float(), confidence], dim=1)


def resample_geometry(geometry, size):
    """把高分辨率证据变成区域统计，不能最近邻抽点后丢弃稀疏N。

    V/O/N/U是该特征格内有真实支持/车辆/背景/未知的面积比例；
    D/Q只在已有支持上归一化。非零比例不是整个格都已观测，更不是补出新证据。
    此处不接F或身份特征，不混合不同actor的外观。
    """
    pooled = F.adaptive_avg_pool2d(geometry.float(), size)
    coverage = pooled[:, 1:2]
    known = coverage > 0
    depth = torch.where(known, pooled[:, 0:1]/coverage.clamp_min(1e-12), 0.)
    confidence = torch.where(known, pooled[:, 5:6]/coverage.clamp_min(1e-12), 0.)
    return torch.cat([depth, pooled[:, 1:5], confidence], dim=1)


class StateResidualAdapter(nn.Module):
    channels = (320, 640, 1280, 1280)
    block_indices = (2, 5, 8, 11)

    def __init__(self, width=32):
        super().__init__()
        self.features = nn.ModuleList([nn.Sequential(nn.Conv2d(6, width, 3, padding=1), nn.SiLU(),
                                                    nn.Conv2d(width, width, 3, padding=1), nn.SiLU()) for _ in self.channels])
        self.zero_heads = nn.ModuleList([nn.Conv2d(width, c, 1) for c in self.channels])
        for layer in self.zero_heads:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.enabled = True
        self._geometry = None
        self._hole = None
        self._handles = []
        self._scales = {}

    def set_condition(self, geometry, hole):
        if geometry.ndim != 4 or geometry.shape[1] != 6: raise ValueError('B臂只接受6个空间状态通道')
        if hole.shape != (geometry.shape[0], 1, *geometry.shape[-2:]): raise ValueError('洞尺寸不一致')
        if not torch.isfinite(geometry).all(): raise ValueError('非法几何条件')
        self._geometry, self._hole = geometry.detach(), hole.detach()
        self._scales = {}

    def clear_condition(self):
        self._geometry = self._hole = None
        self._scales = {}

    def residual(self, index, activation):
        if not self.enabled or self._geometry is None: return activation
        if activation.shape[0] != self._geometry.shape[0]:
            raise ValueError('状态帧数与UNet帧数不一致；必须顺序CFG或显式复制同条件')
        if activation.shape[1] != self.channels[index]: raise ValueError('UNet结构与登记接口不符')
        size = activation.shape[-2:]
        if size not in self._scales:
            self._scales[size] = (resample_geometry(self._geometry, size),
                                  F.adaptive_max_pool2d(self._hole.float(), size))
        geom, hole = self._scales[size]
        delta = self.zero_heads[index](self.features[index](geom))*hole
        return activation+delta.to(activation.dtype)

    def attach(self, unet):
        if self._handles: raise RuntimeError('同一adapter重复接入')
        if len(unet.input_blocks) != 12: raise ValueError('只验证过官方DriveEditor12个输入块')
        for level, index in enumerate(self.block_indices):
            self._handles.append(unet.input_blocks[index].register_forward_hook(
                lambda module, args, output, level=level: self.residual(level, output)))

    def detach_hooks(self):
        for handle in self._handles: handle.remove()
        self._handles = []


def contract_checks():
    torch.manual_seed(6201)
    a = StateResidualAdapter(width=4)
    state = {k: torch.zeros(2, 8, 8) for k in ['D', 'V', 'O', 'N', 'U', 'Q']}
    state['U'][:] = 1
    for k in ['V', 'O', 'Q']: state[k][:, 2:6, 2:6] = 1
    state['U'][:, 2:6, 2:6] = 0; state['D'][:, 2:6, 2:6] = 10
    geometry = pack_geometry(state); hole = torch.zeros(2, 1, 8, 8); hole[:, :, 2:6, 2:6] = 1
    a.set_condition(geometry, hole)
    x = torch.randn(2, 320, 8, 8)
    y = a.residual(0, x); assert torch.equal(x, y)
    y.square().mean().backward()
    assert a.zero_heads[0].weight.grad.abs().sum() > 0
    with torch.no_grad(): a.zero_heads[0].weight.fill_(.03)
    y = a.residual(0, x); assert (y-x).abs().sum() > 0
    assert torch.equal(y.permute(0, 2, 3, 1)[~hole[:, 0].bool()], x.permute(0, 2, 3, 1)[~hole[:, 0].bool()])
    a.enabled = False; assert torch.equal(a.residual(0, x), x)
    a.enabled = True; a.clear_condition(); assert torch.equal(a.residual(0, x), x)
    conflict = {k: v.clone() for k, v in state.items()}; conflict['N'][:] = 1
    try: pack_geometry(conflict)
    except ValueError: pass
    else: raise AssertionError('O与N冲突未拒绝')
    # 单个背景证据落在最近邻采样点之外，仍应到达粗格；不能把全格伪称已知。
    sparse = {k: torch.zeros(1, 8, 8) for k in ['D', 'V', 'O', 'N', 'U', 'Q']}
    sparse['U'][:] = 1
    for k in ['V', 'N', 'Q']: sparse[k][0, 7, 7] = 1
    sparse['U'][0, 7, 7] = 0; sparse['D'][0, 7, 7] = 12
    packed = pack_geometry(sparse); reduced = resample_geometry(packed, (1, 1))
    assert F.interpolate(packed, (1, 1), mode='nearest')[0, 3, 0, 0] == 0
    assert reduced[0, 3, 0, 0] == 1/64 and reduced[0, 4, 0, 0] == 63/64
    assert torch.allclose(reduced[:, 2:5].sum(1), torch.ones(1, 1, 1))
    assert torch.equal(reduced[:, 0, 0, 0], packed[:, 0, 7, 7])
    assert reduced[0, 5, 0, 0] == 1
    a.set_condition(packed, torch.ones(1, 1, 8, 8))
    a.residual(0, torch.zeros(1, 320, 1, 1)); assert a._scales
    a.set_condition(packed, torch.zeros(1, 1, 8, 8)); assert not a._scales
    return {'zero_init_exact': True, 'zero_head_receives_gradient': True, 'disabled_exact': True,
            'missing_condition_exact': True, 'outside_hole_no_direct_residual': True, 'ON_conflict_rejected': True,
            'sparse_background_survives': True, 'unknown_fraction_preserved': True,
            'known_depth_not_diluted': True, 'scale_cache_resets_per_condition': True,
            'hidden_RGB_interface': False, 'quality_claim': False}


if __name__ == '__main__':
    import json
    print(json.dumps(contract_checks(), indent=2))
