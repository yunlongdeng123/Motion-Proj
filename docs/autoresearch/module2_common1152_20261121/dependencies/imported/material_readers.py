"""相同世界场的两种读取；native adapter 显式传入，无默认场景加载。"""
import copy
import numpy as np
import torch

from world_texture_runtime import (FixedGaussianReadLayout, FixedPixelReadLayout, compose_native_eligible)


def frozen_tensor(value, device, dtype=None):
    tensor = torch.as_tensor(np.array(value, copy=True), device=device, dtype=dtype)
    if tensor.requires_grad or not bool(torch.isfinite(tensor).all()):
        raise ValueError('原生常量须为冻结 finite tensor')
    return tensor


class GaussianAverageReader:
    kind = 'avg'
    renderer_mode = 'shared_world_field_GH31_native_GS'

    def __init__(self, operator, gate, active_world_rows, qualified_active, parent_rgb, adapter, manifest):
        self.operator, self.gate, self.adapter = operator, gate, adapter
        if gate not in operator.KH:
            raise ValueError('固定平均算子缺许可 gate')
        self.parent_rgb = parent_rgb.detach().clone()
        if parent_rgb.requires_grad or parent_rgb.shape != (len(active_world_rows), 3):
            raise ValueError('父高斯颜色/拓扑不同')
        self.layout = FixedGaussianReadLayout(operator.row_ids, active_world_rows,
                                             row_id_kind='global_world_row').to(parent_rgb.device)
        scope = np.asarray(qualified_active)
        if scope.dtype != bool or scope.shape != (len(active_world_rows),):
            raise ValueError('必须显式传 actual-qualified active mask')
        covered = np.zeros(len(scope), bool)
        covered[self.layout.active_positions.cpu().numpy()] = True
        if np.any(scope & ~covered):
            raise ValueError('actual scope 含平均算子之外的 world ID')
        self.scope = torch.as_tensor(scope.copy(), device=parent_rgb.device)
        self.record = copy.deepcopy(manifest)
        self.versions = (self.parent_rgb._version, self.scope._version)

    def colors(self, model):
        low, high = model.fields()
        values = self.operator.torch_query(low, high, self.gate)
        colors = self.layout(self.parent_rgb, values)
        return torch.where(self.scope[:, None], colors, self.parent_rgb)

    def render(self, model, raw, joint=False):
        colors = self.colors(model)
        road = self.adapter.render_average(int(raw), colors, joint=False)
        result = dict(road_rgb=road['rgb'], road_alpha=road['alpha'].squeeze(0), K=road['K'])
        if joint:
            out = self.adapter.render_average(int(raw), colors, joint=True)
            result.update(joint_rgb=out['rgb'], joint_alpha=out['alpha'].squeeze(0), K=out['K'])
        return result

    def require_execution(self, role):
        self.adapter.require_execution(role)

    def assert_frozen(self):
        if (self.parent_rgb._version, self.scope._version) != self.versions:
            raise RuntimeError('父颜色/资格 mask 被改动')
        self.adapter.assert_frozen()

    def manifest(self):
        return dict(self.record, reader=self.kind, renderer_mode=self.renderer_mode,
                    row_id_kind='global_world_row', gate=self.gate,
                    source_normalized=True, independent_ci_parameters=False)


class PixelWorldReader:
    kind = 'pixel'
    renderer_mode = 'shared_world_field_pixel_box_native_eligible_weights'

    def __init__(self, operators, gate, basis, device, adapter, manifest):
        self.operators, self.gate, self.adapter = operators, gate, adapter
        self.record = copy.deepcopy(manifest)
        self.basis, self.layouts, self.source_mass = {}, {}, {}
        for raw, operator in operators.items():
            if gate not in operator.KH:
                raise ValueError('固定 pixel 算子缺许可 gate')
            packets = {}
            for mode in ('road', 'joint'):
                source = basis[int(raw)][mode]
                packet = {key: frozen_tensor(source[key], device,
                          torch.bool if key == 'K' else torch.float32)
                          for key in ('baseline_rgb', 'eligible_rgb', 'eligible_weight', 'alpha', 'K')}
                shape = packet['alpha'].shape
                if len(shape) == 3 and shape[0] == 1:
                    packet['alpha'] = packet['alpha'][0]
                    shape = packet['alpha'].shape
                if packet['eligible_weight'].shape == (1, *shape):
                    packet['eligible_weight'] = packet['eligible_weight'][0]
                if packet['baseline_rgb'].shape != (3, *shape) or packet['eligible_rgb'].shape != (3, *shape):
                    raise ValueError('native basis 不是 CHW')
                if packet['eligible_weight'].shape != shape or packet['K'].shape != shape:
                    raise ValueError('native basis 权重/K shape 不同')
                if mode == 'joint' and bool((packet['eligible_weight'][packet['K']] != 0).any()):
                    raise ValueError('joint K 上 eligible 权重非零；不得屏幕 mask 修补')
                packets[mode] = packet
            self.basis[int(raw)] = packets
            height, width = packets['road']['alpha'].shape
            self.layouts[int(raw)] = FixedPixelReadLayout(operator.row_ids, operator.mass,
                height, width, row_id_kind='pixel_flat').to(device)
            mass = np.zeros(height * width, np.float64)
            mass[np.asarray(operator.row_ids, np.int64)] = operator.mass
            self.source_mass[int(raw)] = frozen_tensor(mass.reshape(height, width), device, torch.float32)
        self.versions = {(raw, mode, key): value._version for raw, modes in self.basis.items()
                         for mode, packet in modes.items() for key, value in packet.items()}

    def render(self, model, raw, joint=False):
        raw = int(raw)
        low, high = model.fields()
        values = self.operators[raw].torch_query(low, high, self.gate)
        colors, valid = self.layouts[raw](values)
        road = self.basis[raw]['road']
        result = dict(road_rgb=compose_native_eligible(road['baseline_rgb'], road['eligible_rgb'],
                          road['eligible_weight'], colors, valid),
                      road_alpha=road['alpha'], K=self.basis[raw]['joint']['K'],
                      reader_source_mass=self.source_mass[raw], fallback_mask=~valid,
                      eligible_weight_road=road['eligible_weight'])
        if joint:
            packet = self.basis[raw]['joint']
            result.update(joint_rgb=compose_native_eligible(packet['baseline_rgb'], packet['eligible_rgb'],
                          packet['eligible_weight'], colors, valid), joint_alpha=packet['alpha'],
                          eligible_weight_joint=packet['eligible_weight'])
        return result

    def require_execution(self, role):
        self.adapter.require_execution(role)

    def assert_frozen(self):
        for (raw, mode, key), version in self.versions.items():
            if self.basis[raw][mode][key]._version != version:
                raise RuntimeError('原生冻结 basis 被改动')
        self.adapter.assert_frozen()

    def manifest(self):
        return dict(self.record, reader=self.kind, renderer_mode=self.renderer_mode,
                    row_id_kind='pixel_flat', gate=self.gate, pixel_ID_scatter=True,
                    basis_source='actual_native_eligible_alpha_times_transmittance',
                    zero_material_mass_fallback='original_eligible_ci', independent_ci_parameters=False)
