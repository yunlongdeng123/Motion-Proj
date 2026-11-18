"""Q 新授权 overlay 的显式 native 接口；导入本文件不创建上下文。"""
from pathlib import Path
import json
import sys

from field_training_state import file_sha


class NativeTrainingAdapter:
    def __init__(self, config, ctx=None):
        self.config = dict(config)
        self._require_overlay('smoke')
        if ctx is None:
            sys.path.insert(0, str(Path(config['native_helper']).parent))
            from road_capacity_native import load_context
            ctx = load_context(asset_dir=config['coverage_parent'], frames=(86, 110, 134, 179, 198),
                               lease_file=config['gpu_lease'], lease_owner='road_asset')
        if Path(ctx.asset_dir).resolve() != Path(config['coverage_parent']).resolve():
            raise ValueError('ctx coverage parent 不同')
        self.ctx = ctx
        self.parent_rgb = ctx.base_rgb.detach()
        self._parent_version = self.parent_rgb._version

    def _require_overlay(self, role):
        config = self.config
        auth = json.loads(Path(config['authorization']).read_text())
        lease = json.loads(Path(config['gpu_lease']).read_text())
        if auth.get('GPU_authorized') is not True or auth.get('smoke_authorized') is not True:
            raise RuntimeError('新 GPU overlay 未授权')
        if lease.get('gpu_authorized') is not True or lease.get('active') is not True or lease.get('owner') != 'road_asset':
            raise RuntimeError('当前 field_training lease 未激活')
        for relative, digest in auth['frozen_inputs_sha256'].items():
            if file_sha(Path(config['Q']) / relative) != digest:
                raise RuntimeError('CPU 冻结输入已改变: ' + relative)
        if role == 'formal':
            if lease.get('formal_allowed') is not True:
                raise RuntimeError('formal 尚未通过真实生命周期门')
            launch = config.get('formal_launch_contract')
            if launch is None:
                raise RuntimeError('缺 root formal launch 合同')
            contract = json.loads(Path(launch).read_text())
            if contract.get('formal_authorized') is not True:
                raise RuntimeError('formal launch 未授权')
        elif role != 'smoke':
            raise ValueError('只允许显式 smoke/formal')

    def require_execution(self, role):
        self._require_overlay(role)
        self.ctx.require_lease()

    def render_average(self, raw, colors, joint=False):
        self.ctx.require_lease()
        out = self.ctx.render_joint(raw, colors) if joint else self.ctx.render_road(raw, colors)
        # 直接返回实际 preclamp raster RGB；不再 de-premultiply 或附加 clamp。
        return dict(rgb=out['rgb'], alpha=out['alpha'], K=self.ctx.K_masks[int(raw)])

    def assert_frozen(self):
        if self.parent_rgb._version != self._parent_version:
            raise RuntimeError('readonly native 父颜色已改变')
        self.ctx.assert_frozen()
