"""单次前瞻性 M12：在反传内固定过滤 unmodulated H；旧运行时不改字节。"""
from pathlib import Path
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F

from world_texture_runtime import WorldTextureRuntime
from field_training_state import TrainingState, cpu_clone, file_sha, max_tree_difference
from material_readers import PixelWorldReader

SIGMA_UV = (.8225284847010147, .29998550363258897)
TRUNCATE = 3.


class SourceFootprintWorldTexture(WorldTextureRuntime):
    """同底色/低频/高频参数；只给基类 H 添加固定线性 source-footprint 读取。"""
    def __init__(self, *args, source_sigma_uv=SIGMA_UV, **kwargs):
        super().__init__(*args, **kwargs)
        if tuple(source_sigma_uv) != SIGMA_UV:
            raise ValueError('单一登记 source sigma，不允许参数搜索')
        self._source_sigma_uv = SIGMA_UV
        self._source_kernel_cache = {}

    @property
    def source_sigma_uv(self):
        return self._source_sigma_uv

    def filter_settings(self):
        return dict(schema='source_footprint_unmodulated_H_filter.v1', sigma_uv=list(SIGMA_UV),
            scipy_axis_sigma_v_u_channel=[SIGMA_UV[1], SIGMA_UV[0], 0.], truncate=TRUNCATE,
            radius_uv=[int(TRUNCATE*s+.5) for s in SIGMA_UV],
            boundary='replicate / scipy nearest', axis_order='v then u, no RGB filtering',
            filtering_before_mH=True, trainable_filter_parameters=False)

    def filter_high(self, high):
        if high.ndim != 3 or high.shape[-1] != 3:
            raise ValueError('H 必须为 HWC RGB')
        key = (str(high.device), high.dtype)
        if key not in self._source_kernel_cache:
            kernels = []
            for sigma in (SIGMA_UV[1], SIGMA_UV[0]):
                # 精确复用 scipy ndimage 的 truncate=3 半径取整规则，非 ceil。
                radius = int(TRUNCATE*sigma+.5)
                x = torch.arange(-radius, radius+1, device=high.device, dtype=torch.float64)
                weights = torch.exp(-.5*(x/sigma)**2)
                kernels.append((radius, (weights/weights.sum()).to(high.dtype)))
            self._source_kernel_cache[key] = kernels
        value = high.permute(2, 0, 1)[None]
        (rv, kv), (ru, ku) = self._source_kernel_cache[key]
        value = F.conv2d(F.pad(value, (0, 0, rv, rv), mode='replicate'),
                         kv[None, None, :, None].expand(3, 1, -1, 1), groups=3)
        value = F.conv2d(F.pad(value, (ru, ru, 0, 0), mode='replicate'),
                         ku[None, None, None, :].expand(3, 1, 1, -1), groups=3)
        return value[0].permute(1, 2, 0)

    def fields(self):
        low, high = super().fields()
        return low, self.filter_high(high)

    def save(self, path, metadata):
        # 不使用旧 loader 的 state/settings 顶层键，防止静默按未过滤模式重载。
        torch.save(dict(schema='source_footprint_world_texture.v1', field_state=self.state_dict(),
            metadata=dict(metadata, material_model='SourceFootprintWorldTexture'),
            runtime_settings=dict(low_stride=self.low_stride, low_bound=self.low_bound, high_bound=self.high_bound),
            source_filter=self.filter_settings()), Path(path))

    @classmethod
    def load(cls, path, device='cuda'):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved.get('schema') != 'source_footprint_world_texture.v1':
            raise ValueError('不是 source-filtered 权威 field')
        state = saved['field_state']
        model = cls(state['base'].numpy(), state['valid'][..., 0].numpy(), state['H_gate'][..., 0].numpy(),
            state['latent_mask'][..., 0].numpy(), state['reg_support'][..., 0].numpy(),
            low_mask=state['low_mask'][..., 0].numpy(), **saved['runtime_settings'])
        if saved['source_filter'] != model.filter_settings():
            raise ValueError('source filter recipe 不同')
        model.load_state_dict(state)
        return model.to(device), saved['metadata']


class SourceFilteredTrainingState(TrainingState):
    def __init__(self, *args, arm='M12', **kwargs):
        if arm != 'M12':
            raise ValueError('本合同只有 M12')
        # 基类不改 ARM_SPEC；仅继承其完整状态工程，再使用此独立方法身份。
        super().__init__(*args, arm='M11', **kwargs)
        if not isinstance(self.model, SourceFootprintWorldTexture) or self.reader.kind != 'pixel':
            raise ValueError('M12 必须为 source-filtered pixel world field')
        self.arm = 'M12'


def make_source_filtered_state(prepared, adapter, *, role='smoke', arm='M12',
                               stage_counts=(200, 300, 300), config=None):
    config = dict(prepared.config if config is None else config)
    d = prepared.ledger
    model = SourceFootprintWorldTexture(d['common_base_srgb'], d['common_material_valid'], d['mH_union'],
        d['common_latent_mask'], d['common_reg_support'], low_mask=prepared.low_mask,
        low_stride=6, low_bound=.30, high_bound=.34).to(prepared.device)
    bindings = copy.deepcopy(prepared.bindings)
    bindings[str(Path(__file__).resolve())] = file_sha(__file__)
    contract = config.get('source_filter_contract')
    if contract is None:
        contract = Path(config['Q'])/'gpu_20261119/source_filtered_training/prospective_contract.json'
    contract = Path(contract)
    if not contract.is_file():
        raise ValueError('缺新前瞻性 M12 合同')
    bindings[str(contract)] = file_sha(contract)
    record = dict(source_bindings=bindings, reader='pixel', permission='teacher',
                  source_filter=model.filter_settings(), independent_ci_parameters=False)
    reader = PixelWorldReader(prepared.pixel, 'observed_plus_teacher_H', prepared.basis,
                              prepared.device, adapter, record)
    identity = dict(reader='pixel', source_bindings=bindings, seed=101,
                    scope_rows=int(prepared.qualified.sum()), active_world_rows_kind='global_world_row',
                    material_model='SourceFootprintWorldTexture.v1', source_filter=model.filter_settings())
    return SourceFilteredTrainingState(model, reader, prepared.targets, identity, prepared.optimizer_spec,
        role=role, arm=arm, stage_counts=stage_counts, seed=101)


def rebase_zero_H_common_A(common_snapshot, state, output_receipt=None):
    """仅在 A200 零H等价及原来源精确核对后 rebinding，不执行 fork/optimizer。"""
    old = common_snapshot
    if not isinstance(state, SourceFilteredTrainingState):
        raise ValueError('目标不是 M12 状态')
    if old['role'] != state.role or state.role != 'formal':
        raise ValueError('正式 A200 只用于正式；smoke 必须从零独立运行')
    if old['arm'] != 'common_pixel' or old['cursor'] != 200 or old['stage'] != 'A':
        raise ValueError('不是原 pixel common_A200')
    if old['stage_counts'] != state.stage_counts or old['schedule'] != state.schedule or old['trace'] != state.schedule[:200]:
        raise ValueError('原 A200 sampler/stage 不同')
    if old['optimizer_spec'] != state.optimizer_spec or old['optimizer']['state'].get(1):
        raise ValueError('A200 Adam 角色不同或 H 已更新')
    if not bool(torch.isfinite(old['params']['high']).all()) or bool((old['params']['high'] != 0).any()):
        raise ValueError('A200 不是严格 H0')
    current = state.snapshot()
    if old['settings'] != current['settings']:
        raise ValueError('A200 bounds/stride 不同')
    if old['frozen_signature'].keys() != current['frozen_signature'].keys():
        raise ValueError('共同 frozen buffer 身份集合不同')
    for key, digest in old['frozen_signature'].items():
        if key != 'H_gate' and digest != current['frozen_signature'][key]:
            raise ValueError('共同底色/support 改变: '+key)
    for key, value in old['identity'].items():
        if key != 'source_bindings' and state.identity.get(key) != value:
            raise ValueError('A200 共同 identity 不同: '+key)
    for path, digest in old['identity']['source_bindings'].items():
        if state.identity['source_bindings'].get(path) != digest or file_sha(path) != digest:
            raise ValueError('原绑定来源改变/被替换: '+path)
    # 在真实 A200 参数上证明新算子与原算子输出完全相同；不改变 optimizer/RNG。
    saved_params = cpu_clone(dict(low=state.model.low, high=state.model.high))
    try:
        with torch.no_grad():
            for key in ('low', 'high'):
                target = getattr(state.model, key)
                target.copy_(old['params'][key].to(target.device))
            base_L, base_H = WorldTextureRuntime.fields(state.model)
            new_L, new_H = state.model.fields()
            low_error = float((base_L-new_L).abs().max())
            high_error = float((base_H-new_H).abs().max())
            if low_error != 0 or high_error != 0 or bool((base_H != 0).any()):
                raise ValueError('新算子不满足原 A200 零H等价')
    finally:
        with torch.no_grad():
            for key in ('low', 'high'):
                target = getattr(state.model, key)
                target.copy_(saved_params[key].to(target.device))
    rebased = cpu_clone(old)
    rebased['identity'] = copy.deepcopy(state.identity)
    # common_pixel 身份保留；随后由原 fork_common_A 严格检查并保持新 teacher gate。
    changed = [key for key in old if key != 'identity' and max_tree_difference(old[key], rebased[key]) != 0]
    if changed:
        raise RuntimeError('rebase 越界修改 checkpoint: '+str(changed))
    receipt = dict(passed=True, operation='validated_zero_H_A200_identity_rebase_only',
        role=state.role, original_common_arm=old['arm'], params_Adam_RNG_sampler_exact=True,
        original_bound_files_unchanged=True, frozen_buffers_equal_except_H_gate=True,
        low_field_output_max=low_error, high_field_output_max=high_error,
        new_source_filter=state.model.filter_settings(), optimizer_updates=0,
        fork_executed=False, source_identity_before=old['identity'], source_identity_after=state.identity)
    if output_receipt is not None:
        path = Path(output_receipt); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError('保留既有 rebase receipt')
        path.write_text(json.dumps(receipt, indent=2)+'\n')
    return rebased
