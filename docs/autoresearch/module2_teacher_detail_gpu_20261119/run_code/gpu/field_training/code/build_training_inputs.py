"""显式组装共同场/固定读取/监督；native adapter 由调度方传入。"""
from pathlib import Path
import json
import numpy as np
import torch

from fixed_material_read_operator import FixedReadOperator
from world_texture_runtime import WorldTextureRuntime
from field_training_state import TrainingState, file_sha
from material_readers import GaussianAverageReader, PixelWorldReader

FRAMES = (86, 110, 134, 179, 198)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def common_low_mask(latent, stride=6):
    height, width = latent.shape
    mask = np.zeros((int(np.ceil((height - 1) / stride)) + 1,
                     int(np.ceil((width - 1) / stride)) + 1), np.float32)
    y, x = np.nonzero(latent)
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        mask[np.minimum(y // stride + dy, len(mask) - 1),
             np.minimum(x // stride + dx, mask.shape[1] - 1)] = 1
    return mask


class PreparedInputs:
    def __init__(self, config, device='cpu'):
        self.config = dict(config)
        self.device = torch.device(device)
        self.ledger = load_arrays(config['permission_ledger'])
        self.protocol = json.loads(Path(config['protocol']).read_text())
        self.loss_contract = json.loads(Path(config['P_loss_contract']).read_text())
        self.optimizer_spec = self.loss_contract['optimizer']
        self.low_mask = common_low_mask(self.ledger['common_latent_mask'])
        self.avg = FixedReadOperator.load(config['average_operator'])
        self.pixel = {raw: FixedReadOperator.load(Path(config['pixel_operator_root']) / ('%03d' % raw)) for raw in FRAMES}
        self.targets, self.basis = {}, {}
        files = [Path(config['permission_ledger']), Path(config['protocol']), Path(config['P_loss_contract'])]
        for root in [Path(config['average_operator']), *[Path(config['pixel_operator_root']) / ('%03d' % raw) for raw in FRAMES]]:
            files.extend(path for path in root.iterdir() if path.is_file())
        for raw in (86, 110):
            path = Path(config['targets']) / ('%03d_targets.npz' % raw)
            source = load_arrays(path)
            self.targets[raw] = {key: torch.as_tensor(source[key], device=self.device,
                dtype=torch.bool if key in ('A', 'C', 'S', 'fullS') else torch.float32)
                for key in ('A', 'C', 'S', 'fullS', 'fixed_alpha', 'target_A', 'target_C', 'target_S')}
            files.append(path)
        actual = {key: sum(int(value[key].sum()) for value in self.targets.values()) for key in ('A', 'C', 'S', 'fullS')}
        if actual != dict(A=16904, C=10830, S=3131, fullS=3187):
            raise ValueError('固定完整监督分母改变: ' + str(actual))
        self.active_world_rows = self.ledger['active_world_rows']
        candidate = self.ledger['common_write_active']
        if not np.array_equal(self.avg.row_ids, self.active_world_rows[candidate]):
            raise ValueError('平均 row_ids 不是共同 scope 的 global world IDs')
        for raw in FRAMES:
            directory = Path(config['basis_root']) / ('%03d' % raw)
            self.basis[raw] = {}
            for mode in ('road', 'joint'):
                path = directory / (mode + '_basis.npz')
                self.basis[raw][mode] = load_arrays(path)
                files.append(path)
            path = directory / 'row_scope.npz'
            scope = load_arrays(path)
            if not np.array_equal(scope['active_world_rows'], self.active_world_rows):
                raise ValueError('native scope active world identity 不同')
            qualified = np.asarray(scope['eligible_active'], bool)
            if raw == FRAMES[0]:
                self.qualified = qualified
            elif not np.array_equal(self.qualified, qualified):
                raise ValueError('不同帧未绑定同一持久 actual scope')
            files.append(path)
        if not np.array_equal(self.qualified, candidate):
            raise ValueError('native qualification 与候选共同 scope 不同；先报告，不静默缩域')
        self.K_count = sum(int(self.basis[raw]['joint']['K'].sum()) for raw in FRAMES)
        if self.K_count != 77266:
            raise ValueError('五帧 K 分母改变')
        for name in ('field_training_state.py', 'material_readers.py', 'build_training_inputs.py',
                     'training_factory.py', 'native_training_adapter.py'):
            files.append(Path(__file__).with_name(name))
        for name in ('world_texture_runtime.py', 'fixed_material_read_operator.py'):
            files.append(Path(config['frozen_code_root']) / name)
        files.extend(Path(value) for value in config.get('additional_source_bindings', []))
        self.bindings = {str(path): file_sha(path) for path in sorted(set(files))}

    def model(self, gate):
        if gate not in ('observed', 'teacher'):
            raise ValueError('H 许可名称错误')
        source = self.ledger
        return WorldTextureRuntime(source['common_base_srgb'], source['common_material_valid'],
            source['observed_H_prior'] if gate == 'observed' else source['mH_union'],
            source['common_latent_mask'], source['common_reg_support'], low_mask=self.low_mask,
            low_stride=6, low_bound=.30, high_bound=.34).to(self.device)

    def state(self, reader_kind, gate, adapter, *, role, arm, stage_counts=(200, 300, 300)):
        model = self.model(gate)
        record = dict(source_bindings=self.bindings, scientific_protocol=self.config['protocol'],
                      active_world_rows_sha256=file_sha(self.config['permission_ledger']),
                      K_pixel_frames=self.K_count, scope_rows=int(self.qualified.sum()),
                      permission=gate, source_ledger_unchanged=True)
        if reader_kind == 'avg':
            reader = GaussianAverageReader(self.avg, gate, self.active_world_rows, self.qualified,
                adapter.parent_rgb, adapter, record)
        elif reader_kind == 'pixel':
            key = 'observed_H' if gate == 'observed' else 'observed_plus_teacher_H'
            reader = PixelWorldReader(self.pixel, key, self.basis, self.device, adapter, record)
        else:
            raise ValueError('未知 reader')
        # 共同 A identity 不含 arm/gate；所有组的 gate maps 都已在共同输入中绑定。
        identity = dict(reader=reader_kind, source_bindings=self.bindings, seed=101,
                        scope_rows=int(self.qualified.sum()), active_world_rows_kind='global_world_row')
        return TrainingState(model, reader, self.targets, identity, self.optimizer_spec,
                             role=role, arm=arm, stage_counts=stage_counts, seed=101)
