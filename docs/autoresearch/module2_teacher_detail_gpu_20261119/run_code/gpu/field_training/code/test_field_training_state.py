"""小型 CPU 生命周期：真实 autograd/Adam，但不是场景/native 质量证据。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
import json
import numpy as np
import pytest
import torch

from fixed_material_read_operator import ReadCompiler
from world_texture_runtime import WorldTextureRuntime
from field_training_state import TrainingState, max_tree_difference
from material_readers import GaussianAverageReader, PixelWorldReader

torch.set_num_threads(1)
SPEC = dict(low_lr=.01, high_lr=.003, betas=[.9, .999], eps=1e-8, weight_decay=0.,
    regularizers=dict(low_adjacent_delta_squared=.01, effective_high_energy=.1,
        effective_high_lowfrequency_leakage=.1, effective_high_weighted_DC=.1))


class Adapter:
    def __init__(self):
        self.parent_rgb = torch.full((4, 3), .4)
    def require_execution(self, role):
        assert role == 'smoke'
    def assert_frozen(self):
        pass
    def render_average(self, raw, colors, joint=False):
        return dict(rgb=colors.T.reshape(3, 1, 4), alpha=torch.ones((1, 4)), K=torch.zeros((1, 4), dtype=torch.bool))


def make_state(gate='teacher', arm='M10', reader_kind='avg'):
    h = w = 16
    base = np.full((h, w, 3), .4)
    valid = np.ones((h, w))
    gates = dict(observed=np.zeros((h, w)), teacher=np.ones((h, w)))
    model = WorldTextureRuntime(base, valid, gates[gate], valid, valid, low_mask=np.ones((4, 4)))
    uv = np.array([[[2.2, 3.3]], [[6.4, 3.7]], [[10.1, 9.2]], [[13.5, 12.2]]])
    compiler = ReadCompiler(base, valid, gates)
    compiler.add(uv, np.ones(1))
    rows = np.array([2, 4, 9, 20])
    operator = compiler.finish(rows, dict(role='synthetic_CPU_unit'))
    adapter = Adapter()
    if reader_kind == 'avg':
        reader = GaussianAverageReader(operator, gate, rows, np.ones(4, bool), adapter.parent_rgb, adapter, {})
    else:
        # 显式打乱顺序，验证分组查询必须 scatter 而非 reshape。
        operator.row_ids = np.array([2, 0, 3, 1])
        basis = {}
        for raw in (86, 110):
            packet = dict(baseline_rgb=np.full((3, 1, 4), .34), eligible_rgb=np.full((3, 1, 4), .32),
                          eligible_weight=np.full((1, 4), .8), alpha=np.ones((1, 4)), K=np.zeros((1, 4), bool))
            basis[raw] = dict(road=packet, joint=packet)
        reader = PixelWorldReader({86: operator, 110: operator}, gate, basis, 'cpu', adapter, {})
    targets = {}
    for raw in (86, 110):
        targets[raw] = dict(fixed_alpha=torch.ones((1, 4)), **{key: torch.ones((1, 4), dtype=torch.bool) for key in ('A', 'C', 'S')},
                            **{'target_' + key: torch.full((3, 1, 4), value) for key, value in [('A', .46), ('C', .61), ('S', .65)]})
    state = TrainingState(model, reader, targets, {'reader': reader_kind, 'shared': 'same_inputs'}, SPEC,
                          role='smoke', arm=arm, stage_counts=(2, 2, 2))
    return state


def test_complete_disk_restore_next_step_and_snapshot_not_aliased(tmp_path):
    state = make_state()
    state.step(); state.step()
    state.save(tmp_path / 'A')
    saved = torch.load(tmp_path / 'A/checkpoint.pt', map_location='cpu', weights_only=False)
    pristine = copy.deepcopy(saved)
    state.step()
    expected = state.snapshot()
    assert max_tree_difference(saved, pristine) == 0
    state.restore(saved)
    assert max_tree_difference(state.snapshot(), saved) == 0
    state.step()
    assert max_tree_difference(state.snapshot(), expected) == 0


def test_common_A_fork_keeps_teacher_gate_and_B_freezes_low_Adam():
    common = make_state('observed', 'common_avg')
    common.step(); common.step()
    a = common.snapshot()
    teacher = make_state()
    teacher.fork_common_A(a)
    assert torch.all(teacher.model.H_gate == 1)
    low = teacher.model.low.detach().clone()
    adam = copy.deepcopy(teacher.optimizer.state_dict()['state'][0])
    teacher.step(); teacher.step()
    assert torch.equal(teacher.model.low, low)
    assert max_tree_difference(teacher.optimizer.state_dict()['state'][0], adam) == 0
    assert teacher.model.high.abs().max() > 0
    teacher.step()
    assert not torch.equal(teacher.model.low, low)


def test_T11_only_changes_H_seed_no_low_or_AdamH():
    common = make_state('observed', 'common_pixel', 'pixel')
    common.step(); common.step()
    a = common.snapshot()
    target = make_state('teacher', 'T11', 'pixel')
    seed = torch.full_like(target.model.high, .01)
    target.fork_common_A(a, high_seed=seed)
    assert torch.equal(target.model.high, seed)
    assert torch.equal(target.model.low, a['params']['low'])
    assert target.optimizer.state_dict()['state'].get(1) is None
    assert torch.all(target.model.H_gate == 1)


def test_restore_rejects_wrong_bounds_trace_role_and_nonfinite():
    state = make_state(); state.step()
    good = state.snapshot()
    cases = []
    for key, value in [('role', 'formal'), ('trace', [110 if good['trace'][0] == 86 else 86])]:
        bad = copy.deepcopy(good); bad[key] = value; cases.append(bad)
    bad = copy.deepcopy(good); bad['settings']['high_bound'] = .04; cases.append(bad)
    bad = copy.deepcopy(good); bad['optimizer']['state'][0]['exp_avg'][0, 0, 0] = float('nan'); cases.append(bad)
    for bad in cases:
        with pytest.raises(ValueError): state.restore(bad)


def test_pixel_layout_scatter_and_source_mass_fallback():
    state = make_state('teacher', 'M11', 'pixel')
    op = state.reader.operators[86]
    op.mass[0] = 0
    # layout固定于构建时，单独用固定mask构造未知源回退，不能训练时动态改mass。
    state.reader.layouts[86].material_valid_grid[0, 2] = False
    out = state.reader.render(state.model, 86, joint=True)
    assert torch.equal(out['road_rgb'][:, 0, 2], torch.full((3,), .34))
    loss, _ = state.objective(86); loss.backward()
    assert torch.isfinite(state.model.low.grad).all()


def test_field_save_reload_same_gate_and_parameters(tmp_path):
    state = make_state(); [state.step() for _ in range(6)]
    state.save(tmp_path / 'C')
    model, metadata = WorldTextureRuntime.load(tmp_path / 'C/field.pt')
    assert metadata['authoritative_world_field'] and metadata['cursor'] == 6
    for key, value in state.model.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
