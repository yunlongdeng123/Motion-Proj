"""root 显式调度使用的工厂；无命令行训练/自动启动行为。"""
from pathlib import Path
import json
import sys

Q = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
P = Q.parent / 'road_multiscale_material_20261118_r1'
sys.path.insert(0, str(Q / 'code'))

from build_training_inputs import PreparedInputs
from native_training_adapter import NativeTrainingAdapter


def default_config():
    native = Q / 'gpu_20261119/native_qualification_r2'
    helper = Q.parent / 'road_capacity_gpu_20261118_r1/code/road_capacity_native.py'
    return dict(Q=str(Q), permission_ledger=str(Q / 'permission/final_r1/teacher_evidence.npz'),
        protocol=str(Q / 'protocol.json'), P_loss_contract=str(P / 'execution_contract.json'),
        average_operator=str(Q / 'permission/avg_sparse_gh31_r1/operator'),
        pixel_operator_root=str(Q / 'pixel_render/read_operator_box_r1'),
        targets=str(P / 'stage0_native_r1'), basis_root=str(native / 'basis'),
        frozen_code_root=str(Q / 'code'), authorization=str(Q / 'gpu_20261119/authorization.json'),
        gpu_lease=str(Q / 'gpu_20261119/field_training/gpu_lease.json'),
        native_helper=str(helper), coverage_parent=str(Q.parent / 'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'),
        T11_seed=str(Q / 'trellis_hf/seed_r5_nodal_b034/H_seed.npz'),
        additional_source_bindings=[str(helper), str(native / 'completed.json'),
            str(native / 'constant_identity.json'), str(native / 'native_gradient.json'),
            str(Q / 'trellis_hf/seed_r5_nodal_b034/H_seed.npz'),
            str(Q / 'gpu_20261119/cuda_read_qualification_r1/receipt.json')])


def _config(config):
    if config is None:
        return default_config()
    if isinstance(config, (str, Path)):
        return json.loads(Path(config).read_text())
    return dict(config)


def prepare_inputs(config=None, device='cuda'):
    """只加载冻结场/系数/教师/basis；不加载 native 场景。CPU 可显式 device='cpu'。"""
    return PreparedInputs(_config(config), device=device)


def load_native_adapter(config=None, ctx=None):
    return NativeTrainingAdapter(_config(config), ctx=ctx)


def make_state(reader_kind, gate, role, arm, *, prepared=None, adapter=None, ctx=None,
               config=None, device='cuda', stage_counts=(200, 300, 300)):
    """reader_kind=avg|pixel, gate=observed|teacher；返回完整 TrainingState。"""
    config = _config(config)
    if prepared is None:
        prepared = prepare_inputs(config, device=device)
    if adapter is None:
        adapter = load_native_adapter(config, ctx=ctx)
    return prepared.state(reader_kind, gate, adapter, role=role, arm=arm, stage_counts=stage_counts)
