"""权威 pixel field 重读 + 同诊断相机 fresh native weights；只显式调用。"""
from pathlib import Path
import json
import sys
import numpy as np
import torch

from world_texture_runtime import WorldTextureRuntime
from material_readers import PixelWorldReader
from field_training_state import file_sha


def verify_reload_179(field_path, prepared, adapter, output, *, authorization, expected_native_npz=None):
    """调用者独占 lease；恰2个 native forward，0更新，不创建第二个ctx。

    179 是训练86/110之外的冻结诊断相机。其他新相机/编辑还须重新生成其 G_s pixel operator。
    authorization 为新的显式 native_smoke 合同；不自动把训练授权扩成其他相机。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = prepared.config
    sys.path.insert(0, str(Path(config['Q']) / 'pixel_render/code'))
    from native_basis_adapter import export_native_basis
    before = adapter.ctx.native_calls
    record = export_native_basis(adapter.ctx, 179, prepared.qualified, output / 'fresh_basis',
                                 authorization=authorization)
    # 原 API 在 fresh_basis 内分别保存 road/joint，读取实际产物而非复用旧权重。
    fresh = {}
    for mode in ('road', 'joint'):
        with np.load(output / 'fresh_basis' / (mode + '_basis.npz'), allow_pickle=False) as saved:
            fresh[mode] = {key: saved[key].copy() for key in saved.files}
    model, metadata = WorldTextureRuntime.load(field_path, device=prepared.device)
    saved_gate = model.H_gate.detach().cpu().numpy()[..., 0]
    if np.array_equal(saved_gate, prepared.ledger['observed_H_prior']):
        gate = 'observed_H'
    elif np.array_equal(saved_gate, prepared.ledger['mH_union']):
        gate = 'observed_plus_teacher_H'
    else:
        raise ValueError('持久 field 的 H_gate 与冻结许可都不同')
    reader = PixelWorldReader({179: prepared.pixel[179]}, gate, {179: fresh}, prepared.device, adapter,
                             dict(field=str(field_path), fresh_native_basis=True))
    cached = PixelWorldReader({179: prepared.pixel[179]}, gate, {179: prepared.basis[179]}, prepared.device, adapter, {})
    with torch.no_grad():
        result = reader.render(model, 179, joint=True)
        old = cached.render(model, 179, joint=True)
    values = {key: value.detach().cpu().numpy() for key, value in result.items()}
    np.savez_compressed(output / '179_native.npz', **values)
    errors = {key: float((result[key] - old[key]).abs().max()) for key in ('road_rgb', 'joint_rgb', 'road_alpha', 'joint_alpha')}
    K = values['K']
    Kerror = float(np.abs(values['joint_rgb'][:, K] - fresh['joint']['baseline_rgb'][:, K]).max())
    expected_errors = None
    if expected_native_npz is not None:
        with np.load(expected_native_npz, allow_pickle=False) as saved:
            expected_errors = {key: float(np.abs(values[key] - saved[key]).max()) for key in ('road_rgb', 'joint_rgb')}
    passed = max(errors.values()) <= 1e-6 and Kerror == 0 and (expected_errors is None or max(expected_errors.values()) <= 1e-6)
    receipt = dict(passed=passed, field=str(field_path), field_sha256=file_sha(field_path),
        camera=179, diagnostic_not_train_camera=True, authoritative_field_reloaded=True,
        actual_native_basis_refreshed=True, native_forward_calls=adapter.ctx.native_calls - before,
        optimizer_updates=0, same_camera_cached_basis_differences=errors,
        expected_saved_render_differences=expected_errors, joint_K_max=Kerror,
        new_arbitrary_camera_operator_generated=False,
        arbitrary_camera_requires_new_G_s_mapping_and_native_export=True)
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not passed:
        raise RuntimeError('persistent field/fresh basis mismatch: ' + json.dumps(receipt))
    return receipt
