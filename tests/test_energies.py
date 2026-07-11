"""稀疏轨迹（含缺席帧 NaN）下的能量项健壮性测试。

回归目标：`e_obj` / `e_prior` 在目标缺席帧（`present=False`、坐标为 NaN）下
不应产生 NaN。历史 bug 是 `0 * NaN = NaN` 会污染整个求和，导致缓存
metadata 里的 `obj_before` / `prior_before` 变成 NaN。
"""
import torch

from motion_proj.auditor.state import Track
from motion_proj.projector.energies import e_obj, e_prior

NAN = float("nan")


def _track(present: list[bool], fill_absent: float = NAN) -> Track:
    """构造一条轨迹：缺席帧的 xyxy 填充为 ``fill_absent``（默认 NaN）。"""
    k = len(present)
    boxes = []
    u = 10.0
    for i, p in enumerate(present):
        if p:
            u0 = u + 2.0 * i
            boxes.append([u0, 10.0 + i, u0 + 20.0, 30.0 + i])
        else:
            boxes.append([fill_absent] * 4)
    xyxy = torch.tensor(boxes, dtype=torch.float32)
    depth = torch.tensor(
        [12.0 if p else NAN for p in present], dtype=torch.float32
    )
    return Track("inst0", "car", xyxy, depth, torch.tensor(present, dtype=torch.bool))


def test_e_obj_finite_with_missing_middle_frame():
    tr = _track([True, True, False, True, True, True])
    val = e_obj([tr])
    assert torch.isfinite(val), f"e_obj should be finite, got {val}"


def test_e_prior_finite_with_missing_middle_frame():
    tr = _track([True, True, False, True, True, True])
    val = e_prior([tr])
    assert torch.isfinite(val), f"e_prior should be finite, got {val}"


def test_e_obj_finite_with_leading_and_trailing_gaps():
    tr = _track([False, True, True, False, True, True, False])
    assert torch.isfinite(e_obj([tr]))
    assert torch.isfinite(e_prior([tr]))


def test_absent_frame_values_do_not_affect_energy():
    """被掩掉的窗口不应对能量有任何贡献：缺席帧填 NaN 还是巨大垃圾值，结果一致。"""
    present = [True, True, False, True, True, True]
    tr_nan = _track(present, fill_absent=NAN)
    tr_garbage = _track(present, fill_absent=1e9)
    assert torch.allclose(e_obj([tr_nan]), e_obj([tr_garbage]), atol=1e-4)
    assert torch.allclose(e_prior([tr_nan]), e_prior([tr_garbage]), atol=1e-4)


def test_mixed_tracks_stay_finite():
    """多条轨迹（含全缺席、稀疏、稠密）聚合后仍为有限值。"""
    tracks = [
        _track([True] * 6),
        _track([True, False, True, False, True, False]),
        _track([False, False, True, True, False, False]),
    ]
    assert torch.isfinite(e_obj(tracks))
    assert torch.isfinite(e_prior(tracks))
