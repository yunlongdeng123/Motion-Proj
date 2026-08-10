"""V3.3 背景 Gaussian 到 actor 的确定性、fail-closed 重分配。"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def resolve_background_assignments(
    *,
    instance_ids: Sequence[int],
    scores: np.ndarray,
    eligible: np.ndarray,
    minimum_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 winner instance、winner score 与冲突 mask。

    `scores/eligible` 均为 ``[actor, gaussian]``。两个 actor 的证据差小于
    `minimum_margin` 时拒绝分配，避免同类车辆之间产生静默身份交换。
    """
    values = np.asarray(scores, dtype=np.float32)
    allowed = np.asarray(eligible, dtype=bool)
    actor_ids = np.asarray(instance_ids, dtype=np.int32)
    if values.ndim != 2 or allowed.shape != values.shape:
        raise ValueError("reassignment scores/eligible 必须是同 shape 的二维数组")
    if actor_ids.shape != (values.shape[0],):
        raise ValueError("instance_ids 与 actor 维度不一致")
    if len(set(int(value) for value in actor_ids)) != actor_ids.size:
        raise ValueError("instance_ids 必须唯一")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("semantic score 必须是 [0, 1] 有限值")
    if minimum_margin < 0:
        raise ValueError("minimum_margin 不能为负")

    masked = np.where(allowed, values, -np.inf)
    winner_index = np.argmax(masked, axis=0)
    winner_score = masked[winner_index, np.arange(masked.shape[1])]
    has_candidate = np.isfinite(winner_score)
    if masked.shape[0] > 1:
        runner_up = np.partition(masked, -2, axis=0)[-2]
    else:
        runner_up = np.full(masked.shape[1], -np.inf, dtype=np.float32)
    comparable = has_candidate & np.isfinite(runner_up)
    margin = np.full(masked.shape[1], np.inf, dtype=np.float32)
    margin[comparable] = winner_score[comparable] - runner_up[comparable]
    conflict = comparable & (margin < float(minimum_margin))
    accepted = has_candidate & ~conflict
    assignment = np.full(masked.shape[1], -1, dtype=np.int32)
    assignment[accepted] = actor_ids[winner_index[accepted]]
    score = np.zeros(masked.shape[1], dtype=np.float32)
    score[accepted] = winner_score[accepted]
    return assignment, score, conflict
