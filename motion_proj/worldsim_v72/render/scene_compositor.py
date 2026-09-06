"""按世界深度组合背景与 Actor 返回。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ComposedReturns:
    range_m: np.ndarray
    source: np.ndarray


def compose_nearest_returns(
    background_range_m: np.ndarray,
    actor_ranges_m: Mapping[str, np.ndarray],
) -> ComposedReturns:
    background = np.asarray(background_range_m, dtype=np.float32).reshape(-1)
    candidates = [background]
    labels = ["background"]
    for actor_id, values in actor_ranges_m.items():
        actor = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(actor) != len(background):
            raise ValueError(f"Actor {actor_id} 的射线数量不一致")
        candidates.append(actor)
        labels.append(str(actor_id))
    stacked = np.stack(candidates, axis=0)
    finite = np.where(np.isfinite(stacked), stacked, np.inf)
    selected = np.argmin(finite, axis=0)
    minimum = finite[selected, np.arange(len(background))]
    source = np.asarray(labels, dtype="U128")[selected]
    missing = ~np.isfinite(minimum)
    minimum[missing] = np.nan
    source[missing] = "no_return"
    return ComposedReturns(range_m=minimum.astype(np.float32), source=source)
