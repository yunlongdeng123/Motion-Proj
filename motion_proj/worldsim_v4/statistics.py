"""Scene-level 描述统计与配对检验；禁止 pixel pseudoreplication。"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import wilcoxon


VALID_STATES = {"done", "failed", "blocked", "abstain"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def summarize_scenes(
    rows: Iterable[Mapping[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 40_117,
) -> dict[str, Any]:
    records = list(rows)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples 必须为正数")
    for row in records:
        if row.get("status") not in VALID_STATES:
            raise ValueError(f"非法 scene 状态：{row.get('status')}")
    values = np.asarray([float(row[metric]) for row in records if row.get("status") == "done" and _finite(row.get(metric))], dtype=np.float64)
    states = {state: sum(row.get("status") == state for row in records) for state in VALID_STATES}
    result: dict[str, Any] = {
        "unit": "scene",
        "metric": metric,
        "attempted": len(records),
        "valid": int(values.size),
        "states": states,
        "mean": None,
        "median": None,
        "std": None,
        "iqr": None,
        "ci95": None,
    }
    if values.size == 0:
        return result
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(bootstrap_samples, values.size), replace=True), axis=1)
    q25, q75 = np.percentile(values, [25.0, 75.0])
    result.update(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        iqr=float(q75 - q25),
        ci95=[float(x) for x in np.percentile(means, [2.5, 97.5])],
    )
    return result

def paired_scene_test(
    candidate: Mapping[str, float | None],
    baseline: Mapping[str, float | None],
    *,
    higher_is_better: bool,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 20_000,
    seed: int = 40_117,
) -> dict[str, Any]:
    scenes = sorted(set(candidate) | set(baseline))
    paired = [scene for scene in scenes if _finite(candidate.get(scene)) and _finite(baseline.get(scene))]
    delta = np.asarray([float(candidate[s]) - float(baseline[s]) for s in paired], dtype=np.float64)
    if not higher_is_better:
        delta = -delta
    result: dict[str, Any] = {
        "unit": "scene",
        "attempted_union": len(scenes),
        "paired": len(paired),
        "missing_or_invalid": len(scenes) - len(paired),
        "paired_scenes": paired,
        "oriented_delta": "candidate_minus_baseline_higher_is_better",
        "mean_delta": None,
        "median_delta": None,
        "bootstrap_ci95": None,
        "sign_flip_p_two_sided": None,
        "wilcoxon_p_two_sided": None,
    }
    if delta.size == 0:
        return result
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(delta, size=(bootstrap_samples, delta.size), replace=True), axis=1)
    observed = abs(float(delta.mean()))
    if delta.size <= 16:
        patterns = np.arange(1 << delta.size, dtype=np.uint64)[:, None]
        bits = ((patterns >> np.arange(delta.size, dtype=np.uint64)) & 1).astype(np.int8)
        signs = bits * 2 - 1
    else:
        signs = rng.choice((-1, 1), size=(permutation_samples, delta.size))
    permuted = np.abs(np.mean(signs * delta[None, :], axis=1))
    p_sign = float((np.count_nonzero(permuted >= observed) + 1) / (permuted.size + 1))
    try:
        p_wilcoxon = float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        p_wilcoxon = 1.0
    result.update(
        mean_delta=float(delta.mean()),
        median_delta=float(np.median(delta)),
        bootstrap_ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
        sign_flip_p_two_sided=p_sign,
        wilcoxon_p_two_sided=p_wilcoxon,
    )
    return result
