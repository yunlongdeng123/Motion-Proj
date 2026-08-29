"""Train density with empirical short-H and progressively scene-balanced long-H sampling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import scripts.run_worldsim_v67_p182_log_cost_mixture_density as p182
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import _trajectory_horizon


_ORIGINAL_RANDINT = torch.randint
_TRAJECTORY_COUNT = 0
_BATCH_SIZE = 0
_HORIZON_CODE: torch.Tensor | None = None
_BALANCE_PROBABILITY: torch.Tensor | None = None
_INDEX_TABLES: list[torch.Tensor] = []
_LENGTH_TABLES: list[torch.Tensor] = []


def _conditioned_randint(high: int, size: tuple[int, ...], *args, **kwargs) -> torch.Tensor:
    if high != _TRAJECTORY_COUNT or tuple(size) != (_BATCH_SIZE,) or _HORIZON_CODE is None:
        return _ORIGINAL_RANDINT(high, size, *args, **kwargs)
    device = kwargs.get("device", "cuda")
    indices = _ORIGINAL_RANDINT(high, size, device=device)
    codes = _HORIZON_CODE[indices]
    replace = torch.rand(size, device=device) < _BALANCE_PROBABILITY[codes]
    for code, (table, lengths) in enumerate(zip(_INDEX_TABLES, _LENGTH_TABLES)):
        positions = torch.nonzero(replace & (codes == code), as_tuple=False).flatten()
        if len(positions) == 0:
            continue
        scene = _ORIGINAL_RANDINT(len(lengths), (len(positions),), device=device)
        local = torch.floor(
            torch.rand((len(positions),), device=device) * lengths[scene].to(torch.float32)
        ).to(torch.long)
        indices[positions] = table[scene, local]
    return indices


def _rewrite_summary(config_path: Path, runs_root: Path, run_id: str) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    control = json.loads((runs_root / config["frozen_p182_summary"]["run"] / "summary.json").read_text())
    reductions = []
    for name, row in summary["consumed_development_evaluations"].items():
        base = control["consumed_development_evaluations"][name]
        brier, base_brier = float(row["log_cost_mixture_integrated_brier"]), float(base["log_cost_mixture_integrated_brier"])
        error = float(row["log_cost_mixture_mean_absolute_reliability_error"])
        base_error = float(base["log_cost_mixture_mean_absolute_reliability_error"])
        reduction = (base_error - error) / max(base_error, 1e-12)
        row.update({
            "horizon_conditioned_integrated_brier": brier, "p182_integrated_brier": base_brier,
            "brier_change_vs_p182": (brier - base_brier) / max(base_brier, 1e-12),
            "horizon_conditioned_mean_absolute_reliability_error": error,
            "p182_mean_absolute_reliability_error": base_error,
            "calibration_error_reduction_vs_p182": reduction,
        })
        reductions.append(reduction)
    checks = {
        "brier_noninferior_to_p182_every_cohort": all(
            row["horizon_conditioned_integrated_brier"] <= row["p182_integrated_brier"]
            for row in summary["consumed_development_evaluations"].values()
        ),
        "minimum_mean_calibration_error_reduction_vs_p182": float(np.mean(reductions))
        >= float(config["decision"]["minimum_mean_calibration_error_reduction_vs_p182"]),
    }
    summary.update({
        "schema_version": config["output_schema_version"],
        "verdict": config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"],
        "role": config["role"], "decision_checks": checks,
        "mean_calibration_error_reduction_vs_p182": float(np.mean(reductions)),
        "claim_boundary": config["claim_boundary"],
    })
    summary["training"]["scene_sampling"] = config["sampling"]
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": summary["verdict"]}, indent=2))


def main() -> None:
    global _TRAJECTORY_COUNT, _BATCH_SIZE, _HORIZON_CODE, _BALANCE_PROBABILITY
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    runs_root = Path(sys.argv[sys.argv.index("--runs-root") + 1])
    run_id = sys.argv[sys.argv.index("--run-id") + 1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with np.load(runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"], allow_pickle=False) as loaded:
        source = {name: loaded[name] for name in loaded.files}
    _, scenes = _continuous_cost(source, float(config["boundary_state_cost"]["clearance_floor_m"]))
    horizons = _trajectory_horizon(source)
    frozen_horizons = np.asarray(config["sampling"]["source_horizons_seconds"], dtype=np.float32)
    horizon_code = np.argmin(np.abs(horizons[:, None] - frozen_horizons[None]), axis=1)
    for code in range(len(frozen_horizons)):
        horizon_mask = horizon_code == code
        groups = [np.flatnonzero(horizon_mask & (scenes == scene)) for scene in np.unique(scenes[horizon_mask])]
        maximum = max(len(group) for group in groups)
        table = np.zeros((len(groups), maximum), dtype=np.int64)
        lengths = np.asarray([len(group) for group in groups], dtype=np.int64)
        for index, group in enumerate(groups):
            table[index, :len(group)] = group
        _INDEX_TABLES.append(torch.from_numpy(table).cuda())
        _LENGTH_TABLES.append(torch.from_numpy(lengths).cuda())
    _TRAJECTORY_COUNT = int(len(scenes))
    _BATCH_SIZE = int(config["model"]["batch_size"])
    _HORIZON_CODE = torch.from_numpy(horizon_code.astype(np.int64)).cuda()
    _BALANCE_PROBABILITY = torch.tensor(config["sampling"]["scene_balanced_probability"], device="cuda")
    torch.randint = _conditioned_randint
    try:
        p182.main()
    finally:
        torch.randint = _ORIGINAL_RANDINT
    _rewrite_summary(config_path, runs_root, run_id)


if __name__ == "__main__":
    main()
