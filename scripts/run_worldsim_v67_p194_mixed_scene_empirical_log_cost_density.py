"""Train P182 density with a fixed half empirical, half scene-balanced sampler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import scripts.run_worldsim_v67_p182_log_cost_mixture_density as p182
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost


_ORIGINAL_RANDINT = torch.randint
_SCENE_INDEX_TABLE: torch.Tensor | None = None
_SCENE_LENGTHS: torch.Tensor | None = None
_TRAJECTORY_COUNT = 0
_BATCH_SIZE = 0


def _mixed_randint(high: int, size: tuple[int, ...], *args, **kwargs) -> torch.Tensor:
    if (
        high == _TRAJECTORY_COUNT and tuple(size) == (_BATCH_SIZE,)
        and _SCENE_INDEX_TABLE is not None and _SCENE_LENGTHS is not None
    ):
        device = kwargs.get("device", "cuda")
        empirical_count = _BATCH_SIZE // 2
        empirical = _ORIGINAL_RANDINT(high, (empirical_count,), device=device)
        balanced_count = _BATCH_SIZE - empirical_count
        scene = _ORIGINAL_RANDINT(len(_SCENE_LENGTHS), (balanced_count,), device=device)
        local = torch.floor(
            torch.rand((balanced_count,), device=device) * _SCENE_LENGTHS[scene].to(torch.float32)
        ).to(torch.long)
        balanced = _SCENE_INDEX_TABLE[scene, local]
        return torch.cat((empirical, balanced), dim=0)
    return _ORIGINAL_RANDINT(high, size, *args, **kwargs)


def _rewrite_summary(config_path: Path, runs_root: Path, run_id: str) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = runs_root / "worldsim_v67" / config["task_id"] / run_id
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    control = json.loads((runs_root / config["frozen_p182_summary"]["run"] / "summary.json").read_text())
    calibration_reductions = []
    for name, row in summary["consumed_development_evaluations"].items():
        control_row = control["consumed_development_evaluations"][name]
        brier = float(row["log_cost_mixture_integrated_brier"])
        control_brier = float(control_row["log_cost_mixture_integrated_brier"])
        error = float(row["log_cost_mixture_mean_absolute_reliability_error"])
        control_error = float(control_row["log_cost_mixture_mean_absolute_reliability_error"])
        reduction = (control_error - error) / max(control_error, 1e-12)
        row.update({
            "mixed_integrated_brier": brier, "p182_integrated_brier": control_brier,
            "brier_change_vs_p182": (brier - control_brier) / max(control_brier, 1e-12),
            "mixed_mean_absolute_reliability_error": error,
            "p182_mean_absolute_reliability_error": control_error,
            "calibration_error_reduction_vs_p182": reduction,
        })
        calibration_reductions.append(reduction)
    checks = {
        "brier_noninferior_to_p182_every_cohort": all(
            row["mixed_integrated_brier"] <= row["p182_integrated_brier"]
            for row in summary["consumed_development_evaluations"].values()
        ),
        "minimum_mean_calibration_error_reduction_vs_p182": float(
            np.mean(calibration_reductions)
        ) >= float(config["decision"]["minimum_mean_calibration_error_reduction_vs_p182"]),
    }
    summary.update({
        "schema_version": config["output_schema_version"],
        "verdict": config["verdict_on_pass"] if all(checks.values()) else config["verdict_on_failure"],
        "role": config["role"], "decision_checks": checks,
        "mean_calibration_error_reduction_vs_p182": float(np.mean(calibration_reductions)),
        "claim_boundary": config["claim_boundary"],
    })
    summary["training"]["scene_sampling"] = "fixed_half_empirical_half_uniform_scene"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": summary["verdict"]}, indent=2))


def main() -> None:
    global _SCENE_INDEX_TABLE, _SCENE_LENGTHS, _TRAJECTORY_COUNT, _BATCH_SIZE
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    runs_root = Path(sys.argv[sys.argv.index("--runs-root") + 1])
    run_id = sys.argv[sys.argv.index("--run-id") + 1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with np.load(
        runs_root / config["source_rows"]["run"] / config["source_rows"]["artifact"],
        allow_pickle=False,
    ) as loaded:
        source = {name: loaded[name] for name in loaded.files}
    _, scenes = _continuous_cost(source, float(config["boundary_state_cost"]["clearance_floor_m"]))
    groups = [np.flatnonzero(scenes == scene) for scene in np.unique(scenes)]
    maximum = max(len(group) for group in groups)
    table = np.zeros((len(groups), maximum), dtype=np.int64)
    lengths = np.asarray([len(group) for group in groups], dtype=np.int64)
    for index, group in enumerate(groups):
        table[index, :len(group)] = group
    _SCENE_INDEX_TABLE = torch.from_numpy(table).cuda()
    _SCENE_LENGTHS = torch.from_numpy(lengths).cuda()
    _TRAJECTORY_COUNT = int(len(scenes))
    _BATCH_SIZE = int(config["model"]["batch_size"])
    torch.randint = _mixed_randint
    try:
        p182.main()
    finally:
        torch.randint = _ORIGINAL_RANDINT
    _rewrite_summary(config_path, runs_root, run_id)


if __name__ == "__main__":
    main()
