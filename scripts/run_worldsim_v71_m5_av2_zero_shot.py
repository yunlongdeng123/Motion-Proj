"""在新下载AV2日志到达后逐条执行冻结M5 zero-shot确认。"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.dataset_av2 import compile_av2_log_v71, load_frozen_av2_cohort
from motion_proj.worldsim_v71.evaluate_surface import summarize_surface_rows
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class _RelocationEvaluationAdapter(torch.nn.Module):
    """复用M0 evaluator，但固定UNKNOWN logit为已知。"""

    def __init__(self, model: RaySurfaceRelocationMLP) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        displacement = self.model(features)
        known = torch.full(
            (len(displacement), 1),
            -100.0,
            dtype=displacement.dtype,
            device=displacement.device,
        )
        return torch.cat([displacement, known], dim=1)


def _external_decisions(summary: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    reduction = summary["hazard"]["relative_early_reduction"]
    return {
        "hazard_literal_first_return_relative_reduction": reduction is not None
        and float(reduction) >= float(config["minimum_hazard_literal_relative_reduction"]),
        "chamfer_non_degradation": float(summary["chamfer_delta_m"])
        <= float(config["maximum_chamfer_delta_m"]),
        "actor_state_retention": float(summary["minimum_actor_state_retention"])
        == float(config["required_actor_state_retention"]),
        "hazard_state_retention": float(summary["minimum_hazard_state_retention"])
        == float(config["required_hazard_state_retention"]),
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cohort = load_frozen_av2_cohort(repo_root / config["cohort_config"])
    if len(cohort["logs"]) != int(config["expected_log_count"]):
        raise RuntimeError("frozen V7.1 AV2 cohort count changed")
    compiler = yaml.safe_load(
        (repo_root / config["p2_config"]).read_text(encoding="utf-8")
    )
    m0_runner._deep_update(compiler, config["compiler_overrides"])
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "phase": "waiting_fresh_av2", "completed_logs": 0},
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M5 AV2 zero-shot requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        checkpoint = torch.load(
            Path(config["m5_run"]) / "MODEL.pt", map_location=device, weights_only=False
        )
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        model = RaySurfaceRelocationMLP(
            int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        adapter = _RelocationEvaluationAdapter(model)
        state = Path(config["download_state"])
        rows: list[dict[str, Any]] = []
        processed_logs: list[str] = []
        for position, cohort_row in enumerate(cohort["logs"]):
            log_id = str(cohort_row["log_id"])
            marker = state / f"{log_id}.complete"
            wait_started = time.monotonic()
            while not marker.is_file():
                if time.monotonic() - wait_started > int(config["wait_timeout_seconds"]):
                    raise TimeoutError(f"download timeout for {log_id}")
                _write_json(
                    run_dir / "status.json",
                    {
                        "status": "running",
                        "phase": "waiting_fresh_av2",
                        "completed_logs": len(processed_logs),
                        "current_log": log_id,
                    },
                )
                time.sleep(int(config["poll_seconds"]))
            bundles = compile_av2_log_v71(
                Path(compiler["dataset_root"]) / log_id, compiler, device
            )
            log_rows = 0
            for bundle in bundles:
                bundle["scene_name"] = log_id
                row = m0_runner._evaluate_bundle(
                    bundle,
                    adapter,
                    standardizer,
                    config["model"],
                    config["evaluation"],
                    device,
                )
                if row is None:
                    continue
                row["log_id"] = log_id
                row["external_role"] = str(cohort_row["role"])
                rows.append(row)
                log_rows += 1
            processed_logs.append(log_id)
            _write_jsonl(run_dir / "EXTERNAL_ACTORS.partial.jsonl", rows)
            _write_json(
                run_dir / "status.json",
                {
                    "status": "running",
                    "phase": "fresh_av2_evaluation",
                    "completed_logs": len(processed_logs),
                    "current_log": log_id,
                    "actor_rows": len(rows),
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "m5_fresh_av2",
                        "progress": f"{position + 1}/{len(cohort['logs'])}",
                        "log_id": log_id,
                        "log_actors": log_rows,
                        "total_actors": len(rows),
                    }
                ),
                flush=True,
            )
        metrics = summarize_surface_rows(rows)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["log_id"])].append(row)
        per_log = {
            log_id: summarize_surface_rows(log_rows)
            for log_id, log_rows in grouped.items()
        }
        decisions = _external_decisions(metrics, config["decision"])
        passed = all(decisions.values())
        _write_jsonl(run_dir / "EXTERNAL_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.m5_fresh_av2_zero_shot.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "zero_shot_pareto_transfer_supported"
            if passed
            else "source_only_geometry_improvement_cross_sensor_transfer_rejected",
            "cohort_config": config["cohort_config"],
            "log_count": len(processed_logs),
            "actor_count": len(rows),
            "processed_logs": processed_logs,
            "external": metrics,
            "per_log": per_log,
            "decisions": decisions,
            "m5_checkpoint": config["m5_run"],
            "fine_tuning": False,
            "calibration": False,
            "threshold_selection": False,
            "failed_log_deletion": False,
            "selection_read": False,
            "source_final_read": False,
            "external_read": True,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "fresh_av2_evaluation",
                "completed_logs": len(processed_logs),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "phase": "fresh_av2_evaluation",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.repo_root.resolve(), args.run_id),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
