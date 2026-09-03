"""运行 V7.1 S1-A 连续 candidate displacement oracle。"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene
from motion_proj.worldsim_v71.evaluate_surface import summarize_surface_rows
from motion_proj.worldsim_v71.surface_oracle import optimize_candidate_displacement


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _decisions(summary: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    hazard_reduction = summary["hazard"]["relative_early_reduction"]
    return {
        "hazard_literal_first_return_relative_reduction": (
            hazard_reduction is not None
            and float(hazard_reduction) >= float(config["minimum_hazard_literal_relative_reduction"])
        ),
        "chamfer_non_degradation": float(summary["chamfer_delta_m"]) <= float(config["maximum_chamfer_delta_m"]),
        "actor_state_retention": float(summary["minimum_actor_state_retention"]) == float(config["required_actor_state_retention"]),
        "hazard_state_retention": float(summary["minimum_hazard_state_retention"]) == float(config["required_hazard_state_retention"]),
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    _deep_update(compiler, config["compiler_overrides"])
    split = json.loads((repo_root / config["source_split"]).read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "oracle"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("S1 oracle requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    try:
        index = build_v71_index(Path(config["source"]["dataset_root"]), split)
        scenes = split["roles"]["train"]
        maximum = int(config["source"]["maximum_oracle_actors"])
        for scene_position, scene_name in enumerate(scenes):
            bundles = compile_source_scene(
                scene_name,
                index,
                config["actors"],
                compiler,
                device,
            )
            for bundle in bundles:
                result = optimize_candidate_displacement(
                    bundle["row"], bundle["diagnostics"], config["oracle"], device=device
                )
                if result is None:
                    continue
                result.row["scene_name"] = scene_name
                rows.append(result.row)
                payloads.append(
                    {
                        "track_id": result.row["track_id"],
                        "scene_name": scene_name,
                        "moved_candidates": torch.from_numpy(result.moved_candidates),
                        "displacement": torch.from_numpy(result.displacement),
                        "ray_directions": torch.from_numpy(result.ray_directions),
                        "normals": torch.from_numpy(result.normals),
                    }
                )
                print(
                    json.dumps(
                        {
                            "stage": "s1_oracle",
                            "actor": len(rows),
                            "limit": maximum,
                            "scene": scene_name,
                            "track_id": result.row["track_id"],
                        }
                    ),
                    flush=True,
                )
                if len(rows) >= maximum:
                    break
            if len(rows) >= maximum:
                break
            print(
                json.dumps(
                    {"stage": "source_scene", "progress": f"{scene_position + 1}/{len(scenes)}", "scene": scene_name}
                ),
                flush=True,
            )
        metrics = summarize_surface_rows(rows)
        decisions = _decisions(metrics, config["decision"])
        feasible = all(decisions.values())
        torch.save(payloads, run_dir / "ORACLE_DISPLACEMENTS.pt")
        _write_jsonl(run_dir / "ORACLE_ACTORS.jsonl", rows)
        summary = {
            "schema_version": "worldsim_v71.s1_displacement_oracle.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "displacement_action_space_feasible" if feasible else "displacement_action_space_not_yet_feasible",
            "actor_count": len(rows),
            "scene_count": len(set(row["scene_name"] for row in rows)),
            "metrics": metrics,
            "decisions": decisions,
            "failure_ledger_refs": config["failure_ledger_refs"],
            "failure_ledger_delta": "none" if feasible else "V71-F01-required-at-closeout",
            "target_roles_read": ["train_oracle_fit", "train_oracle_check"],
            "selection_read": False,
            "source_final_read": False,
            "external_read": False,
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
            {"status": "done", "phase": "oracle", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return {"run_dir": str(run_dir), "verdict": summary["verdict"], "decisions": decisions}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "oracle", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.repo_root.resolve(), args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
