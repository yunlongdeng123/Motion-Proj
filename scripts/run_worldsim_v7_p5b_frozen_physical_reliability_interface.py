"""Describe the frozen P4-to-V6.7 physical/reliability interface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return None
    x_rank = _rankdata(x[finite])
    y_rank = _rankdata(y[finite])
    if float(np.std(x_rank)) == 0.0 or float(np.std(y_rank)) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _source_metrics(source: Any, indices: np.ndarray) -> dict[str, float | int]:
    target = source["target_cost"][indices].astype(np.float64)
    error = source["raw_actor_state_error_m"][indices].astype(np.float64)
    false_safe = source["occupancy_false_safe"][indices].astype(np.float64)
    decision_flip = source["occupancy_decision_flip"][indices].astype(np.float64)
    return {
        "source_row_count": int(len(indices)),
        "mean_target_cost": float(np.mean(target)),
        "q90_target_cost": float(np.quantile(target, 0.9)),
        "mean_actor_state_error_m": float(np.mean(error)),
        "q90_actor_state_error_m": float(np.quantile(error, 0.9)),
        "false_safe_rate": float(np.mean(false_safe)),
        "decision_flip_rate": float(np.mean(decision_flip)),
        "false_safe_count": int(np.sum(false_safe)),
        "decision_flip_count": int(np.sum(decision_flip)),
    }


def _group_summary(rows: list[dict[str, Any]], source: Any) -> dict[str, Any]:
    if not rows:
        return {"actor_count": 0, "source_row_count": 0}
    indices = np.concatenate([np.asarray(row["_indices"], dtype=np.int64) for row in rows])
    pooled = _source_metrics(source, indices)
    return {
        "actor_count": len(rows),
        **pooled,
        "equal_actor_mean_target_cost": float(np.mean([row["mean_target_cost"] for row in rows])),
        "equal_actor_mean_q90_target_cost": float(np.mean([row["q90_target_cost"] for row in rows])),
        "equal_actor_mean_false_safe_rate": float(np.mean([row["false_safe_rate"] for row in rows])),
        "equal_actor_mean_decision_flip_rate": float(np.mean([row["decision_flip_rate"] for row in rows])),
        "p4_mean_repair_score": float(np.mean([row["factorized_repair_score"] for row in rows])),
        "p4_selected_actor_count": int(sum(bool(row["factorized_selected"]) for row in rows)),
        "geometric_harm_actor_count": int(sum(bool(row["geometric_repair_harmful"]) for row in rows)),
    }


def _group_members(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if name == "all":
        return rows
    if name == "factorized_selected":
        return [row for row in rows if row["factorized_selected"]]
    if name == "factorized_abstained":
        return [row for row in rows if not row["factorized_selected"]]
    if name == "geometric_repair_helpful":
        return [row for row in rows if not row["geometric_repair_harmful"]]
    if name == "geometric_repair_harmful":
        return [row for row in rows if row["geometric_repair_harmful"]]
    if name == "selected_and_harmful":
        return [row for row in rows if row["factorized_selected"] and row["geometric_repair_harmful"]]
    raise ValueError(f"Unknown fixed group: {name}")


def _role_summary(
    rows: list[dict[str, Any]], source: Any, groups: list[str], horizons: list[float]
) -> dict[str, Any]:
    correlations = {
        metric: _spearman(
            [row["factorized_repair_score"] for row in rows],
            [row[metric] for row in rows],
        )
        for metric in (
            "chamfer_gain_m",
            "mean_target_cost",
            "q90_target_cost",
            "mean_actor_state_error_m",
            "q90_actor_state_error_m",
            "false_safe_rate",
            "decision_flip_rate",
        )
    }
    group_summary = {
        name: _group_summary(_group_members(rows, name), source) for name in groups
    }
    by_horizon: dict[str, Any] = {}
    source_horizon = source["horizon_seconds"].astype(np.float64)
    for horizon in horizons:
        horizon_groups: dict[str, Any] = {}
        for name in groups:
            members = _group_members(rows, name)
            local: list[dict[str, Any]] = []
            for row in members:
                indices = np.asarray(row["_indices"], dtype=np.int64)
                kept = indices[np.isclose(source_horizon[indices], horizon, atol=1e-5)]
                if len(kept):
                    clone = dict(row)
                    clone["_indices"] = kept.tolist()
                    clone.update(_source_metrics(source, kept))
                    local.append(clone)
            horizon_groups[name] = _group_summary(local, source)
        by_horizon[f"{horizon:.1f}"] = horizon_groups
    return {
        "actor_count": len(rows),
        "scene_count": len({row["scene_index"] for row in rows}),
        "groups": group_summary,
        "spearman_repair_score": correlations,
        "by_horizon_seconds": by_horizon,
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v7" / str(config["task_id"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    try:
        alignment = _read_json(Path(config["alignment_run"]) / "ALIGNED_ACTORS.json")
        roles = set(str(role) for role in config["roles"])
        aligned = [row for row in alignment if str(row["p4_role"]) in roles]
        aligned_by_identity = {
            (str(row["p4_role"]), str(row["scene_name"]), str(row["track_id"])): row
            for row in aligned
        }

        score_rows = _read_jsonl(Path(config["p4_run"]) / "SELECTIVE_SCORES.jsonl")
        scores = {
            (str(row["role"]), str(row["scene_or_log"]), str(row["track_id"])): row
            for row in score_rows
            if row.get("dataset") == "nuScenes" and str(row.get("role")) in roles
        }
        with np.load(str(config["v67_source_rows"]), allow_pickle=False) as archive:
            source = {
                key: archive[key]
                for key in (
                    "scene_index",
                    "actor_id",
                    "horizon_seconds",
                    "target_cost",
                    "raw_actor_state_error_m",
                    "occupancy_false_safe",
                    "occupancy_decision_flip",
                )
            }
        source_scene = source["scene_index"].astype(np.int64)
        source_actor = source["actor_id"].astype(np.int64)
        source_keys = (source_scene << 32) | source_actor
        order = np.argsort(source_keys, kind="mergesort")
        sorted_keys = source_keys[order]
        horizons = sorted(float(value) for value in np.unique(source["horizon_seconds"]))

        actor_rows: list[dict[str, Any]] = []
        for identity, aligned_row in sorted(aligned_by_identity.items()):
            score = scores.get(identity)
            if score is None:
                raise RuntimeError(f"Missing frozen P4 score for {identity}")
            key = (int(aligned_row["scene_index"]) << 32) | int(aligned_row["v67_actor_id"])
            left = int(np.searchsorted(sorted_keys, key, side="left"))
            right = int(np.searchsorted(sorted_keys, key, side="right"))
            indices = order[left:right]
            if not len(indices):
                raise RuntimeError(f"Aligned Actor has no V6.7 rows: {identity}")
            query_chamfer = float(score["query_chamfer_m"])
            compiled_chamfer = float(score["compiled_chamfer_m"])
            row = {
                **aligned_row,
                "factorized_repair_score": float(score["factorized_repair_score"]),
                "factorized_hazard_score": float(score["factorized_hazard_score"]),
                "factorized_selected": bool(score["factorized_selected"]),
                "repairable": bool(score["repairable"]),
                "hazardous": bool(score["hazardous"]),
                "query_chamfer_m": query_chamfer,
                "compiled_chamfer_m": compiled_chamfer,
                "chamfer_gain_m": query_chamfer - compiled_chamfer,
                "geometric_repair_harmful": bool(compiled_chamfer > query_chamfer),
                "_indices": indices.tolist(),
                **_source_metrics(source, indices),
            }
            actor_rows.append(row)

        by_role: dict[str, Any] = {}
        for role in config["roles"]:
            rows = [row for row in actor_rows if row["p4_role"] == role]
            by_role[str(role)] = _role_summary(rows, source, list(config["fixed_groups"]), horizons)

        public_rows = [{key: value for key, value in row.items() if key != "_indices"} for row in actor_rows]
        _write_jsonl(run_dir / "ACTOR_INTERFACE_ROWS.jsonl", public_rows)
        summary = {
            "schema_version": "worldsim_v7.p5b_frozen_physical_reliability_interface.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "descriptive_interface_completed",
            "claim_boundary": config["claim_boundary"],
            "frozen_v67_reliability_run": config["frozen_v67_reliability_run"],
            "p346_executed": False,
            "aligned_actor_count": len(actor_rows),
            "primary_role": config["primary_role"],
            "roles": by_role,
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return {"run_dir": str(run_dir), **summary}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
