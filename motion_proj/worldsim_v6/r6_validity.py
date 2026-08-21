"""WorldSim V6 R6 task-conditioned factorized validity 正式实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy.stats import rankdata


TASK_ID = "WS-V6-R6-FACTORIZED-VALIDITY-01"
METHODS = (
    "V0_distance_to_log",
    "V1_view_count",
    "V2_reconstruction_residual",
    "V3_single_scalar_confidence",
    "factorized",
)


class R6ExperimentError(RuntimeError):
    """R6 输入、metric 或冻结 gate 失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    if not uri.startswith("runs://") or ".." in Path(uri[7:]).parts:
        raise R6ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[7:]).resolve()


def _verify_manifest_member(path: Path) -> str:
    manifest_path = path.parent / "MANIFEST.json"
    if not manifest_path.is_file():
        raise R6ExperimentError(f"来源 manifest 缺失：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["files"].get(path.name)
    digest = _sha256(path)
    if record is None or record["bytes"] != path.stat().st_size or record["sha256"] != digest:
        raise R6ExperimentError(f"来源文件 manifest 漂移：{path}")
    return digest


def _auroc(predicted_failure: np.ndarray, failures: np.ndarray) -> float | None:
    positives = int(failures.sum())
    negatives = int(failures.size - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(predicted_failure, method="average")
    value = (float(ranks[failures].sum()) - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    return float(value)


def _auprc(predicted_failure: np.ndarray, failures: np.ndarray) -> float | None:
    positives = int(failures.sum())
    if positives == 0:
        return None
    order = np.argsort(-predicted_failure, kind="stable")
    ranked = failures[order].astype(np.int64)
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked.astype(bool)].mean())


def _ece(predicted_failure: np.ndarray, failures: np.ndarray, bins: int = 5) -> float:
    total = failures.size
    value = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (predicted_failure >= lower) & (
            predicted_failure <= upper if index == bins - 1 else predicted_failure < upper
        )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(predicted_failure[mask].mean()) - float(failures[mask].mean())
            )
    return value


def _confidence(
    method: str,
    task: str,
    row: Mapping[str, Any],
    logged: Mapping[tuple[str, str, int], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> float | None:
    if method == "V0_distance_to_log":
        scale = float(config["baselines"][method]["scale_m"])
        return math.exp(-float(row["lateral_offset_m"]) / scale)
    if method == "V1_view_count":
        return float(row["q_observed_projection"])
    if method == "V2_reconstruction_residual":
        scale = float(config["baselines"][method]["scale"])
        base = logged[(row["scene"], row["frontend"], int(row["frame_index"]))]
        return math.exp(-float(base["downstream_error"]) / scale)
    if method == "V3_single_scalar_confidence":
        return float(row["support_score"])
    task_cfg = config["tasks"][task]
    if task == "photo":
        return math.exp(-float(row["global_rgb_disagreement"]) / float(task_cfg["scale"]))
    if task == "geometry":
        base = logged[(row["scene"], row["frontend"], int(row["frame_index"]))]
        return math.exp(
            -float(base["lidar_relative_depth_error"])
            / float(task_cfg["logged_residual_scale"])
        ) * math.exp(
            -float(row["nearest_training_camera_distance_m"])
            / float(task_cfg["camera_distance_scale_m"])
        )
    if task == "semantic":
        if not row["actor_lidar_denominator_passed"]:
            return 0.0
        return math.exp(
            -float(row["actor_boundary_rgb_disagreement"]) / float(task_cfg["scale"])
        )
    if task == "dynamics":
        return None
    raise R6ExperimentError(f"未知 task：{task}")


def _metrics(
    task: str,
    method: str,
    rows: list[Mapping[str, Any]],
    confidences: list[float | None],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    risk_field = config["tasks"][task]["risk_field"]
    threshold = float(config["tasks"][task]["failure_threshold"])
    accept_threshold = float(config["decision"]["accept_threshold"])
    known_indices = [index for index, value in enumerate(confidences) if value is not None]
    accepted_indices = [
        index for index in known_indices if float(confidences[index]) >= accept_threshold
    ]
    risks = np.asarray([float(row[risk_field]) for row in rows], dtype=np.float64)
    failures = risks > threshold
    accepted_failures = sum(bool(failures[index]) for index in accepted_indices)
    accepted_count = len(accepted_indices)
    known_conf = np.asarray([float(confidences[index]) for index in known_indices], dtype=np.float64)
    known_failures = failures[known_indices]
    predicted_failure = 1.0 - known_conf
    per_scene = []
    for scene in sorted({row["scene"] for row in rows}):
        scene_indices = [index for index, row in enumerate(rows) if row["scene"] == scene]
        scene_accepted = [index for index in scene_indices if index in accepted_indices]
        scene_false = sum(bool(failures[index]) for index in scene_accepted)
        per_scene.append(
            {
                "scene": scene,
                "row_count": len(scene_indices),
                "accept_count": len(scene_accepted),
                "coverage": len(scene_accepted) / len(scene_indices),
                "false_safe_rate": scene_false / len(scene_accepted) if scene_accepted else 0.0,
            }
        )
    return {
        "schema_version": "worldsim_v6.r6_task_method_metrics.v1",
        "task": task,
        "method": method,
        "row_count": len(rows),
        "known_count": len(known_indices),
        "accept_count": accepted_count,
        "abstain_count": len(rows) - accepted_count,
        "coverage": accepted_count / len(rows),
        "failure_count": int(failures.sum()),
        "accepted_failure_count": accepted_failures,
        "false_safe_rate": accepted_failures / accepted_count if accepted_count else 0.0,
        "selective_risk": (
            float(np.mean(risks[accepted_indices])) if accepted_indices else None
        ),
        "brier": (
            float(np.mean((predicted_failure - known_failures.astype(np.float64)) ** 2))
            if known_indices
            else None
        ),
        "ece_5bin": _ece(predicted_failure, known_failures) if known_indices else None,
        "auroc": _auroc(predicted_failure, known_failures) if known_indices else None,
        "auprc": _auprc(predicted_failure, known_failures) if known_indices else None,
        "per_scene": per_scene,
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R6ExperimentError("正式 R6 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R6ExperimentError("R6 task_id 漂移")
    metric_path = _resolve_runs_uri(config["source"]["r3_case_metrics"])
    provenance_path = _resolve_runs_uri(config["source"]["r5_provenance_summary"])
    metric_sha = _verify_manifest_member(metric_path)
    provenance_sha = _verify_manifest_member(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if "unknown" not in provenance["sensor_support_status"] or "unknown" not in provenance["view_support_status"]:
        raise R6ExperimentError("R5 unknown support 边界漂移")
    rows = [
        json.loads(line)
        for line in metric_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != int(config["source"]["expected_rows"]):
        raise R6ExperimentError("R3 row count 漂移")
    if len({row["scene"] for row in rows}) != int(config["source"]["expected_scenes"]):
        raise R6ExperimentError("R3 scene count 漂移")
    if len({row["frontend"] for row in rows}) != int(config["source"]["expected_frontends"]):
        raise R6ExperimentError("R3 frontend count 漂移")
    logged = {
        (row["scene"], row["frontend"], int(row["frame_index"])): row
        for row in rows
        if float(row["lateral_offset_m"]) == 0.0
    }
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R6ExperimentError("R6 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__factorized-validity-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        metric_rows = []
        score_rows = []
        confidence_cache = {}
        for task in config["tasks"]:
            for method in METHODS:
                values = [
                    _confidence(method, task, row, logged, config) for row in rows
                ]
                confidence_cache[(task, method)] = values
                metric_rows.append(_metrics(task, method, rows, values, config))
            for row, value in zip(rows, confidence_cache[(task, "factorized")]):
                score_rows.append(
                    {
                        "task": task,
                        "scene": row["scene"],
                        "frontend": row["frontend"],
                        "frame_index": row["frame_index"],
                        "lateral_offset_m": row["lateral_offset_m"],
                        "confidence": value,
                        "decision": (
                            config["decision"]["unknown_action"]
                            if value is None
                            else (
                                "ACCEPT"
                                if value >= float(config["decision"]["accept_threshold"])
                                else "ABSTAIN"
                            )
                        ),
                    }
                )
        lookup = {(row["task"], row["method"]): row for row in metric_rows}
        factor_rows = [lookup[(task, "factorized")] for task in config["tasks"]]
        v3_rows = [lookup[(task, "V3_single_scalar_confidence")] for task in config["tasks"]]
        factor_worst = max(row["false_safe_rate"] for row in factor_rows)
        v3_worst = max(row["false_safe_rate"] for row in v3_rows)
        gain = v3_worst - factor_worst
        required_coverage = {
            task: lookup[(task, "factorized")]["coverage"]
            for task in config["gate"]["required_nonzero_factors"]
        }
        dynamics_accept = lookup[("dynamics", "factorized")]["accept_count"]
        gates = {
            "maximum_factor_false_safe_passed": factor_worst
            <= float(config["gate"]["maximum_factor_false_safe_rate"]),
            "nonzero_factor_coverage_passed": all(
                value >= float(config["gate"]["minimum_nonzero_factor_coverage"])
                for value in required_coverage.values()
            ),
            "dynamics_abstain_passed": dynamics_accept
            == int(config["gate"]["dynamics_accept_count"]),
            "worst_case_gain_over_v3_passed": gain
            >= float(config["gate"]["minimum_worst_case_false_safe_gain_over_v3"]),
        }
        gate = {
            "schema_version": "worldsim_v6.r6_factorized_validity_gate.v1",
            "factorized_worst_case_false_safe_rate": factor_worst,
            "v3_worst_case_false_safe_rate": v3_worst,
            "worst_case_false_safe_gain_over_v3": gain,
            "required_factor_coverage": required_coverage,
            "dynamics_factor_accept_count": dynamics_accept,
            "gates": {**gates, "passed": all(gates.values())},
            "decision": (
                "factorized_validity_promising_conservative_development_result"
                if all(gates.values())
                else "reject_or_revise_factorized_validity_before_confirmation"
            ),
        }
        _write_jsonl(run_dir / "PER_TASK_METHOD_METRICS.jsonl", metric_rows)
        _write_jsonl(run_dir / "FACTORIZED_SCORES.jsonl", score_rows)
        _write_json(run_dir / "VALIDITY_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.r6_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "hypothesis_outcome": "accepted_development" if gate["gates"]["passed"] else "rejected",
            "source_commit": source_commit,
            "r3_metrics_sha256": metric_sha,
            "r5_summary_sha256": provenance_sha,
            "row_count": len(rows),
            "gate_passed": gate["gates"]["passed"],
            "decision": gate["decision"],
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "PER_TASK_METHOD_METRICS.jsonl",
            "FACTORIZED_SCORES.jsonl",
            "VALIDITY_GATE.json",
            "SUMMARY.json",
        ]
        manifest = {
            "schema_version": "worldsim_v6.r6_run_manifest.v1",
            "files": {
                relative: {
                    "bytes": (run_dir / relative).stat().st_size,
                    "sha256": _sha256(run_dir / relative),
                }
                for relative in tracked
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "done",
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r6_factorized_validity_v0.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
