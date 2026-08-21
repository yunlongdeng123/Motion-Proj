"""WorldSim V6 R38：对 actor 轨迹平移独立验证 self-kinematics 与 AABB interaction。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once


TASK_ID = "WS-V6-R38-ACTOR-INTERACTION-FACTOR-01"


class R38ExperimentError(RuntimeError):
    """R38 正式实验合同失败。"""


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


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    relative = Path(uri[len(prefix) :]) if uri.startswith(prefix) else Path("..")
    if not uri.startswith(prefix) or relative.is_absolute() or ".." in relative.parts:
        raise R38ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / relative).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R38ExperimentError(f"冻结输入漂移：{path}")


def _collision_rows(actor_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_time: dict[int, list[dict[str, Any]]] = {}
    for row in actor_states:
        by_time.setdefault(int(row["timestamp_us"]), []).append(row)
    collisions: list[dict[str, Any]] = []
    for timestamp, states in sorted(by_time.items()):
        ordered = sorted(states, key=lambda row: row["actor_id"])
        for left_index, left in enumerate(ordered):
            left_min = np.asarray(left["aabb_min_world_m"], dtype=np.float64)
            left_max = np.asarray(left["aabb_max_world_m"], dtype=np.float64)
            for right in ordered[left_index + 1 :]:
                right_min = np.asarray(right["aabb_min_world_m"], dtype=np.float64)
                right_max = np.asarray(right["aabb_max_world_m"], dtype=np.float64)
                collisions.append(
                    {
                        "timestamp_us": timestamp,
                        "actor_pair": [left["actor_id"], right["actor_id"]],
                        "aabb_overlap": bool(
                            np.all(np.minimum(left_max, right_max) >= np.maximum(left_min, right_min))
                        ),
                    }
                )
    return collisions


def _compile_intervention(
    base_states: list[dict[str, Any]], target_id: str, delta: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states = json.loads(json.dumps(base_states))
    for row in states:
        if row["actor_id"] != target_id:
            continue
        for field in ("centroid_world_m", "aabb_min_world_m", "aabb_max_world_m"):
            row[field] = (
                np.asarray(row[field], dtype=np.float64) + delta.astype(np.float64)
            ).tolist()
    return states, _collision_rows(states)


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R38ExperimentError("正式 R38 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R38ExperimentError("R38 task_id 漂移")
    sources = config["sources"]
    r37_run = _resolve_runs_uri(sources["r37_run"])
    binding_run = _resolve_runs_uri(sources["sceneir_binding_run"])
    base_package = binding_run / sources["base_sceneir_package"]
    frozen_files = {
        r37_run / "MANIFEST.json": sources["r37_manifest_sha256"],
        r37_run / "R37_GATE.json": sources["r37_gate_sha256"],
        r37_run / "SUMMARY.json": sources["r37_summary_sha256"],
        r37_run / "INTERVENTION_METRICS.jsonl": sources["r37_metrics_sha256"],
        binding_run / "MANIFEST.json": sources["sceneir_binding_manifest_sha256"],
        binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources[
            "sceneir_binding_gate_sha256"
        ],
        base_package / "MANIFEST.json": sources["base_sceneir_manifest_sha256"],
        base_package / "sceneir.json": sources["base_sceneir_document_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r37_gate = json.loads((r37_run / "R37_GATE.json").read_text(encoding="utf-8"))
    binding_gate = json.loads(
        (binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json").read_text(encoding="utf-8")
    )
    if not r37_gate["checks"]["passed"] or not binding_gate["checks"]["passed"]:
        raise R38ExperimentError("R37 或 SceneIR binding authority 未通过")
    base_manifest = json.loads((base_package / "MANIFEST.json").read_text(encoding="utf-8"))
    package_files = {
        base_package / relative: record["sha256"]
        for relative, record in base_manifest["files"].items()
    }
    for path, expected_sha in package_files.items():
        _verify(path, expected_sha)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R38ExperimentError("R38 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__interaction-factor-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        base = _replay_once(base_package, 1)
        target_id = str(config["cohort"]["actor_id"])
        target_states = sorted(
            (row for row in base["actor_states"] if row["actor_id"] == target_id),
            key=lambda row: int(row["timestamp_us"]),
        )
        if len(target_states) != int(config["cohort"]["expected_trajectory_rows"]):
            raise R38ExperimentError("target trajectory denominator 漂移")
        base_positions = np.asarray(
            [row["centroid_world_m"] for row in target_states], dtype=np.float64
        )
        base_collision_keys = {
            (int(row["timestamp_us"]), tuple(row["actor_pair"]))
            for row in base["collision_labels"]
            if row["aabb_overlap"] and target_id in row["actor_pair"]
        }
        decision_rows: list[dict[str, Any]] = []
        compiled_payloads: dict[str, Any] = {}
        dt_seconds = float(config["cohort"]["timestamp_step_seconds"])
        tolerance = float(config["thresholds"]["maximum_kinematic_invariance_error"])
        for intervention in config["interventions"]:
            delta = np.asarray(intervention["translation_delta_m"], dtype=np.float64)
            states_1, collisions_1 = _compile_intervention(base["actor_states"], target_id, delta)
            states_2, collisions_2 = _compile_intervention(base["actor_states"], target_id, delta)
            target_edited = sorted(
                (row for row in states_1 if row["actor_id"] == target_id),
                key=lambda row: int(row["timestamp_us"]),
            )
            edited_positions = np.asarray(
                [row["centroid_world_m"] for row in target_edited], dtype=np.float64
            )
            base_velocity = np.diff(base_positions, axis=0) / dt_seconds
            edited_velocity = np.diff(edited_positions, axis=0) / dt_seconds
            base_acceleration = np.diff(base_velocity, axis=0) / dt_seconds
            edited_acceleration = np.diff(edited_velocity, axis=0) / dt_seconds
            velocity_error = float(np.max(np.abs(base_velocity - edited_velocity)))
            acceleration_error = float(
                np.max(np.abs(base_acceleration - edited_acceleration))
            )
            edited_collision_keys = {
                (int(row["timestamp_us"]), tuple(row["actor_pair"]))
                for row in collisions_1
                if row["aabb_overlap"] and target_id in row["actor_pair"]
            }
            new_collisions = sorted(edited_collision_keys - base_collision_keys)
            removed_collisions = sorted(base_collision_keys - edited_collision_keys)
            self_kinematics = (
                "ACCEPT" if velocity_error <= tolerance and acceleration_error <= tolerance else "REJECT"
            )
            interaction = "ACCEPT" if not new_collisions else "REJECT"
            joint = "ACCEPT_CONFORMANCE" if self_kinematics == interaction == "ACCEPT" else "REJECT"
            compile_repeat_exact = _content_sha256(
                {"states": states_1, "collisions": collisions_1}
            ) == _content_sha256({"states": states_2, "collisions": collisions_2})
            decision_rows.append(
                {
                    "intervention_id": intervention["id"],
                    "translation_delta_m": delta.tolist(),
                    "q_self_kinematics": self_kinematics,
                    "q_aabb_interaction": interaction,
                    "joint_conformance_decision": joint,
                    "q_road_support": "ABSTAIN",
                    "physical_trajectory_validity": "ABSTAIN",
                    "maximum_velocity_invariance_error": velocity_error,
                    "maximum_acceleration_invariance_error": acceleration_error,
                    "baseline_target_collision_events": len(base_collision_keys),
                    "edited_target_collision_events": len(edited_collision_keys),
                    "new_collision_events": len(new_collisions),
                    "removed_collision_events": len(removed_collisions),
                    "new_collision_examples": [
                        {"timestamp_us": row[0], "actor_pair": list(row[1])}
                        for row in new_collisions[:20]
                    ],
                    "compile_repeat_exact": compile_repeat_exact,
                    "false_safe": joint == "ACCEPT_CONFORMANCE" and bool(new_collisions),
                }
            )
            compiled_payloads[intervention["id"]] = {
                "target_actor_states": target_edited,
                "target_collision_rows": [
                    row for row in collisions_1 if target_id in row["actor_pair"]
                ],
            }
        _write_jsonl(run_dir / "FACTOR_DECISIONS.jsonl", decision_rows)
        _write_json(run_dir / "INTERACTION_PAYLOADS.json", compiled_payloads)
        wall_seconds = time.monotonic() - started
        checks = {
            "r37_and_sceneir_authorities_accepted": r37_gate["checks"]["passed"]
            and binding_gate["checks"]["passed"],
            "intervention_denominator_exact": len(decision_rows) == len(config["interventions"]),
            "target_trajectory_denominator_exact": len(target_states)
            == int(config["cohort"]["expected_trajectory_rows"]),
            "self_kinematics_accepts_constant_translations": all(
                row["q_self_kinematics"] == "ACCEPT" for row in decision_rows
            ),
            "interaction_decisions_match_recomputed_truth": all(
                (row["q_aabb_interaction"] == "ACCEPT") == (row["new_collision_events"] == 0)
                for row in decision_rows
            ),
            "zero_false_safe": not any(row["false_safe"] for row in decision_rows),
            "compile_repeat_exact": all(row["compile_repeat_exact"] for row in decision_rows),
            "road_and_physical_validity_abstain": all(
                row["q_road_support"] == "ABSTAIN"
                and row["physical_trajectory_validity"] == "ABSTAIN"
                for row in decision_rows
            ),
            "source_immutable": all(
                _sha256(path) == expected_sha for path, expected_sha in frozen_files.items()
            )
            and all(_sha256(path) == expected_sha for path, expected_sha in package_files.items()),
            "wall_within_budget": wall_seconds
            <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R38_GATE.json",
            {
                "schema_version": "worldsim_v6.r38_gate.v1",
                "checks": checks,
                "decision": "accept_factorized_actor_interaction_verifier"
                if checks["passed"]
                else "reject_or_repair_actor_interaction_factor",
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r38_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r38_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_factorized_actor_interaction"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "intervention_count": len(decision_rows),
            "self_kinematics_accept_count": sum(
                row["q_self_kinematics"] == "ACCEPT" for row in decision_rows
            ),
            "interaction_accept_count": sum(
                row["q_aabb_interaction"] == "ACCEPT" for row in decision_rows
            ),
            "interaction_reject_count": sum(
                row["q_aabb_interaction"] == "REJECT" for row in decision_rows
            ),
            "false_safe_count": sum(row["false_safe"] for row in decision_rows),
            "physical_trajectory_validity": "ABSTAIN",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R38_GATE.json",
            "SUMMARY.json",
            "FACTOR_DECISIONS.jsonl",
            "INTERACTION_PAYLOADS.json",
            "RESOURCE_AUDIT.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r38_manifest.v1",
                "files": {
                    name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                    for name in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r38_actor_interaction_factor_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0

