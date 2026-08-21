"""WorldSim V6 R39：用 observed static Gaussian 邻域验证 actor ground-contact support。"""

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
from scipy.spatial import cKDTree

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once


TASK_ID = "WS-V6-R39-ACTOR-STATIC-CONTACT-FACTOR-01"


class R39ExperimentError(RuntimeError):
    """R39 正式实验合同失败。"""


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
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256((payload + "\n").encode()).hexdigest()


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
        raise R39ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / relative).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R39ExperimentError(f"冻结输入漂移：{path}")


def _contact_rows(
    states: list[dict[str, Any]],
    tree: cKDTree,
    static_means: np.ndarray,
    config: dict[str, Any],
    intervention_id: str,
) -> list[dict[str, Any]]:
    vertical_axis = int(config["coordinate_contract"]["vertical_axis_index"])
    horizontal_axes = [int(value) for value in config["coordinate_contract"]["horizontal_axis_indices"]]
    query_k = int(config["support_query"]["nearest_horizontal_candidates"])
    radius = float(config["support_query"]["maximum_horizontal_radius_m"])
    vertical_tolerance = float(config["support_query"]["maximum_vertical_contact_error_m"])
    minimum_points = int(config["support_query"]["minimum_contact_points"])
    rows: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda row: int(row["timestamp_us"])):
        center = np.asarray(state["centroid_world_m"], dtype=np.float64)
        bottom = float(np.asarray(state["aabb_min_world_m"], dtype=np.float64)[vertical_axis])
        distances, indices = tree.query(center[horizontal_axes], k=query_k, distance_upper_bound=radius)
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        valid = np.isfinite(distances) & (indices < static_means.shape[0])
        vertical_errors = np.abs(static_means[indices[valid], vertical_axis] - bottom)
        contact_points = int(np.count_nonzero(vertical_errors <= vertical_tolerance))
        rows.append(
            {
                "intervention_id": intervention_id,
                "timestamp_us": int(state["timestamp_us"]),
                "actor_bottom_vertical_m": bottom,
                "horizontal_candidate_count": int(np.count_nonzero(valid)),
                "contact_point_count": contact_points,
                "minimum_vertical_error_m": float(vertical_errors.min())
                if vertical_errors.size
                else None,
                "supported": contact_points >= minimum_points,
            }
        )
    return rows


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R39ExperimentError("正式 R39 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R39ExperimentError("R39 task_id 漂移")
    sources = config["sources"]
    r38_run = _resolve_runs_uri(sources["r38_run"])
    binding_run = _resolve_runs_uri(sources["sceneir_binding_run"])
    base_package = binding_run / sources["base_sceneir_package"]
    frozen_files = {
        r38_run / "MANIFEST.json": sources["r38_manifest_sha256"],
        r38_run / "R38_GATE.json": sources["r38_gate_sha256"],
        r38_run / "SUMMARY.json": sources["r38_summary_sha256"],
        r38_run / "FACTOR_DECISIONS.jsonl": sources["r38_decisions_sha256"],
        r38_run / "INTERACTION_PAYLOADS.json": sources["r38_payloads_sha256"],
        binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources[
            "sceneir_binding_gate_sha256"
        ],
        base_package / "MANIFEST.json": sources["base_sceneir_manifest_sha256"],
        base_package / "sceneir.json": sources["base_sceneir_document_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r38_gate = json.loads((r38_run / "R38_GATE.json").read_text(encoding="utf-8"))
    if not r38_gate["checks"]["passed"]:
        raise R39ExperimentError("R38 interaction factor authority 未通过")
    base_manifest = json.loads((base_package / "MANIFEST.json").read_text(encoding="utf-8"))
    package_files = {
        base_package / relative: record["sha256"]
        for relative, record in base_manifest["files"].items()
    }
    for path, expected_sha in package_files.items():
        _verify(path, expected_sha)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R39ExperimentError("R39 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__static-contact-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        document = json.loads((base_package / "sceneir.json").read_text(encoding="utf-8"))
        static_chunks = [row for row in document["chunks"] if row["role"] == "static"]
        if len(static_chunks) != 1:
            raise R39ExperimentError("R39 要求唯一 observed static chunk")
        static_reference = static_chunks[0]["arrays"]["means_m"]
        static_means = np.load(base_package / static_reference["path"], allow_pickle=False).astype(
            np.float64
        )
        if static_means.shape[0] != int(static_chunks[0]["primitive_count"]):
            raise R39ExperimentError("static primitive denominator 漂移")
        horizontal_axes = [
            int(value) for value in config["coordinate_contract"]["horizontal_axis_indices"]
        ]
        tree = cKDTree(static_means[:, horizontal_axes])
        base = _replay_once(base_package, 1)
        target_id = str(config["cohort"]["actor_id"])
        baseline_states = [row for row in base["actor_states"] if row["actor_id"] == target_id]
        payloads = json.loads((r38_run / "INTERACTION_PAYLOADS.json").read_text(encoding="utf-8"))
        all_rows_1 = _contact_rows(
            baseline_states, tree, static_means, config, "logged_baseline"
        )
        decision_rows: list[dict[str, Any]] = []
        for intervention in config["interventions"]:
            states = payloads[intervention["id"]]["target_actor_states"]
            rows = _contact_rows(states, tree, static_means, config, intervention["id"])
            all_rows_1.extend(rows)
        all_rows_2 = _contact_rows(
            baseline_states, tree, static_means, config, "logged_baseline"
        )
        for intervention in config["interventions"]:
            all_rows_2.extend(
                _contact_rows(
                    payloads[intervention["id"]]["target_actor_states"],
                    tree,
                    static_means,
                    config,
                    intervention["id"],
                )
            )
        by_id: dict[str, list[dict[str, Any]]] = {}
        for row in all_rows_1:
            by_id.setdefault(row["intervention_id"], []).append(row)
        baseline_coverage = float(
            np.mean([row["supported"] for row in by_id["logged_baseline"]])
        )
        for intervention in config["interventions"]:
            rows = by_id[intervention["id"]]
            coverage = float(np.mean([row["supported"] for row in rows]))
            retention = coverage / baseline_coverage if baseline_coverage > 0 else 0.0
            accepted = coverage >= float(config["gate"]["minimum_absolute_support_coverage"])
            accepted = accepted and retention >= float(config["gate"]["minimum_support_retention"])
            decision_rows.append(
                {
                    "intervention_id": intervention["id"],
                    "translation_delta_m": intervention["translation_delta_m"],
                    "logged_support_coverage": baseline_coverage,
                    "edited_support_coverage": coverage,
                    "support_retention": retention,
                    "q_static_contact": "ACCEPT" if accepted else "REJECT",
                    "expected_decision": intervention["expected_static_contact_decision"],
                    "q_semantic_road": "ABSTAIN",
                    "physical_trajectory_validity": "ABSTAIN",
                }
            )
        repeat_exact = _content_sha256(all_rows_1) == _content_sha256(all_rows_2)
        _write_jsonl(run_dir / "STATIC_CONTACT_ROWS.jsonl", all_rows_1)
        _write_jsonl(run_dir / "STATIC_CONTACT_DECISIONS.jsonl", decision_rows)
        wall_seconds = time.monotonic() - started
        checks = {
            "r38_authority_accepted": r38_gate["checks"]["passed"],
            "streetgs_coordinate_contract_frozen": config["coordinate_contract"]["world_up_axis"]
            == "y"
            and horizontal_axes == [0, 2],
            "static_chunk_denominator_exact": static_means.shape[0]
            == int(config["cohort"]["expected_static_primitive_count"]),
            "trajectory_denominators_exact": all(
                len(rows) == int(config["cohort"]["expected_trajectory_rows"])
                for rows in by_id.values()
            ),
            "logged_support_coverage_sufficient": baseline_coverage
            >= float(config["gate"]["minimum_logged_support_coverage"]),
            "pre_registered_directional_control": all(
                row["q_static_contact"] == row["expected_decision"] for row in decision_rows
            ),
            "repeat_exact": repeat_exact,
            "semantic_road_and_physical_validity_abstain": all(
                row["q_semantic_road"] == "ABSTAIN"
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
            run_dir / "R39_GATE.json",
            {
                "schema_version": "worldsim_v6.r39_gate.v1",
                "checks": checks,
                "decision": "accept_actor_static_contact_factor"
                if checks["passed"]
                else "reject_or_repair_actor_static_contact_factor",
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r39_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r39_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_actor_static_contact_factor"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "static_primitive_count": int(static_means.shape[0]),
            "logged_support_coverage": baseline_coverage,
            "static_contact_accept_count": sum(
                row["q_static_contact"] == "ACCEPT" for row in decision_rows
            ),
            "static_contact_reject_count": sum(
                row["q_static_contact"] == "REJECT" for row in decision_rows
            ),
            "physical_trajectory_validity": "ABSTAIN",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R39_GATE.json",
            "SUMMARY.json",
            "STATIC_CONTACT_ROWS.jsonl",
            "STATIC_CONTACT_DECISIONS.jsonl",
            "RESOURCE_AUDIT.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r39_manifest.v1",
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
        default=Path("configs/worldsim_v6/r39_actor_static_contact_factor_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0

