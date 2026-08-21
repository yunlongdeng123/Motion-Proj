"""WorldSim V6 R41：融合 renderer、self、interaction 与 contact 因子。"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


TASK_ID = "WS-V6-R41-ACTOR-EDIT-FACTOR-FUSION-01"


class R41ExperimentError(RuntimeError):
    """R41 正式实验合同失败。"""


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    relative = Path(uri[len(prefix) :]) if uri.startswith(prefix) else Path("..")
    if not uri.startswith(prefix) or relative.is_absolute() or ".." in relative.parts:
        raise R41ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / relative).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R41ExperimentError(f"冻结输入漂移：{path}")


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {row["intervention_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise R41ExperimentError("intervention_id 不唯一")
    return indexed


def _fuse(factors: dict[str, str]) -> str:
    values = list(factors.values())
    if "REJECT" in values:
        return "REJECT"
    if "ABSTAIN" in values:
        return "ABSTAIN"
    return "ACCEPT_CONFORMANCE"


def _assemble(
    intervention_id: str,
    expected: dict[str, Any],
    r37: dict[str, Any],
    r38: dict[str, Any],
    r40: dict[str, Any],
) -> dict[str, Any]:
    deltas = [r37["translation_delta_m"], r38["translation_delta_m"], r40["translation_delta_m"]]
    renderer_execution = (
        "ACCEPT"
        if r37["compiled_repeat_exact"]
        and r37["native_translation_state_restored_exact"]
        and int(r37["actor_effect_pixels"]) > 0
        and int(r37["counterfactual_changed_pixels_vs_logged"]) > 0
        and all(
            math.isfinite(float(r37[key]))
            for key in ["full_sensor_rgb_mae", "full_sensor_depth_mae_m", "full_sensor_opacity_mae"]
        )
        else "REJECT"
    )
    factors = {
        "renderer_execution": renderer_execution,
        "self_kinematics": r38["q_self_kinematics"],
        "aabb_interaction": r38["q_aabb_interaction"],
        "lidar_contact": r40["q_lidar_contact"],
    }
    rejecting = [name for name, decision in factors.items() if decision == "REJECT"]
    return {
        "intervention_id": intervention_id,
        "translation_delta_m": deltas[0],
        "translation_binding_exact": deltas[0] == deltas[1] == deltas[2] == expected["translation_delta_m"],
        "factor_decisions": factors,
        "joint_admissibility": _fuse(factors),
        "rejecting_factors": rejecting,
        "unique_rejecting_factor": rejecting[0] if len(rejecting) == 1 else None,
        "expected": {
            "factor_decisions": {
                name: expected[name]
                for name in ["renderer_execution", "self_kinematics", "aabb_interaction", "lidar_contact"]
            },
            "joint_admissibility": expected["joint_admissibility"],
            "unique_rejecting_factor": expected["unique_rejecting_factor"],
        },
        "physical_trajectory_validity": "ABSTAIN",
        "safety_validity": "ABSTAIN",
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R41ExperimentError("正式 R41 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R41ExperimentError("R41 task_id 漂移")
    sources = config["sources"]
    runs = {name: _resolve_runs_uri(sources[f"{name}_run"]) for name in ["r37", "r38", "r40"]}
    files = {
        runs["r37"] / "MANIFEST.json": sources["r37_manifest_sha256"],
        runs["r37"] / "R37_GATE.json": sources["r37_gate_sha256"],
        runs["r37"] / "SUMMARY.json": sources["r37_summary_sha256"],
        runs["r37"] / "INTERVENTION_METRICS.jsonl": sources["r37_metrics_sha256"],
        runs["r38"] / "MANIFEST.json": sources["r38_manifest_sha256"],
        runs["r38"] / "R38_GATE.json": sources["r38_gate_sha256"],
        runs["r38"] / "SUMMARY.json": sources["r38_summary_sha256"],
        runs["r38"] / "FACTOR_DECISIONS.jsonl": sources["r38_decisions_sha256"],
        runs["r40"] / "MANIFEST.json": sources["r40_manifest_sha256"],
        runs["r40"] / "R40_GATE.json": sources["r40_gate_sha256"],
        runs["r40"] / "SUMMARY.json": sources["r40_summary_sha256"],
        runs["r40"] / "LIDAR_CONTACT_DECISIONS.jsonl": sources["r40_decisions_sha256"],
    }
    for path, expected_sha in files.items():
        _verify(path, expected_sha)
    gates = [json.loads((runs[name] / f"{name.upper()}_GATE.json").read_text(encoding="utf-8")) for name in ["r37", "r38", "r40"]]
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R41ExperimentError("R41 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__factor-fusion-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        r37 = _index(_read_jsonl(runs["r37"] / "INTERVENTION_METRICS.jsonl"))
        r38 = _index(_read_jsonl(runs["r38"] / "FACTOR_DECISIONS.jsonl"))
        r40_all = _index(_read_jsonl(runs["r40"] / "LIDAR_CONTACT_DECISIONS.jsonl"))
        expected = config["expected_interventions"]
        source_ids = set(r37) & set(r38) & set(r40_all)
        if source_ids != set(expected):
            raise R41ExperimentError("冻结 factor intervention cohort 不一致")
        rows_1 = [_assemble(key, expected[key], r37[key], r38[key], r40_all[key]) for key in sorted(expected)]
        rows_2 = [_assemble(key, expected[key], r37[key], r38[key], r40_all[key]) for key in sorted(expected)]
        _write_jsonl(run_dir / "FUSED_EDIT_DECISIONS.jsonl", rows_1)
        wall_seconds = time.monotonic() - started
        checks = {
            "all_source_gates_accepted": all(gate["checks"]["passed"] for gate in gates),
            "intervention_count_exact": len(rows_1) == int(config["fusion_contract"]["expected_intervention_count"]),
            "translation_binding_exact": all(row["translation_binding_exact"] for row in rows_1),
            "pre_registered_factor_decisions_exact": all(
                row["factor_decisions"] == row["expected"]["factor_decisions"] for row in rows_1
            ),
            "pre_registered_joint_decisions_exact": all(
                row["joint_admissibility"] == row["expected"]["joint_admissibility"] for row in rows_1
            ),
            "joint_reject_count_exact": sum(row["joint_admissibility"] == "REJECT" for row in rows_1)
            == int(config["fusion_contract"]["expected_joint_reject_count"]),
            "complementary_unique_reject_reasons": len(
                {row["unique_rejecting_factor"] for row in rows_1 if row["unique_rejecting_factor"]}
            ) == int(config["fusion_contract"]["expected_unique_reject_reason_count"])
            and all(
                row["unique_rejecting_factor"] == row["expected"]["unique_rejecting_factor"] for row in rows_1
            ),
            "repeat_exact": _content_sha256(rows_1) == _content_sha256(rows_2),
            "physical_and_safety_validity_abstain": all(
                row["physical_trajectory_validity"] == "ABSTAIN" and row["safety_validity"] == "ABSTAIN"
                for row in rows_1
            ),
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in files.items()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R41_GATE.json", {
            "schema_version": "worldsim_v6.r41_gate.v1",
            "checks": checks,
            "decision": "accept_factorized_actor_edit_admissibility_fusion" if checks["passed"] else "reject_or_repair_factorized_actor_edit_fusion",
        })
        _write_json(run_dir / "RESOURCE_AUDIT.json", {
            "schema_version": "worldsim_v6.r41_resource_audit.v1", "gpu_used": False,
            "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib,
            "training_started": False, "confirmation_content_read": False,
        })
        summary = {
            "schema_version": "worldsim_v6.r41_summary.v1", "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_factorized_actor_edit_admissibility_fusion" if checks["passed"] else "rejected",
            "source_commit": source_commit, "intervention_count": len(rows_1),
            "joint_reject_count": sum(row["joint_admissibility"] == "REJECT" for row in rows_1),
            "joint_accept_count": sum(row["joint_admissibility"] == "ACCEPT_CONFORMANCE" for row in rows_1),
            "physical_trajectory_validity": "ABSTAIN", "safety_validity": "ABSTAIN",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R41_GATE.json", "SUMMARY.json", "FUSED_EDIT_DECISIONS.jsonl", "RESOURCE_AUDIT.json"]
        _write_json(run_dir / "MANIFEST.json", {
            "schema_version": "worldsim_v6.r41_manifest.v1",
            "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked},
        })
        _write_json(run_dir / "TERMINAL.json", {
            "schema_version": "worldsim_v6.terminal.v1", "status": summary["status"],
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        })
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(run_dir / "TERMINAL.json", {
            "schema_version": "worldsim_v6.terminal.v1", "status": "failed",
            "error_type": type(error).__name__, "error": str(error),
        })
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r41_actor_edit_factor_fusion_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

