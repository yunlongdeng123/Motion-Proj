"""V6 PT4 冻结风险策略的一次性独立场景 confirmation。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from motion_proj.worldsim_v6.pt2_risk_policy import (
    _canonical,
    _evaluate,
    _git,
    _load_scene_sources,
    _scene_rows,
    _sha256,
    _write_json,
    _write_jsonl,
)


TASK_ID = "WS-V6-PT4-RISK-POLICY-CONFIRMATION-01"
ALLOWED_TASK_IDS = {TASK_ID, "WS-V6-PT5-RISK-POLICY-TEST-01"}


class PT4ConfirmationError(RuntimeError):
    """PT4 confirmation 正式合同失败。"""


def _factor_sets(geometry: dict[str, Any]) -> dict[str, set[Any]]:
    return {
        "forward": set(geometry["synthetic_clone_forward_offsets_m"]),
        "lateral": set(geometry["synthetic_clone_lateral_offsets_m"]),
        "size": {tuple(value) for value in geometry["synthetic_actor_sizes_m"]},
        "yaw": set(geometry["synthetic_clone_yaw_offsets_deg"]),
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    config_path = (repo_root / config_path).resolve() if not config_path.is_absolute() else config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_id = str(config["task_id"])
    phase = str(config.get("evaluation_phase", "confirmation"))
    if task_id not in ALLOWED_TASK_IDS or phase not in {"confirmation", "test"}:
        raise PT4ConfirmationError("task_id 不匹配")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / task_id / f"{stamp}__risk-policy-{phase}-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    # attempt 必须先于任何 confirmation scene 内容或质量结果读取落盘。
    attempt = {
        "schema_version": f"worldsim_v6.{phase}_attempt.v1",
        "task_id": task_id,
        "hypothesis_id": config["hypothesis_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_created_before_quality_read": True,
        "quality_read": False,
        "evaluation_phase": phase,
        "candidate_gate_sha256": config["frozen_candidate"]["gate_sha256"],
        "candidate_policy_arms_sha256": config["frozen_candidate"]["policy_arms_sha256"],
        f"{phase}_scene": config["confirmation_scene"]["scene"],
        "gate": config["gate"],
        "status": "created",
    }
    _write_json(run_dir / "ATTEMPT.json", attempt)
    try:
        if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
            raise PT4ConfirmationError("磁盘资源不足")
        source_commit = _git(repo_root, "rev-parse", "HEAD")
        if _git(repo_root, "status", "--short"):
            raise PT4ConfirmationError("正式运行要求 clean worktree")
        frozen = config["frozen_candidate"]
        candidate_gate_path = Path(frozen["gate_path"])
        policy_arms_path = Path(frozen["policy_arms_path"])
        if _sha256(candidate_gate_path) != frozen["gate_sha256"]:
            raise PT4ConfirmationError("冻结 candidate gate hash 漂移")
        if _sha256(policy_arms_path) != frozen["policy_arms_sha256"]:
            raise PT4ConfirmationError("冻结 candidate policy hash 漂移")
        candidate_gate = json.loads(candidate_gate_path.read_text(encoding="utf-8"))
        if candidate_gate["decision"] != "accept_factorized_intervention_policy_training":
            raise PT4ConfirmationError("冻结 candidate 未通过 development gate")
        policy_arms = json.loads(policy_arms_path.read_text(encoding="utf-8"))
        prior_gate_path = None
        if config.get("required_prior_gate"):
            prior = config["required_prior_gate"]
            prior_gate_path = Path(prior["path"])
            if _sha256(prior_gate_path) != prior["sha256"]:
                raise PT4ConfirmationError("前序 gate hash 漂移")
            prior_gate = json.loads(prior_gate_path.read_text(encoding="utf-8"))
            if prior_gate["decision"] != prior["required_decision"]:
                raise PT4ConfirmationError("前序 gate decision 不匹配")

        expected_frames = int(config["partition"]["expected_frame_count_per_scene"])
        spec = config["confirmation_scene"]
        instances, poses, paths = _load_scene_sources(
            Path(config["confirmation_dataset_root"]), spec, expected_frames
        )
        scene_config = dict(config)
        scene_config["geometry"] = config["confirmation_geometry"]
        real_rows, synthetic_rows = _scene_rows(spec, instances, poses, scene_config)
        confirmation_rows = [*real_rows, *synthetic_rows]
        frozen_paths = [config_path, candidate_gate_path, policy_arms_path, *paths]
        if prior_gate_path is not None:
            frozen_paths.append(prior_gate_path)
        hashes_before = {str(path): _sha256(path) for path in frozen_paths}
        metrics1 = {
            arm: _evaluate(value["model"], confirmation_rows)
            for arm, value in policy_arms.items()
        }
        metrics2 = {
            arm: _evaluate(value["model"], confirmation_rows)
            for arm, value in policy_arms.items()
        }
        repeat_exact = _canonical(metrics1) == _canonical(metrics2)
        _write_json(run_dir / "CONFIRMATION_ARMS.json", metrics1)
        _write_jsonl(run_dir / "CONFIRMATION_ROWS.jsonl", confirmation_rows)
        _write_json(
            run_dir / "SOURCE_AUDIT.json",
            {
                f"{phase}_scene": spec["scene"],
                "real_rows": len(real_rows),
                "synthetic_rows": len(synthetic_rows),
                "source_sha256": hashes_before,
            },
        )
        v6 = metrics1["real_plus_v6_verified_compiled"]
        real = metrics1["real_only"]
        naive = metrics1["real_plus_naive_synthetic"]
        reduction_real = real["false_safe_rate"] - v6["false_safe_rate"]
        reduction_naive = naive["false_safe_rate"] - v6["false_safe_rate"]
        confirmation_sets = _factor_sets(config["confirmation_geometry"])
        development_sets = [_factor_sets(value) for value in config["development_geometries"]]
        factors_disjoint = all(
            confirmation_sets[key].isdisjoint(dev[key])
            for dev in development_sets
            for key in confirmation_sets
        )
        cfg = config["gate"]
        wall_seconds = time.monotonic() - started
        checks = {
            "attempt_created_before_quality_read": True,
            "candidate_frozen_exact": True,
            "candidate_development_gate_accepted": True,
            f"new_{phase}_scene": spec["scene"] not in set(config["development_scenes"]),
            f"{phase}_factors_disjoint": factors_disjoint,
            f"{phase}_has_both_outcomes": v6["hazard_count"] > 0 and v6["safe_count"] > 0,
            "v6_balanced_accuracy": v6["balanced_accuracy"] >= float(cfg["require_v6_balanced_accuracy_at_least"]),
            "v6_false_safe": v6["false_safe_rate"] <= float(cfg["require_v6_false_safe_rate_at_most"]),
            "v6_safe_route_completion": v6["safe_route_completion"] >= float(cfg["require_v6_safe_route_completion_at_least"]),
            "false_safe_reduction_vs_real_only": reduction_real >= float(cfg["require_false_safe_reduction_vs_real_only_at_least"]),
            "false_safe_reduction_vs_naive": reduction_naive >= float(cfg["require_false_safe_reduction_vs_naive_at_least"]),
            "evaluation_repeat_exact": repeat_exact,
            "source_immutable": hashes_before == {str(path): _sha256(path) for path in frozen_paths},
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in config["unsupported_metrics"].values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
        }
        checks["passed"] = all(checks.values())
        accepted_decision = (
            "accept_risk_policy_confirmation"
            if phase == "confirmation"
            else "accept_risk_policy_exact_once_test"
        )
        rejected_decision = (
            "reject_risk_policy_confirmation"
            if phase == "confirmation"
            else "reject_risk_policy_exact_once_test"
        )
        gate = {
            "schema_version": f"worldsim_v6.{phase}_risk_policy_gate.v1",
            "checks": checks,
            "decision": accepted_decision if checks["passed"] else rejected_decision,
            "false_safe_reduction_vs_real_only": reduction_real,
            "false_safe_reduction_vs_naive": reduction_naive,
            f"{phase}_attempt_consumed": True,
            "unsupported_metrics": config["unsupported_metrics"],
        }
        gate_name = "PT4_CONFIRMATION_GATE.json" if phase == "confirmation" else "PT5_TEST_GATE.json"
        _write_json(run_dir / gate_name, gate)
        summary = {
            "schema_version": f"worldsim_v6.{phase}_risk_policy_summary.v1",
            "task_id": task_id,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "frozen_candidate": frozen,
            f"{phase}_scene": spec["scene"],
            "method_arms": metrics1,
            "wall_seconds": wall_seconds,
            f"{phase}_attempt_consumed": True,
            f"{phase}_content_read": True,
            "claim_boundary": config["claim_boundary"],
            "unsupported_metrics": config["unsupported_metrics"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "ATTEMPT.json",
            "CONFIRMATION_ARMS.json",
            "CONFIRMATION_ROWS.jsonl",
            "SOURCE_AUDIT.json",
            gate_name,
            "SUMMARY.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": f"worldsim_v6.{phase}_risk_policy_manifest.v1",
                "source_commit": source_commit,
                "config": str(config_path),
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
                "task_id": task_id,
                "hypothesis_id": config["hypothesis_id"],
                f"{phase}_attempt_consumed": True,
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "blocked",
                "task_id": task_id,
                f"{phase}_attempt_consumed": True,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    print(run_experiment(args.repo_root, args.config, args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
