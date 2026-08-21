"""V6 PT3 二维 factorized typed-edit risk policy post-training。"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from motion_proj.worldsim_v6.pt2_risk_policy import (
    _canonical,
    _git,
    _load_scene_sources,
    _scene_rows,
    _sha256,
    _train_and_evaluate,
    _write_json,
    _write_jsonl,
)


TASK_ID = "WS-V6-PT3-RISK-POLICY-ROBUSTNESS-01"


class PT3TrainingError(RuntimeError):
    """PT3 training 正式合同失败。"""


def _with_geometry(config: dict, key: str) -> dict:
    value = dict(config)
    value["geometry"] = config[key]
    return value


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    config_path = (repo_root / config_path).resolve() if not config_path.is_absolute() else config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise PT3TrainingError("task_id 不匹配")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{stamp}__factorized-policy-training-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
            raise PT3TrainingError("磁盘资源不足")
        source_commit = _git(repo_root, "rev-parse", "HEAD")
        if _git(repo_root, "status", "--short"):
            raise PT3TrainingError("正式运行要求 clean worktree")
        expected_frames = int(config["partition"]["expected_frame_count_per_scene"])
        train_config = _with_geometry(config, "train_geometry")
        heldout_config = _with_geometry(config, "heldout_geometry")
        frozen_paths = [config_path]
        train_real, train_synthetic = [], []
        source_audit = []
        for spec in config["train_scenes"]:
            instances, poses, paths = _load_scene_sources(
                Path(config["train_dataset_root"]), spec, expected_frames
            )
            real, synthetic = _scene_rows(spec, instances, poses, train_config)
            train_real.extend(real)
            train_synthetic.extend(synthetic)
            frozen_paths.extend(paths)
            source_audit.append({"scene": spec["scene"], "real_rows": len(real), "synthetic_rows": len(synthetic)})
        spec = config["heldout_scene"]
        instances, poses, paths = _load_scene_sources(
            Path(config["heldout_dataset_root"]), spec, expected_frames
        )
        heldout_real, heldout_synthetic = _scene_rows(spec, instances, poses, heldout_config)
        heldout_rows = [*heldout_real, *heldout_synthetic]
        frozen_paths.extend(paths)
        hashes_before = {str(path): _sha256(path) for path in frozen_paths}
        arms1 = _train_and_evaluate(
            train_real, train_synthetic, heldout_rows, list(config["arms"]), config["training"]
        )
        arms2 = _train_and_evaluate(
            train_real, train_synthetic, heldout_rows, list(config["arms"]), config["training"]
        )
        repeat_exact = _canonical(arms1) == _canonical(arms2)
        _write_json(run_dir / "POLICY_ARMS.json", arms1)
        _write_jsonl(run_dir / "TRAIN_REAL_ROWS.jsonl", train_real)
        _write_jsonl(run_dir / "TRAIN_SYNTHETIC_ROWS.jsonl", train_synthetic)
        _write_jsonl(run_dir / "HELDOUT_ROWS.jsonl", heldout_rows)
        _write_json(run_dir / "SOURCE_AUDIT.json", {"train": source_audit,
                    "heldout": {"scene": spec["scene"], "real_rows": len(heldout_real),
                    "synthetic_rows": len(heldout_synthetic)}, "source_sha256": hashes_before})
        v6 = arms1["real_plus_v6_verified_compiled"]["heldout"]
        real = arms1["real_only"]["heldout"]
        naive = arms1["real_plus_naive_synthetic"]["heldout"]
        reduction_real = real["false_safe_rate"] - v6["false_safe_rate"]
        reduction_naive = naive["false_safe_rate"] - v6["false_safe_rate"]
        wall_seconds = time.monotonic() - started
        cfg = config["gate"]
        train_sizes = {tuple(value) for value in config["train_geometry"].get("synthetic_actor_sizes_m", [])}
        heldout_sizes = {tuple(value) for value in config["heldout_geometry"].get("synthetic_actor_sizes_m", [])}
        checks = {
            "new_heldout_scene": spec["scene"] not in {row["scene"] for row in config["train_scenes"]},
            "disjoint_intervention_grids": set(config["train_geometry"]["synthetic_clone_forward_offsets_m"]).isdisjoint(config["heldout_geometry"]["synthetic_clone_forward_offsets_m"])
            and set(config["train_geometry"]["synthetic_clone_lateral_offsets_m"]).isdisjoint(config["heldout_geometry"]["synthetic_clone_lateral_offsets_m"])
            and (not train_sizes or train_sizes.isdisjoint(heldout_sizes))
            and set(config["train_geometry"].get("synthetic_clone_yaw_offsets_deg", [])).isdisjoint(config["heldout_geometry"].get("synthetic_clone_yaw_offsets_deg", [])),
            "heldout_has_both_outcomes": v6["hazard_count"] > 0 and v6["safe_count"] > 0,
            "v6_balanced_accuracy": v6["balanced_accuracy"] >= float(cfg["require_v6_balanced_accuracy_at_least"]),
            "v6_false_safe": v6["false_safe_rate"] <= float(cfg["require_v6_false_safe_rate_at_most"]),
            "v6_safe_route_completion": v6["safe_route_completion"] >= float(cfg["require_v6_safe_route_completion_at_least"]),
            "false_safe_reduction_vs_real_only": reduction_real >= float(cfg["require_false_safe_reduction_vs_real_only_at_least"]),
            "false_safe_reduction_vs_naive": reduction_naive >= float(cfg["require_false_safe_reduction_vs_naive_at_least"]),
            "training_repeat_exact": repeat_exact,
            "source_immutable": hashes_before == {str(path): _sha256(path) for path in frozen_paths},
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in config["unsupported_metrics"].values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {"schema_version": "worldsim_v6.pt3_risk_policy_training_gate.v1", "checks": checks,
                "decision": "accept_factorized_intervention_policy_training" if checks["passed"] else "reject_pt3_factorized_policy_training",
                "false_safe_reduction_vs_real_only": reduction_real,
                "false_safe_reduction_vs_naive": reduction_naive,
                "unsupported_metrics": config["unsupported_metrics"]}
        _write_json(run_dir / "PT3_POLICY_TRAINING_GATE.json", gate)
        summary = {"schema_version": "worldsim_v6.pt3_risk_policy_training_summary.v1", "task_id": TASK_ID,
                   "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
                   "source_commit": source_commit, "train_scenes": [row["scene"] for row in config["train_scenes"]],
                   "heldout_scene": spec["scene"], "method_arms": arms1, "wall_seconds": wall_seconds,
                   "training_started": True, "confirmation_content_read": False,
                   "claim_boundary": config["claim_boundary"], "unsupported_metrics": config["unsupported_metrics"]}
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["POLICY_ARMS.json", "TRAIN_REAL_ROWS.jsonl", "TRAIN_SYNTHETIC_ROWS.jsonl", "HELDOUT_ROWS.jsonl",
                   "SOURCE_AUDIT.json", "PT3_POLICY_TRAINING_GATE.json", "SUMMARY.json"]
        _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.pt3_policy_training_manifest.v1",
                    "source_commit": source_commit, "config": str(config_path),
                    "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": summary["status"],
                    "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "manifest_sha256": _sha256(run_dir / "MANIFEST.json")})
        return run_dir
    except Exception as error:
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "blocked",
                    "task_id": TASK_ID, "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/pt3_risk_policy_training_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    print(run_experiment(args.repo_root, args.config, args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
