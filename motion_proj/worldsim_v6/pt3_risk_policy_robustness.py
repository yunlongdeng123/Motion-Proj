"""冻结 PT2 policy 在新场景与非零 lateral edit 上的正式鲁棒性实验。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

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


TASK_ID = "WS-V6-PT3-RISK-POLICY-ROBUSTNESS-01"


class PT3RobustnessError(RuntimeError):
    """PT3 正式合同失败。"""


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise PT3RobustnessError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _evaluate_all(models: dict, rows: list[dict]) -> dict:
    return {arm: {"model": value["model"], "heldout": _evaluate(value["model"], rows)} for arm, value in models.items()}


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    repo_root = repo_root.resolve()
    config_path = (repo_root / config_path).resolve() if not config_path.is_absolute() else config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["task_id"] != TASK_ID:
        raise PT3RobustnessError("task_id 不匹配")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{stamp}__intervention-robustness-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
            raise PT3RobustnessError("磁盘资源不足")
        source_commit = _git(repo_root, "rev-parse", "HEAD")
        if _git(repo_root, "status", "--short"):
            raise PT3RobustnessError("正式运行要求 clean worktree")
        source_run = _resolve_runs_uri(config["policy_source"]["run"])
        gate_path = source_run / "PT2_RISK_POLICY_GATE.json"
        manifest_path = source_run / "MANIFEST.json"
        arms_path = source_run / "POLICY_ARMS.json"
        if _sha256(gate_path) != config["policy_source"]["gate_sha256"]:
            raise PT3RobustnessError("PT2 gate hash 漂移")
        if _sha256(manifest_path) != config["policy_source"]["manifest_sha256"]:
            raise PT3RobustnessError("PT2 manifest hash 漂移")
        if _sha256(arms_path) != config["policy_source"]["policy_arms_sha256"]:
            raise PT3RobustnessError("PT2 policy arms hash 漂移")
        if json.loads(gate_path.read_text(encoding="utf-8"))["decision"] != "accept_small_risk_policy_post_training":
            raise PT3RobustnessError("PT2 source policy 未接受")
        models = json.loads(arms_path.read_text(encoding="utf-8"))
        expected_frames = int(config["partition"]["expected_frame_count_per_scene"])
        spec = config["heldout_scene"]
        instances, poses, scene_paths = _load_scene_sources(
            Path(config["heldout_dataset_root"]), spec, expected_frames
        )
        clean_rows, edit_rows = _scene_rows(spec, instances, poses, config)
        heldout_rows = [*clean_rows, *edit_rows]
        frozen = [config_path, gate_path, manifest_path, arms_path, *scene_paths]
        hashes_before = {str(path): _sha256(path) for path in frozen}
        results1 = _evaluate_all(models, heldout_rows)
        results2 = _evaluate_all(models, heldout_rows)
        repeat_exact = _canonical(results1) == _canonical(results2)
        _write_json(run_dir / "ROBUSTNESS_ARMS.json", results1)
        _write_jsonl(run_dir / "HELDOUT_ROWS.jsonl", heldout_rows)
        _write_json(run_dir / "SOURCE_AUDIT.json", {"source_sha256": hashes_before,
                    "clean_rows": len(clean_rows), "edit_rows": len(edit_rows), "scene": spec["scene"]})
        v6 = results1["real_plus_v6_verified_compiled"]["heldout"]
        real = results1["real_only"]["heldout"]
        naive = results1["real_plus_naive_synthetic"]["heldout"]
        reduction_real = real["false_safe_rate"] - v6["false_safe_rate"]
        reduction_naive = naive["false_safe_rate"] - v6["false_safe_rate"]
        wall_seconds = time.monotonic() - started
        cfg = config["gate"]
        checks = {
            "new_scene_not_pt2_train_or_heldout": spec["scene"] not in {"scene-0230", "scene-0242", "scene-0255"},
            "nonzero_lateral_intervention": all(float(value) != 0.0 for value in config["geometry"]["synthetic_clone_lateral_offsets_m"]),
            "heldout_has_both_outcomes": v6["hazard_count"] > 0 and v6["safe_count"] > 0,
            "v6_balanced_accuracy": v6["balanced_accuracy"] >= float(cfg["require_v6_balanced_accuracy_at_least"]),
            "v6_false_safe": v6["false_safe_rate"] <= float(cfg["require_v6_false_safe_rate_at_most"]),
            "v6_safe_route_completion": v6["safe_route_completion"] >= float(cfg["require_v6_safe_route_completion_at_least"]),
            "false_safe_reduction_vs_real_only": reduction_real >= float(cfg["require_false_safe_reduction_vs_real_only_at_least"]),
            "false_safe_reduction_vs_naive": reduction_naive >= float(cfg["require_false_safe_reduction_vs_naive_at_least"]),
            "repeat_exact": repeat_exact,
            "source_immutable": hashes_before == {str(path): _sha256(path) for path in frozen},
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in config["unsupported_metrics"].values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {"schema_version": "worldsim_v6.pt3_risk_policy_robustness_gate.v1", "checks": checks,
                "decision": "accept_frozen_policy_intervention_robustness" if checks["passed"] else "reject_pt3_intervention_robustness",
                "false_safe_reduction_vs_real_only": reduction_real,
                "false_safe_reduction_vs_naive": reduction_naive,
                "unsupported_metrics": config["unsupported_metrics"]}
        _write_json(run_dir / "PT3_ROBUSTNESS_GATE.json", gate)
        summary = {"schema_version": "worldsim_v6.pt3_risk_policy_robustness_summary.v1", "task_id": TASK_ID,
                   "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
                   "source_commit": source_commit, "heldout_scene": spec["scene"], "method_arms": results1,
                   "clean_rows": len(clean_rows), "edit_rows": len(edit_rows), "wall_seconds": wall_seconds,
                   "training_started": False, "confirmation_content_read": False,
                   "claim_boundary": config["claim_boundary"], "unsupported_metrics": config["unsupported_metrics"]}
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["ROBUSTNESS_ARMS.json", "HELDOUT_ROWS.jsonl", "SOURCE_AUDIT.json", "PT3_ROBUSTNESS_GATE.json", "SUMMARY.json"]
        _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.pt3_robustness_manifest.v1",
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
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/pt3_risk_policy_robustness_v0.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    print(run_experiment(args.repo_root, args.config, args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
