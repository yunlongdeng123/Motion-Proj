"""Report whether a frozen E2 cohort has enough visual evidence for scoring.

This audit is deliberately target-free.  It does not load checkpoints, actor target
surfaces, or evidence labels, and therefore cannot select a model or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    visual_run = Path(config["visual_run"])
    visual_summary = _read_json(visual_run / "summary.json")
    if not visual_summary["source_test_read"]:
        raise PermissionError("support audit requires the frozen source_test visual cache")
    if visual_summary["external_test_read"]:
        raise PermissionError("support audit does not read external_test")

    architecture_lock = REPO_ROOT / config["architecture_lock"]
    lock = _read_json(architecture_lock)
    lock_checks = {
        "source_payload_was_unread_at_freeze": not bool(lock["source_payload_read"]),
        "visual_scenes_match_lock": sorted(row["window_id"] for row in visual_summary["rows"])
        == sorted(lock["source_sample_tokens"]),
        "locked_evaluator_unchanged": _sha256(
            REPO_ROOT / "scripts/evaluate_worldsim_v72_e2_frozen.py"
        )
        == lock["code_sha256"]["scripts/evaluate_worldsim_v72_e2_frozen.py"],
    }
    rows = list(visual_summary["rows"])
    observed_candidates = sum(int(row["observed_candidate_count"]) for row in rows)
    camera_observations = sum(int(row["camera_observation_count"]) for row in rows)
    matched_actor_poses = sum(int(row["matched_actor_count"]) for row in rows)
    minimum = int(config["minimum_observed_candidates"])
    sufficient = observed_candidates >= minimum and camera_observations > 0
    verdict = (
        "frozen_source_support_sufficient"
        if sufficient
        else "frozen_confirmation_inconclusive_insufficient_visual_support"
    )
    summary = {
        "schema_version": "worldsim_v72.e2_frozen_support_audit.v1",
        "task_id": config["task_id"],
        "run_id": run_id,
        "run_uri": f"run://worldsim_v72/{config['task_id']}/{run_id}",
        "status": "done",
        "role": "source_test",
        "verdict": verdict,
        "support": {
            "window_count": int(visual_summary["window_count"]),
            "scene_count": int(visual_summary["scene_count"]),
            "matched_actor_pose_count": matched_actor_poses,
            "observed_candidate_count": observed_candidates,
            "camera_observation_count": camera_observations,
            "minimum_observed_candidates": minimum,
            "formal_evidence_metrics_defined": sufficient,
        },
        "architecture_lock": {
            "path": str(architecture_lock),
            "sha256": _sha256(architecture_lock),
            "checks": lock_checks,
        },
        "protocol": {
            "target_or_label_read": False,
            "checkpoint_loaded": False,
            "model_or_threshold_selected": False,
            "source_test_read": True,
            "external_test_read": False,
            "failed_formal_attempt": config["failed_formal_attempt"],
            "interpretation": (
                "No source EAS-vs-fallback quality claim is defined when the frozen "
                "visual cohort contains no observed actor candidates."
            ),
        },
        "visual_cache": str(visual_run),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "wall_seconds": time.monotonic() - started,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", summary["protocol"])
    _write_json(run_dir / "status.json", {"status": "done", "phase": "complete"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config.resolve(), arguments.run_id), ensure_ascii=False))


if __name__ == "__main__":
    main()
