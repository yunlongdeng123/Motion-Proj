"""WorldSim V6 R53：冻结语义感知下的 lifecycle 回归效用。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import _git, _resolve_runs_uri, _sha256, _verify, _write_json


TASK_ID = "WS-V6-R53-LIFECYCLE-PERCEPTION-REGRESSION-01"


class R53ExperimentError(RuntimeError):
    """R53 正式实验合同失败。"""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R53ExperimentError("正式 R53 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R53ExperimentError("R53 task_id 漂移")
    sources = config["sources"]
    r52_run = _resolve_runs_uri(sources["r52_run"])
    r49_run = _resolve_runs_uri(sources["r49_run"])
    r51_run = _resolve_runs_uri(sources["r51_run"])
    model_root = Path(sources["semantic_model_root"])
    frozen_files = {
        r52_run / "R52_GATE.json": sources["r52_gate_sha256"],
        r52_run / "SUMMARY.json": sources["r52_summary_sha256"],
        r49_run / "R49_GATE.json": sources["r49_gate_sha256"],
        r49_run / "direct_detached/FRAME_METRICS.jsonl": sources["r49_metrics_sha256"],
        r51_run / "R51_GATE.json": sources["r51_gate_sha256"],
        r51_run / "worker/FRAME_METRICS.jsonl": sources["r51_metrics_sha256"],
        model_root / sources["semantic_model_file"]: sources["semantic_model_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R53ExperimentError("R53 磁盘资源不足")
    r52_gate = json.loads((r52_run / "R52_GATE.json").read_text(encoding="utf-8"))
    r49_gate = json.loads((r49_run / "R49_GATE.json").read_text(encoding="utf-8"))
    r51_gate = json.loads((r51_run / "R51_GATE.json").read_text(encoding="utf-8"))
    naive_rows = {int(row["frame_index"]): row for row in _load_rows(r49_run / "direct_detached/FRAME_METRICS.jsonl")}
    v6_rows = {int(row["frame_index"]): row for row in _load_rows(r51_run / "worker/FRAME_METRICS.jsonl")}
    evaluation = config["evaluation"]
    frames = [int(evaluation["active_control_frame"])] + [int(value) for value in evaluation["inactive_frames"]]
    for frame in frames:
        _verify(r49_run / "direct_detached" / naive_rows[frame]["sensor_path"], naive_rows[frame]["sensor_sha256"])
        _verify(r51_run / "worker" / v6_rows[frame]["sensor_path"], v6_rows[frame]["sensor_sha256"])
        frozen_files[r49_run / "direct_detached" / naive_rows[frame]["sensor_path"]] = naive_rows[frame]["sensor_sha256"]
        frozen_files[r51_run / "worker" / v6_rows[frame]["sensor_path"]] = v6_rows[frame]["sensor_sha256"]

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__lifecycle-perception-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    perception_dir = run_dir / "perception"
    index_rows: list[dict[str, Any]] = []
    for frame in frames:
        for variant, root, row in (("naive", r49_run / "direct_detached", naive_rows[frame]), ("v6", r51_run / "worker", v6_rows[frame])):
            for repeat in range(int(evaluation["repeat_count"])):
                index_rows.append({
                    "case_id": f"{variant}_frame{frame:03d}",
                    "repeat_index": repeat,
                    "render_path": str(root / row["sensor_path"]),
                })
    _write_jsonl(run_dir / "PERCEPTION_INPUT_INDEX.jsonl", index_rows)
    log_path = run_dir / "perception.log"
    command = [
        sys.executable,
        str(repo_root / "scripts/worldsim_v6/r53_perception_worker.py"),
        "--index", str(run_dir / "PERCEPTION_INPUT_INDEX.jsonl"),
        "--model-root", str(model_root),
        "--output-dir", str(perception_dir),
    ]
    with log_path.open("w", encoding="utf-8") as log_stream:
        subprocess.run(command, cwd=repo_root, stdout=log_stream, stderr=subprocess.STDOUT, check=True)
    worker = json.loads((perception_dir / "WORKER_RESULT.json").read_text(encoding="utf-8"))
    rows = _load_rows(perception_dir / "PERCEPTION_OUTPUTS.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    repeat_exact = all(len(items) == int(evaluation["repeat_count"]) and len({item["label_array_sha256"] for item in items}) == 1 for items in grouped.values())
    comparisons = []
    for frame in frames:
        naive_label = np.load(perception_dir / sorted(grouped[f"naive_frame{frame:03d}"], key=lambda item: item["repeat_index"])[0]["label_path"], allow_pickle=False)
        v6_label = np.load(perception_dir / sorted(grouped[f"v6_frame{frame:03d}"], key=lambda item: item["repeat_index"])[0]["label_path"], allow_pickle=False)
        changed = naive_label != v6_label
        comparisons.append({
            "frame_index": frame,
            "actor_frame_valid": bool(v6_rows[frame]["package_actor_frame_valid"]),
            "changed_label_pixels": int(changed.sum()),
            "changed_label_fraction": float(changed.mean()),
            "naive_label_sha256": grouped[f"naive_frame{frame:03d}"][0]["label_array_sha256"],
            "v6_label_sha256": grouped[f"v6_frame{frame:03d}"][0]["label_array_sha256"],
        })
    _write_json(run_dir / "PERCEPTION_COMPARISON.json", {"schema_version": "worldsim_v6.r53_perception_comparison.v1", "rows": comparisons})
    by_frame = {row["frame_index"]: row for row in comparisons}
    active_frame = int(evaluation["active_control_frame"])
    inactive_frames = [int(value) for value in evaluation["inactive_frames"]]
    inactive_detected = [frame for frame in inactive_frames if by_frame[frame]["changed_label_pixels"] >= int(evaluation["minimum_inactive_changed_label_pixels_per_frame"])]
    checks = {
        "r52_gate3_and_r51_authorities_accepted_r49_rejected": bool(r52_gate["checks"]["passed"] and r51_gate["checks"]["passed"] and not r49_gate["checks"]["passed"]),
        "case_and_repeat_denominator_exact": len(rows) == len(frames) * 2 * int(evaluation["repeat_count"]),
        "perception_repeat_exact": repeat_exact,
        "active_sensor_control_exact": naive_rows[active_frame]["sensor_sha256"] == v6_rows[active_frame]["sensor_sha256"],
        "active_perception_control_exact": by_frame[active_frame]["changed_label_pixels"] == int(evaluation["expected_active_changed_label_pixels"]),
        "both_inactive_lifecycle_regressions_detected": inactive_detected == inactive_frames,
        "v6_inactive_native_sensor_exact": all(v6_rows[frame]["full_sensor_rgb_mae"] == 0.0 and v6_rows[frame]["full_sensor_depth_mae_m"] == 0.0 and v6_rows[frame]["full_sensor_opacity_mae"] == 0.0 for frame in inactive_frames),
        "semantic_correctness_abstain": True,
        "physical_planning_and_safety_abstain": True,
        "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()),
        "gpu_within_budget": int(worker["peak_gpu_memory_mib"]) <= int(config["resources"]["maximum_peak_gpu_memory_mib"]),
        "training_not_started": True,
        "confirmation_not_read": True,
    }
    checks["wall_within_budget"] = (time.monotonic() - started) <= float(config["resources"]["maximum_wall_seconds"])
    checks["passed"] = all(checks.values())
    _write_json(run_dir / "R53_GATE.json", {"schema_version": "worldsim_v6.r53_gate.v1", "checks": checks, "decision": "accept_lifecycle_perception_regression_utility" if checks["passed"] else "reject_lifecycle_perception_regression_utility"})
    _write_json(run_dir / "RESOURCE_AUDIT.json", {"schema_version": "worldsim_v6.r53_resource_audit.v1", "peak_gpu_memory_mib": worker["peak_gpu_memory_mib"], "perception_elapsed_seconds": worker["elapsed_seconds"], "wall_seconds": time.monotonic() - started, "training_started": False, "confirmation_content_read": False})
    summary = {
        "schema_version": "worldsim_v6.r53_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected", "hypothesis_outcome": "accepted_development_lifecycle_perception_regression_utility" if checks["passed"] else "rejected",
        "source_commit": source_commit, "active_control_changed_label_pixels": by_frame[active_frame]["changed_label_pixels"],
        "inactive_changed_label_pixels": {str(frame): by_frame[frame]["changed_label_pixels"] for frame in inactive_frames},
        "inactive_changed_label_fractions": {str(frame): by_frame[frame]["changed_label_fraction"] for frame in inactive_frames},
        "semantic_correctness": "ABSTAIN", "physical_planning_safety": "ABSTAIN", "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R53_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "PERCEPTION_INPUT_INDEX.jsonl", "PERCEPTION_COMPARISON.json", "perception/PERCEPTION_OUTPUTS.jsonl", "perception/WORKER_RESULT.json", "perception.log"]
    tracked.extend(str(path.relative_to(run_dir)) for path in sorted(perception_dir.glob("*.npy")))
    _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r53_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
    _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": summary["status"], "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
    print(str(run_dir), flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r53_lifecycle_perception_regression_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
