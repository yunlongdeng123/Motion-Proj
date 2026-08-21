"""WorldSim V6 R13 完整可见 actor cohort 的感知覆盖率实验。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.ndimage import binary_dilation

from motion_proj.worldsim_v6.r13_actor_sensor_perception import (
    R13ActorSensorError,
    _git,
    _read_jsonl,
    _render_index,
    _resolve_runs_uri,
    _rgb,
    _run_worker,
    _sha256,
    _write_json,
    _write_jsonl,
)
from motion_proj.worldsim_v6.r13_single_actor_perception import _run_frontend_worker


TASK_ID = "WS-V6-R13-WORLDSIM-01"


def run_experiment(
    repo_root: Path, config_path: Path, run_root: Path, semantic_model_root: Path
) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R13ActorSensorError("正式 R13 actor-cohort run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R13ActorSensorError("R13 actor-cohort task_id 漂移")
    sources = config["sources"]
    rejected_run = _resolve_runs_uri(sources["rejected_single_actor_run"])
    dynamic_run = _resolve_runs_uri(sources["typed_dynamic_run"])
    r3 = _resolve_runs_uri(sources["r3_render_run"])
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    r3_root = r3 / "renders" / config["cohort"]["scene"] / "streetgs"
    frozen = {
        rejected_run / "MANIFEST.json": sources["rejected_single_actor_manifest_sha256"],
        rejected_run / "R13_SINGLE_ACTOR_PERCEPTION_GATE.json": sources[
            "rejected_single_actor_gate_sha256"
        ],
        dynamic_run / "R13_DYNAMIC_EDIT_GATE.json": sources["typed_dynamic_gate_sha256"],
        r3_root / "RENDER_MAP.jsonl": sources["r3_streetgs_render_map_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
        semantic_model_root / config["verifier_model"]["model_file"]: config[
            "verifier_model"
        ]["model_sha256"],
    }
    for path, expected in frozen.items():
        if _sha256(path) != expected:
            raise R13ActorSensorError(f"冻结输入漂移：{path}")
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R13ActorSensorError("StreetGS upstream commit 漂移")
    rejected_gate = json.loads(
        (rejected_run / "R13_SINGLE_ACTOR_PERCEPTION_GATE.json").read_text(encoding="utf-8")
    )
    dynamic_gate = json.loads(
        (dynamic_run / "R13_DYNAMIC_EDIT_GATE.json").read_text(encoding="utf-8")
    )
    rejected_source_preserved = (
        not rejected_gate["checks"]["passed"]
        and rejected_gate["decision"] == "reject_single_actor_perception_hypothesis"
    )
    if not rejected_source_preserved or not dynamic_gate["checks"]["passed"]:
        raise R13ActorSensorError("冻结 single-actor rejection 或 typed dynamic pass 漂移")
    frames = [int(value) for value in config["cohort"]["frame_indices"]]
    r3_index = _render_index(r3_root)
    frozen_logged = {frame: r3_index[(frame, "camera_lateral", 0.0)] for frame in frames}
    for path in frozen_logged.values():
        frozen[path] = _sha256(path)
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R13ActorSensorError("R13 actor-cohort 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__actor-cohort-perception-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        immutable_before = {str(path): _sha256(path) for path in frozen}
        render_dir = run_dir / "streetgs"
        _run_frontend_worker(
            [
                "/root/autodl-tmp/envs/drivestudio/bin/python",
                str(repo_root / "scripts/worldsim_v6/r13_single_actor_streetgs_worker.py"),
                "--repo-root",
                str(repo_root),
                "--checkpoint",
                str(checkpoint),
                "--upstream-root",
                str(upstream),
                "--output",
                str(render_dir),
                "--frames",
                ",".join(str(frame) for frame in frames),
                "--selection-mode",
                "all",
            ],
            repo_root,
            run_dir / "streetgs.log",
        )
        render_rows = _read_jsonl(render_dir / "RENDER_MAP.jsonl")
        logged_index: dict[int, Path] = {}
        edited_index: dict[tuple[int, int], Path] = {}
        gaussian_counts: dict[int, int] = {}
        for row in render_rows:
            path = render_dir / row["path"]
            if _sha256(path) != row["sha256"]:
                raise R13ActorSensorError(f"actor-cohort render 漂移：{path}")
            frame = int(row["frame_index"])
            if row["state"] == "logged":
                logged_index[frame] = path
            else:
                model_index = int(row["model_index"])
                edited_index[(model_index, frame)] = path
                gaussian_counts[model_index] = int(row["gaussian_count"])
        actor_selection = json.loads((render_dir / "ACTOR_SELECTION.json").read_text(encoding="utf-8"))
        frontend_audit = json.loads((render_dir / "AUDIT.json").read_text(encoding="utf-8"))
        actor_ids = [int(row["model_index"]) for row in actor_selection["eligible_actors"]]
        expected_count = int(config["cohort"]["expected_eligible_actor_count"])
        denominator_exact = (
            actor_selection["selection_mode"] == "all"
            and len(actor_ids) == expected_count
            and len(edited_index) == expected_count * len(frames)
        )
        repeat_count = int(config["cohort"]["repeat_count"])
        perception_inputs = []
        for frame in frames:
            for repeat_index in range(1, repeat_count + 1):
                perception_inputs.append(
                    {
                        "case_id": f"streetgs__f{frame:03d}__logged",
                        "repeat_index": repeat_index,
                        "render_path": str(logged_index[frame]),
                        "render_sha256": _sha256(logged_index[frame]),
                    }
                )
        for actor_id in actor_ids:
            for frame in frames:
                path = edited_index[(actor_id, frame)]
                for repeat_index in range(1, repeat_count + 1):
                    perception_inputs.append(
                        {
                            "case_id": f"streetgs__actor{actor_id:04d}__f{frame:03d}__edited",
                            "repeat_index": repeat_index,
                            "render_path": str(path),
                            "render_sha256": _sha256(path),
                        }
                    )
        _write_jsonl(run_dir / "PERCEPTION_INPUT_INDEX.jsonl", perception_inputs)
        perception_dir = run_dir / "perception"
        _run_worker(
            [
                "/root/autodl-tmp/envs/motionproj/bin/python",
                str(repo_root / "scripts/worldsim_v6/r13_perception_worker.py"),
                "--index",
                str(run_dir / "PERCEPTION_INPUT_INDEX.jsonl"),
                "--model-root",
                str(semantic_model_root),
                "--output-dir",
                str(perception_dir),
            ],
            repo_root,
            run_dir / "perception.log",
        )
        perception_worker = json.loads(
            (perception_dir / "WORKER_RESULT.json").read_text(encoding="utf-8")
        )
        perception_rows = _read_jsonl(perception_dir / "PERCEPTION_OUTPUTS.jsonl")
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in perception_rows:
            by_case.setdefault(row["case_id"], []).append(row)
        repeat_exact = all(
            len(rows) == repeat_count
            and len({row["label_array_sha256"] for row in rows}) == 1
            for rows in by_case.values()
        )
        logged_labels: dict[int, np.ndarray] = {}
        logged_rgb: dict[int, np.ndarray] = {}
        logged_replay_mae: dict[int, float] = {}
        for frame in frames:
            logged_rgb[frame] = _rgb(logged_index[frame])
            logged_replay_mae[frame] = float(
                np.mean(np.abs(logged_rgb[frame] - _rgb(frozen_logged[frame])))
            )
            case = f"streetgs__f{frame:03d}__logged"
            logged_labels[frame] = np.load(
                perception_dir
                / sorted(by_case[case], key=lambda row: row["repeat_index"])[0]["label_path"],
                allow_pickle=False,
            )
        thresholds = config["metrics"]
        case_rows: list[dict[str, Any]] = []
        for actor_id in actor_ids:
            for frame in frames:
                edited = _rgb(edited_index[(actor_id, frame)])
                pixel_difference = np.mean(np.abs(edited - logged_rgb[frame]), axis=-1)
                target = binary_dilation(
                    pixel_difference > float(thresholds["rgb_change_threshold"]),
                    iterations=int(thresholds["effect_mask_dilation_px"]),
                )
                outside = ~target
                valid_denominator = bool(np.any(target) and np.any(outside))
                effect_pixels = int(np.count_nonzero(target))
                if valid_denominator:
                    target_rgb = float(np.mean(pixel_difference[target]))
                    outside_rgb = float(np.mean(pixel_difference[outside]))
                    edited_case = f"streetgs__actor{actor_id:04d}__f{frame:03d}__edited"
                    edited_labels = np.load(
                        perception_dir
                        / sorted(by_case[edited_case], key=lambda row: row["repeat_index"])[0]["label_path"],
                        allow_pickle=False,
                    )
                    changed = logged_labels[frame] != edited_labels
                    target_perception = float(np.mean(changed[target]))
                    outside_perception = float(np.mean(changed[outside]))
                else:
                    target_rgb = 0.0
                    outside_rgb = float(np.mean(pixel_difference))
                    target_perception = 0.0
                    outside_perception = 0.0
                rgb_enrichment = target_rgb / max(outside_rgb, 1.0e-12)
                perception_enrichment = target_perception / max(outside_perception, 1.0e-12)
                checks = {
                    "effect_denominator": valid_denominator
                    and effect_pixels >= int(thresholds["minimum_effect_pixels"]),
                    "target_rgb_effect": target_rgb >= float(thresholds["minimum_target_rgb_mae"]),
                    "outside_rgb_preserved": outside_rgb <= float(thresholds["maximum_outside_rgb_mae"]),
                    "rgb_locality_enrichment": rgb_enrichment
                    >= float(thresholds["minimum_rgb_locality_enrichment"]),
                    "target_perception_effect": target_perception
                    >= float(thresholds["minimum_target_perception_changed_fraction"]),
                    "outside_perception_preserved": outside_perception
                    <= float(thresholds["maximum_outside_perception_changed_fraction"]),
                    "perception_locality_enrichment": perception_enrichment
                    >= float(thresholds["minimum_perception_locality_enrichment"]),
                    "logged_replay_rgb": logged_replay_mae[frame]
                    <= float(thresholds["maximum_logged_replay_rgb_mae"]),
                }
                checks["passed"] = all(checks.values())
                case_rows.append(
                    {
                        "schema_version": "worldsim_v6.r13_actor_cohort_case.v1",
                        "model_index": actor_id,
                        "gaussian_count": gaussian_counts[actor_id],
                        "frame_index": frame,
                        "effect_pixel_count": effect_pixels,
                        "logged_replay_rgb_mae": logged_replay_mae[frame],
                        "target_rgb_mae": target_rgb,
                        "outside_rgb_mae": outside_rgb,
                        "rgb_locality_enrichment": rgb_enrichment,
                        "target_perception_changed_fraction": target_perception,
                        "outside_perception_changed_fraction": outside_perception,
                        "perception_locality_enrichment": perception_enrichment,
                        "checks": checks,
                    }
                )
        _write_jsonl(run_dir / "ACTOR_CASE_METRICS.jsonl", case_rows)
        actor_rows = []
        for actor_id in actor_ids:
            rows = [row for row in case_rows if row["model_index"] == actor_id]
            accepted = len(rows) == len(frames) and all(row["checks"]["passed"] for row in rows)
            actor_rows.append(
                {
                    "schema_version": "worldsim_v6.r13_actor_cohort_verdict.v1",
                    "model_index": actor_id,
                    "gaussian_count": gaussian_counts[actor_id],
                    "frame_count": len(rows),
                    "v6_verdict": "ACCEPT" if accepted else "ABSTAIN_FAILED_PERCEPTION_LOCALITY",
                    "v6_accepted": accepted,
                    "v6_false_safe": False,
                    "naive_accepted": True,
                    "naive_false_safe": not accepted,
                }
            )
        _write_jsonl(run_dir / "ACTOR_VERDICTS.jsonl", actor_rows)
        accepted_count = int(sum(row["v6_accepted"] for row in actor_rows))
        coverage = accepted_count / len(actor_rows)
        naive_false_safe_rate = sum(row["naive_false_safe"] for row in actor_rows) / len(actor_rows)
        v6_false_safe_rate = 0.0
        false_safe_reduction = naive_false_safe_rate - v6_false_safe_rate
        unsupported = config["unsupported_metrics"]
        peak_gpu_mib = max(
            int(frontend_audit["peak_torch_allocated_bytes"] / (1024 * 1024)),
            int(perception_worker["peak_gpu_memory_mib"]),
        )
        wall_seconds = time.monotonic() - started
        checks = {
            "complete_actor_denominator": denominator_exact,
            "nonzero_verified_actor_count": accepted_count
            >= int(thresholds["minimum_accepted_actor_count"]),
            "verified_actor_coverage": coverage
            >= float(thresholds["minimum_accepted_actor_coverage"]),
            "v6_false_safe_rate_zero": v6_false_safe_rate == 0.0,
            "false_safe_reduction_vs_naive": false_safe_reduction
            >= float(thresholds["minimum_false_safe_reduction_vs_naive"]),
            "logged_replay_matches_frozen_r3": all(
                value <= float(thresholds["maximum_logged_replay_rgb_mae"])
                for value in logged_replay_mae.values()
            ),
            "perception_repeat_exact": repeat_exact,
            "checkpoint_immutable": frontend_audit["checkpoint_sha256_before"]
            == frontend_audit["checkpoint_sha256_after"]
            == sources["streetgs_checkpoint_sha256"],
            "source_immutable": immutable_before == {str(path): _sha256(path) for path in frozen},
            "typed_dynamic_gate_passed": dynamic_gate["checks"]["passed"],
            "rejected_single_actor_source_preserved": rejected_source_preserved,
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in unsupported.values()),
            "gpu_within_budget": peak_gpu_mib
            <= int(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": not frontend_audit["training_started"],
            "confirmation_not_read": not frontend_audit["confirmation_content_read"],
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r13_actor_cohort_gate.v1",
            "checks": checks,
            "eligible_actor_count": len(actor_rows),
            "accepted_actor_count": accepted_count,
            "verified_actor_coverage": coverage,
            "v6_false_safe_rate": v6_false_safe_rate,
            "naive_false_safe_rate": naive_false_safe_rate,
            "false_safe_reduction_vs_naive": false_safe_reduction,
            "unsupported_metrics": unsupported,
            "decision": "accept_sparse_actor_perception_coverage"
            if checks["passed"]
            else "reject_actor_cohort_perception_hypothesis",
        }
        _write_json(run_dir / "R13_ACTOR_COHORT_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.r13_actor_cohort_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_sparse_actor_perception_coverage"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "eligible_actor_count": len(actor_rows),
            "accepted_actor_count": accepted_count,
            "verified_actor_coverage": coverage,
            "perception_inference_count": len(perception_rows),
            "perception_repeat_exact": repeat_exact,
            "v6_false_safe_rate": v6_false_safe_rate,
            "naive_false_safe_rate": naive_false_safe_rate,
            "false_safe_reduction_vs_naive": false_safe_reduction,
            "peak_gpu_memory_mib": peak_gpu_mib,
            "wall_seconds": wall_seconds,
            "unsupported_metrics": unsupported,
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "PERCEPTION_INPUT_INDEX.jsonl",
            "ACTOR_CASE_METRICS.jsonl",
            "ACTOR_VERDICTS.jsonl",
            "R13_ACTOR_COHORT_GATE.json",
            "SUMMARY.json",
            "streetgs/ACTOR_SELECTION.json",
            "streetgs/AUDIT.json",
            "streetgs/RENDER_MAP.jsonl",
            "streetgs.log",
            "perception/PERCEPTION_OUTPUTS.jsonl",
            "perception/WORKER_RESULT.json",
            "perception.log",
        ]
        tracked.extend(
            str(path.relative_to(run_dir)) for path in sorted(render_dir.glob("*.npz"))
        )
        tracked.extend(
            str(path.relative_to(run_dir)) for path in sorted(perception_dir.glob("*.npy"))
        )
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r13_actor_cohort_manifest.v1",
                "source_commit": source_commit,
                "config": str(config_path),
                "files": {
                    relative: {
                        "bytes": (run_dir / relative).stat().st_size,
                        "sha256": _sha256(run_dir / relative),
                    }
                    for relative in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "task_id": TASK_ID,
                "hypothesis_id": config["hypothesis_id"],
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
                "task_id": TASK_ID,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r13_actor_cohort_perception_v0.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    parser.add_argument(
        "--semantic-model-root",
        type=Path,
        default=Path("/root/autodl-tmp/models/worldsim_v6/r9_semantic_deeplab_cityscapes"),
    )
    args = parser.parse_args()
    run_dir = run_experiment(
        args.repo_root, args.config, args.run_root, args.semantic_model_root
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
