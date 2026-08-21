"""WorldSim V6 R13 actor edit 的冻结 sensor/perception 局部性实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy.ndimage import binary_dilation

from motion_proj.worldsim_v6.r13_worldspace_route import _render_index


TASK_ID = "WS-V6-R13-WORLDSIM-01"


class R13ActorSensorError(RuntimeError):
    """R13 actor sensor/perception 正式合同失败。"""


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
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R13ActorSensorError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _rgb(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive["rgb"])
    if value.ndim == 3 and value.shape[0] == 3:
        value = np.transpose(value, (1, 2, 0))
    value = value.astype(np.float32)
    # AD-GS 浮点 RGB 可轻微 overshoot，2.0 以下仍按归一化辐射值处理。
    if float(np.nanmax(value)) > 2.0:
        value /= 255.0
    return np.clip(value, 0.0, 1.0)


def _dynamic(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        value = np.squeeze(np.asarray(archive["dynamic_opacity"], dtype=np.float32))
    if value.ndim != 2:
        raise R13ActorSensorError(f"dynamic opacity 非二维：{value.shape}")
    return value


def _run_worker(command: list[str], repo_root: Path, log_path: Path) -> None:
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(command, cwd=repo_root, env=env, capture_output=True, text=True)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise R13ActorSensorError(f"感知 worker 失败，见 {log_path}")


def run_experiment(
    repo_root: Path, config_path: Path, run_root: Path, semantic_model_root: Path
) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R13ActorSensorError("正式 R13 actor-sensor run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R13ActorSensorError("R13 actor-sensor task_id 漂移")
    sources = config["sources"]
    dynamic_run = _resolve_runs_uri(sources["r13_dynamic_run"])
    r3 = _resolve_runs_uri(sources["r3_render_run"])
    scene = config["cohort"]["scene"]
    roots = {
        frontend: r3 / "renders" / scene / frontend for frontend in config["cohort"]["frontends"]
    }
    frozen = {
        dynamic_run / "MANIFEST.json": sources["r13_dynamic_manifest_sha256"],
        dynamic_run / "R13_DYNAMIC_EDIT_GATE.json": sources["r13_dynamic_gate_sha256"],
        roots["streetgs"] / "RENDER_MAP.jsonl": sources["streetgs_render_map_sha256"],
        roots["ad_gs"] / "RENDER_MAP.jsonl": sources["ad_gs_render_map_sha256"],
        semantic_model_root / config["verifier_model"]["model_file"]: config["verifier_model"]["model_sha256"],
    }
    for path, expected in frozen.items():
        if _sha256(path) != expected:
            raise R13ActorSensorError(f"冻结输入漂移：{path}")
    dynamic_gate = json.loads(
        (dynamic_run / "R13_DYNAMIC_EDIT_GATE.json").read_text(encoding="utf-8")
    )
    if not dynamic_gate["checks"]["passed"]:
        raise R13ActorSensorError("H-R13-005 typed dynamic gate 未通过")
    indexes = {frontend: _render_index(root) for frontend, root in roots.items()}
    operation = config["cohort"]["operation"]
    frames = [int(value) for value in config["cohort"]["frame_indices"]]
    selected: dict[tuple[str, int, str], Path] = {}
    for frontend in config["cohort"]["frontends"]:
        for frame in frames:
            selected[(frontend, frame, "logged")] = indexes[frontend][
                (frame, "camera_lateral", 0.0)
            ]
            selected[(frontend, frame, "edited")] = indexes[frontend][
                (frame, operation, 0.0)
            ]
    for path in selected.values():
        frozen[path] = _sha256(path)
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R13ActorSensorError("R13 actor-sensor 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__actor-sensor-perception-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        immutable_before = {str(path): _sha256(path) for path in frozen}
        index_rows = []
        repeat_count = int(config["cohort"]["repeat_count"])
        for (frontend, frame, state), path in sorted(selected.items()):
            case_id = f"{frontend}__f{frame:03d}__{state}"
            for repeat_index in range(1, repeat_count + 1):
                index_rows.append(
                    {
                        "case_id": case_id,
                        "frontend": frontend,
                        "frame_index": frame,
                        "state": state,
                        "repeat_index": repeat_index,
                        "render_path": str(path),
                        "render_sha256": _sha256(path),
                    }
                )
        _write_jsonl(run_dir / "PERCEPTION_INPUT_INDEX.jsonl", index_rows)
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
        worker = json.loads((perception_dir / "WORKER_RESULT.json").read_text(encoding="utf-8"))
        perception_rows = _read_jsonl(perception_dir / "PERCEPTION_OUTPUTS.jsonl")
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in perception_rows:
            by_case.setdefault(row["case_id"], []).append(row)
        perception_repeat_exact = all(
            len(rows) == repeat_count
            and len({row["label_array_sha256"] for row in rows}) == 1
            for rows in by_case.values()
        )
        metrics = config["metrics"]
        metric_rows: list[dict[str, Any]] = []
        for frontend in config["cohort"]["frontends"]:
            for frame in frames:
                logged_path = selected[(frontend, frame, "logged")]
                edited_path = selected[(frontend, frame, "edited")]
                logged = _rgb(logged_path)
                edited = _rgb(edited_path)
                target = binary_dilation(
                    _dynamic(logged_path) > float(metrics["actor_opacity_threshold"]),
                    iterations=int(metrics["actor_mask_dilation_px"]),
                )
                outside = ~target
                if not np.any(target) or not np.any(outside):
                    raise R13ActorSensorError("actor target/outside 分母为空")
                rgb_difference = np.mean(np.abs(edited - logged), axis=-1)
                target_rgb_mae = float(np.mean(rgb_difference[target]))
                outside_rgb_mae = float(np.mean(rgb_difference[outside]))
                rgb_enrichment = target_rgb_mae / max(outside_rgb_mae, 1.0e-12)
                outside_changed = float(np.mean(rgb_difference[outside] > (1.0 / 255.0)))
                logged_case = f"{frontend}__f{frame:03d}__logged"
                edited_case = f"{frontend}__f{frame:03d}__edited"
                logged_labels = np.load(
                    perception_dir / sorted(by_case[logged_case], key=lambda row: row["repeat_index"])[0]["label_path"],
                    allow_pickle=False,
                )
                edited_labels = np.load(
                    perception_dir / sorted(by_case[edited_case], key=lambda row: row["repeat_index"])[0]["label_path"],
                    allow_pickle=False,
                )
                label_changed = logged_labels != edited_labels
                target_perception = float(np.mean(label_changed[target]))
                outside_perception = float(np.mean(label_changed[outside]))
                perception_enrichment = target_perception / max(outside_perception, 1.0e-12)
                checks = {
                    "target_rgb_effect": target_rgb_mae >= float(metrics["minimum_target_rgb_mae"]),
                    "outside_rgb_preserved": outside_rgb_mae <= float(metrics["maximum_outside_rgb_mae"]),
                    "rgb_locality_enrichment": rgb_enrichment
                    >= float(metrics["minimum_rgb_locality_enrichment"]),
                    "outside_rgb_changed_fraction": outside_changed
                    <= float(metrics["maximum_outside_rgb_changed_fraction"]),
                    "target_perception_effect": target_perception
                    >= float(metrics["minimum_target_perception_changed_fraction"]),
                    "outside_perception_preserved": outside_perception
                    <= float(metrics["maximum_outside_perception_changed_fraction"]),
                    "perception_locality_enrichment": perception_enrichment
                    >= float(metrics["minimum_perception_locality_enrichment"]),
                }
                checks["passed"] = all(checks.values())
                metric_rows.append(
                    {
                        "schema_version": "worldsim_v6.r13_actor_sensor_metric.v1",
                        "case_id": f"{frontend}__f{frame:03d}",
                        "frontend": frontend,
                        "frame_index": frame,
                        "target_pixel_count": int(np.count_nonzero(target)),
                        "outside_pixel_count": int(np.count_nonzero(outside)),
                        "target_rgb_mae": target_rgb_mae,
                        "outside_rgb_mae": outside_rgb_mae,
                        "rgb_locality_enrichment": rgb_enrichment,
                        "outside_rgb_changed_fraction": outside_changed,
                        "target_perception_changed_fraction": target_perception,
                        "outside_perception_changed_fraction": outside_perception,
                        "perception_locality_enrichment": perception_enrichment,
                        "checks": checks,
                    }
                )
        _write_jsonl(run_dir / "SENSOR_PERCEPTION_METRICS.jsonl", metric_rows)
        inherited_v6_false_safe = float(dynamic_gate["v6_false_safe_rate"])
        inherited_naive_false_safe = float(dynamic_gate["naive_false_safe_rate"])
        unsupported = config["unsupported_metrics"]
        wall_seconds = time.monotonic() - started
        checks = {
            "all_four_sensor_cases_pass": len(metric_rows) == 4
            and all(row["checks"]["passed"] for row in metric_rows),
            "perception_repeat_exact": perception_repeat_exact,
            "both_frontends_present": {row["frontend"] for row in metric_rows}
            == set(config["cohort"]["frontends"]),
            "source_immutable": immutable_before == {str(path): _sha256(path) for path in frozen},
            "typed_dynamic_gate_passed": dynamic_gate["checks"]["passed"],
            "inherited_v6_false_safe_rate_zero": inherited_v6_false_safe == 0.0,
            "inherited_naive_false_safe_rate_one": inherited_naive_false_safe == 1.0,
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in unsupported.values()),
            "gpu_within_budget": int(worker["peak_gpu_memory_mib"])
            <= int(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r13_actor_sensor_gate.v1",
            "checks": checks,
            "inherited_v6_false_safe_rate": inherited_v6_false_safe,
            "inherited_naive_false_safe_rate": inherited_naive_false_safe,
            "unsupported_metrics": unsupported,
            "decision": "accept_actor_sensor_perception_locality"
            if checks["passed"]
            else "reject_actor_sensor_perception_hypothesis",
        }
        _write_json(run_dir / "R13_ACTOR_SENSOR_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.r13_actor_sensor_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_actor_sensor_perception_locality"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "sensor_case_count": len(metric_rows),
            "perception_inference_count": len(perception_rows),
            "perception_repeat_exact": perception_repeat_exact,
            "peak_gpu_memory_mib": int(worker["peak_gpu_memory_mib"]),
            "wall_seconds": wall_seconds,
            "unsupported_metrics": unsupported,
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "PERCEPTION_INPUT_INDEX.jsonl",
            "SENSOR_PERCEPTION_METRICS.jsonl",
            "R13_ACTOR_SENSOR_GATE.json",
            "SUMMARY.json",
            "perception/PERCEPTION_OUTPUTS.jsonl",
            "perception/WORKER_RESULT.json",
            "perception.log",
        ]
        tracked.extend(
            str(path.relative_to(run_dir)) for path in sorted(perception_dir.glob("*.npy"))
        )
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r13_actor_sensor_manifest.v1",
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
        default=Path("configs/worldsim_v6/r13_actor_sensor_perception_v0.yaml"),
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
