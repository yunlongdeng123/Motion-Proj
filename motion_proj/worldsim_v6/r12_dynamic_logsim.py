"""WorldSim V6 R12 SceneIR 动态 LogSim 正式实验。"""

from __future__ import annotations

import argparse
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

from motion_proj.worldsim_v6.sceneir import load_sceneir


TASK_ID = "WS-V6-R12-LOGSIM-01"


class R12DynamicExperimentError(RuntimeError):
    """R12 dynamic LogSim 正式合同失败。"""


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


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix):]).parts:
        raise R12DynamicExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix):]).resolve()


def _rotation_matrix(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _replay_once(package: Path, repeat_index: int) -> dict[str, Any]:
    document, arrays = load_sceneir(package)
    transforms = {
        (row["name"], int(row["timestamp_us"])): row
        for row in document["transforms"]
    }
    chunks = {row["id"]: row for row in document["chunks"]}
    actor_states: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    bounds_by_time: dict[int, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for actor in sorted(document["actors"], key=lambda row: row["id"]):
        if len(actor["chunk_ids"]) != 1:
            raise R12DynamicExperimentError("fixture actor chunk 分母漂移")
        chunk = chunks[actor["chunk_ids"][0]]
        means_ref = chunk["arrays"]["means_m"]
        local_means = np.asarray(arrays[means_ref["sha256"]], dtype=np.float64)
        semantic_rows.append({"actor_id": actor["id"], "class": actor["class"]})
        visibility = {
            int(row["timestamp_us"]): bool(row["visible"])
            for row in actor["visibility"]
        }
        for key in actor["trajectory"]:
            timestamp = int(key["timestamp_us"])
            transform = transforms[(key["transform_name"], timestamp)]
            rotation = _rotation_matrix(transform["rotation_wxyz"])
            translation = np.asarray(transform["translation_m"], dtype=np.float64)
            world_means = local_means @ rotation.T + translation
            lower = world_means.min(axis=0)
            upper = world_means.max(axis=0)
            centroid = world_means.mean(axis=0)
            state = {
                "actor_id": actor["id"],
                "timestamp_us": timestamp,
                "visible": visibility[timestamp],
                "class": actor["class"],
                "primitive_count": int(local_means.shape[0]),
                "centroid_world_m": centroid.tolist(),
                "aabb_min_world_m": lower.tolist(),
                "aabb_max_world_m": upper.tolist(),
            }
            actor_states.append(state)
            trajectory_rows.append(
                {
                    "actor_id": actor["id"],
                    "timestamp_us": timestamp,
                    "transform_name": key["transform_name"],
                    "rotation_wxyz": transform["rotation_wxyz"],
                    "translation_m": transform["translation_m"],
                    "visible": visibility[timestamp],
                }
            )
            bounds_by_time.setdefault(timestamp, []).append((actor["id"], lower, upper))
    collision_rows: list[dict[str, Any]] = []
    for timestamp, states in sorted(bounds_by_time.items()):
        for left_index in range(len(states)):
            for right_index in range(left_index + 1, len(states)):
                left_id, left_min, left_max = states[left_index]
                right_id, right_min, right_max = states[right_index]
                overlaps = bool(np.all(np.minimum(left_max, right_max) >= np.maximum(left_min, right_min)))
                collision_rows.append(
                    {
                        "timestamp_us": timestamp,
                        "actor_pair": [left_id, right_id],
                        "aabb_overlap": overlaps,
                    }
                )
    sensor_rows = [
        {
            "sensor_id": sensor["id"],
            "sensor_type": sensor["sensor_type"],
            "frame_id": sensor["frame_id"],
            "camera_model": sensor.get("camera_model"),
            "resolution_px": sensor.get("resolution_px"),
            "calibration": sensor.get("calibration"),
        }
        for sensor in sorted(document["sensors"], key=lambda row: row["id"])
    ]
    payload = {
        "schema_version": "worldsim_v6.r12_dynamic_replay.v1",
        "repeat_index": repeat_index,
        "sceneir_content_sha256": document["content_sha256"],
        "actor_states": actor_states,
        "trajectories": trajectory_rows,
        "sensor_calibrations": sensor_rows,
        "semantic_labels": semantic_rows,
        "collision_labels": collision_rows,
        "event_status": "ABSTAIN_NOT_REPRESENTED"
        if not document.get("events")
        else "REPRESENTED",
    }
    comparable = dict(payload)
    comparable.pop("repeat_index")
    payload["replay_content_sha256"] = hashlib.sha256(_canonical(comparable)).hexdigest()
    return payload


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R12DynamicExperimentError("正式 R12 dynamic run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R12DynamicExperimentError("R12 dynamic task_id 漂移")
    r2_run = _resolve_runs_uri(config["sources"]["r2_run"])
    package = r2_run / config["sources"]["sceneir_package"]
    frozen = {
        r2_run / "MANIFEST.json": config["sources"]["r2_manifest_sha256"],
        package / "MANIFEST.json": config["sources"]["sceneir_package_manifest_sha256"],
        package / "sceneir.json": config["sources"]["sceneir_document_sha256"],
    }
    for path, expected in frozen.items():
        if _sha256(path) != expected:
            raise R12DynamicExperimentError(f"冻结输入漂移：{path}")
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R12DynamicExperimentError("R12 dynamic 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__dynamic-logsim-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        immutable_before = {str(path): _sha256(path) for path in frozen}
        replays = [
            _replay_once(package, repeat_index)
            for repeat_index in range(1, int(config["cohort"]["replay_count"]) + 1)
        ]
        for replay in replays:
            _write_json(
                run_dir / f"DYNAMIC_REPLAY_REPEAT{replay['repeat_index']}.json",
                replay,
            )
        document, _ = load_sceneir(package)
        actor_states = replays[0]["actor_states"]
        timestamps = {row["timestamp_us"] for row in actor_states}
        actor_ids = {row["actor_id"] for row in actor_states}
        primitive_counts = {row["primitive_count"] for row in actor_states}
        cohort = config["cohort"]
        denominators_exact = (
            len(actor_ids) == int(cohort["expected_actor_count"])
            and len(timestamps) == int(cohort["expected_timestamp_count"])
            and primitive_counts == {int(cohort["expected_actor_primitive_count"])}
            and len(replays[0]["sensor_calibrations"]) == int(cohort["expected_sensor_count"])
            and len(document.get("events", [])) == int(cohort["expected_event_count"])
        )
        hashes = [row["replay_content_sha256"] for row in replays]
        repeat_exact = len(set(hashes)) == 1
        factor_hashes: dict[str, list[str]] = {}
        for factor in [
            "actor_states",
            "trajectories",
            "sensor_calibrations",
            "semantic_labels",
            "collision_labels",
        ]:
            factor_hashes[factor] = [
                hashlib.sha256(_canonical(replay[factor])).hexdigest() for replay in replays
            ]
        factor_exact = {factor: len(set(values)) == 1 for factor, values in factor_hashes.items()}
        wall_seconds = time.monotonic() - started
        checks = {
            "expected_denominators": denominators_exact,
            "fresh_package_reload_per_repeat": True,
            "trajectory_exact": factor_exact["trajectories"],
            "actor_world_state_exact": factor_exact["actor_states"],
            "sensor_calibration_exact": factor_exact["sensor_calibrations"],
            "semantic_label_exact": factor_exact["semantic_labels"],
            "collision_label_exact": factor_exact["collision_labels"],
            "repeated_run_exact": repeat_exact,
            "source_immutable": immutable_before == {str(path): _sha256(path) for path in frozen},
            "event_abstain": all(row["event_status"] == "ABSTAIN_NOT_REPRESENTED" for row in replays),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_jsonl(
            run_dir / "FACTOR_HASHES.jsonl",
            [
                {"factor": factor, "repeat_hashes": values, "exact": factor_exact[factor]}
                for factor, values in factor_hashes.items()
            ],
        )
        _write_json(
            run_dir / "R12_DYNAMIC_GATE.json",
            {
                "schema_version": "worldsim_v6.r12_dynamic_gate.v1",
                "checks": checks,
                "event_status": "ABSTAIN_NOT_REPRESENTED",
                "decision": "proceed_to_real_scene_dynamic_logsim"
                if checks["passed"]
                else "reject_dynamic_logsim_fixture_hypothesis",
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r12_dynamic_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_conformance_fixture"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "actor_count": len(actor_ids),
            "timestamp_count": len(timestamps),
            "sensor_count": len(replays[0]["sensor_calibrations"]),
            "collision_label_count": len(replays[0]["collision_labels"]),
            "event_status": "ABSTAIN_NOT_REPRESENTED",
            "wall_seconds": wall_seconds,
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "DYNAMIC_REPLAY_REPEAT1.json",
            "DYNAMIC_REPLAY_REPEAT2.json",
            "FACTOR_HASHES.jsonl",
            "R12_DYNAMIC_GATE.json",
            "SUMMARY.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r12_dynamic_manifest.v1",
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
        default=Path("configs/worldsim_v6/r12_dynamic_logsim_v0.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_dir = run_experiment(args.repo_root, args.config, args.run_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
