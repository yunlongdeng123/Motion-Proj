"""用现有DriveStudio资产补齐V7.1纯训练Actor corpus。"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v71.actor_corpus import materialize_actor_cache
from motion_proj.worldsim_v71.dataset_drivestudio import (
    compile_processed_scene,
    discover_processed_train_scenes,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _complete_actor_paths(cache_root: Path) -> list[Path]:
    return sorted(
        path
        for path in (cache_root / "train").glob("*/*.npz")
        if not path.name.endswith(".tmp.npz")
    )


def _existing_actor_ids(paths: list[Path]) -> set[str]:
    actor_ids: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            actor_ids.add(str(payload["track_id"].item()))
    return actor_ids


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = json.loads((repo_root / config["source_split"]).read_text(encoding="utf-8"))
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    _deep_update(compiler, config["compiler_overrides"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("processed corpus recovery requires CUDA")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cache_root = Path(config["cache_root"])
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "processed_recovery"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    minimum = int(config["source"]["minimum_tracklet_count"])
    original_paths = _complete_actor_paths(cache_root)
    original_count = len(original_paths)
    actor_ids = _existing_actor_ids(original_paths)
    if original_count >= minimum:
        raise RuntimeError("Actor corpus已达到冻结门槛，不应启动recovery")
    stale_marker = cache_root / "TRAIN_COMPLETE"
    if stale_marker.is_file():
        stale_marker.replace(cache_root / "TRAIN_INCOMPLETE_BELOW_TARGET")

    protected = {
        str(name)
        for names in split["roles"].values()
        for name in names
    }
    protected.update(str(name) for name in split.get("train_reserve", []))
    candidates = discover_processed_train_scenes(
        [Path(value) for value in config["source"]["processed_roots"]],
        Path(config["source"]["scene_metadata_path"]),
        protected,
    )
    oracle_payload = torch.load(
        Path(config["s1_oracle_run"]) / "ORACLE_DISPLACEMENTS.pt",
        map_location="cpu",
        weights_only=False,
    )
    oracle_by_track = {str(row["track_id"]): row for row in oracle_payload}
    started = time.monotonic()
    added_rows: list[dict[str, Any]] = []
    completed_scenes: list[str] = []
    try:
        for candidate in candidates:
            if original_count + len(added_rows) >= minimum:
                break
            scene_name = str(candidate["scene_name"])
            bundles = compile_processed_scene(
                scene_name,
                Path(candidate["scene_root"]),
                config["actors"],
                compiler,
                device,
                keyframe_stride=int(config["source"]["keyframe_stride"]),
                lidar_record_width=int(config["source"]["lidar_record_width"]),
            )
            for bundle in bundles:
                track_id = str(bundle["row"]["track_id"])
                if track_id in actor_ids:
                    continue
                result = materialize_actor_cache(
                    bundle,
                    cache_root / "train" / scene_name / f"{track_id}.npz",
                    config["corpus"],
                    oracle_by_track=oracle_by_track,
                    device=device,
                )
                result.update(
                    {
                        "scene_name": scene_name,
                        "source": "drivestudio_processed_10hz_keyframe_stride5",
                    }
                )
                actor_ids.add(track_id)
                added_rows.append(result)
            completed_scenes.append(scene_name)
            current_count = original_count + len(added_rows)
            _write_json(
                cache_root / "manifest.json",
                {
                    "schema_version": "worldsim_v71.corpus_manifest.v2",
                    "status": "running_recovery",
                    "verdict": "train_corpus_recovery_running",
                    "actor_count": current_count,
                    "minimum_tracklet_count": minimum,
                    "original_raw_actor_count": original_count,
                    "processed_added_actor_count": len(added_rows),
                    "processed_completed_scenes": completed_scenes,
                    "selection_read": False,
                    "source_final_read": False,
                    "external_read": False,
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "s2_processed_recovery",
                        "scene": scene_name,
                        "scene_compiled_actors": len(bundles),
                        "actors": current_count,
                        "target": minimum,
                    }
                ),
                flush=True,
            )

        actor_count = original_count + len(added_rows)
        passed = actor_count >= minimum
        summary = {
            "schema_version": "worldsim_v71.s2_processed_recovery.v1",
            "task_id": config["task_id"],
            "status": "done",
            "verdict": (
                "train_corpus_target_met_with_processed_recovery"
                if passed
                else "processed_recovery_exhausted_below_target"
            ),
            "actor_count": actor_count,
            "minimum_tracklet_count": minimum,
            "original_raw_actor_count": original_count,
            "processed_added_actor_count": len(added_rows),
            "processed_scene_count": len(completed_scenes),
            "processed_candidate_scene_count": len(candidates),
            "selection_read": False,
            "source_final_read": False,
            "external_read": False,
            "failure_ledger_refs": config["failure_ledger_refs"],
            "resources": {
                "device": str(device),
                "wall_seconds": time.monotonic() - started,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "peak_gpu_gib": (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                    if device.type == "cuda"
                    else 0.0
                ),
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "ADDED_ACTOR_INDEX.json", added_rows)
        _write_json(
            cache_root / "manifest.json",
            {
                **summary,
                "schema_version": "worldsim_v71.corpus_manifest.v2",
                "processed_completed_scenes": completed_scenes,
            },
        )
        if passed:
            (cache_root / "TRAIN_COMPLETE").write_text("done\n", encoding="utf-8")
        else:
            (cache_root / "TRAIN_INCOMPLETE_BELOW_TARGET").write_text(
                "processed recovery exhausted\n", encoding="utf-8"
            )
        _write_json(
            run_dir / "status.json",
            {
                "status": "done" if passed else "failed",
                "phase": "processed_recovery",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if not passed:
            raise RuntimeError("processed recovery仍未达到冻结Actor门槛")
        return {
            "run_dir": str(run_dir),
            "verdict": summary["verdict"],
            "actor_count": actor_count,
        }
    except Exception as error:
        status_path = run_dir / "status.json"
        current = json.loads(status_path.read_text(encoding="utf-8"))
        if current.get("status") != "failed":
            _write_json(
                status_path,
                {
                    "status": "failed",
                    "phase": "processed_recovery",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.repo_root.resolve(), args.run_id),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
