"""流式物化 V7.1 train Actor corpus。"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v71.actor_corpus import materialize_actor_cache
from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = json.loads((repo_root / config["source_split"]).read_text(encoding="utf-8"))
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    _deep_update(compiler, config["compiler_overrides"])
    device = torch.device(config["device"])
    cache_root = Path(config["cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "train_corpus"})
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    oracle_payload = torch.load(Path(config["s1_oracle_run"]) / "ORACLE_DISPLACEMENTS.pt", map_location="cpu", weights_only=False)
    oracle_by_track = {str(row["track_id"]): row for row in oracle_payload}
    index_split = dict(split)
    index_split["roles"] = {"train": list(split["roles"]["train"]) + list(split.get("train_reserve", []))}
    index = build_v71_index(Path(config["source"]["dataset_root"]), index_split)
    primary_scenes = list(split["roles"]["train"])
    reserve_scenes = list(split.get("train_reserve", []))
    minimum = int(config["source"]["minimum_tracklet_count"])
    rows: list[dict[str, Any]] = []
    completed_scenes: list[str] = []
    started = time.monotonic()
    try:
        for phase, scenes in (("primary", primary_scenes), ("reserve", reserve_scenes)):
            if phase == "reserve" and len(rows) >= minimum:
                break
            for scene_name in scenes:
                bundles = compile_source_scene(
                    scene_name, index, config["actors"], compiler, device
                )
                for bundle in bundles:
                    track_id = str(bundle["row"]["track_id"])
                    result = materialize_actor_cache(
                        bundle,
                        cache_root / "train" / scene_name / f"{track_id}.npz",
                        config["corpus"],
                        oracle_by_track=oracle_by_track,
                        device=device,
                    )
                    result["scene_name"] = scene_name
                    rows.append(result)
                completed_scenes.append(scene_name)
                manifest = {
                    "schema_version": "worldsim_v71.corpus_manifest.v1",
                    "status": "running",
                    "phase": phase,
                    "actor_count": len(rows),
                    "hazard_actor_count": sum(int(row["hazardous"]) for row in rows),
                    "oracle_target_count": sum(int(row["oracle_target"]) for row in rows),
                    "completed_scene_count": len(completed_scenes),
                    "completed_scenes": completed_scenes,
                    "minimum_tracklet_count": minimum,
                }
                _write_json(cache_root / "manifest.json", manifest)
                print(json.dumps({"stage": "s2_corpus", "scene": scene_name, "actors": len(rows), "phase": phase}), flush=True)
        total_bytes = sum(int(row["bytes"]) for row in rows)
        summary = {
            "schema_version": "worldsim_v71.s2_actor_corpus.v1",
            "task_id": config["task_id"],
            "status": "done",
            "verdict": "train_corpus_target_met" if len(rows) >= minimum else "train_corpus_exhausted_below_target",
            "actor_count": len(rows),
            "hazard_actor_count": sum(int(row["hazardous"]) for row in rows),
            "oracle_target_count": sum(int(row["oracle_target"]) for row in rows),
            "scene_count": len(completed_scenes),
            "reserve_scene_count": sum(int(scene in reserve_scenes) for scene in completed_scenes),
            "cache_bytes": total_bytes,
            "selection_read": False,
            "source_final_read": False,
            "external_read": False,
            "failure_ledger_refs": config["failure_ledger_refs"],
            "failure_ledger_delta": "none" if len(rows) >= minimum else "V71-F02-required-at-closeout",
            "resources": {
                "device": str(device),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "ACTOR_INDEX.json", rows)
        _write_json(cache_root / "manifest.json", {**summary, "completed_scenes": completed_scenes})
        (cache_root / "TRAIN_COMPLETE").write_text("done\n", encoding="utf-8")
        _write_json(run_dir / "status.json", {"status": "done", "phase": "train_corpus", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return {"run_dir": str(run_dir), "verdict": summary["verdict"], "actor_count": len(rows)}
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "train_corpus", "error": f"{type(error).__name__}: {error}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.repo_root.resolve(), args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
