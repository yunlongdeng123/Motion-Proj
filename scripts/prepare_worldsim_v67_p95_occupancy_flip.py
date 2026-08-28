"""Materialize source and consumed-development occupancy-decision flip rows."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import materialize_actor_query_rows


def _as_flip_target(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = dict(arrays)
    result["actor_position_error_m"] = result["raw_actor_state_error_m"].copy()
    result["raw_actor_state_error_m"] = result["occupancy_decision_flip"].astype(np.float32)
    result["target_cost"] = result["raw_actor_state_error_m"].copy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()

    source_config = config["source_data"]
    source_root = Path(source_config["processed_root"])
    source_scenes = sorted(
        path for path in source_root.iterdir() if path.is_dir()
        and (path / "instances" / "instances_info.json").is_file() and (path / "lidar_pose").is_dir()
        and int(path.name) % int(source_config["source_scene_modulus"])
        != int(source_config["excluded_scene_remainder"])
    )
    source = _as_flip_target(materialize_actor_query_rows(
        source_scenes, source_config["horizons_seconds"], source_config,
    ))
    np.savez_compressed(run_dir / config["row_run"]["source_artifact"], **source)

    development_config = config["development_data"]
    development_root = Path(development_config["processed_root"])
    development_scenes = [development_root / f"{int(index):03d}" for index in development_config["scene_indices"]]
    development = _as_flip_target(materialize_actor_query_rows(
        development_scenes, development_config["horizons_seconds"], development_config,
    ))
    np.savez_compressed(run_dir / config["row_run"]["development_artifact"], **development)
    summary = {
        "schema_version": "worldsim_v67.p95_occupancy_flip_prep_summary.v1", "status": "done",
        "source_scene_count": len(source_scenes), "source_row_count": len(source["features"]),
        "source_flip_count": int(np.count_nonzero(source["occupancy_decision_flip"])),
        "source_false_safe_count": int(np.count_nonzero(source["occupancy_false_safe"])),
        "development_scene_count": len(development_scenes), "development_row_count": len(development["features"]),
        "development_flip_count": int(np.count_nonzero(development["occupancy_decision_flip"])),
        "development_false_safe_count": int(np.count_nonzero(development["occupancy_false_safe"])),
        "wall_seconds": time.monotonic() - started, "hash_checksum_fingerprint_added": False,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
