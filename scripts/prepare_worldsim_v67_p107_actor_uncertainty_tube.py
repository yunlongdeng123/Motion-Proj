"""Materialize Actor-error tubes, saving source before consumed development cohorts."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import materialize_actor_query_rows


def _save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
    )
    started = time.monotonic()

    source_config = config["source_data"]
    source_root = Path(source_config["processed_root"])
    source_scenes = sorted(
        path for path in source_root.iterdir()
        if path.is_dir()
        and (path / "instances" / "instances_info.json").is_file()
        and (path / "lidar_pose").is_dir()
        and int(path.name) % int(source_config["source_scene_modulus"])
        != int(source_config["excluded_scene_remainder"])
    )
    source = materialize_actor_query_rows(
        source_scenes, source_config["horizons_seconds"], source_config,
    )
    _save_npz_atomic(args.run_dir / config["source_rows"]["artifact"], source)
    print(
        f"P107 source ready rows={len(source['features'])} "
        f"actor_tokens={len(source['features']) * source['actor_position_error_profile_m'].shape[1]}",
        flush=True,
    )

    cohort_summaries = {}
    for cohort in config["development_cohorts"]:
        cohort_root = Path(cohort["processed_root"])
        scenes = [cohort_root / f"{int(index):03d}" for index in cohort["scene_indices"]]
        arrays = materialize_actor_query_rows(scenes, cohort["horizons_seconds"], cohort)
        _save_npz_atomic(args.run_dir / cohort["artifact"], arrays)
        cohort_summaries[cohort["name"]] = {
            "scene_count": len(scenes),
            "row_count": len(arrays["features"]),
            "trajectory_flip_rows": int(np.count_nonzero(arrays["occupancy_decision_flip"])),
        }
        print(f"P107 cohort ready name={cohort['name']} rows={len(arrays['features'])}", flush=True)

    summary = {
        "schema_version": "worldsim_v67.p107_actor_uncertainty_tube_prep_summary.v1",
        "status": "done",
        "source_scene_count": len(source_scenes),
        "source_row_count": len(source["features"]),
        "source_actor_error_token_count": int(source["actor_position_error_profile_m"].size),
        "development_cohorts": cohort_summaries,
        "wall_seconds": time.monotonic() - started,
        "new_sensor_or_unread_confirmation_target_read": False,
        "hash_checksum_fingerprint_added": False,
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    (args.run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
