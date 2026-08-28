"""Materialize source Actor rows with anchor identities while another GPU trial runs."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import materialize_actor_query_rows


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
    data = config["source_data"]
    root = Path(data["processed_root"])
    scenes = sorted(path for path in root.iterdir() if path.is_dir()
        and (path / "instances" / "instances_info.json").is_file() and (path / "lidar_pose").is_dir())
    divisor = int(data["source_scene_modulus"])
    remainder = int(data["excluded_scene_remainder"])
    selected = [path for path in scenes if int(path.name) % divisor != remainder]
    arrays = materialize_actor_query_rows(selected, data["horizons_seconds"], data)
    np.savez_compressed(run_dir / "SOURCE_TRAJECTORY_ACTOR_ROWS.npz", **arrays)
    summary = {"schema_version": "worldsim_v67.p86_trajectory_source_summary.v1", "status": "done",
        "scene_count": int(len(selected)), "row_count": int(len(arrays["features"])),
        "horizons_seconds": data["horizons_seconds"], "wall_seconds": time.monotonic() - started,
        "hash_checksum_fingerprint_added": False}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
