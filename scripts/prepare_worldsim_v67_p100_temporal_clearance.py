"""Append frozen analytic temporal-clearance query features to P95 rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def _augment(path: Path) -> dict[str, np.ndarray]:
    arrays = dict(np.load(path, allow_pickle=False))
    features = np.asarray(arrays["features"], dtype=np.float32)
    horizon = features[:, 14].clip(min=1e-3)
    relative_position = features[:, 6:8]
    relative_velocity = features[:, 8:10].copy()
    ego_speed = features[:, 10]
    progress_ratio = features[:, 19]
    lateral_offset = features[:, 20]
    relative_velocity[:, 0] -= ego_speed * (progress_ratio - 1.0)
    relative_velocity[:, 1] -= lateral_offset / horizon
    denominator = np.square(relative_velocity).sum(axis=1).clip(min=1e-6)
    time_to_closest = (-np.sum(relative_position * relative_velocity, axis=1) / denominator).clip(0.0, horizon)
    normalized_time = time_to_closest / horizon
    signed_clearance = np.asarray(arrays["predicted_minimum_separation_m"], dtype=np.float32) - np.asarray(
        arrays["occupancy_interaction_radius_m"], dtype=np.float32,
    )
    boundary_distance = np.abs(signed_clearance)
    arrays["features"] = np.concatenate((
        features, normalized_time[:, None], signed_clearance[:, None], boundary_distance[:, None],
    ), axis=1).astype(np.float32)
    arrays["analytic_time_to_closest_fraction"] = normalized_time.astype(np.float32)
    arrays["predicted_signed_occupancy_clearance_m"] = signed_clearance.astype(np.float32)
    arrays["predicted_occupancy_boundary_distance_m"] = boundary_distance.astype(np.float32)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source_root = args.runs_root / config["row_run"]["source_run"]
    source = _augment(source_root / config["row_run"]["source_input"])
    development = _augment(source_root / config["row_run"]["development_input"])
    np.savez_compressed(args.run_dir / config["source_rows"]["artifact"], **source)
    np.savez_compressed(args.run_dir / config["evaluation_rows"]["artifact"], **development)
    summary = {"schema_version": "worldsim_v67.p100_temporal_clearance_prep_summary.v1", "status": "done",
        "source_row_count": len(source["features"]), "development_row_count": len(development["features"]),
        "query_feature_count": int(source["features"].shape[1]), "new_sensor_or_target_read": False,
        "hash_checksum_fingerprint_added": False}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
