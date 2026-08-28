"""Add the held-out source-scene remainder to P102 temporal interaction rows."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v67.actor_state_reliability import materialize_actor_query_rows


def _augment(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = dict(arrays)
    features = np.asarray(result["features"], dtype=np.float32)
    predicted = np.asarray(result["predicted_separation_profile_m"], dtype=np.float32)
    radius = np.asarray(result["occupancy_interaction_radius_m"], dtype=np.float32)[:, None]
    signed = predicted - radius
    result["features"] = np.concatenate((features, signed, np.abs(signed)), axis=1).astype(np.float32)
    result["actor_position_error_m"] = np.asarray(result["raw_actor_state_error_m"], dtype=np.float32).copy()
    result["raw_actor_state_error_m"] = np.asarray(result["occupancy_decision_flip"], dtype=np.float32)
    result["target_cost"] = result["raw_actor_state_error_m"].copy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()
    base_root = args.runs_root / config["base_rows"]["run"]
    base = dict(np.load(base_root / config["base_rows"]["source_artifact"], allow_pickle=False))
    data = config["supplement_data"]
    processed_root = Path(data["processed_root"])
    supplement_scenes = sorted(
        path for path in processed_root.iterdir() if path.is_dir()
        and (path / "instances" / "instances_info.json").is_file() and (path / "lidar_pose").is_dir()
        and int(path.name) % int(data["source_scene_modulus"]) == int(data["included_scene_remainder"])
    )
    supplement = materialize_actor_query_rows(supplement_scenes, data["horizons_seconds"], data)
    combined = {key: np.concatenate((value, supplement[key]), axis=0) for key, value in base.items()}
    development = dict(np.load(base_root / config["base_rows"]["development_artifact"], allow_pickle=False))
    source_augmented = _augment(combined)
    development_augmented = _augment(development)
    np.savez_compressed(args.run_dir / config["source_rows"]["artifact"], **source_augmented)
    np.savez_compressed(args.run_dir / config["evaluation_rows"]["artifact"], **development_augmented)
    summary = {
        "schema_version": "worldsim_v67.p106_all_source_scaleup_prep_summary.v1", "status": "done",
        "base_source_row_count": int(len(base["features"])),
        "supplement_scene_count": int(len(supplement_scenes)),
        "supplement_row_count": int(len(supplement["features"])),
        "combined_source_row_count": int(len(combined["features"])),
        "development_row_count": int(len(development["features"])),
        "query_feature_count": int(source_augmented["features"].shape[1]),
        "new_sensor_or_confirmation_target_read": False,
        "wall_seconds": time.monotonic() - started, "hash_checksum_fingerprint_added": False,
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
