"""Derive false-safe targets from frozen P95 row artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def _derive(path: Path, target_field: str) -> dict[str, np.ndarray]:
    arrays = dict(np.load(path, allow_pickle=False))
    target = np.asarray(arrays[target_field], dtype=np.float32)
    arrays["raw_actor_state_error_m"] = target
    arrays["target_cost"] = target.copy()
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
    target_field = str(config.get("target_field", "occupancy_false_safe"))
    source = _derive(source_root / config["row_run"]["source_input"], target_field)
    development = _derive(source_root / config["row_run"]["development_input"], target_field)
    np.savez_compressed(args.run_dir / config["source_rows"]["artifact"], **source)
    np.savez_compressed(args.run_dir / config["evaluation_rows"]["artifact"], **development)
    summary = {
        "schema_version": "worldsim_v67.p97_false_safe_prep_summary.v1", "status": "done",
        "source_row_count": len(source["features"]),
        "target_field": target_field,
        "source_target_count": int(np.count_nonzero(source["raw_actor_state_error_m"])),
        "development_row_count": len(development["features"]),
        "development_target_count": int(np.count_nonzero(development["raw_actor_state_error_m"])),
        "new_sensor_or_target_read": False, "hash_checksum_fingerprint_added": False,
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
