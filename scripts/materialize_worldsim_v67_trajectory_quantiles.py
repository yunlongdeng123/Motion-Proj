"""Materialize one horizon-specific trajectory quantile cache."""

from __future__ import annotations

import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
import yaml

from motion_proj.worldsim_v67.trajectory_quantile import materialize_quantiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.runs_root / "worldsim_v67" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()
    result = materialize_quantiles(config["data"], args.runs_root, Path(config["cache_path"]))
    summary = {"schema_version": config["schema_version"], "task_id": config["task_id"], "status": "done",
               "role": config["role"], "materialization": result, "wall_seconds": time.monotonic() - started}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "materialization": result}, indent=2))


if __name__ == "__main__":
    main()
