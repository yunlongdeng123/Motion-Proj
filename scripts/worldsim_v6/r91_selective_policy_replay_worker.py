#!/usr/bin/env python3
"""Fresh-process evaluator for a baked WorldSim selective runtime policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = args.package / name
        if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"policy package manifest mismatch: {name}")
    policy = json.loads((args.package / "POLICY.json").read_text(encoding="utf-8"))
    schema = json.loads((args.package / "FEATURE_SCHEMA.json").read_text(encoding="utf-8"))
    feature_map = json.loads(args.features.read_text(encoding="utf-8"))
    values = feature_map["edited_vs_logged_changed_rgb_pixels_by_frame"]
    threshold = int(policy["threshold_pixels"])
    decisions = [
        {
            "frame_index": frame,
            "feature_value_pixels": int(values[str(frame)]),
            "trigger_expensive_perception": int(values[str(frame)]) >= threshold,
        }
        for frame in range(196)
    ]
    decision_path = args.output / "DECISIONS.jsonl"
    decision_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8"
    )
    write_json(
        args.output / "WORKER_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r91_worker_audit.v1",
            "policy_manifest_sha256": sha256(args.package / "PACKAGE_MANIFEST.json"),
            "feature_map_sha256": sha256(args.features),
            "policy_id": policy["policy_id"],
            "feature_name": schema["name"],
            "threshold_pixels": threshold,
            "frame_count": len(decisions),
            "trigger_count": sum(row["trigger_expensive_perception"] for row in decisions),
            "skip_count": sum(not row["trigger_expensive_perception"] for row in decisions),
            "torch_imported": "torch" in sys.modules,
            "perception_model_loaded": False,
            "training_started": False,
            "confirmation_content_read": False,
            "decisions_sha256": sha256(decision_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
