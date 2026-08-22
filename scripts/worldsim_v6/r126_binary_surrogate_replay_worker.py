#!/usr/bin/env python3
"""Standard-library fresh-process worker for a WorldSim V6 R126 package."""

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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.package / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = args.package / name
        if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"package manifest mismatch: {name}")
    policy = json.loads((args.package / "POLICY.json").read_text(encoding="utf-8"))
    features = read_jsonl(args.package / "FEATURE_ROWS.jsonl")
    targets = read_jsonl(args.package / "TARGET_ROWS.jsonl")
    if len(features) != len(targets):
        raise RuntimeError("feature/target denominator mismatch")
    threshold = int(policy["threshold_pixels"])
    decisions = []
    false_positive = 0
    false_negative = 0
    for feature, target in zip(features, targets):
        if feature["row_id"] != target["row_id"]:
            raise RuntimeError("feature/target row order mismatch")
        predicted = int(feature["changed_rgb_pixels"]) >= threshold
        truth = bool(target["any_changed_frozen_deeplab_label"])
        false_positive += int(predicted and not truth)
        false_negative += int(not predicted and truth)
        decisions.append(
            {
                "row_id": feature["row_id"],
                "predict_any_changed_frozen_deeplab_label": predicted,
            }
        )
    decision_path = args.output / "DECISIONS.jsonl"
    write_jsonl(decision_path, decisions)
    write_json(
        args.output / "WORKER_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r126_worker_audit.v1",
            "package_manifest_sha256": sha256(manifest_path),
            "row_count": len(decisions),
            "threshold_pixels": threshold,
            "positive_predictions": sum(
                row["predict_any_changed_frozen_deeplab_label"] for row in decisions
            ),
            "false_positive": false_positive,
            "false_negative": false_negative,
            "decisions_sha256": sha256(decision_path),
            "torch_imported": "torch" in sys.modules,
            "perception_model_loaded": False,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
