#!/usr/bin/env python3
"""Validate and record reuse of immutable scene-0230 M3 data assets."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


EXPECTED = {
    "images": 1176,
    "lidar": 196,
    "lidar_pose": 196,
    "extrinsics": 1176,
    "sky_masks": 588,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refuse to overwrite stage artifact: {path}")
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_success(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "done" or payload.get("return_code") != 0:
        raise RuntimeError(f"source stage is not reusable: {path}: {payload}")
    return payload


def count_assets(scene: Path) -> dict[str, int]:
    return {
        "images": len(list((scene / "images").glob("*.jpg"))),
        "lidar": len(list((scene / "lidar").glob("*.bin"))),
        "lidar_pose": len(list((scene / "lidar_pose").glob("*.txt"))),
        "extrinsics": len(list((scene / "extrinsics").glob("*.txt"))),
        "sky_masks": len(list((scene / "sky_masks").glob("*.png"))),
    }


def validate_counts(counts: dict[str, int], expected: dict[str, int] = EXPECTED) -> None:
    if counts != expected:
        raise RuntimeError(f"processed asset counts changed: {counts} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/data/dynamic_editing_v2/"
            "drivestudio_processed_10Hz/trainval/179"
        ),
    )
    args = parser.parse_args()

    current_raw = load_success(args.run_dir / "stages" / "raw_prepare.json")
    source_preprocess_path = args.source_run / "stages" / "preprocess.json"
    source_sky_path = args.source_run / "stages" / "sky_masks.json"
    load_success(source_preprocess_path)
    load_success(source_sky_path)

    scene = args.scene_root
    counts = count_assets(scene)
    validate_counts(counts)
    instances = scene / "instances" / "instances_info.json"
    if not instances.is_file():
        raise FileNotFoundError(instances)
    current_manifest = Path(str(current_raw["manifest"]))
    if not current_manifest.is_file():
        raise FileNotFoundError(current_manifest)

    common = {
        "status": "done",
        "return_code": 0,
        "reuse_mode": "verified_existing_assets",
        "source_run": str(args.source_run.resolve()),
        "scene_root": str(scene.resolve()),
        "counts": counts,
        "expected": EXPECTED,
        "raw_manifest": str(current_manifest),
        "raw_manifest_sha256": sha256_file(current_manifest),
        "instances_info_sha256": sha256_file(instances),
        "verified_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
    }
    preprocess = dict(common)
    preprocess.update(
        {
            "stage": "preprocess",
            "source_stage": str(source_preprocess_path),
            "source_stage_sha256": sha256_file(source_preprocess_path),
        }
    )
    sky = dict(common)
    sky.update(
        {
            "stage": "sky_masks",
            "source_stage": str(source_sky_path),
            "source_stage_sha256": sha256_file(source_sky_path),
        }
    )
    atomic_json(args.run_dir / "stages" / "preprocess.json", preprocess)
    atomic_json(args.run_dir / "stages" / "sky_masks.json", sky)
    print(json.dumps({"status": "done", "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
