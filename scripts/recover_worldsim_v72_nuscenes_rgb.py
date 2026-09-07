#!/usr/bin/env python3
"""Recover an explicitly frozen nuScenes RGB subset from camera-only archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import yaml
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--needed", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--provenance-output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    needed_path = Path(args.needed).resolve()
    output_root = Path(args.output_root).resolve()
    requested = [line.strip() for line in needed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(requested) != len(set(requested)):
        raise RuntimeError("The frozen file list contains duplicates")
    for member in requested:
        target = (output_root / member).resolve()
        if output_root not in target.parents:
            raise RuntimeError(f"Unsafe archive member: {member}")

    requested_set = set(requested)
    archive_records = []
    extracted = set()
    for spec in config["archives"]:
        archive = Path(spec["path"]).resolve()
        actual_hash = sha256(archive)
        if actual_hash != spec["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {archive}: {actual_hash}")
        with tarfile.open(archive, "r:gz") as bundle:
            members = {member.name: member for member in bundle.getmembers() if member.isfile()}
            selected = sorted(requested_set.intersection(members))
            for name in selected:
                target = output_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(members[name])
                if source is None:
                    raise RuntimeError(f"Could not read {name} from {archive}")
                with target.open("wb") as stream:
                    while chunk := source.read(8 << 20):
                        stream.write(chunk)
                extracted.add(name)
        archive_records.append(
            {
                "path": str(archive),
                "source_url": spec["source_url"],
                "sha256": actual_hash,
                "selected_file_count": len(selected),
            }
        )

    missing = sorted(requested_set - extracted)
    if missing:
        raise RuntimeError(f"Requested files absent from archives: {missing}")

    image_records = []
    for name in sorted(extracted):
        path = output_root / name
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        if (width, height) != tuple(config["expected_resolution"]):
            raise RuntimeError(f"Unexpected image size for {name}: {(width, height)}")
        image_records.append(
            {
                "relative_path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "width": width,
                "height": height,
                "mode": mode,
            }
        )

    provenance = {
        "schema_version": "worldsim_v72.nuscenes_rgb_recovery.v1",
        "license_note": config["license_note"],
        "selection_policy": "extract_only_predeclared_frozen_paths",
        "needed_manifest": str(needed_path),
        "needed_manifest_sha256": sha256(needed_path),
        "output_root": str(output_root),
        "archive_records": archive_records,
        "image_count": len(image_records),
        "images": image_records,
    }
    out = Path(args.provenance_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"image_count": len(image_records), "provenance": str(out)}, indent=2))


if __name__ == "__main__":
    main()
