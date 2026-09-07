#!/usr/bin/env python3
"""Freeze nuScenes source-test logs and a metadata-only visual cohort after route selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--route-summary", type=Path, required=True)
    parser.add_argument("--cohort-output", type=Path, required=True)
    parser.add_argument("--rgb-needed-output", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path, required=True)
    args = parser.parse_args()

    route = json.loads(args.route_summary.read_text(encoding="utf-8"))
    if route["role"] != "route_select" or route["verdict"] != "frozen_primary_supported":
        raise RuntimeError("source-test may only be frozen after a supported route decision")
    discount = route.get("reliability_discount")
    if not discount or discount["mode"] != "route_select_grid":
        raise RuntimeError("route result does not contain a reliability-discount selection")

    roles = json.loads(args.roles.read_text(encoding="utf-8"))
    nuscenes = roles["datasets"]["nuscenes"]
    groups = nuscenes["group_roles"]
    candidates = sorted(map(str, groups["source_candidate_pool"]))
    if not candidates or groups["source_test"] or nuscenes["frozen_roles"]["source_test"]:
        raise RuntimeError("source-test roles are not in the expected unopened state")

    metadata_root = args.dataset_root / "v1.0-trainval"
    scenes = json.loads((metadata_root / "scene.json").read_text(encoding="utf-8"))
    scenes_by_log: dict[str, list[dict[str, Any]]] = {log: [] for log in candidates}
    for row in scenes:
        log = str(row["log_token"])
        if log in scenes_by_log:
            scenes_by_log[log].append(row)
    if any(not value for value in scenes_by_log.values()):
        raise RuntimeError("a source-test log has no scene metadata")
    chosen = [min(scenes_by_log[log], key=lambda row: str(row["name"])) for log in candidates]
    cohort = {
        "schema_version": "worldsim_v72.visual_cohort.v1",
        "role": "source_test",
        "quality_used_for_selection": False,
        "selection": "lexicographically_first_scene_per_frozen_log",
        "rows": [
            {"role": "source_test", "log_id": str(row["log_token"]), "scene_id": str(row["name"])}
            for row in chosen
        ],
    }

    samples = {row["token"]: row for row in json.loads((metadata_root / "sample.json").read_text())}
    sensors = {row["token"]: row for row in json.loads((metadata_root / "sensor.json").read_text())}
    calibrated = {row["token"]: row for row in json.loads((metadata_root / "calibrated_sensor.json").read_text())}
    sample_data_rows = json.loads((metadata_root / "sample_data.json").read_text())
    by_sample: dict[str, dict[str, str]] = {}
    allowed_channels = {"CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"}
    for row in sample_data_rows:
        channel = str(sensors[calibrated[row["calibrated_sensor_token"]]["sensor_token"]]["channel"])
        if bool(row["is_key_frame"]) and channel in allowed_channels:
            by_sample.setdefault(str(row["sample_token"]), {})[channel] = str(row["filename"])
    needed = []
    sample_tokens = []
    for scene in chosen:
        token = str(scene["first_sample_token"])
        for _ in range(4):
            if not token:
                raise RuntimeError(f"scene {scene['name']} has fewer than four samples")
            sample_tokens.append(token)
            channels = by_sample.get(token, {})
            if set(channels) != allowed_channels:
                raise RuntimeError(f"sample {token} is missing a frozen camera channel")
            needed.extend(channels[channel] for channel in sorted(allowed_channels))
            token = str(samples[token]["next"])

    groups["source_test"] = candidates
    groups["source_candidate_pool"] = []
    nuscenes["frozen_roles"]["source_test"] = True
    nuscenes["notes"] = [
        *nuscenes.get("notes", []),
        "The three source-test logs and one metadata-ordered visual scene per log were frozen after route selection and before source sensor payload access.",
    ]
    roles["status"] = "clean_dev_route_and_source_test_frozen"

    args.rgb_needed_output.parent.mkdir(parents=True, exist_ok=True)
    args.rgb_needed_output.write_text("\n".join(sorted(needed)) + "\n", encoding="utf-8")
    write_json(args.cohort_output, cohort)
    write_json(args.roles, roles)
    repo_root = Path(__file__).resolve().parents[1]
    code_paths = [
        repo_root / "scripts/evaluate_worldsim_v72_e2_frozen.py",
        repo_root / "scripts/build_worldsim_v72_e2_multiview_cache.py",
        repo_root / "scripts/build_worldsim_v72_eas_actor_corpus.py",
        repo_root / "scripts/build_worldsim_v72_clean_actor_data.py",
    ]
    lock = {
        "schema_version": "worldsim_v72.source_test_lock.v1",
        "source_payload_read": False,
        "quality_used_for_selection": False,
        "git_commit_before_source_read": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
        "source_log_ids": candidates,
        "source_scene_ids": [str(row["name"]) for row in chosen],
        "source_sample_tokens": sample_tokens,
        "rgb_needed_manifest": str(args.rgb_needed_output),
        "rgb_needed_manifest_sha256": sha256(args.rgb_needed_output),
        "route_summary": str(args.route_summary),
        "route_summary_sha256": sha256(args.route_summary),
        "selected_reliability_alpha": float(discount["selected_alpha"]),
        "checkpoint_sha256": route["checkpoint_sha256"],
        "code_sha256": {str(path.relative_to(repo_root)): sha256(path) for path in code_paths},
    }
    write_json(args.lock_output, lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
