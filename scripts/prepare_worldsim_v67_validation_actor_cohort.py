"""Extract LIDAR-only validation inputs and build minimal Actor reliability scenes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.build_adgs_nuscenes_assets import _scan_one_shard
from scripts.prepare_dr_v2_drivestudio_scene import collect_required_many


PROJECT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_scene(
    scene_name: str,
    scene_index: int,
    raw_root: Path,
    target_dir: Path,
    processed_root: Path,
    log_dir: Path,
) -> dict[str, object]:
    final_scene = processed_root / f"{scene_index:03d}"
    if (final_scene / "instances" / "instances_info.json").is_file() and (
        final_scene / "lidar_pose"
    ).is_dir():
        return {
            "scene_name": scene_name,
            "scene_index": scene_index,
            "status": "reused_ready",
            "duration_seconds": 0.0,
        }
    started = time.monotonic()
    command = [
        "/root/autodl-tmp/envs/drivestudio/bin/python",
        str(PROJECT / "scripts/preprocess_worldsim_v67_actor_scene.py"),
        "--data-root", str(raw_root),
        "--target-dir", str(target_dir),
        "--scene-index", str(scene_index),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{PROJECT}:/root/autodl-tmp/third_party/drivestudio"
    )
    log_path = log_dir / f"{scene_name}.log"
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command, cwd=PROJECT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{scene_name} actor preprocess returncode={completed.returncode}; log={log_path}"
        )
    if not (final_scene / "instances" / "instances_info.json").is_file() or not (
        final_scene / "lidar_pose"
    ).is_dir():
        raise RuntimeError(f"{scene_name} actor outputs missing at {final_scene}")
    return {
        "scene_name": scene_name,
        "scene_index": scene_index,
        "status": "done",
        "duration_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    log_dir = run_dir / "logs"
    log_dir.mkdir()
    started = time.monotonic()

    data = config["evaluation_data"]
    scene_names = [str(name) for name in data["scene_names"]]
    metadata = Path(data["metadata_root"]) / "v1.0-trainval"
    scenes = json.loads((metadata / "scene.json").read_text(encoding="utf-8"))
    index_by_name = {str(row["name"]): index for index, row in enumerate(scenes)}
    identities = [(name, int(index_by_name[name])) for name in scene_names]
    payloads = collect_required_many(metadata, scene_names)
    required_by_scene = {
        name: {
            row["filename"]
            for row in payloads[name]["sample_data"]
            if row["channel"] == "LIDAR_TOP"
        }
        for name in scene_names
    }
    required = set().union(*required_by_scene.values())
    raw_root = Path(data["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    existing = {name for name in required if (raw_root / name).is_file()}
    archive_root = Path(data["sensor_archive_root"])
    scene_shards = {str(name): str(shard).zfill(2) for name, shard in data["scene_shards"].items()}
    required_by_shard: dict[str, set[str]] = {}
    for name, members in required_by_scene.items():
        required_by_shard.setdefault(scene_shards[name], set()).update(members - existing)
    tasks = [] if data.get("recovery_only", False) else [
        (
            str(archive_root / f"v1.0-trainval{shard}_blobs.tgz"),
            members,
            str(raw_root),
        )
        for shard, members in sorted(required_by_shard.items()) if members
    ]
    found_rows: dict[str, dict[str, object]] = {}
    if tasks:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            for rows in pool.map(_scan_one_shard, tasks):
                found_rows.update(rows)
    missing = required - existing - set(found_rows)
    recovery_shards = [str(value).zfill(2) for value in data.get("recovery_shards", [])]
    if missing and recovery_shards:
        recovery_tasks = [
            (
                str(archive_root / f"v1.0-trainval{shard}_blobs.tgz"),
                missing,
                str(raw_root),
            )
            for shard in recovery_shards
        ]
        with ProcessPoolExecutor(max_workers=len(recovery_tasks)) as pool:
            for rows in pool.map(_scan_one_shard, recovery_tasks):
                found_rows.update(rows)
        missing = required - existing - set(found_rows)
    if missing:
        raise RuntimeError(
            f"exact shard extraction missed {len(missing)} LIDAR files; "
            f"example={sorted(missing)[:5]}"
        )
    mapping = {
        member: scene_shards[scene]
        for scene, members in required_by_scene.items() for member in members if member in existing
    }
    mapping.update({name: str(found_rows[name]["shard"]) for name in sorted(found_rows)})
    extracted = {name for name, row in found_rows.items() if bool(row["extracted"])}
    index_path = Path(data["member_shard_index"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_index = index_path.with_suffix(index_path.suffix + ".partial")
    tmp_index.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_index, index_path)
    print(
        json.dumps({
            "stage": "lidar_extracted", "required": len(required),
            "newly_extracted": len(extracted), "mapped": len(mapping),
        }), flush=True,
    )

    target_dir = Path(data["processed_target_dir"])
    processed_root = Path(data["processed_root"])
    results = []
    with ThreadPoolExecutor(max_workers=int(data["preprocess_workers"])) as pool:
        futures = {
            pool.submit(
                _run_scene, name, index, raw_root, target_dir, processed_root, log_dir
            ): name
            for name, index in identities
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda row: int(row["scene_index"]))
    summary = {
        "schema_version": "worldsim_v67.validation_actor_cohort_summary.v1",
        "status": "done",
        "started_at_utc": _utc_now(),
        "scene_count": len(results),
        "required_lidar_file_count": len(required),
        "newly_extracted_lidar_file_count": len(extracted),
        "processed_root": str(processed_root),
        "scenes": results,
        "wall_seconds": time.monotonic() - started,
        "hash_checksum_fingerprint_added": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({"status": "done", "completed_at_utc": _utc_now()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
