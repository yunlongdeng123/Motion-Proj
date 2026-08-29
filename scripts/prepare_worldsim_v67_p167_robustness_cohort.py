"""Pipeline P167 shard extraction into per-scene preprocessing."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from scripts.build_adgs_nuscenes_assets import _scan_one_shard
from scripts.prepare_dr_v2_drivestudio_scene import collect_required_many
from scripts.prepare_worldsim_v67_validation_actor_cohort import _run_scene, _utc_now


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
    identities = {name: int(index_by_name[name]) for name in scene_names}
    payloads = collect_required_many(metadata, scene_names)
    required_by_scene = {
        name: {
            row["filename"] for row in payloads[name]["sample_data"] if row["channel"] == "LIDAR_TOP"
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
    scenes_by_shard: dict[str, list[str]] = {}
    for name in scene_names:
        shard = scene_shards[name]
        required_by_shard.setdefault(shard, set()).update(required_by_scene[name] - existing)
        scenes_by_shard.setdefault(shard, []).append(name)

    found_rows: dict[str, dict[str, object]] = {}
    preprocess_futures = {}
    results = []
    target_dir = Path(data["processed_target_dir"])
    processed_root = Path(data["processed_root"])
    with ProcessPoolExecutor(max_workers=len(required_by_shard)) as scan_pool, ThreadPoolExecutor(
        max_workers=int(data["preprocess_workers"])
    ) as preprocess_pool:
        scan_futures = {}
        for shard, members in sorted(required_by_shard.items()):
            if members:
                future = scan_pool.submit(
                    _scan_one_shard,
                    (str(archive_root / f"v1.0-trainval{shard}_blobs.tgz"), members, str(raw_root)),
                )
                scan_futures[future] = shard
            else:
                for name in scenes_by_shard[shard]:
                    future = preprocess_pool.submit(
                        _run_scene, name, identities[name], raw_root, target_dir, processed_root, log_dir,
                    )
                    preprocess_futures[future] = name
        for future in as_completed(scan_futures):
            shard = scan_futures[future]
            rows = future.result()
            found_rows.update(rows)
            print(json.dumps({
                "stage": "shard_ready", "shard": shard, "found": len(rows),
                "scenes_released": scenes_by_shard[shard],
            }), flush=True)
            for name in scenes_by_shard[shard]:
                scene_future = preprocess_pool.submit(
                    _run_scene, name, identities[name], raw_root, target_dir, processed_root, log_dir,
                )
                preprocess_futures[scene_future] = name

        missing = required - existing - set(found_rows)
        if missing:
            raise RuntimeError(
                f"exact shard extraction missed {len(missing)} LIDAR files; example={sorted(missing)[:5]}"
            )
        for future in as_completed(preprocess_futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)

    mapping = {
        member: scene_shards[scene]
        for scene, members in required_by_scene.items() for member in members if member in existing
    }
    mapping.update({name: str(found_rows[name]["shard"]) for name in sorted(found_rows)})
    index_path = Path(data["member_shard_index"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    partial = index_path.with_suffix(index_path.suffix + ".partial")
    partial.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, index_path)
    extracted = {name for name, row in found_rows.items() if bool(row["extracted"])}
    results.sort(key=lambda row: int(row["scene_index"]))
    summary = {
        "schema_version": "worldsim_v67.p167_pipelined_robustness_preparation_summary.v1",
        "status": "done", "started_at_utc": _utc_now(), "scene_count": len(results),
        "required_lidar_file_count": len(required), "newly_extracted_lidar_file_count": len(extracted),
        "processed_root": str(processed_root), "shard_to_scene_pipeline": True,
        "scenes": results, "wall_seconds": time.monotonic() - started,
        "hash_checksum_fingerprint_added": False,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "done", "completed_at_utc": _utc_now(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
