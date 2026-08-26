"""为 V6.4 calibration/confirmation cohort 批量准备 DriveStudio 输入。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.prepare_dr_v2_drivestudio_scene import (
    collect_required,
    load_asset_module,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _link_file(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _link_static_dataset(metadata_root: Path, raw_root: Path) -> None:
    metadata = metadata_root / "v1.0-trainval"
    destination = raw_root / "v1.0-trainval"
    destination.mkdir(parents=True, exist_ok=True)
    for source in metadata.iterdir():
        if source.is_file():
            _link_file(source, destination / source.name)
    map_rows = json.loads((metadata / "map.json").read_text(encoding="utf-8"))
    for row in map_rows:
        relative = Path(row["filename"])
        _link_file(metadata_root / relative, raw_root / relative)


def _all_scenes(config: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for partition, payload in config["cohorts"].items():
        for scene in payload["scenes"]:
            rows.append({"partition": partition, **scene})
    return rows


def run(
    config_path: Path,
    repo_root: Path,
    run_dir: Path,
    reuse_temporary_raw: bool = False,
) -> dict[str, object]:
    if run_dir.exists():
        raise FileExistsError(run_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preparation = config["preparation"]
    scenes = _all_scenes(config)
    temporary_root = Path(preparation["temporary_raw_root"]).resolve()
    allowed_parent = Path("/root/autodl-tmp/tmp").resolve()
    allowed_temporary_names = {
        "worldsim_v64_p6_raw_batch",
        "worldsim_v64_p4c_raw_batch",
    }
    if temporary_root.parent != allowed_parent or temporary_root.name not in allowed_temporary_names:
        raise RuntimeError(f"temporary raw path is outside the frozen target: {temporary_root}")
    if temporary_root.exists() and not reuse_temporary_raw:
        raise FileExistsError(temporary_root)
    if reuse_temporary_raw and not temporary_root.exists():
        raise FileNotFoundError(temporary_root)

    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        run_dir / "status.json",
        {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    metadata_root = Path(preparation["metadata_root"])
    metadata = metadata_root / "v1.0-trainval"
    processed_root = Path(preparation["processed_root"])
    metadata_scenes = {
        str(row["name"]): row
        for row in json.loads((metadata / "scene.json").read_text(encoding="utf-8"))
    }
    expected_frames = {
        str(scene["name"]):
        (int(metadata_scenes[str(scene["name"])]["nbr_samples"]) - 1) * 5 + 1
        for scene in scenes
    }
    for scene in scenes:
        destination = processed_root / f"{int(scene['processed_index']):03d}"
        if destination.exists() and not reuse_temporary_raw:
            raise FileExistsError(destination)

    if not reuse_temporary_raw:
        payloads = {
            str(scene["name"]): collect_required(metadata, str(scene["name"]))
            for scene in scenes
        }
        required = {
            row["filename"]
            for payload in payloads.values()
            for row in payload["sample_data"]
        }
        temporary_root.mkdir(parents=True)
        _link_static_dataset(metadata_root, temporary_root)
        helpers = load_asset_module(repo_root)
        helpers.link_existing_files(
            Path(preparation["raw_reuse_root"]), temporary_root, required
        )
        helpers.scan_shards(
            tar_dir=Path(preparation["public_tar_root"]),
            members=required,
            index_path=allowed_parent / "worldsim_v64_p6_member_shards.json",
            dst=temporary_root,
            workers=int(preparation["archive_workers"]),
        )

    scene_rows = []
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{repo_root}:/root/autodl-tmp/third_party/drivestudio"
    )
    for scene in scenes:
        scene_started = time.monotonic()
        destination = processed_root / f"{int(scene['processed_index']):03d}"
        expected_lidar = expected_frames[str(scene["name"])]
        expected_images = expected_lidar * 6
        if destination.exists():
            images = len(list((destination / "images").glob("*.jpg")))
            lidar = len(list((destination / "lidar").glob("*.bin")))
            if images != expected_images or lidar != expected_lidar:
                raise RuntimeError(
                    f"incomplete resume scene {scene['name']}: images={images}/{expected_images}, "
                    f"lidar={lidar}/{expected_lidar}"
                )
            scene_rows.append(
                {
                    "partition": scene["partition"],
                    "scene": scene["name"],
                    "processed_index": int(scene["processed_index"]),
                    "image_count": images,
                    "lidar_count": lidar,
                    "reused_complete_scene": True,
                    "wall_seconds": time.monotonic() - scene_started,
                }
            )
            continue
        command = [
            "/root/autodl-tmp/envs/drivestudio/bin/python",
            str(repo_root / "scripts/preprocess_dr_v2_nuscenes_single.py"),
            "--data-root",
            str(temporary_root),
            "--target-dir",
            str(preparation["processor_target_dir"]),
            "--scene-index",
            str(int(scene["processed_index"])),
        ]
        log_path = run_dir / "logs" / f"{scene['name']}.log"
        with log_path.open("xb") as log:
            process = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=int(preparation["preprocess_timeout_seconds"]),
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError(f"preprocess failed for {scene['name']}: {log_path}")
        images = len(list((destination / "images").glob("*.jpg")))
        lidar = len(list((destination / "lidar").glob("*.bin")))
        if images != expected_images or lidar != expected_lidar:
            raise RuntimeError(
                f"processed count mismatch for {scene['name']}: "
                f"images={images}/{expected_images}, lidar={lidar}/{expected_lidar}"
            )
        scene_rows.append(
            {
                "partition": scene["partition"],
                "scene": scene["name"],
                "processed_index": int(scene["processed_index"]),
                "image_count": images,
                "lidar_count": lidar,
                "reused_complete_scene": False,
                "wall_seconds": time.monotonic() - scene_started,
            }
        )

    shutil.rmtree(temporary_root)
    wall = time.monotonic() - started
    summary = {
        "task_id": config["task_id"],
        "status": "done",
        "scene_count": len(scene_rows),
        "calibration_scene_count": sum(
            row["partition"] == "fresh_calibration" for row in scene_rows
        ),
        "confirmation_scene_count": sum(
            row["partition"] == "fresh_confirmation" for row in scene_rows
        ),
        "temporary_raw_removed_after_success": True,
        "reused_temporary_raw": reuse_temporary_raw,
        "scene_rows": scene_rows,
        "wall_seconds": wall,
        "quality_read": False,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reuse-temporary-raw", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.config.resolve(),
        args.repo_root.resolve(),
        args.run_dir.resolve(),
        args.reuse_temporary_raw,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
