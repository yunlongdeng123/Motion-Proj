"""Prepare the capability-eligible scenes in the frozen V6.5 P2 cohort."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.prepare_dr_v2_drivestudio_scene import collect_required_many, load_asset_module
from scripts.prepare_worldsim_v64_calibration_batch import _link_static_dataset


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(config_path: Path, repo_root: Path, run_dir: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(run_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    prep = config["preparation"]
    temporary_root = Path(prep["temporary_raw_root"]).resolve()
    expected_parent = Path("/root/autodl-tmp/tmp").resolve()
    allowed_temporary_names = {
        "worldsim_v65_p2_raw_batch",
        "worldsim_v65_p3c_raw_batch",
        "worldsim_v65_p10v_raw_batch",
    }
    if temporary_root.parent != expected_parent or temporary_root.name not in allowed_temporary_names:
        raise RuntimeError(f"unexpected temporary root: {temporary_root}")
    reuse_partial_raw = bool(prep.get("reuse_partial_raw", False))
    partial_raw_reused = temporary_root.exists()
    if partial_raw_reused and not reuse_partial_raw:
        raise FileExistsError(temporary_root)
    metadata_root = Path(prep["metadata_root"])
    metadata = metadata_root / "v1.0-trainval"
    scenes = list(config["scenes"])
    processed_root = Path(prep["processed_root"])
    pending = [scene for scene in scenes if not (processed_root / f"{int(scene['processed_index']):03d}").is_dir()]
    payloads = collect_required_many(metadata, [str(scene["name"]) for scene in pending])
    required = {row["filename"] for payload in payloads.values() for row in payload["sample_data"]}
    temporary_root.mkdir(parents=True, exist_ok=reuse_partial_raw)
    _link_static_dataset(metadata_root, temporary_root)
    helpers = load_asset_module(repo_root)
    helpers.link_existing_files(Path(prep["raw_reuse_root"]), temporary_root, required)
    mapping, extracted = helpers.scan_shards(
        tar_dir=Path(prep["public_tar_root"]),
        members=required,
        index_path=Path(prep["member_shard_index_path"]),
        dst=temporary_root,
        workers=int(prep["archive_workers"]),
        shard_numbers=[str(value).zfill(2) for value in prep["archive_shards"]] if prep.get("archive_shards") else None,
    )
    del mapping
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{repo_root}:/root/autodl-tmp/third_party/drivestudio"

    def preprocess(scene: dict[str, object]) -> dict[str, object]:
        destination = processed_root / f"{int(scene['processed_index']):03d}"
        scene_started = time.monotonic()
        if destination.is_dir():
            return {"scene": scene["name"], "processed_index": scene["processed_index"], "reused": True, "wall_seconds": 0.0}
        command = [
            "/root/autodl-tmp/envs/drivestudio/bin/python",
            str(repo_root / "scripts/preprocess_dr_v2_nuscenes_single.py"),
            "--data-root", str(temporary_root),
            "--target-dir", str(prep["processor_target_dir"]),
            "--scene-index", str(int(scene["processed_index"])),
        ]
        with (run_dir / "logs" / f"{scene['name']}.log").open("wb") as handle:
            subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=int(prep["preprocess_timeout_seconds"]),
                check=True,
            )
        if not destination.is_dir():
            raise RuntimeError(f"processed scene missing after command: {scene['name']}")
        return {
            "scene": scene["name"],
            "processed_index": scene["processed_index"],
            "reused": False,
            "wall_seconds": time.monotonic() - scene_started,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(prep["preprocess_workers"])) as executor:
        rows = list(executor.map(preprocess, scenes))
    if temporary_root.parent == expected_parent and temporary_root.name in allowed_temporary_names:
        shutil.rmtree(temporary_root, ignore_errors=True)
    summary = {
        "task_id": config["task_id"],
        "status": "done",
        "scene_count": len(rows),
        "new_scene_count": sum(not row["reused"] for row in rows),
        "partial_raw_reused": partial_raw_reused,
        "extracted_member_count": len(extracted),
        "temporary_raw_removed_after_success": True,
        "quality_read": False,
        "rows": rows,
        "wall_seconds": time.monotonic() - started,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "status.json", {"status": "done", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.repo_root.resolve(), args.run_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
