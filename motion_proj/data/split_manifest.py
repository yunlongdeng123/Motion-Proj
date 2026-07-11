"""构建可复现的 nuScenes scene split 与 clip 清单。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..config import config_fingerprint, load_config, save_resolved_config
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import ExperimentRegistry, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from .nuscenes_dataset import NuScenesFutureVideoDataset

SCHEMA_VERSION = 1
TASK_ID = "P2-FRONT-01"


def build_split_manifest(dataset: NuScenesFutureVideoDataset) -> dict[str, Any]:
    """只基于 metadata 生成稳定清单，不读取 RGB/LiDAR 内容。"""
    scene_clip_counts = Counter(str(row["scene_name"]) for row in dataset.clip_records)
    scene_rows = []
    for scene in sorted(dataset.nusc.scene, key=lambda item: str(item["name"])):
        name = str(scene["name"])
        if name not in dataset.scene_names:
            continue
        scene_rows.append(
            {
                "scene_name": name,
                "scene_token": str(scene["token"]),
                "sample_count": int(scene["nbr_samples"]),
                "clip_count": int(scene_clip_counts[name]),
            }
        )

    clip_rows = [
        {
            "clip_id": str(row["sample_id"]),
            "scene_name": str(row["scene_name"]),
            "scene_token": str(row["scene_token"]),
            "start_index": int(row["start_index"]),
            "sample_tokens": list(row["sample_tokens"]),
        }
        for row in dataset.clip_records
    ]
    metadata_root = Path(dataset.dataroot) / dataset.version
    core = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "nuScenes",
        "version": dataset.version,
        "split": dataset.split,
        "camera": dataset.camera,
        "num_frames": dataset.K,
        "frame_stride": dataset.stride,
        "window_policy": "non-overlapping-keyframes",
        "source_metadata": {
            "scene_json_sha256": file_fingerprint(str(metadata_root / "scene.json")),
            "sample_json_sha256": file_fingerprint(str(metadata_root / "sample.json")),
        },
        "scene_count": len(scene_rows),
        "clip_count": len(clip_rows),
        "scenes": scene_rows,
        "clips": clip_rows,
    }
    return {**core, "split_fingerprint": sha256_json(core)}


def summarize_manifest(manifest: Mapping[str, Any], data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "non_empty": int(manifest["scene_count"]) > 0 and int(manifest["clip_count"]) > 0,
        "unique_clip_ids": len({row["clip_id"] for row in manifest["clips"]})
        == int(manifest["clip_count"]),
    }
    expected_scenes = data_cfg.get("expected_scene_count")
    expected_clips = data_cfg.get("expected_clip_count")
    if expected_scenes is not None:
        checks["expected_scene_count"] = int(manifest["scene_count"]) == int(expected_scenes)
    if expected_clips is not None:
        checks["expected_clip_count"] = int(manifest["clip_count"]) == int(expected_clips)
    return {
        "task_id": TASK_ID,
        "version": manifest["version"],
        "split": manifest["split"],
        "scene_count": manifest["scene_count"],
        "clip_count": manifest["clip_count"],
        "split_fingerprint": manifest["split_fingerprint"],
        "checks": checks,
        "accepted": all(checks.values()),
    }


def run(cfg: Any, output_root: str, run_id: str | None = None) -> tuple[Path, dict[str, Any]]:
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式 split manifest 拒绝在 dirty worktree 上运行")

    dataset = NuScenesFutureVideoDataset(cfg.data)
    split_manifest = build_split_manifest(dataset)
    summary = summarize_manifest(split_manifest, cfg.data)
    split_fingerprint = str(split_manifest["split_fingerprint"])
    if run_id is None:
        run_id = (
            f"p2-data-{dataset.split}-k{dataset.K}-"
            f"{str(git['commit'])[:8]}-{split_fingerprint[:8]}"
        )

    root = Path(output_root)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "COMPLETE"
    if complete_path.is_file():
        previous = complete_path.read_text(encoding="utf-8").strip()
        if previous == split_fingerprint:
            return run_dir, json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        raise RuntimeError(f"run 目录已由不同 split fingerprint 占用: {run_dir}")

    cfg_fingerprint = config_fingerprint(cfg)
    manifest = RunManifest(
        run_id=run_id,
        command=list(sys.argv),
        config_fingerprint=cfg_fingerprint,
        cache_fingerprint=split_fingerprint,
        seed=int(cfg.seed),
        git=git,
        environment=environment_fingerprint(),
        data_split=f"{dataset.version}:{dataset.split}:{dataset.camera}",
    )
    manifest.save(str(run_dir / "manifest.json"))
    save_resolved_config(cfg, str(run_dir / "resolved.yaml"))
    atomic_write_json(str(run_dir / "split_manifest.json"), split_manifest)

    completed_at = utc_now()
    summary.update(
        {
            "run_id": run_id,
            "git_commit": git["commit"],
            "config_fingerprint": cfg_fingerprint,
            "completed_at": completed_at,
        }
    )
    atomic_write_json(str(run_dir / "summary.json"), summary)
    manifest.status = "completed"
    manifest.exit_reason = "acceptance_passed" if summary["accepted"] else "acceptance_failed"
    manifest.ended_at = completed_at
    manifest.save(str(run_dir / "manifest.json"))
    atomic_write_text(str(complete_path), split_fingerprint + "\n")

    registry = ExperimentRegistry(str(root / "experiments.sqlite3"))
    known_run_ids = {row["run_id"] for row in registry.list()}
    if run_id not in known_run_ids:
        registry.register(run_id, "completed", cfg_fingerprint, str(run_dir))
    registry.update(run_id, "completed", exit_reason=manifest.exit_reason, summary=summary)
    return run_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="/root/autodl-tmp/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_dir, summary = run(cfg, args.output_root, args.run_id)
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
