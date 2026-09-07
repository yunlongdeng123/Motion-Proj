"""Build candidate RGB observations with temporal context/holdout roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.data.nuscenes_actor import load_actor_poses_for_samples
from motion_proj.worldsim_v72.data.nuscenes_camera import NuScenesCameraIndex
from motion_proj.worldsim_v72.data.splits import load_data_roles, require_role_access
from motion_proj.worldsim_v72.eas_vggt.appearance import observe_candidate_rgb
from motion_proj.worldsim_v72.eas_vggt.appearance_cache import ActorAppearanceCache, save_actor_appearance_cache
from motion_proj.worldsim_v72.eas_vggt.cache import load_backbone_geometry
from motion_proj.worldsim_v72.eas_vggt.preprocess import resize_camera_window
from motion_proj.worldsim_v72.eas_vggt.visual_pooling import aligned_backbone_points_world


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_windows(index: NuScenesCameraIndex, data: dict[str, Any]):
    split = json.loads(Path(data["visual_split"]).read_text(encoding="utf-8"))
    scene_ids = [str(row["scene_id"]) for row in split["rows"] if row["role"] == data["visual_split_role"]]
    scene_rows = {str(row["name"]): row for row in index.scenes}
    limit = int(data["windows_per_scene"])
    selected = []
    for scene in scene_ids:
        log = str(scene_rows[scene]["log_token"])
        windows = list(
            index.iter_windows(
                [log],
                role="train",
                camera_channels=data["camera_channels"],
                scene_ids=[scene],
                maximum_windows=limit,
            )
        )
        if len(windows) != limit:
            raise RuntimeError(f"insufficient windows for {scene}")
        selected.extend(windows)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["task_id"] != "WS-V72-E3-DECOUPLED-APPEARANCE-01":
        raise ValueError("E3 task id mismatch")
    data = config["data"]
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": "worldsim_v72.e3_appearance_cache.v1",
        "task_id": config["task_id"],
        "run_id": args.run_id,
        "status": "running",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "visual_split_role": data["visual_split_role"],
        "selection_uses_quality": False,
        "rgb_supervision_access": True,
        "physical_state_write": False,
        "source_test_read": False,
        "external_test_read": False,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "index"})
    try:
        roles = load_data_roles(Path(data["roles"]))
        allowed_logs = set(require_role_access(roles, "nuscenes", "train"))
        index = NuScenesCameraIndex(Path(data["dataset_root"]))
        windows = _select_windows(index, data)
        if any(window.log_id not in allowed_logs for window in windows):
            raise ValueError("appearance window escaped the frozen train role")
        corpus = Path(data["actor_corpus_root"])
        paths_by_scene = {
            scene: sorted((corpus / scene).glob("*.npz"))
            for scene in sorted({window.scene_id for window in windows})
        }
        tracks_by_sample = {
            window.window_id: [path.stem for path in paths_by_scene[window.scene_id]]
            for window in windows
        }
        poses = load_actor_poses_for_samples(index.metadata_root, tracks_by_sample)
        candidates = {}
        for paths in paths_by_scene.values():
            for path in paths:
                with np.load(path, allow_pickle=False) as payload:
                    candidates[path.stem] = payload["candidates"]
        ranks = {}
        for window in windows:
            ranks.setdefault(window.scene_id, {})[window.window_id] = len(ranks.setdefault(window.scene_id, {}))
        rows = []
        for backbone in config["backbones"]:
            for ordinal, window in enumerate(windows):
                _write_json(run_dir / "status.json", {"status": "running", "phase": f"rgb_{backbone}", "window_index": ordinal, "window_count": len(windows)})
                geometry_path = Path(config["backbone_run"]) / "backbone_cache" / backbone / f"{window.window_id}.npz"
                geometry = load_backbone_geometry(geometry_path)
                images, transforms = resize_camera_window(
                    window,
                    maximum_pixels=int(config["preprocess"]["maximum_pixels"]),
                    patch_multiple=int(config["preprocess"]["patch_multiple"]),
                )
                if not np.allclose(transforms, geometry.model_from_original_px):
                    raise ValueError("RGB resize/backbone transform mismatch")
                metric_points_world = aligned_backbone_points_world(window, geometry)
                temporal_rank = int(ranks[window.scene_id][window.window_id])
                temporal_role = "context" if temporal_rank < int(data["context_window_count"]) else "heldout_rgb"
                for actor_path in paths_by_scene[window.scene_id]:
                    pose = poses.get((window.window_id, actor_path.stem))
                    if pose is None:
                        continue
                    observation = observe_candidate_rgb(
                        candidates[actor_path.stem],
                        pose.world_from_actor,
                        window,
                        images,
                        geometry,
                        metric_points_world=metric_points_world,
                        occlusion_tolerance_m=float(config["appearance"]["occlusion_tolerance_m"]),
                    )
                    cache = ActorAppearanceCache(
                        track_id=actor_path.stem,
                        scene_id=window.scene_id,
                        window_id=window.window_id,
                        backbone_id=geometry.backbone_id,
                        rgb=observation.pooled_rgb,
                        confidence_sum=observation.confidence_sum,
                        observation_count=observation.observation_count,
                        provenance={
                            "payload_role": temporal_role,
                            "temporal_rank": temporal_rank,
                            "window_fingerprint": window.fingerprint,
                            "geometry_cache_sha256": _sha256(geometry_path),
                            "visibility_contract": "foundation_depth_foreground_test_v1",
                        },
                    )
                    output = run_dir / "appearance_cache" / backbone / window.window_id / f"{cache.track_id}.npz"
                    save_actor_appearance_cache(cache, output)
                    rows.append(
                        {
                            "backbone": backbone,
                            "scene_id": window.scene_id,
                            "window_id": window.window_id,
                            "track_id": cache.track_id,
                            "temporal_rank": temporal_rank,
                            "temporal_role": temporal_role,
                            "candidate_count": len(cache.rgb),
                            "visible_candidate_count": int(np.sum(cache.observation_count > 0)),
                        }
                    )
        summary = {
            **manifest,
            "status": "done",
            "failure_ledger_delta": "none",
            "window_count": len(windows),
            "actor_window_count": len(rows),
            "visible_candidate_count": sum(row["visible_candidate_count"] for row in rows),
            "rows": rows,
            "wall_seconds": time.monotonic() - started,
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "manifest.json", {**manifest, "status": "done", "failure_ledger_delta": "none", "summary": "summary.json"})
        _write_json(run_dir / "status.json", {"status": "done", "phase": "complete"})
        print(json.dumps({key: summary[key] for key in ("status", "window_count", "actor_window_count", "visible_candidate_count", "wall_seconds")}), flush=True)
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "runtime", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
