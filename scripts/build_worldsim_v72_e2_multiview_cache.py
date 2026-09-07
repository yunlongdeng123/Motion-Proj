"""Materialize deterministic multi-window EAS-VGGT actor observations on train logs."""

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
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.data.nuscenes_actor import load_actor_poses_for_samples
from motion_proj.worldsim_v72.data.nuscenes_camera import NuScenesCameraIndex
from motion_proj.worldsim_v72.data.splits import load_data_roles, require_role_access
from motion_proj.worldsim_v72.eas_vggt.actor_visual_cache import ActorVisualCache, save_actor_visual_cache
from motion_proj.worldsim_v72.eas_vggt.backbones import Pi3XBackbone, VGGTBackbone
from motion_proj.worldsim_v72.eas_vggt.cache import save_backbone_geometry
from motion_proj.worldsim_v72.eas_vggt.cache import load_backbone_geometry
from motion_proj.worldsim_v72.eas_vggt.preprocess import resize_camera_window
from motion_proj.worldsim_v72.eas_vggt.visual_pooling import (
    aligned_backbone_points_world,
    observe_actor_candidates,
)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_actor_inputs(path: Path, fields: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = set(fields) - set(payload.files)
        if missing:
            raise ValueError(f"missing fields in {path}: {sorted(missing)}")
        return {name: payload[name] for name in fields}


def _select_windows(index: NuScenesCameraIndex, allowed_logs: set[str], config: dict[str, Any]):
    if "visual_split" in config:
        split = json.loads(Path(config["visual_split"]).read_text(encoding="utf-8"))
        split_role = str(config["visual_split_role"])
        scene_ids = [str(row["scene_id"]) for row in split["rows"] if row["role"] == split_role]
    else:
        scene_ids = [str(value) for value in config["scene_ids"]]
    scene_rows = {str(row["name"]): row for row in index.scenes}
    missing = set(scene_ids) - set(scene_rows)
    if missing:
        raise ValueError(f"unknown scenes: {sorted(missing)}")
    logs = [str(scene_rows[scene]["log_token"]) for scene in scene_ids]
    if not set(logs) <= allowed_logs:
        raise ValueError("frozen E2 scenes are not all in the train role")
    limit = int(config["windows_per_scene"])
    selected = {}
    for scene, log in zip(scene_ids, logs):
        selected[scene] = list(
            index.iter_windows(
                [log],
                role=str(config["role"]),
                camera_channels=config["camera_channels"],
                scene_ids=[scene],
                maximum_windows=limit,
            )
        )
    if any(len(selected[scene]) != limit for scene in scene_ids):
        counts = {scene: len(windows) for scene, windows in selected.items()}
        raise RuntimeError(f"insufficient payload-complete windows: {counts}")
    return [window for scene in scene_ids for window in selected[scene]]


def _make_backbone(name: str, config: dict[str, Any]):
    kwargs = {
        "repository_root": Path(config["repository_root"]),
        "checkpoint": Path(config["checkpoint"]),
        "device": str(config.get("device", "cuda")),
        "keep_loaded": True,
    }
    if name == "vggt":
        return VGGTBackbone(**kwargs)
    if name == "pi3x":
        return Pi3XBackbone(**kwargs)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["task_id"] != "WS-V72-E2-LEARNED-VISUAL-EVIDENCE-01":
        raise ValueError("E2 task id mismatch")
    data = config["data"]
    if data["supervision_access"] or data["source_test_read"] or data["external_test_read"]:
        raise ValueError("multi-window build must not read evaluation or supervision payloads")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": "worldsim_v72.e2_multiview_cache.v1",
        "task_id": config["task_id"],
        "run_id": args.run_id,
        "status": "running",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
        "selection_contract": "explicit_scenes_then_metadata_order_first_k",
        "selection_uses_quality": False,
        "supervision_access": False,
        "source_test_read": False,
        "external_test_read": False,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "window_index"})
    try:
        roles = load_data_roles(Path(data["roles"]))
        allowed_logs = set(require_role_access(roles, "nuscenes", str(data["role"])))
        index = NuScenesCameraIndex(Path(data["dataset_root"]), metadata_version=str(data["metadata_version"]))
        windows = _select_windows(index, allowed_logs, data)
        corpus_root = Path(data["actor_corpus_root"])
        actor_paths = {
            scene: sorted((corpus_root / scene).glob("*.npz"))
            for scene in sorted({window.scene_id for window in windows})
        }
        tracks_by_sample = {
            window.window_id: [path.stem for path in actor_paths[window.scene_id]]
            for window in windows
        }
        _write_json(run_dir / "status.json", {"status": "running", "phase": "actor_pose_stream"})
        actor_poses = load_actor_poses_for_samples(index.metadata_root, tracks_by_sample)
        actor_inputs = {
            path.stem: _load_actor_inputs(path, list(data["selected_actor_fields"]))
            for paths in actor_paths.values()
            for path in paths
        }
        rows: list[dict[str, Any]] = []
        reuse_root = Path(config["reuse_backbone_run"]) if config.get("reuse_backbone_run") else None
        for backbone_name, backbone_config in config["backbones"].items():
            backbone = None if reuse_root is not None else _make_backbone(backbone_name, backbone_config)
            try:
                for window_index, window in enumerate(windows):
                    _write_json(
                        run_dir / "status.json",
                        {
                            "status": "running",
                            "phase": f"infer_{backbone_name}",
                            "window_index": window_index,
                            "window_count": len(windows),
                            "window_id": window.window_id,
                        },
                    )
                    if reuse_root is None:
                        images, transforms = resize_camera_window(
                            window,
                            maximum_pixels=int(config["preprocess"]["maximum_pixels"]),
                            patch_multiple=int(config["preprocess"]["patch_multiple"]),
                        )
                        torch.cuda.reset_peak_memory_stats()
                        inference_started = time.monotonic()
                        geometry = backbone.infer(window, images, transforms)
                        inference_seconds = time.monotonic() - inference_started
                        geometry_path = run_dir / "backbone_cache" / backbone_name / f"{window.window_id}.npz"
                        save_backbone_geometry(
                            geometry,
                            geometry_path,
                            compressed=bool(config.get("compress_backbone_cache", True)),
                        )
                        reused_geometry = False
                    else:
                        geometry_path = reuse_root / "backbone_cache" / backbone_name / f"{window.window_id}.npz"
                        geometry = load_backbone_geometry(geometry_path)
                        if geometry.provenance.get("window_fingerprint") != window.fingerprint:
                            raise ValueError(f"reused {backbone_name} cache/window fingerprint mismatch")
                        inference_seconds = 0.0
                        reused_geometry = True
                    matched = 0
                    observed = 0
                    camera_observations = 0
                    metric_points_world = aligned_backbone_points_world(window, geometry)
                    for actor_path in actor_paths[window.scene_id]:
                        pose = actor_poses.get((window.window_id, actor_path.stem))
                        if pose is None:
                            continue
                        actor = actor_inputs[actor_path.stem]
                        observation = observe_actor_candidates(
                            actor["candidates"],
                            pose.world_from_actor,
                            window,
                            geometry,
                            metric_points_world=metric_points_world,
                        )
                        cache = ActorVisualCache(
                            track_id=str(actor["track_id"]),
                            scene_id=str(actor["scene_name"]),
                            window_id=window.window_id,
                            backbone_id=geometry.backbone_id,
                            candidates_actor_m=actor["candidates"],
                            base_features=actor["base_features"],
                            input_evidence_fou=actor["evidence_masses"],
                            opportunity_count=actor["evidence_opportunities"],
                            visual_features=observation.pooled_features,
                            geometry_features=observation.pooled_geometry_features,
                            confidence_sum=observation.confidence_sum,
                            observation_count=observation.observation_count,
                            world_from_actor=pose.world_from_actor,
                            provenance={
                                "payload_role": "build_input",
                                "actor_annotation_id": pose.annotation_id,
                                "actor_source_sha256": _sha256(actor_path),
                                "backbone_cache_sha256": _sha256(geometry_path),
                                "window_fingerprint": window.fingerprint,
                                "selected_actor_fields": list(data["selected_actor_fields"]),
                                "visibility_contract": "calibrated_frustum_only_v1",
                            },
                        )
                        output_path = run_dir / "actor_cache" / backbone_name / window.window_id / f"{cache.track_id}.npz"
                        save_actor_visual_cache(cache, output_path)
                        matched += 1
                        observed += int(np.sum(cache.observation_count > 0))
                        camera_observations += int(np.sum(cache.observation_count))
                    rows.append(
                        {
                            "backbone": backbone_name,
                            "scene_id": window.scene_id,
                            "window_id": window.window_id,
                            "window_fingerprint": window.fingerprint,
                            "matched_actor_count": matched,
                            "observed_candidate_count": observed,
                            "camera_observation_count": camera_observations,
                            "inference_seconds": inference_seconds,
                            "reused_geometry": reused_geometry,
                            "peak_gpu_memory_gib": 0.0 if reused_geometry else torch.cuda.max_memory_allocated() / 1024**3,
                        }
                    )
            finally:
                if backbone is not None:
                    backbone.close()
        summary = {
            **manifest,
            "status": "done",
            "failure_ledger_delta": "none",
            "window_count": len(windows),
            "scene_count": len({window.scene_id for window in windows}),
            "actor_pose_match_count": len(actor_poses),
            "rows": rows,
            "wall_seconds": time.monotonic() - started,
            "interpretation": "Target-free multi-window feature materialization; no quality claim.",
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "manifest.json", {**manifest, "status": "done", "failure_ledger_delta": "none", "summary": "summary.json"})
        _write_json(run_dir / "status.json", {"status": "done", "phase": "complete"})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "runtime", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
