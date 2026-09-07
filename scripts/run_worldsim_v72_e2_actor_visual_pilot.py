"""E2 pilot: bridge frozen E1 geometry to V7/V7.1 Actor candidates without labels."""

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

from motion_proj.worldsim_v72.data.nuscenes_actor import load_actor_poses_at_sample
from motion_proj.worldsim_v72.data.nuscenes_camera import NuScenesCameraIndex
from motion_proj.worldsim_v72.data.splits import load_data_roles, require_role_access
from motion_proj.worldsim_v72.eas_vggt.actor_visual_cache import ActorVisualCache, save_actor_visual_cache
from motion_proj.worldsim_v72.eas_vggt.cache import load_backbone_geometry
from motion_proj.worldsim_v72.eas_vggt.models import (
    EvidenceConditionedSurfaceAdapter,
    MatchedScalarSurfaceAdapter,
    trainable_parameter_count,
)
from motion_proj.worldsim_v72.eas_vggt.visual_pooling import observe_actor_candidates


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


def _find_window(index: NuScenesCameraIndex, log_ids: list[str], config: dict[str, Any]):
    wanted = str(config["window_id"])
    for window in index.iter_windows(log_ids, role=str(config["role"]), camera_channels=config["camera_channels"]):
        if window.window_id == wanted:
            return window
    raise RuntimeError(f"frozen window unavailable: {wanted}")


def _actor_inputs(path: Path, selected_fields: list[str]) -> dict[str, np.ndarray]:
    required = set(selected_fields)
    with np.load(path, allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"missing actor fields in {path}: {sorted(missing)}")
        return {name: payload[name] for name in selected_fields}


def _tensor_inputs(caches: list[ActorVisualCache], device: torch.device) -> dict[str, torch.Tensor]:
    concatenate = lambda name: np.concatenate([getattr(cache, name) for cache in caches], axis=0)
    return {
        "base_features": torch.from_numpy(concatenate("base_features")).to(device),
        "canonical_xyz": torch.from_numpy(concatenate("candidates_actor_m")).to(device),
        "evidence_fou": torch.from_numpy(concatenate("input_evidence_fou")).to(device),
        "opportunity_count": torch.from_numpy(concatenate("opportunity_count")).to(device),
        "visual_features": torch.from_numpy(concatenate("visual_features")).to(device),
        "visual_observed": torch.from_numpy(concatenate("observation_count") > 0).to(device),
        "geometric_features": torch.from_numpy(concatenate("geometry_features")).to(device),
        "geometry_observed": torch.from_numpy(concatenate("observation_count") > 0).to(device),
    }


def _adapter_smoke(caches: list[ActorVisualCache], config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(str(config["device"]))
    inputs = _tensor_inputs(caches, device)
    kwargs = {
        "base_feature_dim": int(inputs["base_features"].shape[-1]),
        "visual_feature_dim": int(inputs["visual_features"].shape[-1]),
        "hidden_dim": int(config["hidden_dim"]),
        "visual_dim": int(config["visual_dim"]),
        "maximum_surface_delta_m": float(config["maximum_surface_delta_m"]),
    }
    torch.cuda.reset_peak_memory_stats(device)
    model = EvidenceConditionedSurfaceAdapter(**kwargs).to(device)
    scalar = MatchedScalarSurfaceAdapter(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["smoke_learning_rate"]))
    optimizer.zero_grad(set_to_none=True)
    output = model(**inputs)
    evidence_consistency = torch.nn.functional.mse_loss(output.evidence_fou, inputs["evidence_fou"])
    geometry_prior = output.surface_delta_actor_m.square().mean()
    neutral_return_prior = output.blocking_logit.square().mean() + output.detection_logit.square().mean()
    loss = evidence_consistency + 0.01 * geometry_prior + 0.001 * neutral_return_prior
    loss.backward()
    finite_gradients = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    optimizer.step()
    with torch.inference_mode():
        scalar_delta, scalar_response = scalar(**inputs)
    result = {
        "candidate_count": int(len(inputs["canonical_xyz"])),
        "observed_candidate_count": int(inputs["visual_observed"].sum().item()),
        "evidence_consistency_loss": float(evidence_consistency.detach().cpu()),
        "geometry_prior_loss": float(geometry_prior.detach().cpu()),
        "total_smoke_loss": float(loss.detach().cpu()),
        "finite_gradients": bool(finite_gradients),
        "surface_delta_max_abs_m": float(output.surface_delta_actor_m.detach().abs().max().cpu()),
        "fou_sum_max_error": float((output.evidence_fou.detach().sum(dim=-1) - 1.0).abs().max().cpu()),
        "scalar_output_finite": bool(torch.isfinite(scalar_delta).all() and torch.isfinite(scalar_response).all()),
        "eas_parameter_count": trainable_parameter_count(model),
        "scalar_parameter_count": trainable_parameter_count(scalar),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    del model, scalar, optimizer, inputs, output
    torch.cuda.empty_cache()
    return result


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
        raise ValueError("E2 pilot must be build/train-only")
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": "worldsim_v72.e2_actor_visual_pilot.v1",
        "task_id": config["task_id"],
        "run_id": args.run_id,
        "status": "running",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "failure_ledger_refs": list(config["failure_ledger_refs"]),
        "selection_uses_quality": False,
        "supervision_access": False,
        "source_test_read": False,
        "external_test_read": False,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "index"})
    try:
        roles = load_data_roles(Path(data["roles"]))
        log_ids = require_role_access(roles, "nuscenes", str(data["role"]))
        index = NuScenesCameraIndex(Path(data["dataset_root"]), metadata_version=str(data["metadata_version"]))
        window = _find_window(index, log_ids, data)
        actor_paths = sorted(Path(data["actor_corpus_scene"]).glob("*.npz"))
        track_ids = [path.stem for path in actor_paths]
        poses = load_actor_poses_at_sample(index.metadata_root, window.window_id, track_ids)
        if not poses:
            raise RuntimeError("no V7/V7.1 actor is present in the frozen E1 window")
        rows = []
        caches_by_backbone: dict[str, list[ActorVisualCache]] = {}
        for backbone in config["backbones"]:
            geometry_path = Path(config["e1_cache_root"]) / backbone / f"{window.window_id}.npz"
            geometry = load_backbone_geometry(geometry_path)
            if geometry.provenance.get("window_fingerprint") != window.fingerprint:
                raise ValueError(f"{backbone} cache/window fingerprint mismatch")
            caches_by_backbone[backbone] = []
            for actor_path in actor_paths:
                pose = poses.get(actor_path.stem)
                if pose is None:
                    continue
                actor = _actor_inputs(actor_path, list(data["selected_actor_fields"]))
                observation = observe_actor_candidates(actor["candidates"], pose.world_from_actor, window, geometry)
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
                output_path = run_dir / "cache" / backbone / f"{cache.track_id}.npz"
                save_actor_visual_cache(cache, output_path)
                caches_by_backbone[backbone].append(cache)
                rows.append(
                    {
                        "backbone": backbone,
                        "track_id": cache.track_id,
                        "candidate_count": len(cache.candidates_actor_m),
                        "observed_candidate_count": int(np.sum(cache.observation_count > 0)),
                        "camera_observation_count": int(np.sum(cache.observation_count)),
                        "cache_path": str(output_path),
                    }
                )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "cuda_adapter_smoke"})
        smoke = {name: _adapter_smoke(caches, config["adapter"]) for name, caches in caches_by_backbone.items()}
        if not all(row["finite_gradients"] and row["scalar_output_finite"] for row in smoke.values()):
            raise FloatingPointError("adapter smoke produced non-finite values")
        summary = {
            **manifest,
            "status": "done",
            "failure_ledger_delta": "none",
            "window_id": window.window_id,
            "window_fingerprint": window.fingerprint,
            "matched_actor_count": len(poses),
            "rows": rows,
            "adapter_smoke": smoke,
            "wall_seconds": time.monotonic() - started,
            "interpretation": "IO and optimization closure only; no quality claim and no label selection.",
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
