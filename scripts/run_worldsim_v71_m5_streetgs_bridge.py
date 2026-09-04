"""建立只读StreetGS appearance与M5 physical surface的Actor identity bridge。"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m0_ray_displacement as m0_runner
import run_worldsim_v71_m5_pcgrad_relocation as m5_runner
from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.evaluate_surface import evaluate_actor_surface, summarize_surface_rows
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceRelocationMLP


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading_read_only_assets"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("StreetGS bridge requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        checkpoint = torch.load(
            Path(config["m5_run"]) / "MODEL.pt", map_location=device, weights_only=False
        )
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        model = RaySurfaceRelocationMLP(
            int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        streetgs = torch.load(
            Path(config["streetgs_checkpoint"]), map_location="cpu", weights_only=False
        )["models"]["RigidNodes"]
        gaussian_means = streetgs["_means"].detach().cpu().numpy()
        gaussian_actor_ids = streetgs["points_ids"].detach().cpu().numpy().reshape(-1)
        registry = json.loads(Path(config["actor_registry"]).read_text(encoding="utf-8"))
        registry_by_token = {
            str(actor["instance_token"]): actor
            for actor in registry["actors"]
            if actor.get("availability") == "available"
        }

        rows: list[dict[str, Any]] = []
        surface_parts: list[np.ndarray] = []
        offsets = [0]
        actor_tokens: list[str] = []
        cache_dir = Path(config["actor_cache_dir"])
        with torch.inference_mode():
            for path in sorted(cache_dir.glob("*.npz")):
                actor_token = path.stem
                registry_actor = registry_by_token.get(actor_token)
                if registry_actor is None:
                    continue
                actor = m0_runner._prepare_actor(path, standardizer, device)
                if actor is None:
                    continue
                rigid_index = int(registry_actor["rigid_model_index"])
                owned = gaussian_means[gaussian_actor_ids == rigid_index]
                if len(owned) == 0:
                    continue
                _, moved = m5_runner._move(model, actor, config["model"])
                baseline = _voxel_unique(
                    np.concatenate([actor["anchors"], actor["candidates"]], axis=0),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                output = _voxel_unique(
                    torch.cat([actor["anchors_t"], moved], dim=0).cpu().numpy(),
                    float(config["evaluation"]["output_voxel_size_m"]),
                )
                row = evaluate_actor_surface(
                    baseline,
                    output,
                    actor["target"],
                    actor["target_sensor_origins"],
                    hazardous=bool(actor["hazardous"]),
                    device=device,
                    lateral_tolerance_m=float(config["evaluation"]["literal_lateral_tolerance_m"]),
                    depth_tolerance_m=float(config["evaluation"]["literal_depth_tolerance_m"]),
                    distance_chunk_size=int(config["evaluation"]["distance_chunk_size"]),
                )
                displacement = torch.linalg.vector_norm(
                    moved - actor["candidates_t"], dim=1
                )
                row.update(
                    {
                        "scene_name": config["scene_name"],
                        "actor_token": actor_token,
                        "rigid_model_index": rigid_index,
                        "appearance_gaussian_count": int(len(owned)),
                        "appearance_gaussian_extent_min": owned.min(axis=0).tolist(),
                        "appearance_gaussian_extent_max": owned.max(axis=0).tolist(),
                        "baseline_surface_points": int(len(baseline)),
                        "physical_surface_points": int(len(output)),
                        "mean_surface_displacement_m": float(displacement.mean()),
                        "maximum_surface_displacement_m": float(displacement.max()),
                        "appearance_checkpoint_mutated": False,
                        "actor_trajectory_mutated": False,
                    }
                )
                rows.append(row)
                surface_parts.append(output)
                offsets.append(offsets[-1] + len(output))
                actor_tokens.append(actor_token)
        if not rows:
            raise RuntimeError("no identity-matched StreetGS/V7.1 Actors")
        metrics = summarize_surface_rows(rows)
        output_changed = any(float(row["mean_surface_displacement_m"]) > 1.0e-6 for row in rows)
        hazard_present = int(metrics["hazard"]["actor_count"]) > 0
        early_reduced = (
            metrics["hazard"]["relative_early_reduction"] is not None
            and float(metrics["hazard"]["relative_early_reduction"]) > 0.0
        )
        decisions = {
            "identity_matched_appearance_gaussians": all(
                int(row["appearance_gaussian_count"]) > 0 for row in rows
            ),
            "physical_surface_changed": output_changed,
            "hazard_actor_present_and_retained": hazard_present
            and float(metrics["minimum_hazard_state_retention"]) == 1.0,
            "hazard_literal_early_reduced": early_reduced,
            "appearance_and_trajectory_read_only": True,
        }
        passed = all(decisions.values())
        _write_jsonl(run_dir / "ACTOR_BRIDGE_ROWS.jsonl", rows)
        np.savez_compressed(
            run_dir / "PHYSICAL_SURFACE_SIDECAR.npz",
            points=np.concatenate(surface_parts, axis=0).astype(np.float32),
            offsets=np.asarray(offsets, dtype=np.int64),
            actor_tokens=np.asarray(actor_tokens),
            hazardous=np.asarray([bool(row["hazardous"]) for row in rows]),
            rigid_model_indices=np.asarray(
                [int(row["rigid_model_index"]) for row in rows], dtype=np.int64
            ),
        )
        shutil.copy2(Path(config["rendered_asset"]), run_dir / "STREETGS_APPEARANCE.png")
        summary = {
            "schema_version": "worldsim_v71.m5_streetgs_appearance_bridge.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "identity_coupled_appearance_physics_supported"
            if passed
            else "identity_bridge_physical_direction_rejected",
            "scene_name": config["scene_name"],
            "matched_actor_count": len(rows),
            "matched_hazard_actor_count": int(metrics["hazard"]["actor_count"]),
            "appearance_gaussian_count": int(
                sum(int(row["appearance_gaussian_count"]) for row in rows)
            ),
            "streetgs_read_only_fields": [
                "_means",
                "_scales",
                "_quats",
                "_features_dc",
                "_features_rest",
                "_opacities",
                "instances_trans",
                "instances_quats",
            ],
            "streetgs_checkpoint_written": False,
            "m5_surface_sidecar_written": True,
            "rendered_asset_copied_read_only": True,
            "physical": metrics,
            "decisions": decisions,
            "selection_read": False,
            "source_final_read": False,
            "external_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "appearance_physical_bridge",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "appearance_physical_bridge", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
