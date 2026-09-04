"""Decompose anchor and completion-child optical contributions on frozen M35."""

from __future__ import annotations

import argparse
import json
import resource
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
import run_worldsim_v71_m7_gt_supervised_seed_expansion as m7_runner
import run_worldsim_v71_m20_decoder_free_gaussian_ray_energy as m20_runner
import run_worldsim_v71_m22_se3_dynamic_static_composition as m22_runner
import run_worldsim_v71_m34_producer_evidential_anchor_authority as m34_runner
import run_worldsim_v71_m35_transmittance_anchor_authority as m35_runner
from motion_proj.worldsim_v71.evidential_gaussian_authority import (
    EvidentialGaussianAuthority,
    occupied_masses,
)


ARMS = (
    "anchors_unit",
    "anchors_learned",
    "children_unit",
    "all_unit",
    "all_learned",
)


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


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stratum(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        rays = sum(int(row["ray_count"]) for row in selected)
        output: dict[str, Any] = {
            "actor_count": len(selected),
            "ray_count": rays,
            "baseline_early_rate": sum(
                int(row["baseline_early_count"]) for row in selected
            ) / rays,
            "baseline_hit_rate": sum(
                int(row["baseline_hit_count"]) for row in selected
            ) / rays,
        }
        for arm in ARMS:
            output[arm] = {
                "early_rate": sum(int(row[arm]["early_count"]) for row in selected) / rays,
                "hit_rate": sum(int(row[arm]["hit_count"]) for row in selected) / rays,
                "mean_no_return_probability": float(
                    np.mean([row[arm]["mean_no_return_probability"] for row in selected])
                ),
            }
        return output

    return {
        "all": stratum(rows),
        "hazard": stratum([row for row in rows if bool(row["hazardous"])]),
        "clear": stratum([row for row in rows if not bool(row["hazardous"])]),
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M36 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        surface, base, standardizer, surface_config, base_config = m22_runner._load_m8(
            config, device
        )
        paths = m0_runner._paths(
            Path(config["cache_root"]), int(config["maximum_training_actors"])
        )
        actors = [
            actor for path in paths
            if (actor := m0_runner._prepare_actor(path, standardizer, device)) is not None
        ]
        stride = int(config["holdout_stride"])
        holdout_actors = [actor for index, actor in enumerate(actors) if index % stride == 0]
        with torch.inference_mode():
            for actor in holdout_actors:
                _, centers = m5_runner._move(base, actor, base_config)
                actor["m5_centers_t"] = centers
                children, _, scales = m7_runner._predict(surface, actor, surface_config)
                actor["m8_children_t"] = children
                actor["m8_scales_t"] = scales
        for actor in holdout_actors:
            m34_runner._attach_frozen_authority_state(
                actor,
                Path(config["sidecar_root"]),
                float(config["anchor_scale_m"]),
                config["features"],
                device,
            )
        checkpoint = torch.load(
            Path(config["m35_run"]) / "MODEL.pt",
            map_location=device,
            weights_only=False,
        )
        authority = EvidentialGaussianAuthority(
            hidden_dim=int(checkpoint["hidden_dim"]),
            input_dim=int(checkpoint["input_dim"]),
        ).to(device)
        authority.load_state_dict(checkpoint["state_dict"])
        authority.eval()
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for index, actor in enumerate(holdout_actors):
                baseline = m20_runner._energy_partition(
                    actor, config["evaluation"], float(config["anchor_scale_m"]), device
                )
                logits = authority(actor["authority_anchor_features_t"])
                learned_anchor = occupied_masses(logits)
                anchor_count = len(actor["anchors_t"])
                child_count = len(actor["m8_children_t"])
                zeros_anchor = torch.zeros(anchor_count, dtype=torch.float32, device=device)
                zeros_child = torch.zeros(child_count, dtype=torch.float32, device=device)
                ones_anchor = torch.ones_like(zeros_anchor)
                ones_child = torch.ones_like(zeros_child)
                arm_occupied = {
                    "anchors_unit": torch.cat([ones_anchor, zeros_child]),
                    "anchors_learned": torch.cat([learned_anchor, zeros_child]),
                    "children_unit": torch.cat([zeros_anchor, ones_child]),
                    "all_unit": torch.cat([ones_anchor, ones_child]),
                    "all_learned": torch.cat([learned_anchor, ones_child]),
                }
                arm_rows = {}
                for arm, occupied in arm_occupied.items():
                    partition = m35_runner._transmittance_partition(
                        actor, occupied, config["evaluation"]
                    )
                    arm_rows[arm] = {
                        "early_count": int(np.count_nonzero(partition["early"])),
                        "hit_count": int(np.count_nonzero(partition["hit"])),
                        "mean_no_return_probability": partition[
                            "mean_no_return_probability"
                        ],
                    }
                rows.append(
                    {
                        "scene_name": m34_runner._scalar_text(actor["scene_name"]),
                        "track_id": m34_runner._scalar_text(actor["track_id"]),
                        "hazardous": bool(actor["hazardous"]),
                        "ray_count": int(len(actor["target"])),
                        "anchor_count": anchor_count,
                        "child_count": child_count,
                        "baseline_early_count": int(np.count_nonzero(baseline["early"])),
                        "baseline_hit_count": int(np.count_nonzero(baseline["hit"])),
                        **arm_rows,
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(holdout_actors):
                    print(
                        json.dumps(
                            {"stage": "m36_optical_decomposition", "progress": f"{index + 1}/{len(holdout_actors)}"}
                        ),
                        flush=True,
                    )
        metrics = _summarize(rows)
        summary = {
            "schema_version": config["schema_version"],
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "optical_contribution_decomposed",
            "holdout_actor_count": len(holdout_actors),
            "metrics": metrics,
            "training": False,
            "checkpoint_written": False,
            "geometry_centers_and_scales_frozen": True,
            "m35_model_frozen": True,
            "pretrained_holdout_exposure": True,
            "external_read": False,
            "m21_partial_quality_read": False,
            "resources": {
                "device": str(device),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_jsonl(run_dir / "OPTICAL_CONTRIBUTION_ROWS.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "decomposition",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m36", "error": f"{type(error).__name__}: {error}"},
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
