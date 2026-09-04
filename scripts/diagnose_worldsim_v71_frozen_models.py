"""V7.1 冻结模型的训练—部署一致性诊断；不训练、不读取终测。"""

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
import run_worldsim_v71_m1_evidential_surface_field as m1_runner
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.actor_corpus import load_actor_cache
from motion_proj.worldsim_v71.actor_surface_field import EvidentialActorSurfaceField
from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene
from motion_proj.worldsim_v71.evaluate_surface import summarize_surface_rows
from motion_proj.worldsim_v71.ray_displacement import RaySurfaceDisplacementMLP


RUNS_ROOT = Path("/root/autodl-tmp/runs/worldsim_v71")


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


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resources(device: torch.device, started: float) -> dict[str, Any]:
    return {
        "device": str(device),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "wall_seconds": time.monotonic() - started,
    }


def _create_run(task_id: str, run_id: str, phase: str) -> Path:
    run_dir = RUNS_ROOT / task_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": phase})
    return run_dir


def diagnose_m0_no_unknown_mask(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_yaml(args.m0_config)
    source_run = Path(args.m0_run)
    run_dir = _create_run("WS-V71-DIAG-A-M0-NO-UNKNOWN-MASK-01", args.run_id, "frozen_replay")
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "mode": "m0_no_unknown_mask",
                "source_config": str(args.m0_config),
                "source_run": str(source_run),
                "selection_role": "consumed_selection",
                "unknown_mask": "disabled_by_accepting_all_probabilities",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        checkpoint = torch.load(source_run / "MODEL.pt", map_location=device, weights_only=False)
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        model = RaySurfaceDisplacementMLP(
            int(checkpoint["input_dim"]), int(config["model"]["hidden_dim"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        split = json.loads((REPO_ROOT / config["source_split"]).read_text(encoding="utf-8"))
        compiler = _load_yaml(REPO_ROOT / config["p2_config"])
        m0_runner._deep_update(compiler, config["compiler_overrides"])
        index = build_v71_index(Path(config["source"]["dataset_root"]), split)
        deployment = dict(config["model"])
        deployment["unknown_threshold"] = 1.01
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for position, scene_name in enumerate(split["roles"]["selection"]):
                bundles = compile_source_scene(
                    scene_name, index, config["actors"], compiler, device
                )
                for bundle in bundles:
                    row = m0_runner._evaluate_bundle(
                        bundle,
                        model,
                        standardizer,
                        deployment,
                        config["evaluation"],
                        device,
                    )
                    if row is not None:
                        row["unknown_mask_removed"] = True
                        rows.append(row)
                print(
                    json.dumps(
                        {
                            "stage": "diag_a_frozen_selection",
                            "progress": f"{position + 1}/{len(split['roles']['selection'])}",
                            "actors": len(rows),
                        }
                    ),
                    flush=True,
                )
        metrics = summarize_surface_rows(rows)
        source_summary = json.loads((source_run / "summary.json").read_text(encoding="utf-8"))
        summary = {
            "schema_version": "worldsim_v71.diagnostic.m0_no_unknown_mask.v1",
            "status": "done",
            "verdict": "descriptive_frozen_replay_only",
            "source_checkpoint": str(source_run / "MODEL.pt"),
            "selection_actor_count": len(rows),
            "original_m0_source_selection": source_summary["source_selection"],
            "no_unknown_mask_source_selection": metrics,
            "interpretation_contract": {
                "training_changed": False,
                "weights_changed": False,
                "displacement_changed": False,
                "only_unknown_deployment_mask_removed": True,
            },
            "selection_read": True,
            "source_final_read": False,
            "external_read": False,
            "resources": _resources(device, started),
        }
        _write_jsonl(run_dir / "SOURCE_SELECTION_ACTORS.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "frozen_replay",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {"run_dir": str(run_dir), "verdict": summary["verdict"]}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "frozen_replay", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def _quantiles(values: list[np.ndarray]) -> dict[str, float]:
    joined = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    if not len(joined):
        return {}
    return {
        "minimum": float(np.min(joined)),
        "p01": float(np.quantile(joined, 0.01)),
        "p05": float(np.quantile(joined, 0.05)),
        "median": float(np.quantile(joined, 0.50)),
        "p95": float(np.quantile(joined, 0.95)),
    }


def _decode_raw(
    model: EvidentialActorSurfaceField,
    latent: torch.Tensor,
    query: torch.Tensor,
    size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = m1_runner._normalized(query, size)
    expanded_latent = latent.reshape(1, -1).expand(len(query), -1)
    expanded_size = size.reshape(1, 3).expand(len(query), -1)
    raw = model.decoder(torch.cat([expanded_latent, normalized, expanded_size], dim=-1))
    return torch.nn.functional.softplus(raw[:, :2]), raw[:, 2:], raw[:, :2]


def diagnose_m1_field_extraction(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_yaml(args.m1_config)
    source_run = Path(args.m1_run)
    run_dir = _create_run("WS-V71-DIAG-B-M1-FIELD-EXTRACTION-01", args.run_id, "train_field_decomposition")
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "mode": "m1_field_extraction",
                "source_config": str(args.m1_config),
                "source_run": str(source_run),
                "data_role": "train_only",
                "maximum_actors": int(args.maximum_actors),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        checkpoint = torch.load(source_run / "MODEL.pt", map_location=device, weights_only=False)
        standardizer = FeatureStandardizer.from_payload(checkpoint["standardizer"])
        model = EvidentialActorSurfaceField(
            int(checkpoint["evidence_dim"]),
            latent_dim=int(checkpoint["latent_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        cache_root = Path(config["cache_root"])
        paths = sorted(
            path
            for path in (cache_root / "train").glob("*/*.npz")
            if not path.name.endswith(".tmp.npz")
        )
        payloads = []
        for path in paths:
            payload = load_actor_cache(path)
            if len(payload["candidates"]) and len(payload["target"]) and len(payload["anchors"]):
                payloads.append(payload)
            if len(payloads) >= int(args.maximum_actors):
                break

        totals = {
            "grid_points": 0,
            "band_only_points": 0,
            "occupied_only_points": 0,
            "extracted_points": 0,
            "oracle_band_points": 0,
            "surface_probe_points": 0,
            "surface_probe_band_only_points": 0,
            "surface_probe_occupied_only_points": 0,
            "surface_probe_extracted_points": 0,
        }
        samples: dict[str, list[np.ndarray]] = {
            "planar_scf": [],
            "vertical_scf": [],
            "planar_raw": [],
            "vertical_raw": [],
            "occupied_probability": [],
            "surface_planar_scf": [],
            "surface_vertical_scf": [],
            "surface_occupied_probability": [],
        }
        class_counts = np.zeros(3, dtype=np.int64)
        rows: list[dict[str, Any]] = []
        extraction = config["extraction"]
        with torch.inference_mode():
            for position, payload in enumerate(payloads):
                actor = m1_runner._actor_tensors(payload, standardizer, config["model"], device)
                latent = model.encode(actor["evidence_t"])
                grid = torch.as_tensor(
                    m1_runner._grid(payload["size_lwh_m"], extraction),
                    dtype=torch.float32,
                    device=device,
                )
                scf_parts, logit_parts, raw_parts = [], [], []
                for start in range(0, len(grid), int(extraction["extraction_chunk_size"])):
                    scf, logits, raw = _decode_raw(
                        model,
                        latent,
                        grid[start : start + int(extraction["extraction_chunk_size"])],
                        actor["size_t"],
                    )
                    scf_parts.append(scf)
                    logit_parts.append(logits)
                    raw_parts.append(raw)
                scf = torch.cat(scf_parts)
                logits = torch.cat(logit_parts)
                raw = torch.cat(raw_parts)
                probability = logits.softmax(dim=1)
                classes = logits.argmax(dim=1)
                band = (scf[:, 0] <= float(extraction["planar_band_m"])) & (
                    scf[:, 1] <= float(extraction["vertical_band_m"])
                )
                occupied = classes == 1

                reference = torch.cat(
                    [
                        actor["target_t"][: int(config["model"]["maximum_target_points"])],
                        actor["anchors_t"][: int(config["model"]["maximum_anchor_points"])],
                    ]
                )
                oracle_parts = []
                for start in range(0, len(grid), 2048):
                    oracle_parts.append(m1_runner._nearest_scf(grid[start : start + 2048], reference))
                oracle_scf = torch.cat(oracle_parts)
                oracle_band = (oracle_scf[:, 0] <= float(extraction["planar_band_m"])) & (
                    oracle_scf[:, 1] <= float(extraction["vertical_band_m"])
                )

                probe = reference
                probe_scf, probe_logits, _ = _decode_raw(model, latent, probe, actor["size_t"])
                probe_probability = probe_logits.softmax(dim=1)
                probe_band = (probe_scf[:, 0] <= float(extraction["planar_band_m"])) & (
                    probe_scf[:, 1] <= float(extraction["vertical_band_m"])
                )
                probe_occupied = probe_logits.argmax(dim=1) == 1

                actor_counts = {
                    "grid_points": len(grid),
                    "band_only_points": int(torch.count_nonzero(band)),
                    "occupied_only_points": int(torch.count_nonzero(occupied)),
                    "extracted_points": int(torch.count_nonzero(band & occupied)),
                    "oracle_band_points": int(torch.count_nonzero(oracle_band)),
                    "surface_probe_points": len(probe),
                    "surface_probe_band_only_points": int(torch.count_nonzero(probe_band)),
                    "surface_probe_occupied_only_points": int(torch.count_nonzero(probe_occupied)),
                    "surface_probe_extracted_points": int(torch.count_nonzero(probe_band & probe_occupied)),
                }
                for key, value in actor_counts.items():
                    totals[key] += int(value)
                class_counts += np.bincount(classes.cpu().numpy(), minlength=3)

                stride = max(len(grid) // 2048, 1)
                samples["planar_scf"].append(scf[::stride, 0].cpu().numpy())
                samples["vertical_scf"].append(scf[::stride, 1].cpu().numpy())
                samples["planar_raw"].append(raw[::stride, 0].cpu().numpy())
                samples["vertical_raw"].append(raw[::stride, 1].cpu().numpy())
                samples["occupied_probability"].append(probability[::stride, 1].cpu().numpy())
                samples["surface_planar_scf"].append(probe_scf[:, 0].cpu().numpy())
                samples["surface_vertical_scf"].append(probe_scf[:, 1].cpu().numpy())
                samples["surface_occupied_probability"].append(probe_probability[:, 1].cpu().numpy())
                rows.append({"actor_index": position, **actor_counts})
                print(
                    json.dumps(
                        {
                            "stage": "diag_b_train_field",
                            "progress": f"{position + 1}/{len(payloads)}",
                            "extracted": totals["extracted_points"],
                        }
                    ),
                    flush=True,
                )

        summary = {
            "schema_version": "worldsim_v71.diagnostic.m1_field_extraction.v1",
            "status": "done",
            "verdict": "descriptive_train_field_decomposition_only",
            "source_checkpoint": str(source_run / "MODEL.pt"),
            "actor_count": len(payloads),
            "counts": totals,
            "grid_class_fraction": {
                "free": float(class_counts[0] / max(class_counts.sum(), 1)),
                "occupied": float(class_counts[1] / max(class_counts.sum(), 1)),
                "unknown": float(class_counts[2] / max(class_counts.sum(), 1)),
            },
            "quantiles": {key: _quantiles(value) for key, value in samples.items()},
            "interpretation_contract": {
                "training_changed": False,
                "weights_changed": False,
                "train_only": True,
                "band_gate_and_evidence_gate_reported_separately": True,
                "oracle_grid_band_reported": True,
            },
            "selection_read": False,
            "source_final_read": False,
            "external_read": False,
            "resources": _resources(device, started),
        }
        _write_jsonl(run_dir / "TRAIN_ACTORS.jsonl", rows)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {
                "status": "done",
                "phase": "train_field_decomposition",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {"run_dir": str(run_dir), "verdict": summary["verdict"]}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "train_field_decomposition", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("m0_no_unknown_mask", "m1_field_extraction"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--m0-config",
        type=Path,
        default=REPO_ROOT / "configs/worldsim_v71/v71_m0_ray_displacement_v1.yaml",
    )
    parser.add_argument(
        "--m0-run",
        type=Path,
        default=Path("/root/autodl-tmp/runs/worldsim_v71/WS-V71-M0-RAY-SURFACE-DISPLACEMENT-01/20260903T111000Z__m0-ray-displacement-s71101-r3"),
    )
    parser.add_argument(
        "--m1-config",
        type=Path,
        default=REPO_ROOT / "configs/worldsim_v71/v71_m1_evidential_surface_field_v1.yaml",
    )
    parser.add_argument(
        "--m1-run",
        type=Path,
        default=Path("/root/autodl-tmp/runs/worldsim_v71/WS-V71-M1-EVIDENTIAL-SURFACE-FIELD-01/20260903T113000Z__m1-evidential-field-s71102-r1"),
    )
    parser.add_argument("--maximum-actors", type=int, default=64)
    args = parser.parse_args()
    if args.mode == "m0_no_unknown_mask":
        result = diagnose_m0_no_unknown_mask(args)
    else:
        result = diagnose_m1_field_extraction(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
