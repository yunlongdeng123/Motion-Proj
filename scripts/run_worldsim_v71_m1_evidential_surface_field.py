"""训练并按条件协议评估V7.1 M1 Actor-local evidential surface field。"""

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
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v7.av2_four_action_compiler import _voxel_unique
from motion_proj.worldsim_v7.completion_responsibility import FeatureStandardizer
from motion_proj.worldsim_v71.actor_corpus import load_actor_cache
from motion_proj.worldsim_v71.actor_surface_field import EvidentialActorSurfaceField
from motion_proj.worldsim_v71.dataset_nuscenes import build_v71_index, compile_source_scene
from motion_proj.worldsim_v71.evaluate_surface import evaluate_actor_surface, summarize_surface_rows
from motion_proj.worldsim_v71.evidence_volume import build_evidential_queries
from motion_proj.worldsim_v71.surface_extract import extract_zero_crossing_surface, merge_hard_anchors


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _limit(values: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(values)
    if len(values) <= int(maximum):
        return values
    indices = np.linspace(0, len(values) - 1, num=int(maximum), dtype=np.int64)
    return values[indices]


def _limit_paired(
    first: np.ndarray, second: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(first) <= int(maximum):
        return first, second
    indices = np.linspace(0, len(first) - 1, num=int(maximum), dtype=np.int64)
    return first[indices], second[indices]


def _evidence_features(payload: Mapping[str, Any], maximum_points: int) -> np.ndarray:
    size = np.maximum(np.asarray(payload["size_lwh_m"], dtype=np.float32) * 0.5, 1.0e-3)
    candidates = np.asarray(payload["candidates"], dtype=np.float32).reshape(-1, 3)
    base = np.asarray(payload["base_features"], dtype=np.float32).reshape(len(candidates), -1)
    masses = np.asarray(payload["evidence_masses"], dtype=np.float32).reshape(len(candidates), 3)
    candidate_rows = np.concatenate([base, candidates / size[None, :], masses], axis=1)

    stable = _limit(
        np.concatenate(
            [
                np.asarray(payload["canonical"], dtype=np.float32).reshape(-1, 3),
                np.asarray(payload["anchors"], dtype=np.float32).reshape(-1, 3),
            ],
            axis=0,
        ),
        max(int(maximum_points) - len(candidate_rows), 1),
    )
    stable_base = np.zeros((len(stable), base.shape[1]), dtype=np.float32)
    stable_masses = np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (len(stable), 1))
    stable_rows = np.concatenate([stable_base, stable / size[None, :], stable_masses], axis=1)
    return _limit(np.concatenate([candidate_rows, stable_rows], axis=0), int(maximum_points))


def _selection_payload(
    bundle: Mapping[str, Any], device: torch.device, config: Mapping[str, Any]
) -> dict[str, Any]:
    diagnostics = bundle["diagnostics"]
    candidates = np.asarray(diagnostics["completion_candidates"], dtype=np.float32).reshape(-1, 3)
    build_points = np.concatenate(
        [np.asarray(value, dtype=np.float32).reshape(-1, 3) for value in diagnostics["build_frame_points"]],
        axis=0,
    )
    build_origins = np.concatenate(
        [
            np.repeat(np.asarray(origin, dtype=np.float32).reshape(1, 3), len(points), axis=0)
            for points, origin in zip(
                diagnostics["build_frame_points"], diagnostics["build_sensor_origins"]
            )
        ],
        axis=0,
    )
    build_points, build_origins = _limit_paired(
        build_points, build_origins, int(config["maximum_build_evidence_points"])
    )
    masses = build_evidential_queries(
        candidates,
        build_origins,
        build_points,
        beam_radius_m=float(config["beam_radius_m"]),
        endpoint_radius_m=float(config["endpoint_radius_m"]),
        device=device,
        query_chunk_size=int(config["query_chunk_size"]),
    ).masses
    return {
        "track_id": str(bundle["row"]["track_id"]),
        "hazardous": bool(bundle["row"]["hazardous"]),
        "size_lwh_m": np.asarray(diagnostics["track"].size_lwh_m, dtype=np.float32),
        "canonical": np.asarray(diagnostics["canonical"], dtype=np.float32),
        "anchors": np.concatenate(
            [
                np.asarray(diagnostics["kept"], dtype=np.float32).reshape(-1, 3),
                np.asarray(diagnostics["projected"], dtype=np.float32).reshape(-1, 3),
            ],
            axis=0,
        ),
        "candidates": candidates,
        "base_features": np.asarray(diagnostics["completion_features"], dtype=np.float32),
        "evidence_masses": masses,
        "target": np.asarray(diagnostics["target"], dtype=np.float32),
        "target_sensor_origins": np.asarray(diagnostics["target_sensor_origins"], dtype=np.float32),
        "baseline_surface": np.asarray(diagnostics["compiled"], dtype=np.float32),
    }


def _actor_tensors(
    payload: Mapping[str, Any],
    standardizer: FeatureStandardizer,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    features = _evidence_features(payload, int(config["maximum_evidence_points"]))
    return {
        **payload,
        "evidence_t": torch.as_tensor(
            standardizer.transform(features), dtype=torch.float32, device=device
        ),
        "size_t": torch.as_tensor(payload["size_lwh_m"], dtype=torch.float32, device=device),
        "target_t": torch.as_tensor(payload["target"], dtype=torch.float32, device=device),
        "target_origins_t": torch.as_tensor(
            payload["target_sensor_origins"], dtype=torch.float32, device=device
        ),
        "anchors_t": torch.as_tensor(payload["anchors"], dtype=torch.float32, device=device),
    }


def _nearest_scf(query: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
    nearest = torch.cdist(query, surface).argmin(dim=1)
    delta = query - surface[nearest]
    return torch.stack(
        [torch.linalg.vector_norm(delta[:, :2], dim=1), delta[:, 2].abs()], dim=1
    )


def _field_supervision(
    actor: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = actor["target_t"]
    origins = actor["target_origins_t"]
    target, origins = (
        (target, origins)
        if len(target) <= int(config["maximum_target_points"])
        else (
            target[torch.linspace(0, len(target) - 1, steps=int(config["maximum_target_points"]), device=target.device).long()],
            origins[torch.linspace(0, len(origins) - 1, steps=int(config["maximum_target_points"]), device=origins.device).long()],
        )
    )
    anchors = actor["anchors_t"]
    if len(anchors) > int(config["maximum_anchor_points"]):
        indices = torch.linspace(
            0, len(anchors) - 1, steps=int(config["maximum_anchor_points"]), device=anchors.device
        ).long()
        anchors = anchors[indices]
    occupied = torch.cat([target, anchors], dim=0)
    ray_count = min(len(target), int(config["maximum_free_rays"]))
    selected = torch.linspace(0, len(target) - 1, steps=ray_count, device=target.device).long()
    selected_target = target[selected]
    selected_origins = origins[selected]
    direction = selected_target - selected_origins
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    offsets = torch.as_tensor(config["free_offsets_m"], dtype=torch.float32, device=target.device)
    free = selected_target[:, None, :] - direction[:, None, :] * offsets[None, :, None]
    free = free.reshape(-1, 3)
    half = actor["size_t"] * 0.5 + float(config["cuboid_padding_m"])
    free = free[torch.all(free.abs() <= half[None, :], dim=1)]
    unknown = (torch.rand(int(config["unknown_points"]), 3, device=target.device) * 2.0 - 1.0) * half[None, :]
    query = torch.cat([occupied, free, unknown], dim=0)
    labels = torch.cat(
        [
            torch.ones(len(occupied), dtype=torch.long, device=target.device),
            torch.zeros(len(free), dtype=torch.long, device=target.device),
            torch.full((len(unknown),), 2, dtype=torch.long, device=target.device),
        ]
    )
    reference = torch.cat([target, anchors], dim=0)
    scf = _nearest_scf(query, reference)
    return query, labels, scf


def _normalized(query: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    return query / (size.reshape(1, 3) * 0.5).clamp_min(1.0e-3)


def _geometry_loss(
    model: EvidentialActorSurfaceField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    query, labels, target_scf = _field_supervision(actor, config)
    predicted_scf, logits = model(actor["evidence_t"], _normalized(query, actor["size_t"]), actor["size_t"])
    evidence_loss = F.cross_entropy(logits, labels)
    scale = torch.as_tensor(
        [float(config["planar_scf_scale_m"]), float(config["vertical_scf_scale_m"])],
        dtype=torch.float32,
        device=query.device,
    )
    scf_loss = F.smooth_l1_loss(predicted_scf / scale[None, :], target_scf / scale[None, :])
    total = float(config["evidence_loss_weight"]) * evidence_loss + float(config["scf_loss_weight"]) * scf_loss
    return total, {"evidence": evidence_loss, "scf": scf_loss}


def _first_return_loss(
    model: EvidentialActorSurfaceField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> torch.Tensor:
    targets = actor["target_t"]
    origins = actor["target_origins_t"]
    count = min(len(targets), int(config["maximum_render_rays"]))
    selected = torch.linspace(0, len(targets) - 1, steps=count, device=targets.device).long()
    targets = targets[selected]
    origins = origins[selected]
    direction = targets - origins
    target_depth = torch.linalg.vector_norm(direction, dim=1)
    direction /= target_depth[:, None].clamp_min(1.0e-6)
    offsets = torch.linspace(
        -float(config["render_back_m"]),
        float(config["render_front_m"]),
        steps=int(config["render_samples"]),
        device=targets.device,
    )
    depths = (target_depth[:, None] + offsets[None, :]).clamp_min(0.05)
    query = origins[:, None, :] + direction[:, None, :] * depths[:, :, None]
    latent = model.encode(actor["evidence_t"])
    scf, logits = model.decode(
        latent,
        _normalized(query.reshape(-1, 3), actor["size_t"]),
        actor["size_t"],
    )
    scf = scf.reshape(count, -1, 2)
    occupied = logits.softmax(dim=1)[:, 1].reshape(count, -1)
    surface = torch.exp(
        -scf[:, :, 0] / float(config["render_planar_sigma_m"])
        -scf[:, :, 1] / float(config["render_vertical_sigma_m"])
    )
    alpha = 1.0 - torch.exp(-float(config["render_density_scale"]) * occupied * surface)
    transmittance = torch.cumprod(
        torch.cat([torch.ones((count, 1), device=targets.device), 1.0 - alpha + 1.0e-6], dim=1),
        dim=1,
    )
    weights = transmittance[:, :-1] * alpha
    fallback = target_depth + float(config["render_front_m"])
    predicted = (weights * depths).sum(dim=1) + transmittance[:, -1] * fallback
    return F.smooth_l1_loss(
        (predicted - target_depth) / float(config["first_return_scale_m"]),
        torch.zeros_like(target_depth),
    )


def _train_stage(
    model: EvidentialActorSurfaceField,
    actors: list[dict[str, Any]],
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    physical: bool,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    batch_size = int(config["actor_batch_size"])
    for epoch in range(int(epochs)):
        permutation = torch.randperm(len(actors)).tolist()
        total = 0.0
        for start in range(0, len(permutation), batch_size):
            losses = []
            for index in permutation[start : start + batch_size]:
                geometry, _ = _geometry_loss(model, actors[index], config)
                if physical:
                    geometry = geometry + float(config["first_return_loss_weight"]) * _first_return_loss(
                        model, actors[index], config
                    )
                losses.append(geometry)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(losses)
        mean_loss = total / len(actors)
        if epoch in {0, int(epochs) - 1}:
            history.append({"epoch": epoch + 1, "mean_loss": mean_loss})
        print(
            json.dumps(
                {
                    "stage": "m1_physical" if physical else "m1_geometry",
                    "epoch": epoch + 1,
                    "actors": len(actors),
                    "loss": mean_loss,
                }
            ),
            flush=True,
        )
    return history


def _grid(size: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    half = np.asarray(size, dtype=np.float32) * 0.5 + float(config["cuboid_padding_m"])
    voxel = float(config["extraction_voxel_size_m"])
    axes = [np.arange(-value + 0.5 * voxel, value, voxel, dtype=np.float32) for value in half]
    count = int(np.prod([len(axis) for axis in axes]))
    if count > int(config["maximum_extraction_points"]):
        voxel *= (count / float(config["maximum_extraction_points"])) ** (1.0 / 3.0)
        axes = [np.arange(-value + 0.5 * voxel, value, voxel, dtype=np.float32) for value in half]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _extract_surface(
    model: EvidentialActorSurfaceField,
    actor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    query = torch.as_tensor(_grid(actor["size_lwh_m"], config), dtype=torch.float32, device=actor["size_t"].device)
    latent = model.encode(actor["evidence_t"])
    scf_parts, logit_parts = [], []
    for start in range(0, len(query), int(config["extraction_chunk_size"])):
        scf, logits = model.decode(
            latent,
            _normalized(query[start : start + int(config["extraction_chunk_size"])], actor["size_t"]),
            actor["size_t"],
        )
        scf_parts.append(scf)
        logit_parts.append(logits)
    scf = torch.cat(scf_parts, dim=0)
    logits = torch.cat(logit_parts, dim=0)
    extracted = extract_zero_crossing_surface(
        query,
        scf,
        logits,
        planar_band_m=float(config["planar_band_m"]),
        vertical_band_m=float(config["vertical_band_m"]),
    )
    merged = merge_hard_anchors(actor["anchors_t"], extracted).cpu().numpy()
    return _voxel_unique(merged, float(config["output_voxel_size_m"])), len(extracted)


def _decisions(summary: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    hazard_reduction = summary["hazard"]["relative_early_reduction"]
    return {
        "hazard_literal_first_return_relative_reduction": hazard_reduction is not None
        and float(hazard_reduction) >= float(config["minimum_hazard_literal_relative_reduction"]),
        "chamfer_non_degradation": float(summary["chamfer_delta_m"])
        <= float(config["maximum_chamfer_delta_m"]),
        "target_hit_recall": float(summary["hit_recall_delta"])
        >= float(config["minimum_hit_recall_delta"]),
        "actor_state_retention": float(summary["minimum_actor_state_retention"])
        == float(config["required_actor_state_retention"]),
        "hazard_state_retention": float(summary["minimum_hazard_state_retention"])
        == float(config["required_hazard_state_retention"]),
    }


def run(config_path: Path, repo_root: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = json.loads((repo_root / config["source_split"]).read_text(encoding="utf-8"))
    compiler = yaml.safe_load((repo_root / config["p2_config"]).read_text(encoding="utf-8"))
    _deep_update(compiler, config["compiler_overrides"])
    cache_root = Path(config["cache_root"])
    corpus_manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if corpus_manifest.get("verdict") != "train_corpus_target_met":
        raise RuntimeError("M1 requires the frozen 1000-Actor corpus")
    m0_summary = json.loads((Path(config["m0_run"]) / "summary.json").read_text(encoding="utf-8"))
    if m0_summary.get("verdict") != "m0_source_selection_rejected":
        raise RuntimeError("M1 conditional unlock requires rejected M0")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M1 requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["model"]["seed"]))
    np.random.seed(int(config["model"]["seed"]))

    run_dir = Path(config["runs_root"]) / "worldsim_v71" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "loading_train"})
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    started = time.monotonic()
    try:
        paths = sorted(
            path
            for path in (cache_root / "train").glob("*/*.npz")
            if not path.name.endswith(".tmp.npz")
        )[: int(config["model"]["maximum_training_actors"])]
        payloads = [load_actor_cache(path) for path in paths]
        payloads = [
            payload
            for payload in payloads
            if len(payload["candidates"]) and len(payload["target"]) and len(payload["anchors"])
        ]
        feature_arrays = [
            _evidence_features(payload, int(config["model"]["maximum_evidence_points"]))
            for payload in payloads
        ]
        standardizer = FeatureStandardizer.fit(np.concatenate(feature_arrays, axis=0))
        actors = [
            _actor_tensors(payload, standardizer, config["model"], device)
            for payload in payloads
        ]
        model = EvidentialActorSurfaceField(
            actors[0]["evidence_t"].shape[1],
            latent_dim=int(config["model"]["latent_dim"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "geometry_pretraining"})
        geometry_history = _train_stage(
            model,
            actors,
            config["model"],
            optimizer,
            epochs=int(config["model"]["geometry_epochs"]),
            physical=False,
        )
        _write_json(run_dir / "status.json", {"status": "running", "phase": "physical_finetuning"})
        physical_history = _train_stage(
            model,
            actors,
            config["model"],
            optimizer,
            epochs=int(config["model"]["physical_epochs"]),
            physical=True,
        )
        model.eval()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "standardizer": standardizer.payload(),
                "evidence_dim": int(actors[0]["evidence_t"].shape[1]),
                "latent_dim": int(config["model"]["latent_dim"]),
                "hidden_dim": int(config["model"]["hidden_dim"]),
                "seed": int(config["model"]["seed"]),
            },
            run_dir / "MODEL.pt",
        )

        _write_json(run_dir / "status.json", {"status": "running", "phase": "source_selection"})
        index = build_v71_index(Path(config["source"]["dataset_root"]), split)
        selection_rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for position, scene_name in enumerate(split["roles"]["selection"]):
                bundles = compile_source_scene(
                    scene_name, index, config["actors"], compiler, device
                )
                for bundle in bundles:
                    payload = _selection_payload(bundle, device, config["evidence"])
                    if not len(payload["candidates"]):
                        continue
                    actor = _actor_tensors(payload, standardizer, config["model"], device)
                    output, extracted_count = _extract_surface(model, actor, config["extraction"])
                    row = evaluate_actor_surface(
                        payload["baseline_surface"],
                        output,
                        payload["target"],
                        payload["target_sensor_origins"],
                        hazardous=bool(payload["hazardous"]),
                        device=device,
                        lateral_tolerance_m=float(config["evaluation"]["literal_lateral_tolerance_m"]),
                        depth_tolerance_m=float(config["evaluation"]["literal_depth_tolerance_m"]),
                        distance_chunk_size=int(config["evaluation"]["distance_chunk_size"]),
                    )
                    row.update(
                        {
                            "scene_name": scene_name,
                            "track_id": str(payload["track_id"]),
                            "extracted_field_points": int(extracted_count),
                        }
                    )
                    selection_rows.append(row)
                print(
                    json.dumps(
                        {
                            "stage": "m1_source_selection",
                            "progress": f"{position + 1}/{len(split['roles']['selection'])}",
                            "scene": scene_name,
                            "actors": len(selection_rows),
                        }
                    ),
                    flush=True,
                )
        metrics = summarize_surface_rows(selection_rows)
        decisions = _decisions(metrics, config["decision"])
        passed = all(decisions.values())
        _write_jsonl(run_dir / "SOURCE_SELECTION_ACTORS.jsonl", selection_rows)
        summary = {
            "schema_version": "worldsim_v71.m1_evidential_surface_field.v1",
            "task_id": config["task_id"],
            "hypothesis_id": config["hypothesis_id"],
            "status": "done",
            "verdict": "m1_source_selection_passed_frozen" if passed else "m1_source_selection_rejected",
            "training_actor_count": len(actors),
            "corpus_actor_count": int(corpus_manifest["actor_count"]),
            "selection_actor_count": len(selection_rows),
            "geometry_history": geometry_history,
            "physical_history": physical_history,
            "source_selection": metrics,
            "decisions": decisions,
            "m0_run": config["m0_run"],
            "selection_read": True,
            "source_final_read": False,
            "external_read": False,
            "failure_ledger_refs": config["failure_ledger_refs"],
            "failure_ledger_delta": "none" if passed else "failure-id-required-at-closeout",
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
                "phase": "source_selection",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {"run_dir": str(run_dir), "verdict": summary["verdict"], "decisions": decisions}
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "m1", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config.resolve(), args.repo_root.resolve(), args.run_id),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
