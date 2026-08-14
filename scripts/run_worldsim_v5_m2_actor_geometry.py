#!/usr/bin/env python3
"""在 one-actor/one-view request 上重做 V5 M2 G0 staged geometry。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.ndimage import binary_dilation
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v33.spatial_delta import atomic_save_npz
from motion_proj.worldsim_v5.geometry_repair import (
    GeometryRepairError,
    geometry_reference_confidence,
    staged_geometry_metrics,
)
from scripts.run_worldsim_v5_m2_staged_geometry_diagnostic import (
    build_all_rigid_erase_delta,
)
from scripts.run_worldsim_v5_m2_surface_ablation import _arm, _runtime
from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    inventory_files,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M2-GEOMETRY-FIRST-REPAIR-01"
SCHEMA_VERSION = "worldsim_v5_m2_actor_geometry_v1"
MODEL = "G0_ROBUST_PLANE"


class M2ActorGeometryError(RuntimeError):
    """per-actor request、几何输入或 formal provenance 漂移。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M2ActorGeometryError("M2 actor geometry config schema 漂移")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "per_actor_g0_staged_geometry"
        or payload["request_protocol"]["unit"] != "one_actor_one_view_one_hole"
        or payload["request_protocol"]["union_mask_for_geometry_forbidden"] is not True
        or payload["surface"]["active_model"] != MODEL
    ):
        raise M2ActorGeometryError("M2 actor geometry task/request/model 漂移")
    if payload["reference"]["independent_geometry_claim_allowed"] is not False:
        raise M2ActorGeometryError("model proxy 不得声明独立 GT")
    for name in (
        "validation_quality_read",
        "heldout_quality_read",
        "test_quality_read",
        "parameter_search_performed",
        "router_refit_performed",
    ):
        if payload["scope"].get(name) is not False:
            raise M2ActorGeometryError(f"M2 actor geometry restriction 漂移: {name}")
    return payload


def mechanism(rows: list[Mapping[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    done = [row for row in rows if row.get("status") == "done"]
    raw_values = [float(row["staged_metrics"]["raw_geometry_error"]["mae_m"]) for row in done]
    gaussian_deltas = [float(row["staged_metrics"]["gaussianization_delta_mae_m"]) for row in done]
    raw_fail = sum(value >= float(gate["raw_failure_mae_m"]) for value in raw_values)
    gaussian_fail = sum(
        value >= float(gate["gaussianization_primary_delta_mae_m"])
        for value in gaussian_deltas
    )
    count = len(done)
    raw_fraction = raw_fail / count if count else 0.0
    gaussian_fraction = gaussian_fail / count if count else 0.0
    median_raw = float(np.median(raw_values)) if raw_values else math.nan
    enough = count >= int(gate["minimum_evaluable_request_count"])
    g1_unlocked = (
        enough
        and raw_fraction >= float(gate["raw_failure_fraction_to_unlock_g1"])
        and median_raw >= float(gate["median_raw_failure_mae_m_to_unlock_g1"])
    )
    gaussian_primary = enough and gaussian_fraction >= float(
        gate["gaussianization_primary_fraction"]
    )
    if not enough:
        conclusion = "insufficient_per_actor_requests"
    elif gaussian_primary and g1_unlocked:
        conclusion = "per_actor_builder_and_gaussianization_both_fail"
    elif gaussian_primary:
        conclusion = "per_actor_gaussianization_primary"
    elif g1_unlocked:
        conclusion = "per_actor_g0_builder_primary_g1_unlocked"
    else:
        conclusion = "per_actor_g0_and_gaussianization_not_primary"
    return {
        "conclusion": conclusion,
        "evaluable_request_count": count,
        "raw_failure_count": raw_fail,
        "raw_failure_fraction": raw_fraction,
        "median_raw_mae_m": median_raw,
        "gaussianization_primary_count": gaussian_fail,
        "gaussianization_primary_fraction": gaussian_fraction,
        "g1_unlocked_for_next_development_run": g1_unlocked,
        "gaussianization_primary": gaussian_primary,
        "reference_scope": "model_derived_proxy_not_independent_ground_truth",
    }


def _terminal(
    run_dir: Path,
    *,
    status: str,
    source_head: str,
    summary_sha256: str | None,
    manifest_sha256: str | None,
    reason: str | None,
) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v5_m2_actor_geometry_status_v1",
            "task_id": TASK_ID,
            "task_status": "running",
            "status": status,
            "source_commit": source_head,
            "summary_sha256": summary_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": reason,
            "finished_at_utc": utc_now(),
        },
    )


def _load_requests(config: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    path = Path(config["inputs"]["actor_mask_manifest"]["path"])
    payload = json.loads(path.read_text())
    protocol = config["request_protocol"]
    if (
        payload.get("status") != "done"
        or payload.get("scene") != config["scene"]["name"]
        or payload.get("request_unit") != protocol["unit"]
        or payload.get("union_mask_for_geometry_forbidden") is not True
        or len(payload["masks"]) != int(protocol["expected_request_count"])
    ):
        raise M2ActorGeometryError("actor-mask manifest request contract 漂移")
    return path.parent.parent, list(payload["masks"])


def _mask(root: Path, request: Mapping[str, Any]) -> np.ndarray:
    path = root / request["mask"]["path"]
    verify_file(path, request["mask"]["sha256"])
    with np.load(path, allow_pickle=False) as payload:
        actor_id = int(payload["actor_id"].item())
        accepted = bool(int(payload["mask_quality_accepted"].item()))
        binary = payload["binary"].astype(bool)
    if actor_id != int(request["actor_id"]) or accepted is not bool(request["accepted"]):
        raise M2ActorGeometryError(f"actor mask scalar contract 漂移: {path}")
    return binary


def run(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved = write_resolved_config(run_dir, config)
    events: list[dict[str, Any]] = [
        {"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}
    ]
    write_events(run_dir, events)
    try:
        inputs = {
            name: verify_file(binding["path"], binding["sha256"])
            for name, binding in config["inputs"].items()
        }
        actor_summary = json.loads(Path(inputs["actor_mask_summary"]["path"]).read_text())
        if (
            actor_summary.get("status") != "done"
            or actor_summary.get("request_unit") != "one_actor_one_view_one_hole"
            or int(actor_summary.get("actor_mask_count", -1))
            != int(config["request_protocol"]["expected_request_count"])
        ):
            raise M2ActorGeometryError("r004 actor-mask summary binding 漂移")
        mask_root, requests = _load_requests(config)
        accepted_count = sum(bool(row["accepted"]) for row in requests)
        if (
            accepted_count != int(config["request_protocol"]["expected_accepted_mask_count"])
            or len(requests) - accepted_count
            != int(config["request_protocol"]["expected_rejected_mask_count"])
        ):
            raise M2ActorGeometryError("accepted/rejected request denominator 漂移")
        checkpoint = Path(inputs["formal_checkpoint"]["path"])
        checkpoint_before = sha256_file(checkpoint)
        if not torch.cuda.is_available():
            raise M2ActorGeometryError("M2 actor geometry 需要 CUDA")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        _, dataset, trainer = _runtime(config, device)
        from scripts.run_worldsim_v32_s2_3dgic import render_snapshot

        erase_delta = build_all_rigid_erase_delta(trainer.models["RigidNodes"])
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for request in requests:
            grouped[(int(request["frame"]), int(request["camera_id"]))].append(request)
        rows: list[dict[str, Any]] = []
        protocol = config["request_protocol"]
        for view_index, ((frame, camera), view_requests) in enumerate(sorted(grouped.items())):
            base = render_snapshot(
                trainer=trainer, dataset=dataset, frame=frame, camera_id=camera, device=device
            )
            reference = np.asarray(base["background_depth"], dtype=np.float32)
            for request in sorted(view_requests, key=lambda row: int(row["actor_id"])):
                actor_id = int(request["actor_id"])
                row: dict[str, Any] = {"frame": frame, "camera_id": camera, "actor_id": actor_id}
                if request["accepted"] is not True:
                    row.update({"status": "abstain", "reason": "ABSTAIN_SAM_MASK_REJECTED"})
                    rows.append(row)
                    continue
                try:
                    target = binary_dilation(
                        _mask(mask_root, request),
                        iterations=int(protocol["target_mask_dilation_pixels"]),
                    )
                    inner = binary_dilation(
                        target, iterations=int(protocol["support_ring_inner_pixels"])
                    )
                    outer = binary_dilation(
                        target, iterations=int(protocol["support_ring_outer_pixels"])
                    )
                    support = outer & ~inner & ~np.asarray(base["dynamic_mask"], bool)
                    inpaint = cv2.inpaint(
                        cv2.cvtColor(np.asarray(base["groundtruth"], np.uint8), cv2.COLOR_RGB2BGR),
                        target.astype(np.uint8) * 255,
                        3.0,
                        cv2.INPAINT_TELEA,
                    )
                    inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
                    state = _arm(
                        model=MODEL,
                        trainer=trainer,
                        dataset=dataset,
                        device=device,
                        frame=frame,
                        camera=camera,
                        base=base,
                        target=target,
                        support=support,
                        inpaint=inpaint,
                        erase_delta=erase_delta,
                        config=config,
                        candidate_suffix=f"a{actor_id:03d}",
                    )
                    common = (
                        target
                        & np.isfinite(reference)
                        & (reference > 1e-4)
                        & np.isfinite(state["raw"])
                        & np.isfinite(state["pre"])
                        & np.isfinite(state["post"])
                        & (state["post"] > 1e-4)
                    )
                    metrics = staged_geometry_metrics(
                        raw_surface_depth=state["raw"],
                        pre_gaussianization_depth=state["pre"],
                        post_gaussianization_render_depth=state["post"],
                        reference_depth=reference,
                        evaluation_mask=common,
                    )
                    lidar = np.asarray(base["measured_lidar_depth"], dtype=np.float32)
                    lidar_valid = support & np.isfinite(lidar) & (lidar > 1e-4) & np.isfinite(reference) & (reference > 1e-4)
                    lidar_count = int(lidar_valid.sum())
                    lidar_mae = float(np.mean(np.abs(lidar[lidar_valid] - reference[lidar_valid]))) if lidar_count else None
                    confidence = geometry_reference_confidence(
                        observed_reference_pixels=lidar_count,
                        target_pixels=int(target.sum()),
                        lidar_agreement_mae_m=lidar_mae,
                        agreement_scale_m=float(config["reference"]["confidence"]["agreement_scale_m"]),
                    )
                    artifact = run_dir / "artifacts/requests" / f"f{frame:03d}_c{camera}_a{actor_id:03d}.npz"
                    atomic_save_npz(
                        artifact,
                        {
                            "target_mask": target.astype(np.int8),
                            "support_mask": support.astype(np.int8),
                            "common_evaluation_mask": common.astype(np.int8),
                            "reference_depth": reference.astype(np.float16),
                            "raw_surface_depth": state["raw"].astype(np.float16),
                            "pre_gaussianization_depth": state["pre"].astype(np.float16),
                            "post_gaussianization_render_depth": state["post"].astype(np.float16),
                        },
                    )
                    row.update(
                        {
                            "status": "done",
                            "target_pixels": int(target.sum()),
                            "common_evaluation_pixels": int(common.sum()),
                            "reference": {
                                "source": config["reference"]["source"],
                                "confidence": confidence,
                                "observed_reference_pixels": lidar_count,
                                "lidar_agreement_mae_m": lidar_mae,
                                "independent_geometry_claim_allowed": False,
                            },
                            "surface_fit": state["fit"].audit(),
                            "staged_metrics": metrics,
                            "gaussian_count": state["gaussian_count"],
                            "composition_audit": state["composition_audit"],
                            "artifact": {
                                "path": str(artifact.relative_to(run_dir)),
                                "sha256": sha256_file(artifact),
                            },
                        }
                    )
                except GeometryRepairError as error:
                    row.update({"status": "abstain", "reason": str(error)})
                rows.append(row)
            print(
                f"M2 actor G0 view {view_index + 1}/{len(grouped)} frame={frame} camera={camera} requests={len(view_requests)}",
                flush=True,
            )
        decision = mechanism(rows, config["mechanism_gate"])
        diagnostics = {
            "schema_version": "worldsim_v5_m2_actor_geometry_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene": config["scene"]["name"],
            "request_unit": "one_actor_one_view_one_hole",
            "rows": rows,
            "mechanism": decision,
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "test_quality_read": False,
            "parameter_search_performed": False,
            "router_refit_performed": False,
        }
        diagnostics_path = run_dir / "artifacts/diagnostics.json"
        atomic_json(diagnostics_path, diagnostics)
        checkpoint_after = sha256_file(checkpoint)
        if checkpoint_after != checkpoint_before:
            raise M2ActorGeometryError("formal checkpoint 被 actor geometry 修改")
        snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/run_worldsim_v5_m2_actor_geometry.py",
                PROJECT / "scripts/run_worldsim_v5_m2_surface_ablation.py",
                PROJECT / "motion_proj/worldsim_v5/geometry_repair.py",
                PROJECT / "tests/test_run_worldsim_v5_m2_actor_geometry.py",
            ],
            PROJECT,
        )
        summary = {
            "schema_version": "worldsim_v5_m2_actor_geometry_summary_v1",
            "task_id": TASK_ID,
            "task_status": "running",
            "status": "done",
            "phase": config["phase"],
            "scene": config["scene"]["name"],
            "source_commit": source_head,
            "conclusion": decision["conclusion"],
            "request_unit": "one_actor_one_view_one_hole",
            "request_count": len(rows),
            "evaluable_request_count": decision["evaluable_request_count"],
            "abstain_request_count": len(rows) - decision["evaluable_request_count"],
            "mechanism": decision,
            "reference_scope": "model_derived_proxy_not_independent_ground_truth",
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "duration_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": int(torch.cuda.max_memory_allocated(device) / 1024**2),
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "test_quality_read": False,
            "parameter_search_performed": False,
            "router_refit_performed": False,
        }
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v5_m2_actor_geometry_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": source_head,
                "source_clean": True,
                "resolved_config": resolved,
                "inputs": inputs,
                "runtime": {
                    "drivestudio_commit": config["runtime"]["drivestudio_commit"],
                    "drivestudio_status": config["runtime"]["drivestudio_expected_status"],
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device),
                },
                "source_snapshot": snapshot,
            },
        )
        events.append({"event": "run_done", "at_utc": utc_now(), **decision})
        write_events(run_dir, events)
        manifest = {
            "schema_version": "worldsim_v5_m2_actor_geometry_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
        }
        manifest_path = run_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        _terminal(
            run_dir,
            status="done",
            source_head=source_head,
            summary_sha256=sha256_file(summary_path),
            manifest_sha256=sha256_file(manifest_path),
            reason=None,
        )
        return summary
    except Exception as error:
        events.append({"event": "run_blocked", "at_utc": utc_now(), "reason": f"{type(error).__name__}: {error}"})
        write_events(run_dir, events)
        _terminal(
            run_dir,
            status="blocked",
            source_head=source_head,
            summary_sha256=None,
            manifest_sha256=None,
            reason=f"{type(error).__name__}: {error}",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_dir.resolve(), args.device), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
