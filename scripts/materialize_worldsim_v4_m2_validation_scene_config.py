#!/usr/bin/env python3
"""Materialize one read-only nuScenes M2 validation scene from frozen M1 assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_worldsim_v4_m2_scene_config import (  # noqa: E402
    M2MaterializationError,
    _find_inventory,
    _json,
    _verified,
    _yaml,
    sha256_file,
)


TASK_ID = "WS-V4-M2-REPAIR-ROUTER-01"


def _binding(path: str | Path, expected: str, label: str) -> dict[str, Any]:
    return _verified(path, expected, label)


def _validate_root(config: Mapping[str, Any], scene: str) -> tuple[dict, dict, dict]:
    if (
        config.get("schema_version") != "worldsim_v4_m2_validation_v1"
        or config.get("task_id") != TASK_ID
        or config.get("status") != "pending"
        or config.get("partition") != "validation"
        or config.get("dataset") != "nuScenes"
    ):
        raise M2MaterializationError("M2 validation root contract drift")
    protocol = config["protocol"]
    if (
        scene not in protocol["scene_order"]
        or protocol.get("target_partition_within_validation_scene") != "development"
        or protocol.get("support_partition") != "train_only"
        or protocol.get("validation_optimization_forbidden") is not True
        or protocol.get("heldout_content_read") is not False
        or protocol.get("test_quality_read") is not False
    ):
        raise M2MaterializationError("M2 validation partition contract drift")
    candidate_binding = config["candidate_protocol"]
    candidate = _yaml(
        Path(
            _binding(
                candidate_binding["path"],
                candidate_binding["sha256"],
                "M2 candidate protocol",
            )["path"]
        )
    )
    freeze = config["development_freeze"]
    summary_binding = _binding(
        Path(freeze["run"]) / "summary.json",
        freeze["summary_sha256"],
        "M2 development freeze summary",
    )
    _binding(
        Path(freeze["run"]) / "manifest.json",
        freeze["manifest_sha256"],
        "M2 development freeze manifest",
    )
    summary = _json(Path(summary_binding["path"]))
    frozen = summary.get("frozen_router", {})
    if (
        summary.get("status") != "done"
        or summary.get("validation_authorized") is not True
        or summary.get("heldout_content_read") is not False
        or summary.get("test_quality_read") is not False
        or frozen.get("weight_name") != freeze["weight_name"]
        or frozen.get("weights") != freeze["weights"]
        or float(frozen.get("threshold")) != float(freeze["threshold"])
        or frozen.get("tie_priority") != freeze["tie_priority"]
    ):
        raise M2MaterializationError("M2 development freeze drift")
    m1 = config["m1_validation"]
    m1_summary_binding = _binding(
        Path(m1["run"]) / "summary.json",
        m1["summary_sha256"],
        "M1 validation summary",
    )
    m1_summary = _json(Path(m1_summary_binding["path"]))
    if (
        m1_summary.get("status") != "done"
        or m1_summary.get("phase") != "six_scene_validation_confirmation"
        or m1_summary.get("scenes") != protocol["scene_order"]
        or m1_summary.get("heldout_content_read") is not False
        or m1_summary.get("test_quality_read") is not False
    ):
        raise M2MaterializationError("M1 validation cohort drift")
    return candidate, summary, m1_summary


def _request_row(
    *,
    scene: str,
    role: str,
    actor: Mapping[str, Any],
    row: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    frame, camera = int(row["frame"]), int(row["camera_id"])
    if frame % 5 != int(protocol["target_remainder"]):
        raise M2MaterializationError(f"validation target remainder drift: {frame}")
    supports = [[frame + int(offset), camera] for offset in protocol["support_offsets"]]
    if any(value < 0 for value, _ in supports) or any(
        value % 5 in {int(protocol["target_remainder"]), int(protocol["heldout_remainder"])}
        for value, _ in supports
    ):
        raise M2MaterializationError(f"validation support partition drift: {supports}")
    return {
        "request_id": f"{scene}__{role}__f{frame:03d}__c{camera}",
        "hole_id": f"remove_actor_{actor['dataset_instance_id']}",
        "role": role,
        "frame": frame,
        "camera_id": camera,
        "support_views": supports,
        "target_mask": _verified(row["mask"], row["mask_sha256"], "validation target mask"),
        "groundtruth_with_actor": _verified(
            row["source_image"], row["source_image_sha256"], "validation target image"
        ),
        "positive_pixels": int(row["positive_pixels"]),
    }


def materialize_validation_scene(*, config_path: Path, scene: str) -> dict[str, Any]:
    config = _yaml(config_path)
    candidate, freeze_summary, _ = _validate_root(config, scene)
    scene_input = config["scene_inputs"][scene]
    m1_binding = _binding(
        scene_input["m1_scene_config"]["path"],
        scene_input["m1_scene_config"]["sha256"],
        "M1 validation scene config",
    )
    source = _yaml(Path(m1_binding["path"]))
    if (
        source.get("scene") != scene
        or source.get("partition") != "validation"
        or source.get("heldout_content_read") is not False
        or source.get("test_quality_read") is not False
    ):
        raise M2MaterializationError("M1 validation scene identity drift")
    common = {
        "schema_version": "worldsim_v4_m2_scene_v1",
        "task_id": TASK_ID,
        "scene": scene,
        "partition": "validation",
        "source_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "candidate_protocol": config["candidate_protocol"],
        "m1_scene_config": m1_binding,
        "development_freeze": config["development_freeze"],
        "partition_contract": config["protocol"]["partition_contract"],
        "target_remainder": int(config["protocol"]["target_remainder"]),
        "heldout_remainder": int(config["protocol"]["heldout_remainder"]),
        "development_content_read": False,
        "validation_content_read": False,
        "validation_optimization_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    if source.get("status") == "abstain":
        if source.get("reason") != "ABSTAIN_NO_ACTOR" or scene_input["actor_assets"]:
            raise M2MaterializationError("validation scene abstain contract drift")
        return {
            **common,
            "status": "abstain",
            "reason": "ABSTAIN_NO_ACTOR",
            "requests": [],
            "blocked_requests": [],
            "request_count": 0,
            "total_request_count": 0,
            "retained_in_denominator": True,
            "frozen_router": freeze_summary["frozen_router"],
        }
    if source.get("status") != "ready":
        raise M2MaterializationError("validation scene is neither ready nor abstain")
    mask_binding = source["inputs"]["development_evaluation_masks"]
    mask_manifest = _json(
        Path(_binding(mask_binding["path"], mask_binding["sha256"], "validation masks")["path"])
    )
    if (
        mask_manifest.get("evaluation_partition") != "development"
        or mask_manifest.get("optimization_forbidden") is not True
    ):
        raise M2MaterializationError("validation-scene mask seal drift")
    accepted = [row for row in mask_manifest["masks"] if row.get("accepted") is True]
    if len(accepted) != int(mask_manifest["accepted_mask_count"]):
        raise M2MaterializationError("validation accepted-mask accounting drift")
    assets = scene_input["actor_assets"]
    executable_roles = [role for role, spec in assets.items() if "erase_package" in spec]
    if len(executable_roles) != 1:
        raise M2MaterializationError("scene runner requires exactly one executable actor role")
    executable_role = executable_roles[0]
    actor = source["actors"][executable_role]
    asset_spec = assets[executable_role]
    evidence_state = _binding(
        asset_spec["evidence_state"]["path"],
        asset_spec["evidence_state"]["sha256"],
        "validation evidence state",
    )
    package_binding = _binding(
        asset_spec["erase_package"]["path"],
        asset_spec["erase_package"]["sha256"],
        "validation erase package",
    )
    package = _json(Path(package_binding["path"]))
    package_actor = package.get("actor", {})
    if (
        package_actor.get("role") != executable_role
        or int(package_actor.get("dataset_instance_id")) != int(actor["dataset_instance_id"])
    ):
        raise M2MaterializationError("validation erase package actor drift")
    erase_inventory = _find_inventory(package, suffix="erase_indices.npz")
    erase_path = Path(package_binding["path"]).parent / erase_inventory["path"]
    erase_delta = _verified(erase_path, erase_inventory["sha256"], "validation erase delta")
    requests: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for row in accepted:
        role = str(row["role"])
        actor_for_role = source["actors"][role]
        request = _request_row(
            scene=scene,
            role=role,
            actor=actor_for_role,
            row=row,
            protocol=config["protocol"],
        )
        if role == executable_role:
            requests.append(request)
        else:
            reason = assets.get(role, {}).get("abstain_reason")
            if not reason or not str(reason).startswith("ABSTAIN_"):
                raise M2MaterializationError(f"validation role lacks frozen asset decision: {role}")
            blocked.append({**request, "status": "abstain", "reason": reason})
    requests.sort(key=lambda row: (row["frame"], row["camera_id"], row["role"]))
    blocked.sort(key=lambda row: (row["frame"], row["camera_id"], row["role"]))
    if len({row["request_id"] for row in requests + blocked}) != len(requests) + len(blocked):
        raise M2MaterializationError("validation request ID collision")
    checkpoint = source["inputs"]["checkpoint"]
    drive_config = source["inputs"]["drivestudio_source_config"]
    field = source["inputs"]["v33_o1_instance_field"]
    return {
        **common,
        "status": "ready",
        "actor": {"role": executable_role, **actor},
        "inputs": {
            "checkpoint": _binding(checkpoint["path"], checkpoint["sha256"], "checkpoint"),
            "drivestudio_source_config": _binding(
                drive_config["path"], drive_config["sha256"], "DriveStudio config"
            ),
            "v33_o1_instance_field": _binding(field["path"], field["sha256"], "O1 field"),
            "development_mask_manifest": _binding(
                mask_binding["path"], mask_binding["sha256"], "validation mask manifest"
            ),
            "evidence_state": evidence_state,
            "erase_package_manifest": package_binding,
            "erase_delta": erase_delta,
            "processed_scene_dir": source["inputs"]["processed_scene_dir"],
        },
        "runtime": source["runtime"],
        "asset_build": candidate["asset_build"],
        "risk": {
            "weights": config["development_freeze"]["weights"],
            "threshold": float(config["development_freeze"]["threshold"]),
            "tie_priority": config["development_freeze"]["tie_priority"],
        },
        "ablations": candidate["ablations"],
        "candidate_availability": {
            "OBSERVED": "ready_cross_view_train_only",
            "TELEA": "ready_deterministic_full_same_hole",
            "DONOR": "ready_native_checkpoint_builder",
            "GENERATED": "abstain_no_frozen_model",
        },
        "requests": requests,
        "blocked_requests": blocked,
        "request_count": len(requests),
        "blocked_request_count": len(blocked),
        "total_request_count": len(requests) + len(blocked),
        "all_accepted_masks_accounted": len(requests) + len(blocked) == len(accepted),
        "retained_in_denominator": True,
        "frozen_router": freeze_summary["frozen_router"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = materialize_validation_scene(
        config_path=args.config.resolve(), scene=args.scene
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scene": payload["scene"],
                "status": payload["status"],
                "request_count": payload.get("request_count", 0),
                "blocked_request_count": payload.get("blocked_request_count", 0),
                "total_request_count": payload.get("total_request_count", 0),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
