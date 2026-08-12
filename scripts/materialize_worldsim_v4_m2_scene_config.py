#!/usr/bin/env python3
"""把冻结的 M1 nuScenes 输出物化成 M2 development scene 请求。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASK_ID = "WS-V4-M2-REPAIR-ROUTER-01"


class M2MaterializationError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M2MaterializationError(f"YAML root 不是 mapping: {path}")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M2MaterializationError(f"JSON root 不是 mapping: {path}")
    return value


def _verified(path: str | Path, expected: str, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise M2MaterializationError(f"{label} 不存在: {source}")
    actual = sha256_file(source)
    if actual != expected:
        raise M2MaterializationError(
            f"{label} SHA 漂移: expected={expected} actual={actual}"
        )
    return {"path": str(source), "sha256": actual, "bytes": source.stat().st_size}


def _find_inventory(
    package: Mapping[str, Any], *, suffix: str
) -> Mapping[str, Any]:
    matches = [row for row in package["inventory"] if str(row["path"]).endswith(suffix)]
    if len(matches) != 1:
        raise M2MaterializationError(f"package inventory 缺少唯一 {suffix}")
    return matches[0]


def _scene_source_path(config: Mapping[str, Any], scene: str) -> Path:
    run = Path(config["inputs"]["m1_development_run"]["path"])
    return run / "source_snapshot" / f"{scene}.yaml"


def _validate_root_contract(config: Mapping[str, Any], scene: str) -> None:
    if config.get("schema_version") != "worldsim_v4_m2_router_v1":
        raise M2MaterializationError("M2 config schema 漂移")
    if config.get("task_id") != TASK_ID:
        raise M2MaterializationError("M2 task_id 漂移")
    gate = config["execution_gate"]
    if (
        gate.get("dataset") != "nuScenes"
        or gate.get("m1_task_status") != "rejected"
        or gate.get("m1_validation_status") != "done"
        or gate.get("fallback_scope") != "evidence_routed_delta_compiler"
        or gate.get("test_quality_read") is not False
    ):
        raise M2MaterializationError("M2 execution gate 未冻结到 nuScenes rejection fallback")
    protocol = config["protocol"]
    if scene not in protocol["development_scenes"]:
        raise M2MaterializationError(f"scene 不属于冻结 development cohort: {scene}")
    if (
        protocol.get("target_partition") != "development"
        or protocol.get("support_partition") != "train_only"
        or protocol.get("heldout_content_read") is not False
        or protocol.get("test_quality_read") is not False
    ):
        raise M2MaterializationError("M2 partition contract 漂移")


def materialize_scene_config(
    *, config_path: Path, scene: str, m1_scene_path: Path | None = None
) -> dict[str, Any]:
    config = _yaml(config_path)
    _validate_root_contract(config, scene)
    protocol = config["protocol"]
    source_path = m1_scene_path or _scene_source_path(config, scene)
    source = _yaml(source_path)
    if (
        source.get("scene") != scene
        or source.get("partition") != "development"
        or source.get("heldout_content_read") is not False
        or source.get("test_quality_read") is not False
    ):
        raise M2MaterializationError("M1 scene identity/partition provenance 漂移")

    run_spec = config["inputs"]["m1_development_run"]
    run = Path(run_spec["path"])
    summary_binding = _verified(
        run / "summary.json", run_spec["summary_sha256"], "M1 development summary"
    )
    manifest_binding = _verified(
        run / "manifest.json", run_spec["manifest_sha256"], "M1 development manifest"
    )
    summary = _json(Path(summary_binding["path"]))
    if (
        summary.get("status") != "done"
        or summary.get("phase") != "six_scene_development"
        or summary.get("heldout_content_read") is not False
        or summary.get("test_quality_read") is not False
        or scene not in summary.get("scenes", [])
    ):
        raise M2MaterializationError("M1 development terminal contract 漂移")

    common = {
        "schema_version": "worldsim_v4_m2_scene_v1",
        "task_id": TASK_ID,
        "scene": scene,
        "partition": "development",
        "source_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "m1_scene_config": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "m1_development": {
            "run": str(run),
            "summary": summary_binding,
            "manifest": manifest_binding,
        },
        "partition_contract": protocol["partition_contract"],
        "development_content_read": False,
        "development_optimization_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    if source.get("status") == "abstain":
        record = next(
            (row for row in summary["scene_records"] if row.get("scene") == scene), None
        )
        reason = source.get("reason") or (record or {}).get("reason")
        if not reason or not str(reason).startswith("ABSTAIN_"):
            raise M2MaterializationError("M1 abstain scene 缺失冻结原因")
        return {
            **common,
            "status": "abstain",
            "reason": reason,
            "requests": [],
            "candidate_availability": {},
            "retained_in_denominator": True,
        }
    if source.get("status") != "ready":
        raise M2MaterializationError("M1 scene 未 ready/abstain")

    actor_items = list(source.get("actors", {}).items())
    if len(actor_items) != 1:
        raise M2MaterializationError("M2 development 需要唯一冻结 actor")
    role, actor = actor_items[0]
    inputs = source["inputs"]
    checkpoint = _verified(
        inputs["checkpoint"]["path"], inputs["checkpoint"]["sha256"], "checkpoint"
    )
    source_config = _verified(
        inputs["drivestudio_source_config"]["path"],
        inputs["drivestudio_source_config"]["sha256"],
        "DriveStudio config",
    )
    field = _verified(
        inputs["v33_o1_instance_field"]["path"],
        inputs["v33_o1_instance_field"]["sha256"],
        "V3.3 O1 field",
    )
    masks_binding = _verified(
        inputs["development_evaluation_masks"]["path"],
        inputs["development_evaluation_masks"]["sha256"],
        "development mask manifest",
    )
    mask_manifest = _json(Path(masks_binding["path"]))
    if (
        mask_manifest.get("evaluation_partition") != "development"
        or mask_manifest.get("optimization_forbidden") is not True
    ):
        raise M2MaterializationError("mask manifest partition contract 漂移")
    accepted = [row for row in mask_manifest["masks"] if row.get("accepted") is True]
    if not accepted or len(accepted) != int(mask_manifest["accepted_mask_count"]):
        raise M2MaterializationError("accepted mask accounting 漂移")
    if protocol.get("retain_all_accepted_masks") is not True:
        raise M2MaterializationError("M2 必须保留所有 accepted development masks")

    chain_binding = source["v33_scene_chain"]
    chain = _json(Path(_verified(
        chain_binding["path"], chain_binding["sha256"], "V3.3 scene chain"
    )["path"]))
    spatial = chain["stages"]["spatial_delta"]
    if spatial.get("status") != "done":
        raise M2MaterializationError("M2 缺少 V3.3 erase package")
    package_root = Path(spatial["run"]) / "package/artifacts/worldsim_asset"
    package_path = package_root / "package_manifest.json"
    package = _json(package_path)
    if package.get("actor", {}).get("dataset_instance_id") != actor["dataset_instance_id"]:
        raise M2MaterializationError("erase package actor identity 漂移")
    package_binding = {
        "path": str(package_path),
        "sha256": sha256_file(package_path),
        "bytes": package_path.stat().st_size,
    }
    erase_row = _find_inventory(package, suffix="erase_indices.npz")
    erase_binding = _verified(
        package_root / erase_row["path"], erase_row["sha256"], "erase delta"
    )

    state_path = run / "artifacts/states" / scene / f"{role}.npz"
    state_binding = {
        "path": str(state_path),
        "sha256": sha256_file(state_path),
        "bytes": state_path.stat().st_size,
    }
    target_remainder = int(protocol["target_remainder"])
    heldout_remainder = int(protocol["heldout_remainder"])
    support_offsets = [int(value) for value in protocol["support_offsets"]]
    requests: list[dict[str, Any]] = []
    for row in accepted:
        frame = int(row["frame"])
        camera = int(row["camera_id"])
        if frame % 5 != target_remainder:
            raise M2MaterializationError(f"development target remainder 漂移: {frame}")
        supports = [[frame + offset, camera] for offset in support_offsets]
        if any(value < 0 for value, _ in supports) or any(
            value % 5 in {target_remainder, heldout_remainder} for value, _ in supports
        ):
            raise M2MaterializationError(f"support view 混入 development/heldout: {supports}")
        mask = _verified(row["mask"], row["mask_sha256"], "target mask")
        image = _verified(row["source_image"], row["source_image_sha256"], "target image")
        requests.append(
            {
                "request_id": f"{scene}__{role}__f{frame:03d}__c{camera}",
                "hole_id": f"remove_actor_{actor['dataset_instance_id']}",
                "role": role,
                "frame": frame,
                "camera_id": camera,
                "support_views": supports,
                "target_mask": mask,
                "groundtruth_with_actor": image,
                "positive_pixels": int(row["positive_pixels"]),
            }
        )
    requests.sort(key=lambda row: (row["frame"], row["camera_id"], row["role"]))
    if len({row["request_id"] for row in requests}) != len(requests):
        raise M2MaterializationError("M2 request_id 重复")

    return {
        **common,
        "status": "ready",
        "actor": {"role": role, **actor},
        "inputs": {
            "checkpoint": checkpoint,
            "drivestudio_source_config": source_config,
            "v33_o1_instance_field": field,
            "development_mask_manifest": masks_binding,
            "evidence_state": state_binding,
            "erase_package_manifest": package_binding,
            "erase_delta": erase_binding,
            "processed_scene_dir": inputs["processed_scene_dir"],
        },
        "runtime": source["runtime"],
        "asset_build": config["asset_build"],
        "risk": config["risk"],
        "ablations": config["ablations"],
        "candidate_availability": {
            "OBSERVED": "ready_cross_view_train_only",
            "TELEA": "ready_deterministic_full_same_hole",
            "DONOR": "ready_native_checkpoint_builder",
            "GENERATED": "abstain_no_frozen_model",
        },
        "requests": requests,
        "request_count": len(requests),
        "all_accepted_masks_retained": len(requests)
        == int(mask_manifest["accepted_mask_count"]),
        "retained_in_denominator": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--m1-scene-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = materialize_scene_config(
        config_path=args.config.resolve(),
        scene=args.scene,
        m1_scene_path=(args.m1_scene_config.resolve() if args.m1_scene_config else None),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(json.dumps({"scene": args.scene, "status": payload["status"], "request_count": len(payload["requests"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
