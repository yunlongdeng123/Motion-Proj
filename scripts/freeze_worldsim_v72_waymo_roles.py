#!/usr/bin/env python3
"""在读取任何传感器质量前冻结 Waymo context 角色。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motion_proj.worldsim_v72.data.splits import validate_data_roles


SPLIT_SALT = "worldsim-v72-eas-vggt-waymo-v1"


def _read_contexts(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"{path} 包含重复 context")
    if not values or any(not value.startswith("segment-") for value in values):
        raise ValueError(f"{path} 不是有效的 Waymo context 清单")
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(context: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{SPLIT_SALT}\0{context}".encode("utf-8")).hexdigest()
    return digest, context


def freeze_roles(
    payload: dict,
    *,
    train_contexts: list[str],
    validation_contexts: list[str],
    build_count: int,
    development_count: int,
    train_source: dict,
    validation_source: dict,
) -> dict:
    if build_count <= 0 or development_count <= 0:
        raise ValueError("build/development 数量必须为正")
    ranked = sorted(train_contexts, key=_rank)
    route_count = len(ranked) - build_count - development_count
    if route_count <= 0:
        raise ValueError("必须为 route_select 保留至少一个 context")
    payload = json.loads(json.dumps(payload))
    payload["datasets"]["waymo_perception_v2"] = {
        "group_id_kind": "segment_context_name",
        "selection": {
            "method": "sha256_rank_without_sensor_payload_or_quality",
            "salt": SPLIT_SALT,
            "train_manifest_count": len(train_contexts),
            "validation_manifest_count": len(validation_contexts),
            "quality_used_for_selection": False,
        },
        "source_manifests": {
            "training": train_source,
            "validation": validation_source,
        },
        "group_roles": {
            "train": ranked[:build_count],
            "development": ranked[build_count : build_count + development_count],
            "route_select": ranked[build_count + development_count :],
            "source_test": sorted(validation_contexts),
            "external_test": [],
        },
        "frozen_roles": {
            "route_select": True,
            "source_test": True,
            "external_test": False,
        },
        "payload_status": "official_download_authorization_required_not_materialized",
        "notes": [
            "Context names were frozen before downloading camera, LiDAR, label, or quality payloads.",
            "Official validation contexts are source_test and must remain unopened during development.",
            "Waymo account registration and authenticated gcloud access are required for official payloads.",
        ],
    }
    validate_data_roles(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roles", type=Path, required=True)
    parser.add_argument("--train-contexts", type=Path, required=True)
    parser.add_argument("--validation-contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-count", type=int, default=600)
    parser.add_argument("--development-count", type=int, default=99)
    args = parser.parse_args()
    payload = json.loads(args.data_roles.read_text(encoding="utf-8"))
    train = _read_contexts(args.train_contexts)
    validation = _read_contexts(args.validation_contexts)
    frozen = freeze_roles(
        payload,
        train_contexts=train,
        validation_contexts=validation,
        build_count=args.build_count,
        development_count=args.development_count,
        train_source={
            "path": str(args.train_contexts),
            "sha256": _sha256(args.train_contexts),
            "count": len(train),
            "upstream": "https://github.com/NVlabs/GaussianSTORM/tree/main/data/dataset_scene_list",
        },
        validation_source={
            "path": str(args.validation_contexts),
            "sha256": _sha256(args.validation_contexts),
            "count": len(validation),
            "upstream": "https://github.com/NVlabs/GaussianSTORM/tree/main/data/dataset_scene_list",
        },
    )
    args.output.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
