from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_STAGES = (
    "semantic_lift",
    "instance_field",
    "roadpatch",
    "asset_harvester",
    "spatial_delta",
    "semantic_render",
)


class V33ReplayError(RuntimeError):
    """V3.3 逐场景 replay 合同不满足。"""


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise V33ReplayError(f"YAML 顶层必须是 mapping：{path}")
    return payload


def _actor_id_by_token(instances_path: Path, token: str) -> int:
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    matches = [int(key) for key, value in instances.items() if str(value.get("id")) == token]
    if len(matches) != 1:
        raise V33ReplayError(
            f"actor token 必须在 instances_info 中精确命中一次：{token} -> {matches}"
        )
    return matches[0]


def resolve_scene_contracts(
    replay_config: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    algorithm = replay_config.get("algorithm", {})
    if tuple(algorithm.get("required_stages", ())) != REQUIRED_STAGES:
        raise V33ReplayError("V3.3 required stages 漂移")
    if algorithm.get("base_rgb_immutable") is not True:
        raise V33ReplayError("V3.3 replay 必须保持 base RGB immutable")
    partition = replay_config.get("frame_partition", {})
    if partition != {
        "modulus": 5,
        "train_remainders": [0, 1, 3],
        "development_remainder": 2,
        "heldout_remainder": 4,
        "test_quality_read": False,
    }:
        raise V33ReplayError("V3.3 replay 的 mod-5 partition 漂移")

    inputs = replay_config["inputs"]
    cohort_path = root / str(inputs["cohort_config"])
    if sha256_file(cohort_path) != inputs["cohort_config_sha256"]:
        raise V33ReplayError("D0 cohort config SHA 漂移")
    cohort = load_yaml(cohort_path)
    matrix = load_yaml(root / str(inputs["baseline_matrix"]))
    if matrix.get("frame_partition") != {
        "heldout": "sample_index_mod_5_eq_4",
        "development": "sample_index_mod_5_eq_2",
        "train": "remaining",
    }:
        raise V33ReplayError("B0 matrix partition 漂移")
    v33 = matrix["baselines"]["v33_frozen"]
    if v33.get("implementation_commit") != algorithm.get("implementation_commit"):
        raise V33ReplayError("V3.3 implementation commit 漂移")

    expected_scenes = list(
        cohort.get("freeze", {}).get("scene_roles", {}).get("development", [])
    )
    if len(expected_scenes) != int(replay_config["gates"]["expected_scene_count"]):
        raise V33ReplayError("development scene 数量漂移")
    scene_contract = matrix.get("scene_contract", {})
    if set(expected_scenes) != set(scene_contract):
        raise V33ReplayError("D0 与 B0 scene 集合不一致")
    records = {
        str(row["scene"]): row
        for row in cohort["freeze"]["scene_records"]
        if row.get("role") == replay_config["scene_source"]["split"]
    }
    checkpoints = matrix["baselines"]["streetgs"].get("checkpoints", {})
    processed_root = Path(str(inputs["processed_root"]))
    output: list[dict[str, Any]] = []
    for scene in expected_scenes:
        if scene not in records:
            raise V33ReplayError(f"缺少 D0 scene record：{scene}")
        record = records[scene]
        checkpoint = checkpoints.get(scene, {})
        path = Path(str(checkpoint.get("path", "")))
        if not path.is_file() or path.stat().st_size != checkpoint.get("bytes"):
            raise V33ReplayError(f"StreetGS base checkpoint 缺失或 bytes 漂移：{scene}")
        if sha256_file(path) != checkpoint.get("sha256"):
            raise V33ReplayError(f"StreetGS base checkpoint SHA 漂移：{scene}")
        source_config = path.parent / "config.yaml"
        if not source_config.is_file():
            raise V33ReplayError(f"StreetGS source config 缺失：{scene}")
        scene_index = int(scene_contract[scene]["scene_index"])
        processed_scene = processed_root / f"{scene_index:03d}"
        instances_path = processed_scene / "instances" / "instances_info.json"
        if not instances_path.is_file():
            raise V33ReplayError(f"instances_info 缺失：{scene}")
        high = record["actors"][replay_config["scene_source"]["high_support_role"]]
        boundary = record["actors"][replay_config["scene_source"]["boundary_support_role"]]
        high_token = str(high["instance_token"])
        boundary_token = str(boundary["instance_token"])
        if high_token == boundary_token:
            raise V33ReplayError(f"high/boundary actor 不得相同：{scene}")
        output.append(
            {
                "scene": scene,
                "scene_index": scene_index,
                "processed_scene": str(processed_scene),
                "instances_info": {
                    "path": str(instances_path),
                    "sha256": sha256_file(instances_path),
                },
                "base_checkpoint": {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": checkpoint["sha256"],
                    "source_config": str(source_config),
                    "source_config_sha256": sha256_file(source_config),
                },
                "actors": {
                    "high_support": {
                        "instance_token": high_token,
                        "dataset_instance_id": _actor_id_by_token(instances_path, high_token),
                        "class_name": high["category"],
                    },
                    "boundary_support": {
                        "instance_token": boundary_token,
                        "dataset_instance_id": _actor_id_by_token(
                            instances_path, boundary_token
                        ),
                        "class_name": boundary["category"],
                    },
                },
                "continuous_clip": dict(record["continuous_clip"]),
                "test_quality_read": False,
            }
        )
    return output


def bind_actor_registry(
    scene_contract: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    require_high_available: bool = True,
) -> dict[str, Any]:
    actors = registry.get("actors", [])
    by_token = {
        str(row.get("instance_token")): row
        for row in actors
        if isinstance(row, Mapping)
    }
    resolved = dict(scene_contract)
    resolved["actors"] = {}
    for role, actor in scene_contract["actors"].items():
        token = str(actor["instance_token"])
        row = by_token.get(token)
        if row is None:
            raise V33ReplayError(f"actor registry 缺少 {role}：{token}")
        availability = str(row.get("availability"))
        if (
            role == "high_support"
            and require_high_available
            and availability != "available"
        ):
            raise V33ReplayError(f"high actor 不可用：{token} / {availability}")
        if int(row.get("processed_true_instance_id", -1)) != int(
            actor["dataset_instance_id"]
        ):
            raise V33ReplayError(f"actor registry true id 漂移：{token}")
        resolved["actors"][role] = {
            **actor,
            "availability": availability,
            "rigid_model_index": row.get("rigid_model_index"),
            "registry_row": row,
        }
    return resolved
