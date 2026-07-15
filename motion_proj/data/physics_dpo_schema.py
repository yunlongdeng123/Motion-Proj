"""SAP-DPO 的 scene split、provenance schema 与 fail-closed validator。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from omegaconf import OmegaConf

from ..runtime.atomic import atomic_directory, atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json


SCENE_SPLIT_SCHEMA_VERSION = "sap-dpo-scene-split-v1"
PREFERENCE_SCHEMA_VERSION = "sap-dpo-preference-v1"
SCENE_PARTITIONS = (
    "preference_train",
    "preference_dev",
    "screen_eval",
    "formal_test",
)
BRANCH_FAMILIES = frozenset({"common_prefix", "base_renoise", "independent_seed_baseline"})
PAIR_LABELS = frozenset({"a_wins", "b_wins", "tie", "abstain", "invalid"})
SEGMENT_LABELS = frozenset({"a_wins", "b_wins", "tie", "abstain"})
FORBIDDEN_REFERENCE_KEY_TOKENS = (
    "p1_target",
    "hybrid_latent",
    "projected_rgb",
    "projected_cache",
)


class PhysicsDpoSchemaError(ValueError):
    """数据缺 provenance、跨 split 或违反结构对齐约束时立即失败。"""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicsDpoSchemaError(f"{label} 必须是 object")
    return value


def _require(record: Mapping[str, Any], fields: Iterable[str], *, label: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise PhysicsDpoSchemaError(f"{label} 缺少字段: {', '.join(missing)}")


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhysicsDpoSchemaError(f"{label} 必须是非空字符串")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise PhysicsDpoSchemaError(f"{label} 必须是 >= {minimum} 的整数")
    return value


def _finite(value: Any, *, label: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise PhysicsDpoSchemaError(f"{label} 必须是有限数")
    result = float(value)
    if minimum is not None and result < minimum:
        raise PhysicsDpoSchemaError(f"{label} 必须 >= {minimum}")
    if maximum is not None and result > maximum:
        raise PhysicsDpoSchemaError(f"{label} 必须 <= {maximum}")
    return result


def _bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise PhysicsDpoSchemaError(f"{label} 必须是 bool")
    return value


def _unique_ids(records: Sequence[Mapping[str, Any]], field: str, *, label: str) -> None:
    values = [_string(record.get(field), label=f"{label}.{field}") for record in records]
    if len(values) != len(set(values)):
        raise PhysicsDpoSchemaError(f"{label}.{field} 存在重复")


def _canonical_records(records: Iterable[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    return [dict(record) for record in sorted(records, key=lambda item: str(item[field]))]


def _contains_forbidden_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            any(token in str(key).lower() for token in FORBIDDEN_REFERENCE_KEY_TOKENS)
            or _contains_forbidden_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_reference(item) for item in value)
    return False


def _scene_rank(salt: str, scene_token: str) -> tuple[str, str]:
    return hashlib.sha256(f"{salt}|{scene_token}".encode("utf-8")).hexdigest(), scene_token


def _validate_source_manifest(source: Mapping[str, Any], *, expected_split: str) -> None:
    _require(
        source,
        ("split", "camera", "num_frames", "scenes", "clips", "split_fingerprint"),
        label=f"source manifest {expected_split}",
    )
    if source["split"] != expected_split:
        raise PhysicsDpoSchemaError(f"source manifest split 必须是 {expected_split}")
    scenes = source["scenes"]
    clips = source["clips"]
    if not isinstance(scenes, list) or not isinstance(clips, list):
        raise PhysicsDpoSchemaError("source manifest scenes/clips 必须是 list")
    _unique_ids([_mapping(row, label="source scene") for row in scenes], "scene_token", label="source scenes")
    _unique_ids([_mapping(row, label="source clip") for row in clips], "clip_id", label="source clips")
    tokens = {str(row["scene_token"]) for row in scenes}
    for clip in clips:
        mapped = _mapping(clip, label="source clip")
        _require(mapped, ("clip_id", "scene_name", "scene_token"), label="source clip")
        if str(mapped["scene_token"]) not in tokens:
            raise PhysicsDpoSchemaError("source clip 引用了未知 scene_token")


def _partition_from_source(
    source: Mapping[str, Any],
    *,
    partition: str,
    selected_tokens: set[str],
    salt: str,
) -> dict[str, Any]:
    scenes = [dict(row) for row in source["scenes"] if str(row["scene_token"]) in selected_tokens]
    clips = [dict(row) for row in source["clips"] if str(row["scene_token"]) in selected_tokens]
    scenes.sort(key=lambda row: str(row["scene_token"]))
    clips.sort(key=lambda row: (str(row["scene_token"]), str(row["clip_id"])))
    core = {
        "partition": partition,
        "source_split": str(source["split"]),
        "source_manifest_fingerprint": str(source["split_fingerprint"]),
        "selection_salt": salt,
        "scene_tokens": [str(row["scene_token"]) for row in scenes],
        "clip_ids": [str(row["clip_id"]) for row in clips],
    }
    return {
        **core,
        "scene_count": len(scenes),
        "clip_count": len(clips),
        "scene_list_fingerprint": sha256_json(core),
        "scenes": scenes,
        "clips": clips,
    }


def build_scene_split(
    train_manifest: Mapping[str, Any],
    val_manifest: Mapping[str, Any],
    *,
    salt: str,
    preference_dev_scene_count: int = 70,
    screen_eval_scene_count: int = 32,
) -> dict[str, Any]:
    """按 token 的 salted SHA256 冻结 V2 的四个 scene-level partition。"""
    _validate_source_manifest(train_manifest, expected_split="train")
    _validate_source_manifest(val_manifest, expected_split="val")
    _string(salt, label="salt")
    train_tokens = {str(row["scene_token"]) for row in train_manifest["scenes"]}
    val_tokens = {str(row["scene_token"]) for row in val_manifest["scenes"]}
    if train_tokens & val_tokens:
        raise PhysicsDpoSchemaError("official train/val scene_token 不得重叠")
    preference_dev_scene_count = _integer(
        preference_dev_scene_count, label="preference_dev_scene_count", minimum=1
    )
    screen_eval_scene_count = _integer(screen_eval_scene_count, label="screen_eval_scene_count", minimum=1)
    if preference_dev_scene_count >= len(train_tokens):
        raise PhysicsDpoSchemaError("preference_dev 必须小于 official train scenes")
    if screen_eval_scene_count >= len(val_tokens):
        raise PhysicsDpoSchemaError("screen_eval 必须小于 official val scenes")

    ranked_train = sorted(train_tokens, key=lambda token: _scene_rank(salt, token))
    ranked_val = sorted(val_tokens, key=lambda token: _scene_rank(salt, token))
    dev_tokens = set(ranked_train[:preference_dev_scene_count])
    screen_tokens = set(ranked_val[:screen_eval_scene_count])
    partitions = {
        "preference_train": _partition_from_source(
            train_manifest, partition="preference_train", selected_tokens=train_tokens - dev_tokens, salt=salt
        ),
        "preference_dev": _partition_from_source(
            train_manifest, partition="preference_dev", selected_tokens=dev_tokens, salt=salt
        ),
        "screen_eval": _partition_from_source(
            val_manifest, partition="screen_eval", selected_tokens=screen_tokens, salt=salt
        ),
        "formal_test": _partition_from_source(
            val_manifest, partition="formal_test", selected_tokens=val_tokens - screen_tokens, salt=salt
        ),
    }
    core = {
        "schema_version": SCENE_SPLIT_SCHEMA_VERSION,
        "salt": salt,
        "source_manifests": {
            "official_train": {
                "split_fingerprint": str(train_manifest["split_fingerprint"]),
                "camera": str(train_manifest["camera"]),
                "num_frames": int(train_manifest["num_frames"]),
            },
            "official_val": {
                "split_fingerprint": str(val_manifest["split_fingerprint"]),
                "camera": str(val_manifest["camera"]),
                "num_frames": int(val_manifest["num_frames"]),
            },
        },
        "selection_rule": "ascending SHA256(salt + '|' + scene_token); smaller partition receives first ranks",
        "partitions": partitions,
    }
    split_manifest = {**core, "split_fingerprint": sha256_json(core)}
    validate_scene_split(split_manifest)
    return split_manifest


def validate_scene_split(split_manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """验证 scene/clip 不泄漏，并返回 scene_token 与 clip_id 的所属 partition。"""
    _require(
        split_manifest,
        ("schema_version", "salt", "source_manifests", "selection_rule", "partitions", "split_fingerprint"),
        label="scene split",
    )
    if split_manifest["schema_version"] != SCENE_SPLIT_SCHEMA_VERSION:
        raise PhysicsDpoSchemaError("scene split schema_version 不匹配")
    partitions = _mapping(split_manifest["partitions"], label="scene split partitions")
    if set(partitions) != set(SCENE_PARTITIONS):
        raise PhysicsDpoSchemaError("scene split partitions 必须恰好包含四个预注册 partition")
    scene_owner: dict[str, str] = {}
    clip_owner: dict[str, str] = {}
    scene_name_by_token: dict[str, str] = {}
    for partition_name in SCENE_PARTITIONS:
        partition = _mapping(partitions[partition_name], label=f"partition {partition_name}")
        _require(
            partition,
            ("partition", "source_split", "scene_tokens", "clip_ids", "scene_count", "clip_count", "scenes", "clips", "scene_list_fingerprint"),
            label=f"partition {partition_name}",
        )
        if partition["partition"] != partition_name:
            raise PhysicsDpoSchemaError(f"partition 名称不一致: {partition_name}")
        scenes = partition["scenes"]
        clips = partition["clips"]
        if not isinstance(scenes, list) or not isinstance(clips, list):
            raise PhysicsDpoSchemaError(f"partition {partition_name} scenes/clips 必须是 list")
        tokens = [str(_mapping(row, label="scene").get("scene_token", "")) for row in scenes]
        clip_ids = [str(_mapping(row, label="clip").get("clip_id", "")) for row in clips]
        if tokens != list(partition["scene_tokens"]) or clip_ids != list(partition["clip_ids"]):
            raise PhysicsDpoSchemaError(f"partition {partition_name} 的行列表与 ID 列表不一致")
        if len(tokens) != int(partition["scene_count"]) or len(clip_ids) != int(partition["clip_count"]):
            raise PhysicsDpoSchemaError(f"partition {partition_name} 的 count 不一致")
        if len(tokens) != len(set(tokens)) or len(clip_ids) != len(set(clip_ids)):
            raise PhysicsDpoSchemaError(f"partition {partition_name} 含重复 scene 或 clip")
        local_tokens = set(tokens)
        for row in scenes:
            mapped = _mapping(row, label="scene")
            token = _string(mapped.get("scene_token"), label="scene.scene_token")
            scene_name_by_token[token] = _string(mapped.get("scene_name"), label="scene.scene_name")
            if token in scene_owner:
                raise PhysicsDpoSchemaError(f"scene_token 跨 split 泄漏: {token}")
            scene_owner[token] = partition_name
        for row in clips:
            mapped = _mapping(row, label="clip")
            clip_id = _string(mapped.get("clip_id"), label="clip.clip_id")
            token = _string(mapped.get("scene_token"), label="clip.scene_token")
            if token not in local_tokens:
                raise PhysicsDpoSchemaError(f"clip {clip_id} 的 scene 不属于 {partition_name}")
            if clip_id in clip_owner:
                raise PhysicsDpoSchemaError(f"clip_id 跨 split 泄漏: {clip_id}")
            clip_owner[clip_id] = partition_name
        partition_core = {
            "partition": partition_name,
            "source_split": partition["source_split"],
            "source_manifest_fingerprint": partition.get("source_manifest_fingerprint"),
            "selection_salt": partition.get("selection_salt"),
            "scene_tokens": list(partition["scene_tokens"]),
            "clip_ids": list(partition["clip_ids"]),
        }
        if partition["scene_list_fingerprint"] != sha256_json(partition_core):
            raise PhysicsDpoSchemaError(f"partition {partition_name} scene_list_fingerprint 不匹配")
    core = {key: value for key, value in split_manifest.items() if key != "split_fingerprint"}
    if split_manifest["split_fingerprint"] != sha256_json(core):
        raise PhysicsDpoSchemaError("scene split split_fingerprint 不匹配")
    return {"scene_owner": scene_owner, "clip_owner": clip_owner, "scene_name_by_token": scene_name_by_token}


def make_condition_id(record: Mapping[str, Any]) -> str:
    """只依赖 condition provenance，避免 generation seed 改变统计基本单位。"""
    fields = (
        "schema_version", "scene_token", "clip_id", "camera", "conditioning_frame",
        "condition_frame_sha256", "num_frames", "generation_protocol",
        "scheduler_fingerprint", "base_model_fingerprint",
    )
    _require(record, fields, label="condition")
    return "condition-" + sha256_json({field: record[field] for field in fields})


def validate_conditions(
    records: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    owners = validate_scene_split(split_manifest)
    _unique_ids(records, "condition_id", label="conditions")
    result: dict[str, dict[str, Any]] = {}
    required = (
        "schema_version", "condition_id", "scene_id", "scene_token", "clip_id", "split", "camera",
        "conditioning_frame", "condition_frame_sha256", "num_frames", "fps", "generation_protocol",
        "scheduler_fingerprint", "base_model_fingerprint", "uses_future_gt", "git_commit", "config_fingerprint",
    )
    for raw in records:
        record = dict(_mapping(raw, label="condition"))
        _require(record, required, label="condition")
        if _contains_forbidden_reference(record):
            raise PhysicsDpoSchemaError("condition 引用了 P1/hybrid/projected artifact")
        if record["schema_version"] != PREFERENCE_SCHEMA_VERSION:
            raise PhysicsDpoSchemaError("condition schema_version 不匹配")
        condition_id = _string(record["condition_id"], label="condition_id")
        token = _string(record["scene_token"], label="condition.scene_token")
        clip_id = _string(record["clip_id"], label="condition.clip_id")
        split = _string(record["split"], label="condition.split")
        if split not in SCENE_PARTITIONS or owners["scene_owner"].get(token) != split:
            raise PhysicsDpoSchemaError("condition 跨 scene split 或引用未知 scene")
        known_clip_owner = owners["clip_owner"].get(clip_id)
        if known_clip_owner is not None and known_clip_owner != split:
            raise PhysicsDpoSchemaError("condition clip_id 跨 scene split")
        if _string(record["scene_id"], label="condition.scene_id") != owners["scene_name_by_token"][token]:
            raise PhysicsDpoSchemaError("condition.scene_id 与冻结 scene manifest 不一致")
        _string(record["camera"], label="condition.camera")
        frame_count = _integer(record["num_frames"], label="condition.num_frames", minimum=1)
        if frame_count not in {8, 14}:
            raise PhysicsDpoSchemaError("condition.num_frames 只能是 PA1 预注册的 8 或 14")
        frame = _integer(record["conditioning_frame"], label="condition.conditioning_frame", minimum=0)
        if frame >= frame_count:
            raise PhysicsDpoSchemaError("conditioning_frame 超出 num_frames")
        _integer(record["fps"], label="condition.fps", minimum=1)
        if record["generation_protocol"] != "svd_official_v1":
            raise PhysicsDpoSchemaError("condition 必须使用 svd_official_v1")
        for field in ("condition_frame_sha256", "scheduler_fingerprint", "base_model_fingerprint", "git_commit", "config_fingerprint"):
            _string(record[field], label=f"condition.{field}")
        if _bool(record["uses_future_gt"], label="condition.uses_future_gt"):
            raise PhysicsDpoSchemaError("condition.uses_future_gt 必须为 false")
        if condition_id != make_condition_id(record):
            raise PhysicsDpoSchemaError("condition_id 与 canonical provenance 不匹配")
        result[condition_id] = record
    return result


def _same_json(left: Any, right: Any) -> bool:
    return sha256_json(left) == sha256_json(right)


def validate_candidates(
    records: Sequence[Mapping[str, Any]],
    conditions: Mapping[str, Mapping[str, Any]],
    *,
    exact_sibling_count: int | None = None,
) -> dict[str, dict[str, Any]]:
    _unique_ids(records, "candidate_id", label="candidates")
    required = (
        "candidate_id", "condition_id", "scene_id", "split", "candidate_role", "rgb_video_path",
        "vae_latent_path", "diagnostics_path", "generation_protocol", "base_model_fingerprint",
        "scheduler_fingerprint", "initial_latent_hash", "prefix_latent_hash", "prefix_trace_hash",
        "fork_step", "branch_family", "branch_direction", "branch_strength", "perturbation_rms",
        "antithetic_group_id", "generation_seed", "guidance_schedule", "num_frames", "fps",
        "uses_future_gt", "git_commit", "config_fingerprint",
    )
    candidates: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in records:
        record = dict(_mapping(raw, label="candidate"))
        _require(record, required, label="candidate")
        if _contains_forbidden_reference(record):
            raise PhysicsDpoSchemaError("candidate 引用了 P1/hybrid/projected artifact")
        candidate_id = _string(record["candidate_id"], label="candidate_id")
        condition_id = _string(record["condition_id"], label="candidate.condition_id")
        condition = conditions.get(condition_id)
        if condition is None:
            raise PhysicsDpoSchemaError("candidate 引用了未知 condition")
        for field in ("scene_id", "split", "generation_protocol", "base_model_fingerprint", "scheduler_fingerprint", "num_frames", "fps", "git_commit", "config_fingerprint"):
            if record[field] != condition[field]:
                raise PhysicsDpoSchemaError(f"candidate.{field} 必须与 condition 一致")
        for field in ("rgb_video_path", "vae_latent_path", "diagnostics_path", "initial_latent_hash", "prefix_latent_hash", "prefix_trace_hash", "antithetic_group_id"):
            _string(record[field], label=f"candidate.{field}")
        _integer(record["fork_step"], label="candidate.fork_step", minimum=0)
        _integer(record["generation_seed"], label="candidate.generation_seed", minimum=0)
        if _bool(record["uses_future_gt"], label="candidate.uses_future_gt"):
            raise PhysicsDpoSchemaError("candidate.uses_future_gt 必须为 false")
        role = _string(record["candidate_role"], label="candidate.candidate_role")
        direction = _string(record["branch_direction"], label="candidate.branch_direction")
        family = _string(record["branch_family"], label="candidate.branch_family")
        strength = _finite(record["branch_strength"], label="candidate.branch_strength", minimum=0.0)
        rms = _finite(record["perturbation_rms"], label="candidate.perturbation_rms", minimum=0.0)
        if role == "base_guard":
            if family != "base_guard" or direction != "base" or strength != 0.0 or rms != 0.0:
                raise PhysicsDpoSchemaError("base_guard 必须使用 base_guard/base/零 strength 与 RMS")
        elif role == "sibling":
            if family not in BRANCH_FAMILIES or direction not in {"positive", "negative"} or strength <= 0.0 or rms <= 0.0:
                raise PhysicsDpoSchemaError("sibling branch family/direction/strength/RMS 不合法")
        else:
            raise PhysicsDpoSchemaError("candidate_role 只能为 base_guard 或 sibling")
        candidates[candidate_id] = record
        grouped.setdefault(condition_id, []).append(record)

    for condition_id, group in grouped.items():
        guards = [record for record in group if record["candidate_role"] == "base_guard"]
        siblings = [record for record in group if record["candidate_role"] == "sibling"]
        if len(guards) != 1:
            raise PhysicsDpoSchemaError(f"condition {condition_id} 必须恰好有一条 base_guard")
        if exact_sibling_count is not None and len(siblings) != exact_sibling_count:
            raise PhysicsDpoSchemaError(f"condition {condition_id} sibling 数必须为 {exact_sibling_count}")
        shared_fields = ("initial_latent_hash", "prefix_latent_hash", "prefix_trace_hash", "scheduler_fingerprint", "guidance_schedule")
        anchor = guards[0]
        for sibling in siblings:
            for field in shared_fields:
                if not _same_json(sibling[field], anchor[field]):
                    raise PhysicsDpoSchemaError(f"condition {condition_id} 的 sibling 未共享 {field}")
        by_antithetic: dict[str, list[dict[str, Any]]] = {}
        for sibling in siblings:
            by_antithetic.setdefault(str(sibling["antithetic_group_id"]), []).append(sibling)
        for group_id, pair in by_antithetic.items():
            if len(pair) != 2 or {item["branch_direction"] for item in pair} != {"positive", "negative"}:
                raise PhysicsDpoSchemaError(f"antithetic group {group_id} 必须恰好是一正一负两条 sibling")
            if not math.isclose(float(pair[0]["perturbation_rms"]), float(pair[1]["perturbation_rms"]), rel_tol=0.0, abs_tol=1.0e-12):
                raise PhysicsDpoSchemaError(f"antithetic group {group_id} 的 RMS 不相等")
            if not math.isclose(float(pair[0]["branch_strength"]), float(pair[1]["branch_strength"]), rel_tol=0.0, abs_tol=1.0e-12):
                raise PhysicsDpoSchemaError(f"antithetic group {group_id} 的 strength 不相等")
        if siblings and all(item["branch_family"] == "common_prefix" for item in siblings):
            rms_values = [float(item["perturbation_rms"]) for item in siblings]
            reference_rms = rms_values[0]
            # permutation 的归约顺序会带来极小的 float32 RMS 尾差；保持与
            # generator 的理论等范数检查一致，拒绝任何有意义的强度差异。
            if not all(math.isclose(value, reference_rms, rel_tol=1.0e-7, abs_tol=1.0e-12) for value in rms_values[1:]):
                raise PhysicsDpoSchemaError("common_prefix 的四条 sibling 必须在数值容差内等范数")
    return candidates


def _feasible(value: Any) -> bool:
    if value is True or value == "feasible":
        return True
    return isinstance(value, Mapping) and value.get("feasible") is True


def validate_preferences(
    records: Sequence[Mapping[str, Any]],
    conditions: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    _unique_ids(records, "pair_id", label="preferences")
    required = (
        "pair_id", "condition_id", "candidate_a", "candidate_b", "split", "global_label",
        "winner_candidate_id", "loser_candidate_id", "feasibility_a", "feasibility_b", "physics_components",
        "quality_components", "preference_margin", "pair_confidence", "scorer_fingerprint", "human_review_id",
        "abstain_reason", "uses_future_gt",
    )
    preferences: dict[str, dict[str, Any]] = {}
    condition_pair_count: dict[str, int] = {}
    for raw in records:
        record = dict(_mapping(raw, label="preference"))
        _require(record, required, label="preference")
        if _contains_forbidden_reference(record):
            raise PhysicsDpoSchemaError("preference 引用了 P1/hybrid/projected artifact")
        pair_id = _string(record["pair_id"], label="pair_id")
        condition_id = _string(record["condition_id"], label="preference.condition_id")
        condition = conditions.get(condition_id)
        if condition is None:
            raise PhysicsDpoSchemaError("preference 引用了未知 condition")
        if record["split"] != condition["split"]:
            raise PhysicsDpoSchemaError("preference split 与 condition 不一致")
        a = candidates.get(_string(record["candidate_a"], label="candidate_a"))
        b = candidates.get(_string(record["candidate_b"], label="candidate_b"))
        if a is None or b is None or a["candidate_id"] == b["candidate_id"]:
            raise PhysicsDpoSchemaError("preference candidates 必须是两个不同的已知 candidate")
        if a["condition_id"] != condition_id or b["condition_id"] != condition_id:
            raise PhysicsDpoSchemaError("preference candidate 不得跨 condition")
        for field in ("initial_latent_hash", "prefix_latent_hash", "prefix_trace_hash", "scheduler_fingerprint"):
            if not _same_json(a[field], b[field]):
                raise PhysicsDpoSchemaError(f"preference candidates 未通过结构对齐: {field}")
        label = _string(record["global_label"], label="preference.global_label")
        if label not in PAIR_LABELS:
            raise PhysicsDpoSchemaError("global_label 无效")
        if not isinstance(record["physics_components"], Mapping) or not isinstance(record["quality_components"], Mapping):
            raise PhysicsDpoSchemaError("physics_components 与 quality_components 必须是 object")
        _string(record["scorer_fingerprint"], label="preference.scorer_fingerprint")
        if _bool(record["uses_future_gt"], label="preference.uses_future_gt"):
            raise PhysicsDpoSchemaError("preference.uses_future_gt 必须为 false")
        condition_pair_count[condition_id] = condition_pair_count.get(condition_id, 0) + 1
        if condition_pair_count[condition_id] > 1:
            raise PhysicsDpoSchemaError("每个 condition 最多一个 global pair")
        if label in {"a_wins", "b_wins"}:
            expected_winner = a["candidate_id"] if label == "a_wins" else b["candidate_id"]
            expected_loser = b["candidate_id"] if label == "a_wins" else a["candidate_id"]
            if record["winner_candidate_id"] != expected_winner or record["loser_candidate_id"] != expected_loser:
                raise PhysicsDpoSchemaError("winner/loser 与 global_label 不一致")
            if not _feasible(record["feasibility_a"]) or not _feasible(record["feasibility_b"]):
                raise PhysicsDpoSchemaError("只有 feasible/feasible pair 才能进入 winner/loser")
            if _finite(record["preference_margin"], label="preference.preference_margin", minimum=0.0) <= 0.0:
                raise PhysicsDpoSchemaError("decisive preference 的 preference_margin 必须 > 0")
            _finite(record["pair_confidence"], label="preference.pair_confidence", minimum=0.0, maximum=1.0)
            if float(record["pair_confidence"]) <= 0.0:
                raise PhysicsDpoSchemaError("decisive preference 的 pair_confidence 必须 > 0")
        else:
            if record["winner_candidate_id"] is not None or record["loser_candidate_id"] is not None:
                raise PhysicsDpoSchemaError("tie/abstain/invalid preference 不得填 winner/loser")
            _string(record["abstain_reason"], label="preference.abstain_reason")
        preferences[pair_id] = record
    return preferences


def validate_segments(
    records: Sequence[Mapping[str, Any]],
    preferences: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    _unique_ids(records, "segment_id", label="segments")
    required = (
        "segment_id", "pair_id", "start_frame", "end_frame", "label", "winner_candidate_id",
        "loser_candidate_id", "confidence", "violation_decomposition", "frame_alignment_pass", "abstain_reason",
    )
    segments: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = dict(_mapping(raw, label="segment"))
        _require(record, required, label="segment")
        pair_id = _string(record["pair_id"], label="segment.pair_id")
        preference = preferences.get(pair_id)
        if preference is None:
            raise PhysicsDpoSchemaError("segment 引用了未知 pair")
        start = _integer(record["start_frame"], label="segment.start_frame", minimum=0)
        end = _integer(record["end_frame"], label="segment.end_frame", minimum=1)
        condition_frames = int(candidates[preference["candidate_a"]]["num_frames"])
        if end - start != 4 or end > condition_frames:
            raise PhysicsDpoSchemaError("segment 必须是位于视频内的 4-frame window")
        label = _string(record["label"], label="segment.label")
        if label not in SEGMENT_LABELS:
            raise PhysicsDpoSchemaError("segment label 无效")
        if not isinstance(record["violation_decomposition"], Mapping):
            raise PhysicsDpoSchemaError("segment.violation_decomposition 必须是 object")
        aligned = _bool(record["frame_alignment_pass"], label="segment.frame_alignment_pass")
        _finite(record["confidence"], label="segment.confidence", minimum=0.0, maximum=1.0)
        if label in {"a_wins", "b_wins"}:
            a, b = preference["candidate_a"], preference["candidate_b"]
            winner = a if label == "a_wins" else b
            loser = b if label == "a_wins" else a
            if record["winner_candidate_id"] != winner or record["loser_candidate_id"] != loser:
                raise PhysicsDpoSchemaError("segment winner/loser 与 label 不一致")
            if not aligned or float(record["confidence"]) <= 0.0:
                raise PhysicsDpoSchemaError("decisive segment 必须 frame-aligned 且 confidence > 0")
        else:
            if record["winner_candidate_id"] is not None or record["loser_candidate_id"] is not None:
                raise PhysicsDpoSchemaError("tie/abstain segment 不得填 winner/loser")
            _string(record["abstain_reason"], label="segment.abstain_reason")
        segments[_string(record["segment_id"], label="segment_id")] = record
    return segments


def validate_training_dataset(
    split_manifest: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    preferences: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    *,
    require_segments: bool,
    exact_sibling_count: int | None = 4,
) -> dict[str, Any]:
    """训练入口的强校验：非 decisive pair/segment 不可被静默吞入损失。"""
    indexed_conditions = validate_conditions(conditions, split_manifest)
    indexed_candidates = validate_candidates(candidates, indexed_conditions, exact_sibling_count=exact_sibling_count)
    indexed_preferences = validate_preferences(preferences, indexed_conditions, indexed_candidates)
    indexed_segments = validate_segments(segments, indexed_preferences, indexed_candidates)
    segments_by_pair: dict[str, list[dict[str, Any]]] = {}
    for segment in indexed_segments.values():
        segments_by_pair.setdefault(str(segment["pair_id"]), []).append(segment)
    for pair_id, pair in indexed_preferences.items():
        if pair["global_label"] not in {"a_wins", "b_wins"}:
            raise PhysicsDpoSchemaError("训练 dataset 拒绝 tie/abstain/invalid global pair")
        if require_segments:
            decisive = [
                segment for segment in segments_by_pair.get(pair_id, [])
                if segment["label"] in {"a_wins", "b_wins"}
            ]
            if not decisive:
                raise PhysicsDpoSchemaError("训练 dataset 拒绝 all-tie 或无 segment 的 pair")
    core = {
        "schema_version": PREFERENCE_SCHEMA_VERSION,
        "scene_split_fingerprint": split_manifest["split_fingerprint"],
        "conditions": _canonical_records(indexed_conditions.values(), "condition_id"),
        "candidates": _canonical_records(indexed_candidates.values(), "candidate_id"),
        "preferences": _canonical_records(indexed_preferences.values(), "pair_id"),
        "segments": _canonical_records(indexed_segments.values(), "segment_id"),
        "require_segments": require_segments,
    }
    return {
        "condition_count": len(indexed_conditions),
        "candidate_count": len(indexed_candidates),
        "preference_count": len(indexed_preferences),
        "segment_count": len(indexed_segments),
        "dataset_fingerprint": sha256_json(core),
        "core": core,
    }


def materialize_scene_split(config: Mapping[str, Any], *, work_dir: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    """在 clean worktree 落盘唯一 split manifest；不读取 RGB、不写 cache。"""
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式 SAP-DPO scene split 拒绝在 dirty worktree 上运行")
    cfg = dict(config)
    run_id = _string(cfg.get("run_id"), label="run_id")
    target = Path(work_dir if work_dir is not None else _string(cfg.get("work_dir"), label="work_dir"))
    if target.exists():
        raise FileExistsError(f"scene split run 已存在: {target}")
    source = _mapping(cfg.get("source_manifests"), label="source_manifests")
    train_path = Path(_string(source.get("official_train"), label="source_manifests.official_train"))
    val_path = Path(_string(source.get("official_val"), label="source_manifests.official_val"))
    train = json.loads(train_path.read_text(encoding="utf-8"))
    val = json.loads(val_path.read_text(encoding="utf-8"))
    expected = _mapping(cfg.get("expected_source_fingerprints"), label="expected_source_fingerprints")
    if train.get("split_fingerprint") != expected.get("official_train") or val.get("split_fingerprint") != expected.get("official_val"):
        raise PhysicsDpoSchemaError("source split fingerprint 与预注册值不一致")
    split = build_scene_split(
        train,
        val,
        salt=_string(cfg.get("salt"), label="salt"),
        preference_dev_scene_count=_integer(cfg.get("preference_dev_scene_count"), label="preference_dev_scene_count", minimum=1),
        screen_eval_scene_count=_integer(cfg.get("screen_eval_scene_count"), label="screen_eval_scene_count", minimum=1),
    )
    resolved = json.loads(json.dumps(cfg, ensure_ascii=False, default=str))
    config_fingerprint = sha256_json(resolved)
    summary = {
        "status": "done",
        "task_id": str(cfg.get("task_id", "PA0-SCENE-SPLIT-01")),
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "split_fingerprint": split["split_fingerprint"],
        "partition_counts": {
            name: {"scenes": data["scene_count"], "clips": data["clip_count"]}
            for name, data in split["partitions"].items()
        },
        "next_gate": "PA1-HORIZON-01",
    }
    manifest = {
        "run_id": run_id,
        "task_id": summary["task_id"],
        "command": list(sys.argv),
        "config_fingerprint": config_fingerprint,
        "git": git,
        "environment": environment_fingerprint(),
        "source_files_sha256": {"official_train": file_fingerprint(str(train_path)), "official_val": file_fingerprint(str(val_path))},
        "status": "running",
        "started_at": utc_now(),
    }
    with atomic_directory(str(target)) as tmp_dir:
        output = Path(tmp_dir)
        atomic_write_text(str(output / "resolved.yaml"), OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True))
        atomic_write_json(str(output / "manifest.json"), manifest)
        atomic_write_json(str(output / "scene_split_manifest.json"), split)
        metrics = JsonlMetrics(str(output / "metrics.jsonl"))
        for step, name in enumerate(SCENE_PARTITIONS):
            item = split["partitions"][name]
            metrics.append(step, {"event": "partition_materialized", "partition": name, "scene_count": item["scene_count"], "clip_count": item["clip_count"]})
        atomic_write_json(str(output / "summary.json"), summary)
        manifest.update({"status": "done", "ended_at": utc_now(), "exit_reason": "scene_split_validated"})
        atomic_write_json(str(output / "manifest.json"), manifest)
        atomic_write_text(str(output / "COMPLETE"), sha256_json(summary) + "\n")
    return target, summary


def _load_config(path: str) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return dict(_mapping(loaded, label="scene split config"))


def main() -> None:
    parser = argparse.ArgumentParser(description="materialize SAP-DPO scene split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()
    run_dir, summary = materialize_scene_split(_load_config(args.config), work_dir=args.work_dir)
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
