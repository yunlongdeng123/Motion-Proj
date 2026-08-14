from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Any, Mapping, Sequence

from motion_proj.worldsim_v4.datasets.nuscenes import (
    CohortError,
    REQUIRED_CHANNELS,
    _scene_strata,
    build_cohort as build_v4_metadata_candidates,
    canonical_json_sha256,
)


SCHEMA_VERSION = "worldsim_v5_nuscenes_fresh_cohort_v1"
TASK_ID = "WS-V5-D0-NUSCENES-FRESH-COHORT-01"
ROLES = {"development": 8, "validation": 8, "test": 20}
SELECTION_FIELDS = (
    "official_split",
    "location",
    "time_of_day",
    "weather",
    "road_geometry",
    "actor_class",
    "speed_regime",
    "distance_regime",
    "occlusion",
    "donor_support_metadata_proxy",
    "sensor_contract",
    "eligible_actor_count",
)
FORBIDDEN_SELECTION_FIELDS = {
    "reconstruction_quality",
    "edit_quality",
    "m1_quality",
    "m2_quality",
    "m3_quality",
    "test_metric",
    "psnr",
    "ssim",
    "lpips",
    "boundary_f1",
    "iou",
    "model_score",
    "render_quality",
}


def _stable_tiebreak(seed: int, role: str, name: str) -> int:
    digest = hashlib.sha256(f"{seed}|{role}|{name}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def _greedy_diverse_select(
    candidates: Sequence[Mapping[str, Any]], count: int, role: str, seed: int
) -> list[dict[str, Any]]:
    by_name = {str(row["scene"]): dict(row) for row in candidates}
    selected: list[dict[str, Any]] = []
    frequencies = Counter(tag for row in candidates for tag in _scene_strata(row))
    covered: set[str] = set()
    while len(selected) < count:
        if not by_name:
            raise CohortError(f"{role} 候选不足，无法选择 {count} scene")

        def rank(row: Mapping[str, Any]) -> tuple[float, int, str]:
            new_tags = _scene_strata(row) - covered
            diversity = math.fsum(
                1.0 / max(frequencies[tag], 1) for tag in sorted(new_tags)
            )
            actor_bonus = min(int(row["eligible_actor_count"]), 10) * 0.001
            return (
                diversity + actor_bonus,
                -_stable_tiebreak(seed, role, str(row["scene"])),
                str(row["scene"]),
            )

        chosen = max(by_name.values(), key=rank)
        selected.append(chosen)
        covered.update(_scene_strata(chosen))
        by_name.pop(str(chosen["scene"]))
    return selected


def select_fresh_scene_cohort(
    candidates: Sequence[Mapping[str, Any]], *, seed: int, excluded_scenes: Sequence[str]
) -> list[dict[str, Any]]:
    excluded = set(excluded_scenes)
    eligible = [
        row
        for row in candidates
        if str(row["scene"]) not in excluded and row.get("sensor_contract_complete") is True
    ]
    train_pool = [row for row in eligible if row["official_split"] == "train"]
    val_pool = [row for row in eligible if row["official_split"] == "val"]
    development = _greedy_diverse_select(
        train_pool, ROLES["development"], "development", seed
    )
    used = {str(row["scene"]) for row in development}
    validation = _greedy_diverse_select(
        [row for row in train_pool if str(row["scene"]) not in used],
        ROLES["validation"],
        "validation",
        seed,
    )
    test = _greedy_diverse_select(val_pool, ROLES["test"], "test", seed)
    selected: list[dict[str, Any]] = []
    for role, rows in (
        ("development", development),
        ("validation", validation),
        ("test", test),
    ):
        for row in rows:
            item = dict(row)
            item["role"] = role
            selected.append(item)
    return selected


def _validate_scene_record(row: Mapping[str, Any]) -> None:
    role = str(row["role"])
    expected_split = "val" if role == "test" else "train"
    if row.get("official_split") != expected_split:
        raise CohortError(f"{row['scene']} official split 泄漏")
    if row.get("sensor_contract_complete") is not True:
        raise CohortError(f"{row['scene']} sensor contract 不完整")
    if row.get("camera_set") != list(REQUIRED_CHANNELS[:3]) or row.get("lidar") != "LIDAR_TOP":
        raise CohortError(f"{row['scene']} camera/LiDAR contract 漂移")
    partitions = [row["train_frames"], row["development_frames"], row["heldout_frames"]]
    tokens = [str(item["sample_token"]) for part in partitions for item in part]
    if len(tokens) != int(row["sample_count"]) or len(tokens) != len(set(tokens)):
        raise CohortError(f"{row['scene']} frame partition 非完整互斥")
    if set(row.get("edits", {})) != {"remove", "lateral", "insert"}:
        raise CohortError(f"{row['scene']} edit contract 不完整")
    high = row.get("actors", {}).get("high_support", {})
    difficult = row.get("actors", {}).get("difficult", {})
    if not isinstance(high, Mapping) or not isinstance(difficult, Mapping):
        raise CohortError(f"{row['scene']} actor contract 不完整")
    if not (difficult.get("instance_token") or difficult.get("status") == "ABSTAIN_NO_DIFFICULT_ACTOR"):
        raise CohortError(f"{row['scene']} difficult actor/ABSTAIN contract 不完整")
    clip = row.get("continuous_clip", {})
    if high.get("status") == "ABSTAIN_NO_ACTOR":
        if set(row["edits"].values()) != {"ABSTAIN_NO_ACTOR"}:
            raise CohortError(f"{row['scene']} absent actor edit 未 ABSTAIN")
        if clip.get("status") != "ABSTAIN_NO_ACTOR":
            raise CohortError(f"{row['scene']} absent actor clip 未 ABSTAIN")
    else:
        actor_token = high.get("instance_token")
        if not actor_token or set(row["edits"].values()) != {actor_token}:
            raise CohortError(f"{row['scene']} edit actor 与 high-support actor 不一致")
        if clip.get("status") != "ready" or clip.get("actor_instance_token") != actor_token:
            raise CohortError(f"{row['scene']} continuous clip actor 不一致")
        if not 2.0 <= float(clip.get("duration_s", 0.0)) <= 4.0:
            raise CohortError(f"{row['scene']} continuous clip 必须为 2–4 秒")
        if len(clip.get("sample_tokens", [])) < 5:
            raise CohortError(f"{row['scene']} continuous clip 关键帧不足")


def validate_fresh_cohort(
    manifest: Mapping[str, Any], *, excluded_scenes: Sequence[str]
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CohortError("V5 fresh cohort schema drift")
    if manifest.get("task_id") != TASK_ID or manifest.get("status") != "done":
        raise CohortError("V5 fresh cohort task/status drift")
    if manifest.get("selection_uses_model_results") is not False:
        raise CohortError("fresh cohort 不得读取模型结果")
    fields = set(manifest.get("selection_fields", []))
    if fields != set(SELECTION_FIELDS) or fields & FORBIDDEN_SELECTION_FIELDS:
        raise CohortError("fresh cohort selection field contract drift")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != sum(ROLES.values()):
        raise CohortError("fresh cohort 必须恰好包含 36 scenes")
    names = [str(row["scene"]) for row in scenes]
    tokens = [str(row["scene_token"]) for row in scenes]
    if len(names) != len(set(names)) or len(tokens) != len(set(tokens)):
        raise CohortError("fresh cohort scene name/token 必须互斥")
    overlap = set(names) & set(excluded_scenes)
    if overlap:
        raise CohortError(f"fresh cohort 与 V4 overlap: {sorted(overlap)}")
    counts = Counter(str(row["role"]) for row in scenes)
    if dict(counts) != ROLES:
        raise CohortError(f"fresh cohort role count drift: {dict(counts)}")
    for row in scenes:
        _validate_scene_record(row)


def build_fresh_cohort(
    meta_root: str,
    protocol: Mapping[str, Any],
    *,
    seed: int,
    excluded_scenes: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(excluded_scenes) != len(set(excluded_scenes)):
        raise CohortError("V4 exclusion scene 重复")
    v4_protocol = {
        "version": protocol.get("version", "v1.0-trainval"),
        "selection_seed": seed,
        "development_anchors": [],
        "continuous_clip_keyframes": int(protocol.get("continuous_clip_keyframes", 7)),
    }
    v4_manifest, candidates = build_v4_metadata_candidates(meta_root, v4_protocol)
    candidate_names = {str(row["scene"]) for row in candidates}
    if not set(excluded_scenes) <= candidate_names:
        raise CohortError(
            f"V4 exclusion 不在 metadata pool: {sorted(set(excluded_scenes) - candidate_names)}"
        )
    selected = select_fresh_scene_cohort(
        candidates, seed=seed, excluded_scenes=excluded_scenes
    )
    metadata_fingerprints = v4_manifest["metadata_fingerprints"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "done",
        "selection_seed": seed,
        "selection_uses_model_results": False,
        "selection_fields": list(SELECTION_FIELDS),
        "official_split_contract": {
            "development": "nuscenes_train",
            "validation": "nuscenes_train",
            "test": "nuscenes_val",
        },
        "frame_partition_rule": {
            "heldout": "sample_index_mod_5_eq_4",
            "development": "sample_index_mod_5_eq_2",
            "train": "remaining",
        },
        "required_channels": list(REQUIRED_CHANNELS),
        "metadata_root": str(meta_root),
        "metadata_fingerprints": metadata_fingerprints,
        "metadata_inventory_sha256": canonical_json_sha256(metadata_fingerprints),
        "candidate_scene_count": len(candidates),
        "excluded_v4_scene_count": len(excluded_scenes),
        "excluded_v4_scenes": list(excluded_scenes),
        "scene_counts": dict(ROLES),
        "scenes": selected,
    }
    validate_fresh_cohort(manifest, excluded_scenes=excluded_scenes)
    return manifest, candidates
