from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import ijson


SCHEMA_VERSION = "worldsim_v4_nuscenes_cohort_v1"
TASK_ID = "WS-V4-D0-NUSCENES-COHORT-01"
ROLES = {"development": 6, "validation": 6, "test": 18}
REQUIRED_CHANNELS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "LIDAR_TOP")
DYNAMIC_CATEGORY_PREFIXES = ("vehicle.",)
FORBIDDEN_SELECTION_KEYS = {
    "psnr",
    "ssim",
    "lpips",
    "boundary_f1",
    "iou",
    "model_score",
    "render_quality",
}


class CohortError(RuntimeError):
    """nuScenes cohort 合同不满足。"""


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stream_json(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item")


def _local_hour(logfile: str) -> int | None:
    match = re.search(
        r"-\d{4}-\d{2}-\d{2}-(\d{2})-\d{2}-\d{2}(?:[+-]\d{4})?$",
        logfile,
    )
    return int(match.group(1)) if match else None


def classify_scene_context(description: str, logfile: str) -> dict[str, str]:
    text = description.lower()
    hour = _local_hour(logfile)
    if hour is None:
        time_of_day = "unknown"
    elif hour >= 18 or hour < 6:
        time_of_day = "night"
    elif hour >= 16:
        time_of_day = "dusk"
    else:
        time_of_day = "day"

    weather = "rain" if any(word in text for word in ("rain", "wet")) else "dry_or_unspecified"
    if "roundabout" in text:
        road_geometry = "roundabout"
    elif any(word in text for word in ("intersection", "cross junction", "crossing junction")):
        road_geometry = "intersection"
    elif any(word in text for word in ("turn left", "turn right", "curve", "bend")):
        road_geometry = "turn_or_curve"
    elif any(word in text for word in ("straight", "highway", "road")):
        road_geometry = "road_segment"
    else:
        road_geometry = "general_urban"
    return {
        "time_of_day": time_of_day,
        "weather": weather,
        "road_geometry": road_geometry,
    }


def build_frame_partitions(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(samples, key=lambda row: (int(row["timestamp"]), str(row["token"])))
    partitions = {"train_frames": [], "development_frames": [], "heldout_frames": []}
    for index, sample in enumerate(ordered):
        record = {
            "index": index,
            "sample_token": str(sample["token"]),
            "timestamp": int(sample["timestamp"]),
        }
        if index % 5 == 4:
            partitions["heldout_frames"].append(record)
        elif index % 5 == 2:
            partitions["development_frames"].append(record)
        else:
            partitions["train_frames"].append(record)
    return partitions


def _longest_consecutive(observations: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    best: list[Mapping[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    previous = None
    for observation in sorted(observations, key=lambda row: int(row["sample_index"])):
        index = int(observation["sample_index"])
        if previous is None or index == previous + 1:
            current.append(observation)
        else:
            if len(current) > len(best):
                best = current
            current = [observation]
        previous = index
    if len(current) > len(best):
        best = current
    return best


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3])))


def _actor_descriptor(
    instance_token: str,
    category: str,
    observations: Sequence[Mapping[str, Any]],
    ego_by_sample: Mapping[str, Sequence[float]],
    scene_sample_count: int,
    clip_keyframes: int,
) -> dict[str, Any] | None:
    ordered = sorted(observations, key=lambda row: int(row["timestamp"]))
    consecutive = _longest_consecutive(ordered)
    if len(consecutive) < 5:
        return None
    visibility = statistics.fmean(float(row["visibility"]) for row in ordered)
    lidar_mean = statistics.fmean(max(0, int(row["num_lidar_pts"])) for row in ordered)
    radar_mean = statistics.fmean(max(0, int(row["num_radar_pts"])) for row in ordered)
    speeds: list[float] = []
    for left, right in zip(ordered, ordered[1:]):
        dt = (int(right["timestamp"]) - int(left["timestamp"])) / 1_000_000.0
        if dt > 0:
            speeds.append(_distance(left["translation"], right["translation"]) / dt)
    speed = statistics.median(speeds) if speeds else 0.0
    displacement = _distance(ordered[0]["translation"], ordered[-1]["translation"])
    distances = [
        _distance(row["translation"], ego_by_sample[str(row["sample_token"])])
        for row in ordered
        if str(row["sample_token"]) in ego_by_sample
    ]
    median_distance = statistics.median(distances) if distances else None
    support_score = (
        0.30 * min(len(ordered) / max(scene_sample_count, 1), 1.0)
        + 0.25 * min(visibility / 4.0, 1.0)
        + 0.25 * min(math.log1p(lidar_mean) / math.log(21.0), 1.0)
        + 0.15 * min(displacement / 10.0, 1.0)
        + 0.05 * (1.0 if median_distance is not None and median_distance <= 50.0 else 0.0)
    )
    window_size = min(max(5, clip_keyframes), len(consecutive))
    start = max(0, (len(consecutive) - window_size) // 2)
    clip = consecutive[start : start + window_size]
    return {
        "instance_token": instance_token,
        "category": category,
        "observation_count": len(ordered),
        "longest_consecutive_observations": len(consecutive),
        "mean_visibility": round(visibility, 6),
        "mean_lidar_points": round(lidar_mean, 6),
        "mean_radar_points": round(radar_mean, 6),
        "median_speed_mps": round(speed, 6),
        "displacement_m": round(displacement, 6),
        "median_ego_distance_m": round(median_distance, 6) if median_distance is not None else None,
        "support_score": round(support_score, 9),
        "observation_sample_tokens": [str(row["sample_token"]) for row in ordered],
        "clip_sample_tokens": [str(row["sample_token"]) for row in clip],
        "clip_start_index": int(clip[0]["sample_index"]),
        "clip_end_index": int(clip[-1]["sample_index"]),
        "clip_duration_s": round(
            (int(clip[-1]["timestamp"]) - int(clip[0]["timestamp"])) / 1_000_000.0,
            6,
        ),
    }


def _speed_bucket(speed: float | None) -> str:
    if speed is None:
        return "unknown"
    if speed < 0.5:
        return "stationary"
    if speed < 3.0:
        return "low_speed"
    return "normal_speed"


def _distance_bucket(distance: float | None) -> str:
    if distance is None:
        return "unknown"
    if distance < 25.0:
        return "near"
    if distance < 50.0:
        return "mid"
    return "far"


def _short_category(category: str | None) -> str:
    if not category:
        return "none"
    parts = category.split(".")
    return parts[1] if len(parts) > 1 else category


def _scene_strata(record: Mapping[str, Any]) -> set[str]:
    return {
        f"location={record['location']}",
        f"time={record['time_of_day']}",
        f"weather={record['weather']}",
        f"geometry={record['road_geometry']}",
        f"actor={record['actor_class']}",
        f"speed={record['speed_regime']}",
        f"distance={record['distance_regime']}",
        f"occlusion={record['occlusion']}",
        f"donor={record['donor_support']}",
    }


def _stable_tiebreak(seed: int, role: str, name: str) -> int:
    digest = hashlib.sha256(f"{seed}|{role}|{name}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def _greedy_diverse_select(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    role: str,
    seed: int,
    anchors: Sequence[str] = (),
) -> list[dict[str, Any]]:
    by_name = {str(row["scene"]): dict(row) for row in candidates}
    selected: list[dict[str, Any]] = []
    for name in anchors:
        if name not in by_name:
            raise CohortError(f"development anchor 不在可用 pool：{name}")
        selected.append(by_name.pop(name))
    frequencies = Counter(tag for row in candidates for tag in _scene_strata(row))
    covered = set().union(*(_scene_strata(row) for row in selected)) if selected else set()
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


def select_scene_cohort(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    development_anchors: Sequence[str],
) -> list[dict[str, Any]]:
    train_pool = [
        row for row in candidates if row["official_split"] == "train" and row["sensor_contract_complete"]
    ]
    val_pool = [
        row for row in candidates if row["official_split"] == "val" and row["sensor_contract_complete"]
    ]
    development = _greedy_diverse_select(
        train_pool, ROLES["development"], "development", seed, development_anchors
    )
    used = {str(row["scene"]) for row in development}
    validation_pool = [row for row in train_pool if str(row["scene"]) not in used]
    validation = _greedy_diverse_select(
        validation_pool, ROLES["validation"], "validation", seed
    )
    test = _greedy_diverse_select(val_pool, ROLES["test"], "test", seed)
    selected = []
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


def _metadata_fingerprints(meta: Path) -> dict[str, dict[str, Any]]:
    names = (
        "scene.json",
        "log.json",
        "sample.json",
        "sample_data.json",
        "sample_annotation.json",
        "instance.json",
        "category.json",
        "sensor.json",
        "calibrated_sensor.json",
        "ego_pose.json",
        "visibility.json",
    )
    return {
        name: {"bytes": (meta / name).stat().st_size, "sha256": sha256_file(meta / name)}
        for name in names
    }


def build_cohort(meta_root: str | Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta_root = Path(meta_root).resolve()
    meta = meta_root / str(config.get("version", "v1.0-trainval"))
    if not meta.is_dir():
        raise CohortError(f"nuScenes metadata 目录不存在：{meta}")
    scenes = _load_json(meta / "scene.json")
    logs = {row["token"]: row for row in _load_json(meta / "log.json")}
    categories = {row["token"]: row["name"] for row in _load_json(meta / "category.json")}
    instances = {
        row["token"]: categories[row["category_token"]]
        for row in _load_json(meta / "instance.json")
    }
    dynamic_instances = {
        token: category
        for token, category in instances.items()
        if category.startswith(DYNAMIC_CATEGORY_PREFIXES)
    }
    sample_rows = _load_json(meta / "sample.json")
    samples_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_lookup: dict[str, dict[str, Any]] = {}
    for row in sample_rows:
        item = {"token": row["token"], "timestamp": int(row["timestamp"]), "scene_token": row["scene_token"]}
        samples_by_scene[row["scene_token"]].append(item)
        sample_lookup[row["token"]] = item
    for rows in samples_by_scene.values():
        rows.sort(key=lambda item: (item["timestamp"], item["token"]))
        for index, row in enumerate(rows):
            row["sample_index"] = index

    sensor_channels = {row["token"]: row["channel"] for row in _load_json(meta / "sensor.json")}
    calibrated_channels = {
        row["token"]: sensor_channels[row["sensor_token"]]
        for row in _load_json(meta / "calibrated_sensor.json")
    }
    sensors_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    wanted_samples = set(sample_lookup)
    for row in _stream_json(meta / "sample_data.json"):
        sample_token = str(row["sample_token"])
        if sample_token not in wanted_samples or not bool(row["is_key_frame"]):
            continue
        channel = calibrated_channels.get(row["calibrated_sensor_token"])
        if channel not in REQUIRED_CHANNELS:
            continue
        sensors_by_sample[sample_token][channel] = {
            "sample_data_token": str(row["token"]),
            "filename": str(row["filename"]),
            "ego_pose_token": str(row["ego_pose_token"]),
            "timestamp": int(row["timestamp"]),
        }

    ego_tokens = {
        channels["LIDAR_TOP"]["ego_pose_token"]
        for channels in sensors_by_sample.values()
        if "LIDAR_TOP" in channels
    }
    ego_by_token: dict[str, Sequence[float]] = {}
    for row in _stream_json(meta / "ego_pose.json"):
        token = str(row["token"])
        if token in ego_tokens:
            ego_by_token[token] = [float(value) for value in row["translation"]]
    ego_by_sample = {
        sample_token: ego_by_token[channels["LIDAR_TOP"]["ego_pose_token"]]
        for sample_token, channels in sensors_by_sample.items()
        if "LIDAR_TOP" in channels and channels["LIDAR_TOP"]["ego_pose_token"] in ego_by_token
    }

    actor_observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _stream_json(meta / "sample_annotation.json"):
        instance_token = str(row["instance_token"])
        if instance_token not in dynamic_instances:
            continue
        sample = sample_lookup.get(str(row["sample_token"]))
        if sample is None:
            continue
        actor_observations[instance_token].append(
            {
                "annotation_token": str(row["token"]),
                "sample_token": str(row["sample_token"]),
                "sample_index": int(sample["sample_index"]),
                "scene_token": str(sample["scene_token"]),
                "timestamp": int(sample["timestamp"]),
                "visibility": int(row["visibility_token"]),
                "num_lidar_pts": int(row["num_lidar_pts"]),
                "num_radar_pts": int(row["num_radar_pts"]),
                "translation": [float(value) for value in row["translation"]],
            }
        )

    actors_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    clip_keyframes = int(config.get("continuous_clip_keyframes", 7))
    for instance_token, observations in actor_observations.items():
        scene_token = str(observations[0]["scene_token"])
        descriptor = _actor_descriptor(
            instance_token,
            dynamic_instances[instance_token],
            observations,
            ego_by_sample,
            len(samples_by_scene[scene_token]),
            clip_keyframes,
        )
        if descriptor is not None:
            actors_by_scene[scene_token].append(descriptor)

    from nuscenes.utils.splits import create_splits_scenes

    official = create_splits_scenes()
    official_lookup = {name: "train" for name in official["train"]}
    official_lookup.update({name: "val" for name in official["val"]})
    candidates: list[dict[str, Any]] = []
    for scene in sorted(scenes, key=lambda row: row["name"]):
        name = str(scene["name"])
        if name not in official_lookup:
            continue
        scene_token = str(scene["token"])
        samples = samples_by_scene[scene_token]
        actors = sorted(
            actors_by_scene.get(scene_token, []),
            key=lambda row: (-float(row["support_score"]), -int(row["observation_count"]), row["instance_token"]),
        )
        high = actors[0] if actors else None
        difficult_pool = actors[1:] if len(actors) > 1 else []
        difficult = (
            min(
                difficult_pool,
                key=lambda row: (
                    float(row["mean_visibility"]),
                    float(row["mean_lidar_points"]),
                    -int(row["observation_count"]),
                    row["instance_token"],
                ),
            )
            if difficult_pool
            else None
        )
        log = logs[scene["log_token"]]
        context = classify_scene_context(str(scene["description"]), str(log["logfile"]))
        all_sensor_complete = all(
            set(REQUIRED_CHANNELS).issubset(sensors_by_sample.get(str(sample["token"]), {}))
            for sample in samples
        )
        actor_density = sum(int(actor["observation_count"]) for actor in actors) / max(len(samples), 1)
        if actor_density <= 3:
            donor_support = "strong"
        elif actor_density <= 6:
            donor_support = "medium"
        else:
            donor_support = "weak"
        occlusion = (
            "heavy"
            if difficult is not None and float(difficult["mean_visibility"]) <= 2.5
            else "normal"
        )
        partitions = build_frame_partitions(samples)
        continuous_clip = (
            {
                "status": "ready",
                "actor_instance_token": high["instance_token"],
                "sample_tokens": high["clip_sample_tokens"],
                "start_index": high["clip_start_index"],
                "end_index": high["clip_end_index"],
                "duration_s": high["clip_duration_s"],
            }
            if high is not None
            else {"status": "ABSTAIN_NO_ACTOR", "sample_tokens": []}
        )
        candidates.append(
            {
                "scene": name,
                "scene_token": scene_token,
                "official_split": official_lookup[name],
                "description": str(scene["description"]),
                "location": str(log["location"]),
                **context,
                "actor_class": _short_category(high["category"] if high else None),
                "speed_regime": _speed_bucket(high["median_speed_mps"] if high else None),
                "distance_regime": _distance_bucket(high["median_ego_distance_m"] if high else None),
                "occlusion": occlusion,
                "donor_support": donor_support,
                "donor_support_is_metadata_proxy": True,
                "eligible_actor_count": len(actors),
                "actors": {
                    "high_support": high if high is not None else {"status": "ABSTAIN_NO_ACTOR"},
                    "difficult": difficult if difficult is not None else {"status": "ABSTAIN_NO_DIFFICULT_ACTOR"},
                },
                "edits": {
                    "remove": high["instance_token"] if high else "ABSTAIN_NO_ACTOR",
                    "lateral": high["instance_token"] if high else "ABSTAIN_NO_ACTOR",
                    "insert": high["instance_token"] if high else "ABSTAIN_NO_ACTOR",
                },
                "continuous_clip": continuous_clip,
                "camera_set": list(REQUIRED_CHANNELS[:3]),
                "lidar": "LIDAR_TOP",
                "sensor_contract_complete": all_sensor_complete,
                "sample_count": len(samples),
                **partitions,
            }
        )

    selected = select_scene_cohort(
        candidates,
        seed=int(config["selection_seed"]),
        development_anchors=list(config.get("development_anchors", [])),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "done",
        "selection_seed": int(config["selection_seed"]),
        "selection_uses_model_results": False,
        "selection_fields": [
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
        ],
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
        "metadata_fingerprints": _metadata_fingerprints(meta),
        "candidate_scene_count": len(candidates),
        "scene_counts": dict(ROLES),
        "scenes": selected,
    }
    validate_cohort(manifest, development_anchors=config.get("development_anchors", []))
    return manifest, candidates


def validate_cohort(
    manifest: Mapping[str, Any], *, development_anchors: Sequence[str] = ()
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CohortError(f"schema_version 必须为 {SCHEMA_VERSION}")
    if manifest.get("task_id") != TASK_ID or manifest.get("status") != "done":
        raise CohortError("cohort task/status 非法")
    if manifest.get("selection_uses_model_results") is not False:
        raise CohortError("cohort 选择不得读取模型结果")
    serialized_keys = {key.lower() for key in _walk_keys(manifest.get("selection_fields", []))}
    if serialized_keys & FORBIDDEN_SELECTION_KEYS:
        raise CohortError("selection_fields 含模型质量字段")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != sum(ROLES.values()):
        raise CohortError("cohort 必须恰好包含 30 scenes")
    names = [str(row["scene"]) for row in scenes]
    tokens = [str(row["scene_token"]) for row in scenes]
    if len(names) != len(set(names)) or len(tokens) != len(set(tokens)):
        raise CohortError("scene name/token 必须 scene-disjoint")
    counts = Counter(str(row["role"]) for row in scenes)
    if dict(counts) != ROLES:
        raise CohortError(f"role counts 必须为 {ROLES}，实际 {dict(counts)}")
    for row in scenes:
        role = str(row["role"])
        expected_split = "val" if role == "test" else "train"
        if row["official_split"] != expected_split:
            raise CohortError(f"{row['scene']} official split 泄漏")
        if row.get("sensor_contract_complete") is not True:
            raise CohortError(f"{row['scene']} sensor contract 不完整")
        if row.get("camera_set") != list(REQUIRED_CHANNELS[:3]) or row.get("lidar") != "LIDAR_TOP":
            raise CohortError(f"{row['scene']} camera/LiDAR contract 漂移")
        partitions = [row["train_frames"], row["development_frames"], row["heldout_frames"]]
        partition_tokens = [[str(item["sample_token"]) for item in part] for part in partitions]
        flattened = [token for part in partition_tokens for token in part]
        if len(flattened) != int(row["sample_count"]) or len(flattened) != len(set(flattened)):
            raise CohortError(f"{row['scene']} frame partitions 非完整互斥")
        if set(row["edits"]) != {"remove", "lateral", "insert"}:
            raise CohortError(f"{row['scene']} edit contract 不完整")
        actors = row.get("actors", {})
        high_actor = actors.get("high_support", {})
        difficult_actor = actors.get("difficult", {})
        if not isinstance(high_actor, Mapping) or not isinstance(difficult_actor, Mapping):
            raise CohortError(f"{row['scene']} actor contract 不完整")
        high_abstains = high_actor.get("status") == "ABSTAIN_NO_ACTOR"
        difficult_is_valid = bool(difficult_actor.get("instance_token")) or (
            difficult_actor.get("status") == "ABSTAIN_NO_DIFFICULT_ACTOR"
        )
        if not difficult_is_valid:
            raise CohortError(f"{row['scene']} difficult actor/ABSTAIN contract 不完整")
        clip = row.get("continuous_clip", {})
        if high_abstains:
            if set(row["edits"].values()) != {"ABSTAIN_NO_ACTOR"}:
                raise CohortError(f"{row['scene']} actor 缺失时 edit 必须显式 ABSTAIN")
            if clip.get("status") != "ABSTAIN_NO_ACTOR":
                raise CohortError(f"{row['scene']} actor 缺失时 clip 必须显式 ABSTAIN")
        else:
            actor_token = high_actor.get("instance_token")
            if not actor_token or set(row["edits"].values()) != {actor_token}:
                raise CohortError(f"{row['scene']} edit actor 与 high-support actor 不一致")
            if clip.get("status") != "ready" or clip.get("actor_instance_token") != actor_token:
                raise CohortError(f"{row['scene']} continuous clip actor 不一致")
            duration_s = float(clip.get("duration_s", 0.0))
            if not 2.0 <= duration_s <= 4.0:
                raise CohortError(f"{row['scene']} continuous clip 必须为 2–4 秒")
            if len(clip.get("sample_tokens", [])) < 5:
                raise CohortError(f"{row['scene']} continuous clip 关键帧不足")
    role_by_name = {row["scene"]: row["role"] for row in scenes}
    for name in development_anchors:
        if role_by_name.get(name) != "development":
            raise CohortError(f"infrastructure anchor 必须只在 development：{name}")


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)
    elif isinstance(value, str):
        yield value
