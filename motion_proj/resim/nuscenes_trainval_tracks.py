#!/usr/bin/env python
"""从 nuScenes v1.0-trainval 标注流式重建 10Hz actor 轨迹（annotation-only）。

设计动机：full-domain N1 需要与 mini 完全一致的 10Hz cadence，才能沿用 N1 冻结的
帧级阈值（``min_track_frames`` 等）。mini 的 ``instances_info.json`` 由 DriveStudio
``NuScenesProcessor.save_objects`` + ``interpolate_boxes``（interpolate_N=4）生成，
过程**只使用关键帧标注**（global translation/size/rotation），不触碰任何相机/LiDAR
sweep。本模块在 2GB cgroup 限制下用 ``ijson`` 流式复刻同一逻辑：

- ``frame_idx = keyframe_index * (interpolate_N + 1)``；
- ``box_size`` 为 lwh（由 nuScenes 的 wlh 重排 [size[1], size[0], size[2]]）；
- ``obj_to_world`` 为由 quaternion+translation 构造的 4x4；
- class 过滤为 DriveStudio 的 ``NUSCENES_DYNAMIC_CLASSES``；
- 关键帧之间按 ``interpolate_boxes`` 线性插值平移、SLERP 插值旋转、线性插值 box_size。

输出与 DriveStudio ``instances_info.json`` 同 schema，``id`` 为 nuScenes instance_token，
因此可与 mini 既有产物按 token 对拍验证。
"""
from __future__ import annotations

import json
from pathlib import Path

import ijson
import numpy as np
from pyquaternion import Quaternion

# 与 third_party/drivestudio/datasets/nuscenes/nuscenes_preprocess.py 保持一致
NUSCENES_DYNAMIC_CLASSES = [
    "animal",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "vehicle.bicycle",
    "vehicle.motorcycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.trailer",
    "vehicle.truck",
]


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _float_list(values) -> list[float]:
    return [float(value) for value in values]


class TrainvalAnnotationSource:
    """加载 trainval 小表并在需要时流式扫描 sample_annotation。"""

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)
        self.meta = self.dataset_root / "v1.0-trainval"
        if not (self.meta / "scene.json").is_file():
            raise FileNotFoundError(f"缺少 trainval metadata: {self.meta}")
        scenes = _load_json(self.meta / "scene.json")
        logs = {row["token"]: row for row in _load_json(self.meta / "log.json")}
        self.scene_by_token = {row["token"]: row for row in scenes}
        self.scene_by_name = {row["name"]: row for row in scenes}
        self.map_name_by_scene = {
            row["token"]: logs[row["log_token"]]["location"] for row in scenes
        }
        # sample/instance 是 trainval 中最占 Python 对象内存的两张元数据表。
        # 下游只读取三个 sample 字段和 instance->category 映射，因此流式保留
        # 最小投影，避免 json.load 的完整行与最终索引同时驻留造成峰值翻倍。
        self.sample_by_token = {}
        with (self.meta / "sample.json").open("rb") as handle:
            for row in ijson.items(handle, "item"):
                self.sample_by_token[row["token"]] = {
                    "next": row["next"],
                    "timestamp": row["timestamp"],
                }
        categories = {
            row["token"]: row["name"]
            for row in _load_json(self.meta / "category.json")
        }
        self.category_by_instance = {}
        with (self.meta / "instance.json").open("rb") as handle:
            for row in ijson.items(handle, "item"):
                self.category_by_instance[row["token"]] = categories[
                    row["category_token"]
                ]

    def resolve_scene(self, entry: str) -> dict:
        row = self.scene_by_name.get(entry) or self.scene_by_token.get(entry)
        if row is None:
            raise KeyError(f"无法解析 scene: {entry}")
        return row

    def keyframe_order(self, scene: dict) -> dict[str, int]:
        """scene 内 sample_token -> keyframe index（按 first->next 链表顺序）。"""
        order = {}
        token = scene["first_sample_token"]
        index = 0
        while token:
            order[token] = index
            token = self.sample_by_token[token]["next"]
            index += 1
        return order


def _build_keyframe_records(
    source: TrainvalAnnotationSource, scenes: list[dict]
) -> dict[str, dict[str, dict[int, dict]]]:
    """流式扫描 sample_annotation，收集 scene->instance->keyframe->box。"""
    sample_to_scene = {}
    keyframe_index = {}
    for scene in scenes:
        order = source.keyframe_order(scene)
        for sample_token, index in order.items():
            sample_to_scene[sample_token] = scene["token"]
            keyframe_index[sample_token] = index

    records: dict[str, dict[str, dict[int, dict]]] = {
        scene["token"]: {} for scene in scenes
    }
    with (source.meta / "sample_annotation.json").open("rb") as handle:
        for ann in ijson.items(handle, "item"):
            sample_token = ann["sample_token"]
            scene_token = sample_to_scene.get(sample_token)
            if scene_token is None:
                continue
            instance_token = ann["instance_token"]
            category = source.category_by_instance.get(instance_token)
            if category not in NUSCENES_DYNAMIC_CLASSES:
                continue
            kf = keyframe_index[sample_token]
            records[scene_token].setdefault(instance_token, {})[kf] = {
                "category": category,
                "translation": _float_list(ann["translation"]),
                "size": _float_list(ann["size"]),
                "rotation": _float_list(ann["rotation"]),
            }
    return records


def _obj_to_world(translation: list[float], rotation: list[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def _keyframe_instances_info(
    source: TrainvalAnnotationSource,
    scene: dict,
    instance_records: dict[str, dict[int, dict]],
    interpolate_n: int,
) -> dict[str, dict]:
    """复刻 save_objects：按关键帧顺序构建稀疏 instances_info（frame_idx=kf*(N+1)）。"""
    # 稳定 first-appearance 排序：按最早出现 keyframe，再按 instance_token
    def first_key(item: tuple[str, dict[int, dict]]):
        return (min(item[1]), item[0])

    instances_info: dict[str, dict] = {}
    for instance_token, kf_map in sorted(instance_records.items(), key=first_key):
        frame_idx = []
        obj_to_world = []
        box_size = []
        for kf in sorted(kf_map):
            box = kf_map[kf]
            frame_idx.append(kf * (interpolate_n + 1))
            obj_to_world.append(_obj_to_world(box["translation"], box["rotation"]).tolist())
            size = box["size"]
            box_size.append([size[1], size[0], size[2]])  # wlh -> lwh
        instances_info[instance_token] = {
            "id": instance_token,
            "class_name": kf_map[min(kf_map)]["category"],
            "frame_annotations": {
                "frame_idx": frame_idx,
                "obj_to_world": obj_to_world,
                "box_size": box_size,
            },
        }
    return instances_info


def _interpolate_boxes(instances_info: dict[str, dict], interpolate_n: int) -> dict[str, dict]:
    """复刻 interpolate_boxes：关键帧间线性平移、SLERP 旋转、线性 box_size。"""
    result: dict[str, dict] = {}
    for obj_id, obj_info in instances_info.items():
        fa = obj_info["frame_annotations"]
        keyframe_indices = fa["frame_idx"]
        o2w_list = fa["obj_to_world"]
        size_list = fa["box_size"]
        new_frame_idx = []
        new_o2w = []
        new_size = []
        for i in range(len(keyframe_indices) - 1):
            start_frame = keyframe_indices[i]
            start_transform = np.asarray(o2w_list[i])
            end_transform = np.asarray(o2w_list[i + 1])
            start_quat = Quaternion(matrix=start_transform[:3, :3])
            end_quat = Quaternion(matrix=end_transform[:3, :3])
            start_size = np.asarray(size_list[i])
            end_size = np.asarray(size_list[i + 1])
            for j in range(interpolate_n + 1):
                t = j / (interpolate_n + 1)
                current_frame = start_frame + j
                translation = (1 - t) * start_transform[:3, 3] + t * end_transform[:3, 3]
                current_quat = Quaternion.slerp(start_quat, end_quat, t)
                current_transform = np.eye(4)
                current_transform[:3, :3] = current_quat.rotation_matrix
                current_transform[:3, 3] = translation
                current_size = (1 - t) * start_size + t * end_size
                new_frame_idx.append(current_frame)
                new_o2w.append(current_transform.tolist())
                new_size.append(current_size.tolist())
        new_frame_idx.append(keyframe_indices[-1])
        new_o2w.append(o2w_list[-1])
        new_size.append(size_list[-1])
        result[obj_id] = {
            "id": obj_info["id"],
            "class_name": obj_info["class_name"],
            "frame_annotations": {
                "frame_idx": new_frame_idx,
                "obj_to_world": new_o2w,
                "box_size": new_size,
            },
        }
    return result


def build_scene_instances_info(
    dataset_root: Path,
    scene_entries: list[str],
    interpolate_n: int = 4,
    cache_dir: Path | None = None,
    retain: bool = True,
    source: TrainvalAnnotationSource | None = None,
) -> dict[str, dict]:
    """为给定 scene 生成 10Hz interpolated instances_info。

    返回 ``scene_name -> {"scene_token", "map_name", "instances_info"?}``。
    - 若提供 ``cache_dir``，按 scene_name 缓存 instances_info.json；
    - 若 ``retain=False``，dense 轨迹写盘后不驻留内存（大规模评估时控内存），
      返回值中不含 ``instances_info``，仅含 scene_token/map_name 与缓存路径。
    """
    source = source or TrainvalAnnotationSource(dataset_root)
    scenes = [source.resolve_scene(entry) for entry in scene_entries]
    records = _build_keyframe_records(source, scenes)

    output: dict[str, dict] = {}
    for scene in scenes:
        sparse = _keyframe_instances_info(
            source, scene, records[scene["token"]], interpolate_n
        )
        dense = _interpolate_boxes(sparse, interpolate_n) if interpolate_n > 0 else sparse
        entry = {
            "scene_token": scene["token"],
            "scene_name": scene["name"],
            "map_name": source.map_name_by_scene[scene["token"]],
        }
        if cache_dir is not None:
            scene_dir = Path(cache_dir) / scene["name"] / "instances"
            scene_dir.mkdir(parents=True, exist_ok=True)
            cache_path = scene_dir / "instances_info.json"
            cache_path.write_text(
                json.dumps(dense, ensure_ascii=False), encoding="utf-8"
            )
            entry["cache_path"] = str(cache_path)
        if retain:
            entry["instances_info"] = dense
        output[scene["name"]] = entry
        # 释放当前 scene 的原始记录，降低峰值内存
        records[scene["token"]] = {}
    return output
