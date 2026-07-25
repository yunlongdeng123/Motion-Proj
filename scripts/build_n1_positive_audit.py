#!/usr/bin/env python
"""为 N1-EVENT-FULL-01 的 positive 事件生成人工盲评审计清单。

产出（写入 ``<run_dir>/audit/``）：
- ``positive_audit_checklist.md``：评判标准 + 每个 positive 的完整证据文件路径 + verdict 栏；
- ``positive_audit.jsonl``：逐事件结构化记录，含空 ``verdict`` / ``reviewer`` / ``notes`` 供填写。

证据基于**已在本机的关键帧**（``samples/CAM_FRONT`` 与 ``samples/LIDAR_TOP``）+ 官方矢量地图 +
标注轨迹缓存，无需等待 N2 sweeps。相机关键帧按 10Hz relation/crossing 帧就近映射到 2Hz keyframe。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ijson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.resim.nuscenes_trainval_tracks import (
    TrainvalAnnotationSource,
    _load_json,
)


def _channel_by_sd(source: TrainvalAnnotationSource) -> dict[str, str]:
    sensors = {r["token"]: r["channel"] for r in _load_json(source.meta / "sensor.json")}
    return {
        r["token"]: sensors[r["sensor_token"]]
        for r in _load_json(source.meta / "calibrated_sensor.json")
    }


def _keyframe_files(
    source: TrainvalAnnotationSource, scenes: list[str]
) -> dict[str, dict[int, dict[str, str]]]:
    """scene_name -> keyframe_index -> {CAM_FRONT, LIDAR_TOP} 关键帧文件名。"""
    cs = _channel_by_sd(source)
    order_by_scene = {}
    sample_to_scene = {}
    sample_to_kf = {}
    for name in scenes:
        scene = source.resolve_scene(name)
        order = source.keyframe_order(scene)
        order_by_scene[name] = order
        for sample_token, kf in order.items():
            sample_to_scene[sample_token] = name
            sample_to_kf[sample_token] = kf
    result: dict[str, dict[int, dict[str, str]]] = {name: {} for name in scenes}
    with (source.meta / "sample_data.json").open("rb") as handle:
        for row in ijson.items(handle, "item"):
            name = sample_to_scene.get(row["sample_token"])
            if name is None or not row["is_key_frame"]:
                continue
            channel = cs[row["calibrated_sensor_token"]]
            if channel in ("CAM_FRONT", "LIDAR_TOP"):
                kf = sample_to_kf[row["sample_token"]]
                result[name].setdefault(kf, {})[channel] = row["filename"]
    return result


def _actor_token_map(cache_dir: Path, scene_name: str) -> tuple[list[str], dict]:
    info = json.loads(
        (cache_dir / scene_name / "instances" / "instances_info.json").read_text(
            encoding="utf-8"
        )
    )
    tokens = sorted(info.keys())  # 与 driver 内 actor_id 枚举顺序一致
    return tokens, info


def _xy_at_frame(info: dict, instance_token: str, frame: int):
    fa = info[instance_token]["frame_annotations"]
    if frame in fa["frame_idx"]:
        idx = fa["frame_idx"].index(frame)
        m = fa["obj_to_world"][idx]
        return [round(float(m[0][3]), 2), round(float(m[1][3]), 2)]
    return None


def build(run_dir: Path, dataset_root: Path, cache_dir: Path) -> None:
    event_pool = json.loads((run_dir / "event_pool.json").read_text(encoding="utf-8"))
    positives = event_pool["evaluation"]["positives"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    scenes = sorted({p["scene_id"] for p in positives})

    source = TrainvalAnnotationSource(dataset_root)
    kf_files = _keyframe_files(source, scenes)
    token_cache: dict[str, tuple[list[str], dict]] = {}

    def cam_path(scene: str, frame_10hz: int) -> str | None:
        kf = round(frame_10hz / 5)
        entry = kf_files.get(scene, {}).get(kf)
        if entry and "CAM_FRONT" in entry:
            return str(dataset_root / entry["CAM_FRONT"])
        return None

    audit_dir = run_dir / "audit"
    audit_dir.mkdir(exist_ok=True)
    map_dir = dataset_root / "maps" / "expansion"

    rows = []
    for event in sorted(positives, key=lambda e: (e["scene_id"], e["actor_id"])):
        scene = event["scene_id"]
        if scene not in token_cache:
            token_cache[scene] = _actor_token_map(cache_dir, scene)
        tokens, info = token_cache[scene]
        actor_id = int(event["actor_id"])
        instance_token = tokens[actor_id]
        it = event["interaction"]
        front = it.get("front") or {}
        rear = it.get("rear") or {}
        front_token = tokens[int(front["actor_id"])] if front else None
        rear_token = tokens[int(rear["actor_id"])] if rear else None
        rf = int(event["relation_frame"])
        cf = int(event["crossing_frame"])
        scene_obj = source.resolve_scene(scene)
        map_name = source.map_name_by_scene[scene_obj["token"]]
        rows.append(
            {
                "event_id": event["event_id"],
                "scene": scene,
                "scene_token": scene_obj["token"],
                "map_name": map_name,
                "map_file": str(map_dir / f"{map_name}.json"),
                "actor_id": actor_id,
                "subject_instance_token": instance_token,
                "topology_type": event["topology"]["type"],
                "source_lane_token": event["source_run"]["token"],
                "target_lane_token": event["target_run"]["token"],
                "crossing_frame_10hz": cf,
                "relation_frame_10hz": rf,
                "front_gap_m": round(float(it["front_gap_m"]), 2),
                "rear_gap_m": round(float(it["rear_gap_m"]), 2),
                "front_instance_token": front_token,
                "rear_instance_token": rear_token,
                "front_on_exact_target_token": bool(front.get("same_exact_token")),
                "rear_on_exact_target_token": bool(rear.get("same_exact_token")),
                "corridor_neighbor_count": it.get("neighbor_count_on_corridor"),
                "subject_xy_at_relation": _xy_at_frame(info, instance_token, rf),
                "front_xy_at_relation": _xy_at_frame(info, front_token, rf)
                if front_token
                else None,
                "rear_xy_at_relation": _xy_at_frame(info, rear_token, rf)
                if rear_token
                else None,
                "cam_front_crossing": cam_path(scene, cf),
                "cam_front_relation": cam_path(scene, rf),
                "cam_front_relation_prev": cam_path(scene, rf - 5),
                "cam_front_relation_next": cam_path(scene, rf + 5),
                "instances_info_cache": str(
                    cache_dir / scene / "instances" / "instances_info.json"
                ),
                "event_pool_file": str(run_dir / "event_pool.json"),
                "verdict": "",  # TRUE_POSITIVE | FALSE_POSITIVE | UNCERTAIN
                "reviewer": "",
                "notes": "",
            }
        )

    jsonl_path = audit_dir / "positive_audit.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = _render_markdown(run_dir, summary, rows)
    (audit_dir / "positive_audit_checklist.md").write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {
                "audit_dir": str(audit_dir),
                "events": len(rows),
                "checklist": str(audit_dir / "positive_audit_checklist.md"),
                "jsonl": str(jsonl_path),
            },
            ensure_ascii=False,
        )
    )


def _render_markdown(run_dir: Path, summary: dict, rows: list[dict]) -> str:
    cross = sum(
        1
        for r in rows
        if not (r["front_on_exact_target_token"] and r["rear_on_exact_target_token"])
    )
    lines = []
    lines.append("# N1-EVENT-FULL-01 positive 事件人工盲评审计清单\n")
    lines.append(f"> **run**：`{summary['run_id']}`  ")
    lines.append(f"> **event_pool_sha256**：`{summary['event_pool_sha256']}`  ")
    lines.append(f"> **positive 总数**：{len(rows)}（其中 {cross} 个依赖 corridor 跨-token 邻车）  ")
    lines.append("> **审计目的**：独立判定每个 positive 是否为真实 cut-in/merge 且目标 corridor 上确有合理 front+rear。\n")

    lines.append("## 评判标准（每个事件三选一）\n")
    lines.append("- **TRUE_POSITIVE**：同时满足以下全部——")
    lines.append("  1. subject 在 relation 帧附近确实发生 lane change / merge（source→target lane 变更，图像可见并道/汇入）；")
    lines.append("  2. target corridor 上的 **front 与 rear 均为同向、同一可行驶 corridor 上的真实车辆**（非停放、非对向、非横穿/路口无关车）；")
    lines.append("  3. front/rear 纵向 gap 在物理合理范围（冻结 `[2, 60] m`），且与图像/俯视轨迹一致；")
    lines.append("  4. 若 front/rear 位于相邻 token（`*_on_exact_target_token=false`），该相邻 token 与 target 属于**连续同向** corridor，corridor 链未跳到断裂/反向/岔路 lane。")
    lines.append("- **FALSE_POSITIVE**：出现任一——无真实并道（地图/插值假象）；front/rear 为停放/对向/横穿/路口无关车；gap 不合理；corridor 链跨到不连续或反向 lane；同一目标其实是自车前后正常跟车而非并道交互。")
    lines.append("- **UNCERTAIN**：关键帧+地图证据不足以判定（记录缺哪类证据，N2 sweeps 到位后再判）。\n")
    lines.append("填写方式：编辑 `audit/positive_audit.jsonl` 每行的 `verdict`/`reviewer`/`notes`（verdict 取上述三值之一），或在本表末列直接标注。verdict 只能由用户/指定评审填写，不得由本 pipeline 回写。\n")

    lines.append("## 审计要点提示\n")
    lines.append("- `cam_front_relation` 为并道判定帧的前视关键帧；配合 `_prev`/`_next`（±0.5s）看并道过程；`cam_front_crossing` 为跨越帧。")
    lines.append("- `subject/front/rear_xy_at_relation` 为世界坐标（米），可在对应 `map_file` 上核对是否同一 corridor。")
    lines.append("- 逐事件完整字段见 `audit/positive_audit.jsonl`；原始事件记录在 `event_pool_file` 内按 `event_id` 检索。\n")

    lines.append("## 逐事件清单\n")
    for i, r in enumerate(rows, 1):
        lines.append(f"### {i}. `{r['event_id']}`  ({r['topology_type']})\n")
        lines.append(f"- scene / map：`{r['scene']}` / `{r['map_name']}`")
        lines.append(
            f"- subject actor_id={r['actor_id']} token=`{r['subject_instance_token']}`；"
            f"relation_frame(10Hz)={r['relation_frame_10hz']}，crossing={r['crossing_frame_10hz']}"
        )
        lines.append(
            f"- front gap={r['front_gap_m']} m (exact_token={r['front_on_exact_target_token']}, token=`{r['front_instance_token']}`)；"
            f"rear gap={r['rear_gap_m']} m (exact_token={r['rear_on_exact_target_token']}, token=`{r['rear_instance_token']}`)"
        )
        lines.append(
            f"- world xy @relation：subject={r['subject_xy_at_relation']}，"
            f"front={r['front_xy_at_relation']}，rear={r['rear_xy_at_relation']}"
        )
        lines.append(f"- source→target lane：`{r['source_lane_token']}` → `{r['target_lane_token']}`")
        lines.append("- 证据文件（完整路径）：")
        lines.append(f"  - CAM_FRONT @relation：`{r['cam_front_relation']}`")
        lines.append(f"  - CAM_FRONT @relation-0.5s：`{r['cam_front_relation_prev']}`")
        lines.append(f"  - CAM_FRONT @relation+0.5s：`{r['cam_front_relation_next']}`")
        lines.append(f"  - CAM_FRONT @crossing：`{r['cam_front_crossing']}`")
        lines.append(f"  - map：`{r['map_file']}`")
        lines.append(f"  - 轨迹缓存：`{r['instances_info_cache']}`")
        lines.append(f"  - 事件记录：`{r['event_pool_file']}`（event_id 检索）")
        lines.append("- **verdict**：______（TRUE_POSITIVE / FALSE_POSITIVE / UNCERTAIN）  reviewer：______  notes：______\n")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/autodl-tmp/data/nuscenes"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/root/autodl-tmp/data/occgs/processed_10Hz/trainval_annots"),
    )
    args = parser.parse_args()
    build(args.run_dir, args.dataset_root, args.cache_dir)


if __name__ == "__main__":
    main()
