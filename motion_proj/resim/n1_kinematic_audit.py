"""为第三版 N1 生成带身份框、俯视轨迹和完整提示词的盲审包。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ijson
import matplotlib
import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.resim.nuscenes_trainval_tracks import _load_json
from motion_proj.runtime.atomic import atomic_write_json, atomic_write_text
from motion_proj.runtime.fingerprint import file_fingerprint


def _transform(translation: list[float], rotation: list[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def _camera_entries(source, scene_names: list[str]) -> dict[str, dict[int, dict]]:
    sensors = {
        row["token"]: row["channel"] for row in _load_json(source.meta / "sensor.json")
    }
    calibrated_rows = {
        row["token"]: row
        for row in _load_json(source.meta / "calibrated_sensor.json")
    }
    channel_by_calibrated = {
        token: sensors[row["sensor_token"]]
        for token, row in calibrated_rows.items()
    }
    sample_lookup = {}
    result: dict[str, dict[int, dict]] = {name: {} for name in scene_names}
    for name in scene_names:
        scene = source.resolve_scene(name)
        for sample_token, keyframe_index in source.keyframe_order(scene).items():
            sample_lookup[sample_token] = (name, keyframe_index)
    needed_ego_poses = set()
    with (source.meta / "sample_data.json").open("rb") as handle:
        for row in ijson.items(handle, "item"):
            key = sample_lookup.get(row["sample_token"])
            if key is None or not row["is_key_frame"]:
                continue
            if channel_by_calibrated[row["calibrated_sensor_token"]] != "CAM_FRONT":
                continue
            scene_name, keyframe_index = key
            frame = keyframe_index * 5
            result[scene_name][frame] = {
                "filename": row["filename"],
                "ego_pose_token": row["ego_pose_token"],
                "calibrated_sensor_token": row["calibrated_sensor_token"],
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
            needed_ego_poses.add(row["ego_pose_token"])
    ego_poses = {}
    with (source.meta / "ego_pose.json").open("rb") as handle:
        for row in ijson.items(handle, "item"):
            if row["token"] in needed_ego_poses:
                ego_poses[row["token"]] = row
    for by_frame in result.values():
        for entry in by_frame.values():
            calibration = calibrated_rows[entry["calibrated_sensor_token"]]
            pose = ego_poses[entry["ego_pose_token"]]
            entry["camera_intrinsic"] = calibration["camera_intrinsic"]
            entry["T_world_camera"] = (
                _transform(pose["translation"], pose["rotation"])
                @ _transform(calibration["translation"], calibration["rotation"])
            ).tolist()
    return result


def _actor_state(info: dict, instance_token: str, frame: int) -> dict | None:
    actor = info.get(instance_token)
    if actor is None:
        return None
    annotations = actor["frame_annotations"]
    try:
        index = annotations["frame_idx"].index(frame)
    except ValueError:
        return None
    return {
        "transform": np.asarray(annotations["obj_to_world"][index], dtype=float),
        "dimensions_lwh": np.asarray(annotations["box_size"][index], dtype=float),
    }


def _box_corners(state: dict) -> np.ndarray:
    length, width, height = state["dimensions_lwh"]
    local = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (-length / 2, length / 2)
            for y in (-width / 2, width / 2)
            for z in (-height / 2, height / 2)
        ],
        dtype=float,
    ).T
    return (state["transform"] @ local)[:3]


BOX_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def _draw_actor_box(
    draw: ImageDraw.ImageDraw,
    state: dict | None,
    camera: dict,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if state is None:
        return
    corners_world = _box_corners(state)
    homogeneous = np.vstack([corners_world, np.ones((1, corners_world.shape[1]))])
    camera_points = np.linalg.inv(np.asarray(camera["T_world_camera"])) @ homogeneous
    depths = camera_points[2]
    intrinsic = np.asarray(camera["camera_intrinsic"], dtype=float)
    pixels = intrinsic @ camera_points[:3]
    valid_depth = depths > 0.1
    pixels[:2, valid_depth] /= depths[valid_depth]
    for left, right in BOX_EDGES:
        if not (valid_depth[left] and valid_depth[right]):
            continue
        p1 = tuple(float(value) for value in pixels[:2, left])
        p2 = tuple(float(value) for value in pixels[:2, right])
        draw.line((p1, p2), fill=color, width=12)
    valid_indices = np.flatnonzero(valid_depth)
    if valid_indices.size:
        index = int(valid_indices[np.argmin(pixels[1, valid_indices])])
        x, y = (float(value) for value in pixels[:2, index])
        draw.rectangle((x - 4, y - 42, x + 84, y + 5), fill=(0, 0, 0))
        draw.text((x + 3, y - 35), label, fill=color, stroke_width=2)


def _camera_panel(
    dataset_root: Path,
    camera: dict | None,
    info: dict,
    frame: int,
    roles: dict[str, str],
    size: tuple[int, int] = (320, 180),
) -> Image.Image:
    if camera is None or not (dataset_root / camera["filename"]).is_file():
        image = Image.new("RGB", size, (30, 30, 30))
        ImageDraw.Draw(image).text((10, 10), f"missing CAM_FRONT frame={frame}", fill="white")
        return image
    image = Image.open(dataset_root / camera["filename"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {
        "SUBJECT": (255, 0, 255),
        "FRONT": (0, 220, 0),
        "REAR": (0, 160, 255),
    }
    for role, instance_token in roles.items():
        _draw_actor_box(
            draw,
            _actor_state(info, instance_token, frame),
            camera,
            colors[role],
            role[0],
        )
    draw.rectangle((0, 0, 190, 28), fill=(0, 0, 0))
    draw.text((8, 7), f"2Hz keyframe={frame}", fill=(255, 255, 255))
    image.thumbnail(size)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _trajectory_xy(info: dict, token: str, start: int, end: int) -> np.ndarray:
    actor = info[token]["frame_annotations"]
    values = [
        (np.asarray(transform, dtype=float)[0, 3], np.asarray(transform, dtype=float)[1, 3])
        for frame, transform in zip(actor["frame_idx"], actor["obj_to_world"])
        if frame % 5 == 0 and start <= frame <= end
    ]
    return np.asarray(values, dtype=float)


def _topdown_panel(
    output: Path,
    event: dict,
    info: dict,
    lane_index,
    roles: dict[str, str],
) -> None:
    crossing = int(event["crossing_frame"])
    subject_state = _actor_state(info, roles["SUBJECT"], int(round(crossing / 5) * 5))
    if subject_state is None:
        center = np.asarray(event["motion"].get("projected_xy", [0, 0]), dtype=float)
    else:
        center = subject_state["transform"][:2, 3]
    figure, axis = plt.subplots(figsize=(8, 6), dpi=120)
    for token, line in lane_index.centerlines.items():
        if (
            np.min(line[:, 0]) <= center[0] + 65
            and np.max(line[:, 0]) >= center[0] - 65
            and np.min(line[:, 1]) <= center[1] + 65
            and np.max(line[:, 1]) >= center[1] - 65
        ):
            axis.plot(line[:, 0], line[:, 1], color="#cccccc", linewidth=0.6, zorder=1)
    source_token = event["source_run"]["token"]
    target_token = event["target_run"]["token"]
    axis.plot(
        lane_index.centerlines[source_token][:, 0],
        lane_index.centerlines[source_token][:, 1],
        color="#ff8c00",
        linewidth=3,
        label="source lane",
        zorder=2,
    )
    axis.plot(
        lane_index.centerlines[target_token][:, 0],
        lane_index.centerlines[target_token][:, 1],
        color="#00bcd4",
        linewidth=3,
        label="target lane",
        zorder=2,
    )
    colors = {"SUBJECT": "#ff00ff", "FRONT": "#00aa00", "REAR": "#008cff"}
    for role, token in roles.items():
        trajectory = _trajectory_xy(info, token, crossing - 30, crossing + 40)
        if trajectory.size:
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                marker="o",
                markersize=3,
                linewidth=2,
                color=colors[role],
                label=role,
                zorder=3,
            )
    axis.set_xlim(center[0] - 55, center[0] + 55)
    axis.set_ylim(center[1] - 55, center[1] + 55)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    axis.set_title("2 Hz annotation trajectories and vector-map centerlines")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _metrics_text(event: dict, audit_id: str) -> str:
    motion = event["motion"]
    interaction = event["interaction"]
    join = motion["join_geometry"]

    def number(value, digits: int = 2) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    return "\n".join(
        [
            f"AUDIT ITEM: {audit_id}",
            "Colors: SUBJECT=magenta FRONT=green REAR=blue",
            f"machine type: {motion.get('maneuver_mode')}",
            f"source->target: {event['source_run']['token'][:8]} -> {event['target_run']['token'][:8]}",
            f"2Hz pre frames: {motion.get('pre_keyframe_indices')}",
            f"2Hz post frames: {motion.get('post_keyframe_indices')}",
            f"speed median: {number(motion.get('median_speed_mps'))} m/s",
            f"net course/yaw: {number(motion.get('net_course_change_deg'))}/{number(motion.get('net_yaw_change_deg'))} deg",
            f"lane preference pre/post: {number(motion.get('pre_lane_preference_margin_m'))}/{number(motion.get('post_lane_preference_margin_m'))} m",
            f"branch approach error: {number(join.get('source_target_approach_heading_error_deg'))} deg",
            f"branch disadvantage: {number(motion.get('merge_branch_alignment_disadvantage_deg'))}",
            f"front/rear bumper gap: {number(interaction.get('front_bumper_gap_m'))}/{number(interaction.get('rear_bumper_gap_m'))} m",
            f"identity support: {interaction.get('identity_support_keyframes')}/{len(interaction.get('frames', []))}",
            f"subject/front/rear speed: {interaction.get('subject_longitudinal_speed_mps')}/{interaction.get('front_longitudinal_speed_mps')}/{interaction.get('rear_longitudinal_speed_mps')}",
            "",
            "Do not infer TRUE from machine PASS.",
            "Judge the colored subject identity and its actual maneuver first.",
        ]
    )


def _combine_panel(
    output: Path,
    camera_images: list[Image.Image],
    topdown_path: Path,
    metrics: str,
) -> None:
    width = 1600
    camera_height = 180
    bottom_height = 600
    panel = Image.new("RGB", (width, camera_height + bottom_height), "white")
    for index, image in enumerate(camera_images[:5]):
        panel.paste(image, (index * 320, 0))
    topdown = Image.open(topdown_path).convert("RGB")
    topdown.thumbnail((800, bottom_height))
    panel.paste(topdown, ((800 - topdown.width) // 2, camera_height))
    draw = ImageDraw.Draw(panel)
    draw.multiline_text((830, camera_height + 25), metrics, fill=(0, 0, 0), spacing=8)
    panel.save(output, optimize=True)


def _review_prompt(
    run_dir: Path,
    event_pool_sha: str,
    population: int,
    count: int,
    config: dict,
    machine_summary: dict,
    machine_checks: dict,
) -> str:
    gates = config["human_audit"]
    return f"""# N1-KINEMATIC-01 第三次人工盲审完整提示词

## 1. 评测目的与非目标

你要独立判断第三版机器候选是否为真实的 vehicle lane-change / cut-in / converging-branch merge，
并确认 target corridor 上标成 FRONT 与 REAR 的车辆身份、方向、纵向次序和物理 gap 都成立。

本评测**不**评价渲染质量、碰撞安全、反事实可行性、N2 raw evidence、模型训练收益，也不决定任何
传感器下载。机器 kinematic PASS 只表示候选进入人审，绝不等于 TRUE_POSITIVE。

## 2. 盲法与禁止读取的信息

- 只按 `audit/index.html` 或 `audit/REVIEW_CHECKLIST.md` 的盲序 `K3-xxx` 审核；
- 可以读取同一 item 的 panel 与 `evidence/K3-xxx.json`；
- 禁止读取第二次人审文件、`calibration_audit.json`、旧 `N1-EVENT-FULL-01/audit/`、源码中的校准结果，
  或根据旧 reviewer/notes 猜答案；
- 禁止更改 `review_template.jsonl`、panel、evidence 或任何 hash 字段；只编辑
  `audit/review_working.jsonl` 的 verdict/reviewer/notes/failure_codes；
- 不因候选数量、scene 名、机器类型或阈值“应该通过”而给 TRUE。

## 3. 素材范围与颜色

- 每项 panel 顶部最多 5 张原始 CAM_FRONT 2 Hz keyframe，已投影 3D box：
  subject=洋红 `S`，front=绿色 `F`，rear=蓝色 `R`；
- 左下是官方 vector-map centerline 与三辆车的 2 Hz annotation 轨迹；
- 右下是只读运动学与 gap 数值；
- 世界轨迹和 box 来自 nuScenes annotation；10 Hz 插值不参与速度/加速度判定；
- 若角色不在 CAM_FRONT 视野，必须用俯视轨迹与其他时刻核对；证据仍不足则判 `UNCERTAIN`。

## 4. 逐项判定顺序与优先级

按以下顺序，前项失败即 overall=`FALSE_POSITIVE`：

1. **subject identity**：洋红框/轨迹确实对应同一辆目标车，没有把自车、邻车或另一辆并道车认成 subject；
2. **subject maneuver**：subject 在给定时窗内发生相邻同向 lane crossing 或从较不连续的支路收敛到主 corridor。
   主路 lane→connector→lane 正常续接、仅道路转弯、仅 token 边界切换均不是事件；
3. **target corridor**：青色 target 及灰色上下游链连续同向，没有跳到对向、横穿或错误岔路；
4. **front/rear**：绿色 F 位于 target corridor 前方、蓝色 R 位于后方，均为正确身份与同向车辆，
   且至少 2/3 个关键帧保持同一身份和次序；
5. **gap**：panel 中 bumper gap 与俯视几何相容，并落在冻结 `[0.5, 60] m`。

## 5. Overall verdict 定义

- `TRUE_POSITIVE`：上述 1–5 全部成立；
- `FALSE_POSITIVE`：任一项明确失败。必须填至少一个 failure code，并在 notes 指出可见证据；
- `UNCERTAIN`：现有相机+地图+annotation 仍无法确定。不得把“看不清”猜成 TRUE/FALSE，notes 必须写缺什么。

优先 failure codes：
`SUBJECT_IDENTITY_MISMATCH`、`SUBJECT_NO_LATERAL_MANEUVER`、`ROUTE_CONTINUATION`、
`NORMAL_TURN`、`MAP_MATCH_JITTER`、`INTERPOLATION_ONLY`、`WRONG_BRANCH`、
`OPPOSITE_OR_CROSS_TRAFFIC`、`FRONT_INVALID`、`REAR_INVALID`、`GAP_INVALID`、
`IDENTITY_NOT_PERSISTENT`、`INSUFFICIENT_VISUAL_EVIDENCE`、`OTHER`。

边界例：

- target 有两个 incoming，但 subject 沿最顺直的主路 incoming 进入 target：`FALSE_POSITIVE/ROUTE_CONTINUATION`；
- subject 世界轨迹近直，但相邻平行 lane 的距离偏好在多个 2 Hz keyframe 前后稳定翻转：可是真实 lane change，
  需结合框与中心线判断；
- 正常左/右转后进入新 token：`FALSE_POSITIVE/NORMAL_TURN`；
- front/rear 只在单帧出现或身份切换：`FALSE_POSITIVE/IDENTITY_NOT_PERSISTENT`；
- subject 不在前视画面但俯视 annotation 明确：可据俯视判；二者冲突则 `UNCERTAIN` 并说明。

## 6. JSONL 填写格式

逐行保留 `audit_id`、`evidence_sha256`、`panel_sha256`，填写：

```json
{{"audit_id":"K3-001","evidence_sha256":"...","panel_sha256":"...",
"subject_maneuver_verdict":"VALID|INVALID|UNCERTAIN",
"target_corridor_verdict":"VALID|INVALID|UNCERTAIN",
"front_relation_verdict":"VALID|INVALID|UNCERTAIN",
"rear_relation_verdict":"VALID|INVALID|UNCERTAIN",
"overall_verdict":"TRUE_POSITIVE|FALSE_POSITIVE|UNCERTAIN",
"failure_codes":["ROUTE_CONTINUATION"],"reviewer":"你的名字","notes":"基于哪些帧和轨迹作出判断"}}
```

不得删除、增加或重复 item；`FALSE_POSITIVE` 必须至少一个 failure code；所有项 reviewer/notes 非空。

## 7. 聚合阈值（结果查看前已冻结）

population={population}，本次以 SHA256 均匀盲抽/全审 count={count}；
parent event_pool SHA256=`{event_pool_sha}`。

只有同时满足才可建议第三版 N1 人审通过：

- 预注册研究支持 machine gate 全部通过。当前 machine summary：
  `{json.dumps(machine_summary, ensure_ascii=False, sort_keys=True)}`；
  checks：`{json.dumps(machine_checks, ensure_ascii=False, sort_keys=True)}`；
- 完整审核数 ≥ `{gates['min_reviewed_items']}`；
- TRUE_POSITIVE ≥ `{gates['min_true_positive_count']}`，且覆盖 ≥ `{gates['min_true_positive_scenes']}` scenes；
- determinate precision `TP/(TP+FP) ≥ {gates['min_precision']:.2f}`；
- Wilson 95% precision lower bound ≥ `{gates['min_wilson_95_lower_bound']:.2f}`；
- UNCERTAIN fraction ≤ `{gates['max_uncertain_fraction']:.2f}`。

任何一项失败 → 第三版 N1 `REJECTED`。全部通过也只得到“可请求下一步授权”的资格；本 run 明确
`n2_authorized=false`，不得启动 N2。

## 8. 完成后的精确命令与下一阶段影响

在仓库根目录执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
cd /root/autodl-tmp/motion_proj
PYTHONPATH=. python scripts/validate_n1_kinematic_review.py \\
  --run-dir {run_dir} \\
  --review-file {run_dir}/audit/review_working.jsonl
```

命令只校验/汇总人工填写，不会启动 N2。然后把 `review_working.jsonl` 路径和汇总输出交给 Codex；
最终 verdict 由用户确认并写入独立 audit adjudication run。无论输出为何，都不得改写本候选 run。
"""


def build_audit_pack(
    run_dir: Path,
    positives: list[dict],
    config: dict,
    source,
    lane_indices: dict,
    machine_summary: dict,
    machine_checks: dict,
) -> dict:
    audit_dir = run_dir / "audit"
    panels_dir = audit_dir / "panels"
    evidence_dir = audit_dir / "evidence"
    topdown_dir = audit_dir / "topdown"
    for directory in (audit_dir, panels_dir, evidence_dir, topdown_dir):
        directory.mkdir(parents=True, exist_ok=False)
    seed = str(config["human_audit"]["blind_order_seed"])
    ordered = sorted(
        positives,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['event_id']}".encode()).hexdigest(),
            row["event_id"],
        ),
    )
    selected = ordered[: int(config["human_audit"]["max_items"])]
    scenes = sorted({row["scene_id"] for row in selected})
    cameras = _camera_entries(source, scenes)
    dataset_root = Path(config["dataset_root"])
    cache_root = Path(config["cache_dir"])
    scene_cache = {}
    template_rows = []
    index_cards = []
    checklist = ["# N1-KINEMATIC-01 第三次人工审核清单", ""]

    for index, event in enumerate(selected, 1):
        audit_id = f"K3-{index:03d}"
        scene = event["scene_id"]
        if scene not in scene_cache:
            scene_cache[scene] = json.loads(
                (cache_root / scene / "instances" / "instances_info.json").read_text(
                    encoding="utf-8"
                )
            )
        info = scene_cache[scene]
        roles = {
            "SUBJECT": event["subject_instance_token"],
            "FRONT": event["front_instance_token"],
            "REAR": event["rear_instance_token"],
        }
        crossing = int(event["crossing_frame"])
        relation = int(event["interaction"]["center_frame"])
        frames = sorted(
            {
                int(round(value / 5) * 5)
                for value in (
                    crossing - 10,
                    crossing - 5,
                    crossing,
                    crossing + 5,
                    relation,
                )
            }
        )[:5]
        camera_images = [
            _camera_panel(
                dataset_root,
                cameras.get(scene, {}).get(frame),
                info,
                frame,
                roles,
            )
            for frame in frames
        ]
        while len(camera_images) < 5:
            camera_images.append(Image.new("RGB", (320, 180), (30, 30, 30)))
        topdown_path = topdown_dir / f"{audit_id}.png"
        map_name = source.map_name_by_scene[source.resolve_scene(scene)["token"]]
        _topdown_panel(topdown_path, event, info, lane_indices[map_name], roles)
        panel_path = panels_dir / f"{audit_id}.png"
        _combine_panel(
            panel_path,
            camera_images,
            topdown_path,
            _metrics_text(event, audit_id),
        )
        evidence = {
            "audit_id": audit_id,
            "event_id": event["event_id"],
            "scene_id": scene,
            "map_name": map_name,
            "camera_keyframes": [
                {
                    "frame": frame,
                    "path": (
                        str(dataset_root / cameras[scene][frame]["filename"])
                        if frame in cameras.get(scene, {})
                        else None
                    ),
                }
                for frame in frames
            ],
            "roles": roles,
            "source_run": event["source_run"],
            "target_run": event["target_run"],
            "crossing_frame": crossing,
            "relation_frame": relation,
            "topology": event["topology"],
            "motion": event["motion"],
            "interaction": event["interaction"],
            "event_record_sha256": event["event_record_sha256"],
        }
        evidence_path = evidence_dir / f"{audit_id}.json"
        atomic_write_json(str(evidence_path), evidence)
        evidence_sha = file_fingerprint(str(evidence_path))
        panel_sha = file_fingerprint(str(panel_path))
        template = {
            "audit_id": audit_id,
            "evidence_sha256": evidence_sha,
            "panel_sha256": panel_sha,
            "subject_maneuver_verdict": "",
            "target_corridor_verdict": "",
            "front_relation_verdict": "",
            "rear_relation_verdict": "",
            "overall_verdict": "",
            "failure_codes": [],
            "reviewer": "",
            "notes": "",
        }
        template_rows.append(template)
        checklist.extend(
            [
                f"## {audit_id}",
                "",
                f"![{audit_id}](panels/{audit_id}.png)",
                "",
                f"- evidence: `evidence/{audit_id}.json`",
                f"- verdict: ______ reviewer: ______ notes: ______",
                "",
            ]
        )
        index_cards.append(
            f"<section><h2>{audit_id}</h2>"
            f"<img src='panels/{audit_id}.png' alt='{audit_id}'>"
            f"<p><a href='evidence/{audit_id}.json'>evidence JSON</a></p></section>"
        )

    template_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in template_rows
    )
    atomic_write_text(str(audit_dir / "review_template.jsonl"), template_text)
    atomic_write_text(str(audit_dir / "review_working.jsonl"), template_text)
    atomic_write_text(
        str(audit_dir / "REVIEW_CHECKLIST.md"), "\n".join(checklist) + "\n"
    )
    index_html = (
        "<!doctype html><meta charset='utf-8'><title>N1 Kinematic Audit</title>"
        "<style>body{font-family:sans-serif;max-width:1660px;margin:auto}"
        "section{border-bottom:1px solid #ccc;padding:20px 0}"
        "img{width:100%;height:auto}</style><h1>N1-KINEMATIC-01 blind audit</h1>"
        + "".join(index_cards)
    )
    atomic_write_text(str(audit_dir / "index.html"), index_html)
    event_pool = json.loads((run_dir / "event_pool.json").read_text(encoding="utf-8"))
    prompt = _review_prompt(
        run_dir,
        event_pool["event_pool_sha256"],
        len(positives),
        len(selected),
        config,
        machine_summary,
        machine_checks,
    )
    atomic_write_text(str(audit_dir / "HUMAN_REVIEW_PROMPT.md"), prompt)

    immutable_files = [
        path
        for path in audit_dir.rglob("*")
        if path.is_file()
        and path.name not in {"review_working.jsonl", "audit_manifest.json"}
    ]
    file_hashes = {
        str(path.relative_to(audit_dir)): file_fingerprint(str(path))
        for path in sorted(immutable_files)
    }
    manifest = {
        "schema_version": "n1-kinematic-human-audit-pack-v1",
        "run_dir": str(run_dir),
        "event_pool_sha256": event_pool["event_pool_sha256"],
        "candidate_population_count": len(positives),
        "audit_item_count": len(selected),
        "selection": config["human_audit"]["sampling"],
        "blind_order_seed": seed,
        "selected_event_ids_sha256": canonical_sha256(
            [row["event_id"] for row in selected]
        ),
        "immutable_file_hashes": file_hashes,
        "immutable_artifact_set_sha256": canonical_sha256(file_hashes),
        "mutable_review_file": "review_working.jsonl",
        "human_verdict_filled": False,
    }
    atomic_write_json(str(audit_dir / "audit_manifest.json"), manifest)
    return manifest
