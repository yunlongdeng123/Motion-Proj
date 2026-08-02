#!/usr/bin/env python3
"""Build and visually finalize the frozen M2 nuScenes actor-evaluation cohort."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from motion_proj.dynamic_editing_v2.actor_selection import select_cohort
from motion_proj.dynamic_editing_v2.nuscenes_actor_eval import (
    build_actor_candidates,
    load_table,
    scene_clip_samples,
    stream_filter_rows,
    stream_filter_tokens,
)


PROJECT = Path("/root/autodl-tmp/motion_proj")
RAW_ROOT = Path("/root/autodl-tmp/data/dynamic_recon/raw_subset/adgs_nuscenes_v1")
META = RAW_ROOT / "v1.0-trainval"
FRAME_TABLES = Path(
    "/root/autodl-tmp/data/dynamic_recon/manifests/adgs_nuscenes_v1_frame_tables.json"
)
TASK_ID = "DR-V2-M2-ACTOR-EVAL-01"
COMPONENT = "nuscenes-actor-evaluation-adapter-v2"


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def resource_sample(phase: str) -> dict[str, Any]:
    cgroup = Path("/sys/fs/cgroup")
    result: dict[str, Any] = {
        "timestamp": now(),
        "phase": phase,
        "disk_free_bytes": shutil.disk_usage("/root/autodl-tmp").free,
    }
    for name in ("memory.current", "memory.max", "memory.events"):
        path = cgroup / name
        if path.is_file():
            result[name.replace(".", "_")] = path.read_text().strip()
    return result


def initialize(run_dir: Path, config_path: Path) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run 目录非空，禁止覆盖: {run_dir}")
    config = load_json(config_path)
    run_dir.mkdir(parents=True)
    for name in ("artifacts", "environment", "logs", "qa", "source_snapshot", "stages"):
        (run_dir / name).mkdir()
    sources = [
        Path(__file__).resolve(),
        config_path.resolve(),
        PROJECT / "motion_proj/dynamic_editing_v2/__init__.py",
        PROJECT / "motion_proj/dynamic_editing_v2/schema.py",
        PROJECT / "motion_proj/dynamic_editing_v2/frame_mapping.py",
        PROJECT / "motion_proj/dynamic_editing_v2/actor_projection.py",
        PROJECT / "motion_proj/dynamic_editing_v2/actor_selection.py",
        PROJECT / "motion_proj/dynamic_editing_v2/nuscenes_actor_eval.py",
    ]
    source_snapshot = {}
    for source in sources:
        destination = run_dir / "source_snapshot" / source.name
        shutil.copy2(source, destination)
        source_snapshot[source.name] = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    resolved = {
        **config,
        "task_id": TASK_ID,
        "component": COMPONENT,
        "instance_id": run_dir.name,
        "raw_root": str(RAW_ROOT),
        "metadata_root": str(META),
        "frame_tables": str(FRAME_TABLES),
        "selection_time_boundary": "before any M3/M4 edit output exists",
        "coordinate_convention": {
            "box_size": "nuScenes wlh; explicit conversion to local lwh",
            "T_global_ego": "ego -> global",
            "T_ego_camera": "camera -> ego",
            "T_camera_global": "inverse(T_global_ego @ T_ego_camera)",
        },
    }
    resolved["config_fingerprint"] = canonical_sha256(resolved)
    write_json(run_dir / "resolved.yaml", resolved)
    status = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT, text=True)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": COMPONENT,
        "instance_id": run_dir.name,
        "status": "running",
        "started_at": now(),
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "project_git_status": status.splitlines(),
        "config_fingerprint": resolved["config_fingerprint"],
        "source_snapshot": source_snapshot,
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "terminal.json", {"status": "running", "updated_at": now(), "failure": None})
    (run_dir / "environment" / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        encoding="utf-8",
    )
    append_jsonl(run_dir / "resource.jsonl", resource_sample("start"))
    append_jsonl(run_dir / "logs/runner.jsonl", {"timestamp": now(), "event": "initialized"})
    (run_dir / "metrics.jsonl").touch()
    return resolved


def hash_inputs(run_dir: Path) -> dict[str, Any]:
    files = [
        FRAME_TABLES,
        *(META / name for name in (
            "scene.json", "sample.json", "sample_data.json", "sample_annotation.json",
            "instance.json", "category.json", "calibrated_sensor.json", "ego_pose.json",
            "sensor.json", "visibility.json",
        )),
    ]
    records = {}
    for path in files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"输入 metadata 缺失: {path}")
        records[path.name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(run_dir / "input_metadata_hashes.json", records)
    write_json(run_dir / "stages/input_hashes.json", {"status": "done", "file_count": len(records)})
    return records


def draw_trajectory(actor: dict[str, Any], destination: Path, role: str) -> None:
    canvas = Image.new("RGB", (800, 800), "white")
    draw = ImageDraw.Draw(canvas)
    points = [(float(row["translation_global"][0]), float(row["translation_global"][1])) for row in actor["raw_annotations"]]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 10.0)
    center_x, center_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    def pixel(point):
        return (
            int(400 + (point[0] - center_x) * 680 / span),
            int(400 - (point[1] - center_y) * 680 / span),
        )

    projected = [pixel(point) for point in points]
    if len(projected) > 1:
        draw.line(projected, fill=(220, 30, 30), width=4)
    for index, point in enumerate(projected):
        draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=(220, 30, 30))
        if index in (0, len(projected) - 1):
            draw.text((point[0] + 8, point[1] - 8), str(index), fill="black")
    draw.text((20, 20), f"{actor['scene_id']} {role}", fill="black")
    draw.text((20, 45), actor["instance_token"], fill="black")
    draw.text((20, 70), actor["category_name"], fill="black")
    draw.line((20, 750, 70, 750), fill=(220, 30, 30), width=4)
    draw.text((80, 740), "raw nuScenes 2Hz truth", fill="black")
    draw.line((20, 775, 70, 775), fill=(30, 80, 220), width=2)
    draw.text((80, 765), "interpolated: none (not truth)", fill="black")
    canvas.save(destination)


def draw_projection_panel(actor: dict[str, Any], destination: Path, role: str) -> dict[str, Any]:
    by_annotation: dict[str, list[dict[str, Any]]] = {}
    for observation in actor["camera_observations"]:
        by_annotation.setdefault(observation["annotation_token"], []).append(observation)
    annotation_token, observations = max(
        by_annotation.items(),
        key=lambda item: (sum(obs["projection"]["visible_area_px"] for obs in item[1]), item[0]),
    )
    by_camera = {row["camera"]: row for row in observations}
    panels = []
    audit = {"annotation_token": annotation_token, "cameras": {}}
    for camera in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"):
        observation = by_camera[camera]
        source = RAW_ROOT / observation["filename"]
        if not source.is_file() or source.stat().st_size <= 0:
            raise RuntimeError(f"QA 图像缺失: {source}")
        with Image.open(source) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            polygon = observation["projection"]["polygon_after_clip"]
            if len(polygon) >= 3:
                draw.line([tuple(point) for point in polygon + [polygon[0]]], fill=(0, 255, 0), width=6)
            center = observation["projection"]["center_pixel"]
            if center is not None:
                draw.ellipse((center[0] - 8, center[1] - 8, center[0] + 8, center[1] + 8), fill=(255, 255, 0))
            draw.rectangle((0, 0, 1600, 55), fill=(0, 0, 0))
            draw.text((10, 10), f"{camera} raw 2Hz | {role} | {actor['instance_token'][:12]}", fill="white")
            panels.append(image.resize((800, 450), Image.Resampling.LANCZOS))
        audit["cameras"][camera] = {
            "filename": observation["filename"],
            "sample_data_token": observation["sample_data_token"],
            "timestamp_delta_us": observation["timestamp_delta_us"],
            "sample_token_match": observation["sample_token_match"],
            "visible_area_px": observation["projection"]["visible_area_px"],
            "center_inside_image": observation["projection"]["center_inside_image"],
        }
    combined = Image.new("RGB", (2400, 450), "white")
    for index, panel in enumerate(panels):
        combined.paste(panel, (index * 800, 0))
    combined.save(destination)
    return audit


def write_cohort_tables(run_dir: Path, per_scene: dict[str, Any]) -> None:
    columns = [
        "scene", "instance_token", "category_name", "eligible", "selected_role",
        "support_score", "raw_annotation_count", "point_supported_annotation_count",
        "lidar_radar_point_sum", "valid_camera_observation_count",
        "center_inside_observation_count", "median_visible_area_px", "failure_reasons",
    ]
    combined = []
    for scene, selection in per_scene.items():
        roles = {row["instance_token"]: row["role"] for row in selection["selected"]}
        rows = []
        for actor in sorted(selection["actors"], key=lambda value: value["instance_token"]):
            support = actor["support_summary"]
            row = {
                "scene": scene,
                "instance_token": actor["instance_token"],
                "category_name": actor["category_name"],
                "eligible": support["eligible"],
                "selected_role": roles.get(actor["instance_token"], ""),
                "support_score": support["support_score"],
                "raw_annotation_count": support["raw_annotation_count"],
                "point_supported_annotation_count": support["point_supported_annotation_count"],
                "lidar_radar_point_sum": support["lidar_radar_point_sum"],
                "valid_camera_observation_count": support["valid_camera_observation_count"],
                "center_inside_observation_count": support["center_inside_observation_count"],
                "median_visible_area_px": support["median_visible_area_px"],
                "failure_reasons": "|".join(support["failure_reasons"]),
            }
            rows.append(row)
            combined.append(row)
        destination = run_dir / "qa" / f"{scene}__cohort.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    with (run_dir / "cohort_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(combined)


def build(run_dir: Path, config_path: Path) -> dict[str, Any]:
    resolved = initialize(run_dir, config_path)
    started = time.time()
    try:
        input_hashes = hash_inputs(run_dir)
        frame_tables = load_json(FRAME_TABLES)
        scenes = {row["name"]: row for row in load_table(META / "scene.json")}
        samples = load_table(META / "sample.json")
        categories = {row["token"]: row for row in load_table(META / "category.json")}
        wanted_frame_rows = [row for scene in resolved["scenes"] for row in frame_tables[scene]]
        sample_data_tokens = {row["sample_data_token"] for row in wanted_frame_rows}
        sample_data = stream_filter_tokens(META / "sample_data.json", sample_data_tokens)
        calibrated_tokens = {row["calibrated_sensor_token"] for row in sample_data.values()}
        ego_pose_tokens = {row["ego_pose_token"] for row in sample_data.values()}
        calibrated = stream_filter_tokens(META / "calibrated_sensor.json", calibrated_tokens)
        ego_poses = stream_filter_tokens(META / "ego_pose.json", ego_pose_tokens)

        clip_samples_by_scene = {}
        clip_sample_tokens = set()
        for scene_name in resolved["scenes"]:
            if scene_name not in scenes:
                raise RuntimeError(f"scene metadata 缺失: {scene_name}")
            timestamps = [int(row["timestamp"]) for row in frame_tables[scene_name] if row["camera"] == "CAM_FRONT"]
            clip = scene_clip_samples(samples, scenes[scene_name]["token"], min(timestamps), max(timestamps))
            clip_samples_by_scene[scene_name] = clip
            clip_sample_tokens.update(sample["token"] for sample in clip)
        annotation_rows = stream_filter_rows(
            META / "sample_annotation.json", clip_sample_tokens, "sample_token"
        )
        annotations = {row["token"]: row for row in annotation_rows}
        if len(annotations) != len(annotation_rows):
            raise RuntimeError("sample_annotation token 重复")
        instance_tokens = {row["instance_token"] for row in annotation_rows}
        instances = stream_filter_tokens(META / "instance.json", instance_tokens)

        per_scene = {}
        selected_actors = []
        failure_reasons: Counter[str] = Counter()
        for scene_name in resolved["scenes"]:
            clip = clip_samples_by_scene[scene_name]
            clip_tokens = {row["token"] for row in clip}
            scene_annotations = [row for row in annotations.values() if row["sample_token"] in clip_tokens]
            actors = build_actor_candidates(
                scene_name,
                scenes[scene_name]["token"],
                frame_tables[scene_name],
                clip,
                scene_annotations,
                instances,
                categories,
                sample_data,
                calibrated,
                ego_poses,
                int(resolved["max_camera_timestamp_delta_us"]),
            )
            selection = select_cohort(actors, resolved)
            per_scene[scene_name] = selection
            actor_by_token = {actor["instance_token"]: actor for actor in actors}
            for actor in actors:
                failure_reasons.update(actor["support_summary"]["failure_reasons"])
                append_jsonl(
                    run_dir / "metrics.jsonl",
                    {"type": "actor_support", "scene": scene_name, "instance_token": actor["instance_token"], **actor["support_summary"]},
                )
            for selected in selection["selected"]:
                actor = actor_by_token[selected["instance_token"]]
                selected_actors.append({"scene": scene_name, "role": selected["role"], "actor": actor})
            write_json(run_dir / f"actors_{scene_name}.json", selection)

        cohort = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "frozen_at": now(),
            "selection_config_fingerprint": resolved["config_fingerprint"],
            "selection_uses_edit_outputs": False,
            "scenes": {
                scene: {
                    "eligible_instance_tokens": data["eligible_instance_tokens"],
                    "selected": [
                        {
                            **row,
                            "support_summary": next(
                                actor["support_summary"] for actor in data["actors"] if actor["instance_token"] == row["instance_token"]
                            ),
                        }
                        for row in data["selected"]
                    ],
                    "slot_coverage": data["slot_coverage"],
                }
                for scene, data in per_scene.items()
            },
        }
        write_json(run_dir / "cohort_frozen.json", cohort)
        write_cohort_tables(run_dir, per_scene)
        qa_rows = []
        for selected in selected_actors:
            actor, scene, role = selected["actor"], selected["scene"], selected["role"]
            stem = f"{scene}__{role}__{actor['instance_token'][:12]}"
            draw_trajectory(actor, run_dir / "qa" / f"{stem}__trajectory.png", role)
            panel_audit = draw_projection_panel(actor, run_dir / "qa" / f"{stem}__projection.png", role)
            qa_rows.append({"scene": scene, "role": role, "instance_token": actor["instance_token"], **panel_audit})
        exact_token_coverage = sum(
            camera["sample_token_match"]
            for row in qa_rows
            for camera in row["cameras"].values()
        )
        expected_panel_cameras = len(qa_rows) * 3
        automatic_status = "pass" if exact_token_coverage == expected_panel_cameras else "fail"
        write_json(
            run_dir / "qa/automatic_qa.json",
            {
                "status": automatic_status,
                "exact_sample_token_mapping_count": exact_token_coverage,
                "expected_sample_token_mapping_count": expected_panel_cameras,
                "panels": qa_rows,
            },
        )
        if automatic_status != "pass":
            raise RuntimeError("representative panel 未达到 timestamp+sample_token 双重精确映射")
        write_json(
            run_dir / "stages/actor_build.json",
            {
                "status": "done",
                "scene_actor_counts": {scene: len(data["actors"]) for scene, data in per_scene.items()},
                "scene_eligible_counts": {scene: len(data["eligible_instance_tokens"]) for scene, data in per_scene.items()},
                "scene_selected_counts": {scene: len(data["selected"]) for scene, data in per_scene.items()},
            },
        )
        if any(len(per_scene[scene]["selected"]) < 1 for scene in resolved["scenes"]):
            raise RuntimeError("至少一个 pilot scene 没有合格 actor")
        summary = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "component": COMPONENT,
            "instance_id": run_dir.name,
            "status": "running",
            "gate": "awaiting_agent_visual_identity_projection_qa",
            "scene_eligible_counts": {scene: len(data["eligible_instance_tokens"]) for scene, data in per_scene.items()},
            "scene_selected_counts": {scene: len(data["selected"]) for scene, data in per_scene.items()},
            "selected": cohort["scenes"],
            "failure_reason_counts": dict(sorted(failure_reasons.items())),
            "raw_interpolated_separation": "pass",
            "input_metadata_hash_count": len(input_hashes),
            "wall_seconds_before_visual_qa": time.time() - started,
            "next": "visual QA then DR-V2-M3-EDIT-BASELINE-01",
        }
        write_json(run_dir / "summary.json", summary)
        (run_dir / "summary.md").write_text(
            "# DR-V2 M2 actor evaluation adapter\n\n"
            "- status: `running`\n"
            "- gate: `awaiting_agent_visual_identity_projection_qa`\n"
            f"- eligible: `{summary['scene_eligible_counts']}`\n"
            f"- selected: `{summary['scene_selected_counts']}`\n",
            encoding="utf-8",
        )
        append_jsonl(run_dir / "resource.jsonl", resource_sample("awaiting_visual_qa"))
        append_jsonl(run_dir / "logs/runner.jsonl", {"timestamp": now(), "event": "awaiting_visual_qa"})
        write_artifacts(run_dir)
        return summary
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        write_json(run_dir / "summary.json", {"schema_version": 1, "task_id": TASK_ID, "status": "blocked", "failure": failure})
        (run_dir / "summary.md").write_text(
            "# DR-V2 M2 blocked run\n\n"
            f"- failure: `{failure}`\n"
            "- conclusion: engineering failure only; no method-quality claim\n",
            encoding="utf-8",
        )
        write_json(run_dir / "terminal.json", {"status": "blocked", "updated_at": now(), "failure": failure})
        append_jsonl(run_dir / "logs/runner.jsonl", {"timestamp": now(), "event": "blocked", "failure": failure})
        write_artifacts(run_dir)
        raise


def write_artifacts(run_dir: Path) -> None:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial") or path.name == "artifacts.json":
            continue
        artifacts.append({"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(run_dir / "artifacts.json", {"schema_version": 1, "artifacts": artifacts})


def finalize_visual_qa(run_dir: Path, qa_verdict: Path) -> dict[str, Any]:
    terminal = load_json(run_dir / "terminal.json")
    summary = load_json(run_dir / "summary.json")
    verdict = load_json(qa_verdict)
    if terminal.get("status") != "running" or summary.get("gate") != "awaiting_agent_visual_identity_projection_qa":
        raise RuntimeError("run 不在可完成的 visual QA 状态")
    if verdict.get("verdict") not in {"pass", "fail"} or not verdict.get("reviewer") or not verdict.get("panels_reviewed"):
        raise RuntimeError("visual QA 记录不完整")
    destination = run_dir / "qa/visual_qa.json"
    if destination.exists():
        raise RuntimeError("visual QA 记录已存在，禁止覆盖")
    shutil.copy2(qa_verdict, destination)
    if verdict["verdict"] == "fail":
        failure = {"type": "VisualQAFailure", "message": verdict.get("notes", "visual QA failed")}
        summary.update({"status": "blocked", "gate": "visual_qa_failed", "completed_at": now(), "visual_qa": verdict, "failure": failure})
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "terminal.json", {"status": "blocked", "updated_at": now(), "failure": failure})
        append_jsonl(run_dir / "logs/runner.jsonl", {"timestamp": now(), "event": "visual_qa_failed", "failure": failure})
        write_artifacts(run_dir)
        return summary
    summary.update({"status": "done", "gate": "passed", "completed_at": now(), "visual_qa": verdict})
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(
        "# DR-V2 M2 actor evaluation adapter\n\n"
        "- status: `done`\n"
        f"- eligible: `{summary['scene_eligible_counts']}`\n"
        f"- selected: `{summary['scene_selected_counts']}`\n"
        f"- visual QA: `pass` by `{verdict['reviewer']}`\n"
        "- truth: raw 2Hz and visualization interpolation remain physically separated\n",
        encoding="utf-8",
    )
    write_json(run_dir / "terminal.json", {"status": "done", "updated_at": now(), "failure": None})
    manifest = load_json(run_dir / "manifest.json")
    manifest.update({"status": "done", "completed_at": now(), "visual_qa_sha256": sha256_file(destination)})
    write_json(run_dir / "manifest.json", manifest)
    append_jsonl(run_dir / "resource.jsonl", resource_sample("done"))
    append_jsonl(run_dir / "logs/runner.jsonl", {"timestamp": now(), "event": "done"})
    write_artifacts(run_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/dynamic_editing_v2/actor_selection_v1.yaml")
    parser.add_argument("--finalize-visual-qa", type=Path)
    args = parser.parse_args()
    if args.finalize_visual_qa:
        result = finalize_visual_qa(args.run_dir.resolve(), args.finalize_visual_qa.resolve())
    else:
        result = build(args.run_dir.resolve(), args.config.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
