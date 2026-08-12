#!/usr/bin/env python3
"""汇总 V3.3 replay 阶段，生成可注册的 development-only scene chain。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Optional

import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.baseline_scene_evaluator import evaluate_scene_records
from motion_proj.worldsim_v4.v33_replay import V33ReplayError, load_yaml, sha256_file


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, path)


def load_terminal_stage(
    run_dir: Path,
    *,
    expected_scene: str,
    expected_stage: str,
    expected_task_id: Optional[str] = None,
) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    summary_path = run_dir / "stage_summary.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if status.get("status") != "done" or summary.get("status") != "done":
        raise V33ReplayError(f"{expected_stage} 尚未完成")
    if summary.get("scene") != expected_scene or summary.get("stage") != expected_stage:
        raise V33ReplayError(f"{expected_stage} scene/stage 漂移")
    if expected_task_id is not None and any(
        payload.get("task_id") != expected_task_id for payload in (status, summary)
    ):
        raise V33ReplayError(f"{expected_stage} task_id 漂移")
    if summary.get("heldout_content_read") is not False or summary.get(
        "test_quality_read"
    ) is not False:
        raise V33ReplayError(f"{expected_stage} 读取了 heldout/test")
    expected_summary = status.get("stage_summary_sha256")
    if expected_summary is not None and expected_summary != sha256_file(summary_path):
        raise V33ReplayError(f"{expected_stage} stage summary SHA 漂移")
    return {
        "run": str(run_dir),
        "status_sha256": sha256_file(status_path),
        "summary_sha256": sha256_file(summary_path),
        "summary": summary,
    }


def resize_target(source: Path, target: Path, *, width: int, height: int) -> None:
    with Image.open(source) as image:
        rgb = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(target, format="PNG")


def load_development_actor_masks(
    instance_run: Path, *, instance_stage: Mapping[str, Any]
) -> dict[tuple[int, int], dict[str, Any]]:
    """读取并强校验 instance evaluation 产生的 development actor target。"""
    eval_summary_path = instance_run / "eval_targets" / "summary.json"
    mask_manifest_path = (
        instance_run
        / "eval_targets"
        / "artifacts"
        / "masks"
        / "mask_manifest.json"
    )
    for label, path in (
        ("instance eval summary", eval_summary_path),
        ("instance eval mask manifest", mask_manifest_path),
    ):
        if not path.is_file():
            raise V33ReplayError(f"{label} 缺失: {path}")
    if instance_stage.get("eval_summary_sha256") != sha256_file(eval_summary_path):
        raise V33ReplayError("instance eval summary SHA 漂移")
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    if (
        eval_summary.get("status") != "done"
        or eval_summary.get("evaluation_partition") != "development"
        or eval_summary.get("optimization_forbidden") is not True
    ):
        raise V33ReplayError("instance actor target 不是冻结 development evaluation")
    if eval_summary.get("mask_manifest_sha256") != sha256_file(mask_manifest_path):
        raise V33ReplayError("instance eval mask manifest SHA 漂移")
    declared_manifest = Path(eval_summary["mask_manifest"]).resolve()
    if declared_manifest != mask_manifest_path.resolve():
        raise V33ReplayError("instance eval mask manifest 路径漂移")

    manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("evaluation_partition") != "development"
        or manifest.get("optimization_forbidden") is not True
    ):
        raise V33ReplayError("actor mask manifest 不是 development-only")
    masks: dict[tuple[int, int], dict[str, Any]] = {}
    rows = manifest.get("masks", [])
    accepted_count = sum(
        bool(row.get("accepted")) and int(row.get("positive_pixels", 0)) > 0
        for row in rows
    )
    if accepted_count != int(eval_summary.get("accepted_mask_count", -1)):
        raise V33ReplayError("accepted mask 总数与 eval summary 不一致")
    development_frames = {int(value) for value in manifest.get("evaluation_frames", [])}
    for row in rows:
        if row.get("role") != "high_support":
            continue
        if not bool(row.get("accepted")) or int(row.get("positive_pixels", 0)) <= 0:
            continue
        key = (int(row["frame"]), int(row["camera_id"]))
        if key[0] not in development_frames:
            raise V33ReplayError(f"actor mask frame 不在 development partition: {key}")
        if key in masks:
            raise V33ReplayError(f"重复 accepted high_support actor mask: {key}")
        source = Path(row["mask"]).resolve()
        try:
            source.relative_to(instance_run.resolve())
        except ValueError as error:
            raise V33ReplayError(f"actor mask 不在 instance run 内: {source}") from error
        if not source.is_file() or sha256_file(source) != row.get("mask_sha256"):
            raise V33ReplayError(f"actor mask 缺失或 SHA 漂移: {key}")
        masks[key] = dict(row)
    if not masks:
        raise V33ReplayError("缺少 accepted development high_support actor mask")
    return masks


def materialize_actor_mask(
    row: Mapping[str, Any], target: Path, *, width: int, height: int
) -> None:
    source = Path(row["mask"])
    with np.load(source, allow_pickle=False) as archive:
        if "binary" not in archive.files:
            raise V33ReplayError(f"actor mask 缺少 binary array: {source}")
        value = np.asarray(archive["binary"])
    if value.shape != (height, width):
        raise V33ReplayError(
            f"actor mask shape 漂移: expected={(height, width)} actual={value.shape}"
        )
    binary = value.astype(bool, copy=False)
    positive = int(binary.sum())
    if positive <= 0 or positive != int(row["positive_pixels"]):
        raise V33ReplayError("actor mask positive pixel 计数漂移")
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary.astype(np.uint8) * 255, mode="L").save(target, format="PNG")


def build_records(
    *,
    spatial_run: Path,
    spatial_config: Mapping[str, Any],
    actor_masks: Mapping[tuple[int, int], Mapping[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    evaluation = json.loads(
        (spatial_run / "evaluation" / "summary.json").read_text(encoding="utf-8")
    )
    if evaluation.get("state") != "completed" or not evaluation["decision"]["accepted"]:
        raise V33ReplayError("spatial render evaluation 未通过")
    processed = Path(spatial_config["scene"]["processed_root"])
    width = int(spatial_config["scene"]["model_native_width"])
    height = int(spatial_config["scene"]["model_native_height"])
    development = {int(value) for value in spatial_config["scene"]["development_frames"]}
    rows = []
    for row in evaluation["rows"]:
        frame = int(row["frame"])
        camera = int(row["camera_id"])
        if frame not in development:
            raise V33ReplayError("render row 混入非 development frame")
        render_dir = spatial_run / "evaluation" / "artifacts" / "renders" / f"f{frame:03d}_c{camera}"
        prediction = render_dir / "base_only.png"
        source_target = processed / "images" / f"{frame:03d}_{camera}.jpg"
        for label, path in (("prediction", prediction), ("target", source_target)):
            if not path.is_file():
                raise V33ReplayError(f"{label} 缺失: {path}")
        actor_row = actor_masks.get((frame, camera))
        if actor_row is None:
            raise V33ReplayError(
                f"render row 缺少 accepted development actor mask: {(frame, camera)}"
            )
        target = output_dir / "targets" / f"f{frame:03d}_c{camera}.png"
        resize_target(source_target, target, width=width, height=height)
        dynamic_mask = output_dir / "actor_masks" / f"f{frame:03d}_c{camera}.png"
        materialize_actor_mask(actor_row, dynamic_mask, width=width, height=height)
        egocar = output_dir / "egocar_masks" / f"f{frame:03d}_c{camera}.png"
        egocar.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (width, height), color=0).save(egocar, format="PNG")
        rows.append(
            {
                "frame": frame,
                "camera_id": camera,
                "partition": "development",
                "prediction": str(prediction),
                "prediction_sha256": sha256_file(prediction),
                "target": str(target),
                "target_sha256": sha256_file(target),
                "dynamic_mask": str(dynamic_mask),
                "dynamic_mask_sha256": sha256_file(dynamic_mask),
                "dynamic_mask_source": "accepted_high_support_sam2_development_mask",
                "dynamic_mask_source_npz": str(Path(actor_row["mask"]).resolve()),
                "dynamic_mask_source_npz_sha256": actor_row["mask_sha256"],
                "dynamic_mask_positive_pixels": int(actor_row["positive_pixels"]),
                "egocar_mask": str(egocar),
                "egocar_mask_sha256": sha256_file(egocar),
            }
        )
    if not rows:
        raise V33ReplayError("render manifest rows 为空")
    return rows


def build_lpips_model(replay: Mapping[str, Any]):
    evaluator = replay["unified_evaluator"]
    for key in ("alexnet_weight", "lpips_alex_weight"):
        spec = evaluator[key]
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise V33ReplayError(f"{key} SHA 漂移")
    os.environ["TORCH_HOME"] = str(evaluator["torch_home"])
    import lpips

    model = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    model.eval()
    return model


def manifest(run_dir: Path, *, task_id: str = TASK_ID) -> dict[str, Any]:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "status.json"}:
            continue
        files.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "worldsim_v4_v33_scene_chain_manifest_v1",
        "task_id": task_id,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-config", type=Path, required=True)
    parser.add_argument("--spatial-config", type=Path, required=True)
    parser.add_argument("--semantic-run", type=Path, required=True)
    parser.add_argument("--instance-run", type=Path, required=True)
    parser.add_argument("--spatial-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"scene chain run 已存在：{args.run_dir}")
    replay = load_yaml(args.replay_config)
    spatial_config = load_yaml(args.spatial_config)
    task_id = str(replay.get("task_id", TASK_ID))
    if spatial_config.get("task_id") != task_id:
        raise V33ReplayError("spatial config task_id 与 replay 不一致")
    scene = str(spatial_config["scene"]["name"])
    semantic = load_terminal_stage(
        args.semantic_run.resolve(),
        expected_scene=scene,
        expected_stage="semantic_lift",
        expected_task_id=task_id,
    )
    instance = load_terminal_stage(
        args.instance_run.resolve(),
        expected_scene=scene,
        expected_stage="instance_field",
        expected_task_id=task_id,
    )
    spatial = load_terminal_stage(
        args.spatial_run.resolve(),
        expected_scene=scene,
        expected_stage="spatial_delta",
        expected_task_id=task_id,
    )
    args.run_dir.mkdir(parents=True)
    (args.run_dir / "artifacts").mkdir()
    rows = build_records(
        spatial_run=args.spatial_run.resolve(),
        spatial_config=spatial_config,
        actor_masks=load_development_actor_masks(
            args.instance_run.resolve(), instance_stage=instance["summary"]
        ),
        output_dir=args.run_dir / "artifacts",
    )
    render_manifest = {
        "schema_version": "worldsim_v4_v33_render_manifest_v1",
        "task_id": task_id,
        "scene": scene,
        "split": "development",
        "test_quality_read": False,
        "rows": rows,
    }
    atomic_json(args.run_dir / "render_manifest.json", render_manifest)
    metric_rows, metric_summary = evaluate_scene_records(
        rows, lpips_model=build_lpips_model(replay)
    )
    metrics = {
        "schema_version": "worldsim_v4_v33_scene_metrics_v1",
        "task_id": task_id,
        "scene": scene,
        "split": "development",
        "test_quality_read": False,
        "rows": metric_rows,
        "summary": metric_summary,
    }
    atomic_json(args.run_dir / "metrics.json", metrics)
    abstentions = spatial_config["stage_abstentions"]
    chain = {
        "schema_version": "worldsim_v4_v33_scene_chain_v1",
        "task_id": task_id,
        "scene": scene,
        "algorithm_commit": replay["algorithm"]["implementation_commit"],
        "base_checkpoint_sha256": spatial_config["inputs"]["checkpoint"]["sha256"],
        "partition_contract": "sample_index_mod_5",
        "test_quality_read": False,
        "stages": {
            "semantic_lift": {"status": "done", **semantic},
            "instance_field": {"status": "done", **instance},
            "roadpatch": dict(abstentions["roadpatch"]),
            "asset_harvester": dict(abstentions["asset_harvester"]),
            "spatial_delta": {"status": "done", **spatial},
            "semantic_render": {
                "status": "done",
                "run": str(args.spatial_run.resolve()),
                "render_manifest_sha256": sha256_file(args.run_dir / "render_manifest.json"),
                "metrics_sha256": sha256_file(args.run_dir / "metrics.json"),
            },
        },
    }
    atomic_json(args.run_dir / "scene_chain.json", chain)
    now = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": "worldsim_v4_v33_scene_chain_summary_v1",
        "task_id": task_id,
        "scene": scene,
        "status": "done",
        "algorithm_commit": chain["algorithm_commit"],
        "base_checkpoint_sha256": chain["base_checkpoint_sha256"],
        "partition_contract": chain["partition_contract"],
        "development_content_read": True,
        "heldout_content_read": False,
        "test_quality_read": False,
        "finished_at_utc": now,
    }
    atomic_json(args.run_dir / "summary.json", summary)
    atomic_json(
        args.run_dir / "manifest.json", manifest(args.run_dir, task_id=task_id)
    )
    atomic_json(
        args.run_dir / "status.json",
        {
            "schema_version": "worldsim_v4_v33_scene_chain_status_v1",
            "task_id": task_id,
            "scene": scene,
            "status": "done",
            "summary_sha256": sha256_file(args.run_dir / "summary.json"),
            "manifest_sha256": sha256_file(args.run_dir / "manifest.json"),
            "test_quality_read": False,
            "finished_at_utc": now,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
