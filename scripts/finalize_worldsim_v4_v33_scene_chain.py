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
from typing import Any, Mapping

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
    run_dir: Path, *, expected_scene: str, expected_stage: str
) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    summary_path = run_dir / "stage_summary.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if status.get("status") != "done" or summary.get("status") != "done":
        raise V33ReplayError(f"{expected_stage} 尚未完成")
    if summary.get("scene") != expected_scene or summary.get("stage") != expected_stage:
        raise V33ReplayError(f"{expected_stage} scene/stage 漂移")
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


def build_records(
    *, spatial_run: Path, spatial_config: Mapping[str, Any], output_dir: Path
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
        dynamic_mask = processed / "dynamic_masks" / "all" / f"{frame:03d}_{camera}.png"
        for label, path in (
            ("prediction", prediction),
            ("target", source_target),
            ("dynamic_mask", dynamic_mask),
        ):
            if not path.is_file():
                raise V33ReplayError(f"{label} 缺失: {path}")
        target = output_dir / "targets" / f"f{frame:03d}_c{camera}.png"
        resize_target(source_target, target, width=width, height=height)
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


def manifest(run_dir: Path) -> dict[str, Any]:
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
        "task_id": TASK_ID,
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
    scene = str(spatial_config["scene"]["name"])
    semantic = load_terminal_stage(
        args.semantic_run.resolve(), expected_scene=scene, expected_stage="semantic_lift"
    )
    instance = load_terminal_stage(
        args.instance_run.resolve(), expected_scene=scene, expected_stage="instance_field"
    )
    spatial = load_terminal_stage(
        args.spatial_run.resolve(), expected_scene=scene, expected_stage="spatial_delta"
    )
    args.run_dir.mkdir(parents=True)
    (args.run_dir / "artifacts").mkdir()
    rows = build_records(
        spatial_run=args.spatial_run.resolve(),
        spatial_config=spatial_config,
        output_dir=args.run_dir / "artifacts",
    )
    render_manifest = {
        "schema_version": "worldsim_v4_v33_render_manifest_v1",
        "task_id": TASK_ID,
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
        "task_id": TASK_ID,
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
        "task_id": TASK_ID,
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
        "task_id": TASK_ID,
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
    atomic_json(args.run_dir / "manifest.json", manifest(args.run_dir))
    atomic_json(
        args.run_dir / "status.json",
        {
            "schema_version": "worldsim_v4_v33_scene_chain_status_v1",
            "task_id": TASK_ID,
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
