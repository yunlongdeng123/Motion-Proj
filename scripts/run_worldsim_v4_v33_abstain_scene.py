#!/usr/bin/env python3
"""为冻结 V3.3 的不可用 D0 actor 生成可审计的 base-only 弃权链。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.baseline_scene_evaluator import evaluate_scene_records
from motion_proj.worldsim_v4.v33_replay import V33ReplayError, load_yaml, sha256_file
from scripts.lift_worldsim_v32_semantics import build_runtime
from scripts.run_worldsim_v32_s2_3dgic import render_snapshot


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
RUN_ROOT = Path(f"/root/autodl-tmp/runs/worldsim_v4/{TASK_ID}")
ABSTAIN_REASON = "ABSTAIN_NO_ACTOR"
ABSTAIN_STAGES = (
    "semantic_lift",
    "instance_field",
    "roadpatch",
    "asset_harvester",
    "spatial_delta",
)
IMAGE_NAME = re.compile(r"^(?P<frame>\d+)_(?P<camera>\d+)\.jpg$")
SNAPSHOT_FILES = (
    "motion_proj/worldsim_v4/baseline_scene_evaluator.py",
    "motion_proj/worldsim_v4/evaluator.py",
    "motion_proj/worldsim_v4/region_masks.py",
    "motion_proj/worldsim_v4/v33_replay.py",
    "scripts/lift_worldsim_v32_semantics.py",
    "scripts/run_worldsim_v32_s2_3dgic.py",
    "scripts/run_worldsim_v4_v33_abstain_scene.py",
)


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


def verified_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise V33ReplayError(f"{label} 缺失或 SHA 漂移")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise V33ReplayError(f"{label} bytes 漂移")
    return path


def validate_no_actor_contract(
    replay: Mapping[str, Any], bound: Mapping[str, Any]
) -> dict[str, Any]:
    if replay.get("schema_version") != "worldsim_v4_v33_replay_v1":
        raise V33ReplayError("V3.3 replay schema 漂移")
    if bound.get("schema_version") != "worldsim_v4_v33_bound_scene_v1":
        raise V33ReplayError("bound scene schema 漂移")
    if bound.get("partition_contract") != "sample_index_mod_5":
        raise V33ReplayError("bound scene partition 漂移")
    if bound.get("algorithm_commit") != replay["algorithm"]["implementation_commit"]:
        raise V33ReplayError("bound scene algorithm commit 漂移")
    if bound.get("test_quality_read") is not False:
        raise V33ReplayError("bound scene 未证明 test quality 未读")
    contract = replay.get("abstain_no_actor", {})
    if contract != {
        "reason": ABSTAIN_REASON,
        "render_camera_id": 0,
        "development_view_count": 3,
        "view_selection": "first_development_frames",
        "semantic_render_status": "done",
    }:
        raise V33ReplayError("ABSTAIN_NO_ACTOR render 合同漂移")

    checkpoint = verified_file(bound["base_checkpoint"], "base checkpoint")
    registry_path = verified_file(bound["actor_registry"], "actor registry")
    source_config = Path(str(bound["base_checkpoint"]["source_config"])).resolve()
    if (
        not source_config.is_file()
        or sha256_file(source_config)
        != bound["base_checkpoint"]["source_config_sha256"]
    ):
        raise V33ReplayError("base source config 缺失或 SHA 漂移")

    actor = bound.get("actors", {}).get("high_support")
    if not isinstance(actor, Mapping) or actor.get("availability") == "available":
        raise V33ReplayError("ABSTAIN_NO_ACTOR 只允许不可用 high_support actor")
    if actor.get("rigid_model_index") is not None:
        raise V33ReplayError("不可用 high actor 不得绑定 rigid_model_index")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in registry.get("actors", [])
        if str(row.get("instance_token")) == str(actor.get("instance_token"))
        and row.get("processed_true_instance_id") == actor.get("dataset_instance_id")
    ]
    if len(matches) != 1:
        raise V33ReplayError("actor registry 未精确命中请求的 D0 high actor")
    row = matches[0]
    tensor_slice = row.get("checkpoint_tensor_slice", {})
    if (
        row.get("availability") != actor.get("availability")
        or row.get("availability") == "available"
        or row.get("rigid_model_index") is not None
        or tensor_slice.get("gaussian_count") != 0
        or tensor_slice.get("flat_index_ranges_half_open") != []
    ):
        raise V33ReplayError("不可用 D0 actor 的 0-Gaussian registry 证明不成立")
    if registry.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise V33ReplayError("actor registry/base checkpoint SHA 错配")
    return {
        "reason": ABSTAIN_REASON,
        "actor_role": "high_support",
        "actor": {
            "instance_token": str(actor["instance_token"]),
            "dataset_instance_id": int(actor["dataset_instance_id"]),
            "class_name": str(actor["class_name"]),
            "availability": str(actor["availability"]),
            "rigid_model_index": None,
            "checkpoint_gaussian_count": 0,
        },
        "actor_registry": {
            "path": str(registry_path),
            "bytes": registry_path.stat().st_size,
            "sha256": sha256_file(registry_path),
        },
        "base_checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "source_config": {
            "path": str(source_config),
            "sha256": sha256_file(source_config),
        },
        "test_quality_read": False,
    }


def select_development_views(
    replay: Mapping[str, Any], bound: Mapping[str, Any]
) -> list[tuple[int, int]]:
    contract = replay["abstain_no_actor"]
    camera = int(contract["render_camera_id"])
    processed = Path(str(bound["processed_scene"]))
    frames = sorted(
        int(match.group("frame"))
        for path in (processed / "images").glob(f"*_{camera}.jpg")
        if (match := IMAGE_NAME.match(path.name)) is not None
        and int(match.group("camera")) == camera
    )
    if not frames or frames != list(range(len(frames))):
        raise V33ReplayError("base-only render frame 索引必须从 0 连续")
    partition = replay["frame_partition"]
    modulus = int(partition["modulus"])
    remainder = int(partition["development_remainder"])
    development = [frame for frame in frames if frame % modulus == remainder]
    count = int(contract["development_view_count"])
    if len(development) < count:
        raise V33ReplayError("development base-only render views 不足")
    return [(frame, camera) for frame in development[:count]]


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


def save_mask(path: Path, value: np.ndarray, *, shape: tuple[int, int]) -> int:
    mask = np.asarray(value).squeeze().astype(bool, copy=False)
    if mask.shape != shape:
        raise V33ReplayError(f"render mask shape 漂移: {mask.shape} != {shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path, format="PNG")
    return int(mask.sum())


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


def gpu_compute_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run(
    *,
    replay_config_path: Path,
    bound_scene_path: Path,
    project_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"run 目录已存在，禁止覆盖：{run_dir}")
    if RUN_ROOT.resolve() not in run_dir.resolve().parents:
        raise V33ReplayError(f"abstain run 必须位于冻结根目录：{RUN_ROOT}")
    replay = load_yaml(replay_config_path)
    bound = json.loads(bound_scene_path.read_text(encoding="utf-8"))
    proof = validate_no_actor_contract(replay, bound)
    views = select_development_views(replay, bound)
    expected_python = Path(replay["runtimes"]["drivestudio_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise V33ReplayError(f"必须使用冻结 DriveStudio Python: {expected_python}")
    if gpu_compute_processes():
        raise V33ReplayError("GPU preflight 非空闲")
    git_head = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain"], text=True
    ).strip():
        raise V33ReplayError("formal abstain run 要求 project git clean")

    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "source_snapshot").mkdir()
    shutil.copy2(replay_config_path, run_dir / "replay_config.yaml")
    shutil.copy2(bound_scene_path, run_dir / "bound_scene.json")
    for relative in SNAPSHOT_FILES:
        source = project_root / relative
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    started_at = datetime.now(timezone.utc).isoformat()
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v4_v33_scene_chain_status_v1",
            "task_id": TASK_ID,
            "scene": bound["scene"],
            "status": "running",
            "test_quality_read": False,
            "started_at_utc": started_at,
        },
    )
    atomic_json(run_dir / "abstention.json", proof)
    checkpoint = Path(proof["base_checkpoint"]["path"])
    checkpoint_before = sha256_file(checkpoint)
    start = time.monotonic()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    runtime_config = {
        "inputs": {
            "checkpoint": proof["base_checkpoint"]["path"],
            "source_config": proof["source_config"]["path"],
        },
        "runtimes": {
            "drivestudio_checkout": replay["runtimes"]["drivestudio_checkout"]
        },
    }
    dataset, trainer = build_runtime(runtime_config, device)
    if hasattr(trainer, "optimizer"):
        raise V33ReplayError("base-only abstain renderer 禁止 optimizer")
    trainer.set_eval()
    rows = []
    for frame, camera in views:
        snapshot = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=frame,
            camera_id=camera,
            device=device,
        )
        prediction_value = np.asarray(snapshot["rgb"], dtype=np.uint8)
        target_value = np.asarray(snapshot["groundtruth"], dtype=np.uint8)
        if prediction_value.shape != target_value.shape or prediction_value.ndim != 3:
            raise V33ReplayError("base-only prediction/target shape 漂移")
        shape = prediction_value.shape[:2]
        view_dir = run_dir / "artifacts" / "renders" / f"f{frame:03d}_c{camera}"
        view_dir.mkdir(parents=True)
        prediction = view_dir / "base_only.png"
        target = view_dir / "target.png"
        dynamic = view_dir / "dynamic_mask.png"
        egocar = view_dir / "egocar_mask.png"
        Image.fromarray(prediction_value).save(prediction, format="PNG")
        Image.fromarray(target_value).save(target, format="PNG")
        dynamic_pixels = save_mask(dynamic, snapshot["dynamic_mask"], shape=shape)
        egocar_pixels = save_mask(egocar, snapshot["egocar_mask"], shape=shape)
        rows.append(
            {
                "frame": frame,
                "camera_id": camera,
                "partition": "development",
                "prediction": str(prediction),
                "prediction_sha256": sha256_file(prediction),
                "target": str(target),
                "target_sha256": sha256_file(target),
                "dynamic_mask": str(dynamic),
                "dynamic_mask_sha256": sha256_file(dynamic),
                "dynamic_mask_source": "drivestudio_dynamic_masks_all",
                "dynamic_mask_positive_pixels": dynamic_pixels,
                "egocar_mask": str(egocar),
                "egocar_mask_sha256": sha256_file(egocar),
                "egocar_mask_positive_pixels": egocar_pixels,
            }
        )
    render_manifest = {
        "schema_version": "worldsim_v4_v33_render_manifest_v1",
        "task_id": TASK_ID,
        "scene": bound["scene"],
        "split": "development",
        "abstention_reason": ABSTAIN_REASON,
        "test_quality_read": False,
        "rows": rows,
    }
    atomic_json(run_dir / "render_manifest.json", render_manifest)
    metric_rows, metric_summary = evaluate_scene_records(
        rows, lpips_model=build_lpips_model(replay)
    )
    metrics = {
        "schema_version": "worldsim_v4_v33_scene_metrics_v1",
        "task_id": TASK_ID,
        "scene": bound["scene"],
        "split": "development",
        "abstention_reason": ABSTAIN_REASON,
        "test_quality_read": False,
        "rows": metric_rows,
        "summary": metric_summary,
    }
    atomic_json(run_dir / "metrics.json", metrics)
    torch.cuda.synchronize(device)
    duration = time.monotonic() - start
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise V33ReplayError("base checkpoint 在 abstain render 中发生 mutation")
    resources = {
        "duration_seconds": duration,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "base_checkpoint_sha256_before": checkpoint_before,
        "base_checkpoint_sha256_after": checkpoint_after,
        "base_checkpoint_immutable": True,
        "training_started": False,
        "model_inference_started": True,
    }
    atomic_json(run_dir / "resources.json", resources)

    abstain_stage = {
        "status": "abstain",
        "reason": ABSTAIN_REASON,
        "proof_sha256": sha256_file(run_dir / "abstention.json"),
    }
    chain = {
        "schema_version": "worldsim_v4_v33_scene_chain_v1",
        "task_id": TASK_ID,
        "scene": bound["scene"],
        "algorithm_commit": replay["algorithm"]["implementation_commit"],
        "base_checkpoint_sha256": checkpoint_before,
        "partition_contract": "sample_index_mod_5",
        "test_quality_read": False,
        "abstention": proof,
        "stages": {
            **{stage: dict(abstain_stage) for stage in ABSTAIN_STAGES},
            "semantic_render": {
                "status": "done",
                "run": str(run_dir),
                "mode": "base_only_abstain_no_actor",
                "render_manifest_sha256": sha256_file(
                    run_dir / "render_manifest.json"
                ),
                "metrics_sha256": sha256_file(run_dir / "metrics.json"),
            },
        },
    }
    atomic_json(run_dir / "scene_chain.json", chain)
    finished = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": "worldsim_v4_v33_scene_chain_summary_v1",
        "task_id": TASK_ID,
        "scene": bound["scene"],
        "status": "done",
        "algorithm_commit": chain["algorithm_commit"],
        "base_checkpoint_sha256": checkpoint_before,
        "partition_contract": "sample_index_mod_5",
        "abstention_reason": ABSTAIN_REASON,
        "development_content_read": True,
        "heldout_content_read": False,
        "test_quality_read": False,
        "training_started": False,
        "model_inference_started": True,
        "project_git_head": git_head,
        "duration_seconds": duration,
        "finished_at_utc": finished,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "manifest.json", manifest(run_dir))
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v4_v33_scene_chain_status_v1",
            "task_id": TASK_ID,
            "scene": bound["scene"],
            "status": "done",
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "test_quality_read": False,
            "finished_at_utc": finished,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-config", type=Path, required=True)
    parser.add_argument("--bound-scene", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    preexisted = run_dir.exists()
    try:
        summary = run(
            replay_config_path=args.replay_config.resolve(),
            bound_scene_path=args.bound_scene.resolve(),
            project_root=args.project_root.resolve(),
            run_dir=run_dir,
        )
    except Exception as error:
        if not preexisted and run_dir.is_dir():
            try:
                status = json.loads(
                    (run_dir / "status.json").read_text(encoding="utf-8")
                )
            except Exception:
                status = {"task_id": TASK_ID}
            status.update(
                status="failed",
                reason=type(error).__name__,
                error=str(error),
                heldout_content_read=False,
                test_quality_read=False,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(run_dir / "status.json", status)
        raise
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
