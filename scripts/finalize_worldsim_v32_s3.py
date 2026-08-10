#!/usr/bin/env python
"""严格核对 S3 证据链，并在人工质量裁决后关闭 formal run。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import (
    sha256_file,
    validate_actor_identity_contract,
)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_spec(spec: dict, label: str) -> Path:
    path = Path(spec["path"])
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"S3 finalizer 文件缺失或 SHA 漂移: {label} {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--decision",
        choices=["asset_harvester_1view", "asset_harvester_2view", "native_streetgs"],
        required=True,
    )
    parser.add_argument("--decision-reason", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inputs_path = Path(config["inputs"]["input_manifest"])
    inputs = load_json(inputs_path)
    if sha256_file(inputs_path) != config["inputs"]["input_manifest_sha256"]:
        raise RuntimeError("S3 finalizer input manifest SHA 漂移")
    identity = (
        inputs["dataset_instance_id"],
        inputs["instance_token"],
        inputs["rigid_model_index"],
    )
    expected_identity = (
        int(config["actor"]["dataset_instance_id"]),
        config["actor"]["instance_token"],
        int(config["actor"]["rigid_model_index"]),
    )
    if identity != expected_identity:
        raise RuntimeError(f"S3 finalizer actor identity 错配: {identity} != {expected_identity}")

    s1_config_path = Path(inputs["s1_config"])
    if sha256_file(s1_config_path) != inputs["s1_config_sha256"]:
        raise RuntimeError("S3 finalizer S1 config SHA 漂移")
    s1_config = yaml.safe_load(s1_config_path.read_text(encoding="utf-8"))
    actor_config = s1_config["actors"]["high_support"]
    instances_path = Path(s1_config["scene"]["processed_scene_dir"]) / "instances/instances_info.json"
    if sha256_file(instances_path) != inputs["instances_info_sha256"]:
        raise RuntimeError("S3 finalizer instances_info SHA 漂移")
    instances = load_json(instances_path)
    registry_path = Path(s1_config["inputs"]["actor_registry"])
    if sha256_file(registry_path) != s1_config["inputs"]["actor_registry_sha256"]:
        raise RuntimeError("S3 finalizer actor registry SHA 漂移")
    registry = load_json(registry_path)
    registry_matches = [
        row
        for row in registry["actors"]
        if int(row["rigid_model_index"]) == int(actor_config["rigid_model_index"])
    ]
    if len(registry_matches) != 1:
        raise RuntimeError("S3 finalizer actor registry rigid index 非唯一")
    validate_actor_identity_contract(
        role="high_support",
        actor_config=actor_config,
        dataset_instance=instances[str(actor_config["dataset_instance_id"])],
        registry_actor=registry_matches[0],
    )
    heldout = {int(value) for value in s1_config["split"]["heldout_frames"]}
    input_frames = {
        int(view["frame"])
        for sample in inputs["samples"]
        for view in sample["views"]
    }
    if input_frames & heldout:
        raise RuntimeError(f"S3 finalizer heldout 泄漏: {sorted(input_frames & heldout)}")

    environment_path = args.run_dir / "environment_setup_result.json"
    environment = load_json(environment_path)
    if environment["status"] != "done" or environment.get("gsplat_cuda_extension") != "imported":
        raise RuntimeError("S3 finalizer environment 未完成")
    if set(environment.get("transitive_models", {})) != set(config["transitive_models"]):
        raise RuntimeError("S3 finalizer 传递 HF 模型未冻结")

    inference_path = args.run_dir / "artifacts/asset_harvester/inference_manifest.json"
    inference = load_json(inference_path)
    if inference["status"] != "done":
        raise RuntimeError("S3 finalizer inference 未完成")
    if inference["input_manifest_sha256"] != config["inputs"]["input_manifest_sha256"]:
        raise RuntimeError("S3 finalizer inference/input SHA 错配")
    if inference["source_commit"] != config["source"]["commit"]:
        raise RuntimeError("S3 finalizer inference source commit 漂移")
    if not inference.get("hf_offline"):
        raise RuntimeError("S3 finalizer inference 未证明 HF offline")
    for sample, spec in inference["plys"].items():
        verify_spec(spec, f"{sample} PLY")
    resource = inference["runtime"]
    if resource.get("memory_pressure_observed"):
        raise RuntimeError("S3 finalizer inference 命中 cgroup 90% 停止门")
    if resource["memory_events_delta"].get("oom", 0) or resource[
        "memory_events_delta"
    ].get("oom_kill", 0):
        raise RuntimeError("S3 finalizer inference OOM 事件增加")
    if int(resource["disk_free_after_bytes"]) < 20 * 1024**3:
        raise RuntimeError("S3 finalizer disk free 低于 20 GiB")

    assets = {}
    for sample in ("high_support_1view", "high_support_2view"):
        path = args.run_dir / f"artifacts/actor_assets/{sample}/actor_asset_manifest.json"
        manifest = load_json(path)
        if manifest["status"] != "done" or manifest["sample"] != sample:
            raise RuntimeError(f"S3 finalizer actor asset 状态错误: {sample}")
        manifest_identity = (
            int(manifest["dataset_instance_id"]),
            manifest["instance_token"],
            int(manifest["rigid_model_index"]),
        )
        if manifest_identity != expected_identity:
            raise RuntimeError(f"S3 finalizer asset identity 错配: {sample}")
        verify_spec(manifest["asset"], f"{sample} actor asset")
        if not manifest["reload_exact"] or manifest["generation_provenance"] != "GENERATED_ACTOR":
            raise RuntimeError(f"S3 finalizer asset reload/provenance 错误: {sample}")
        bounds = np.asarray(manifest["coordinate_contract"]["bounds_upper_m"]) - np.asarray(
            manifest["coordinate_contract"]["bounds_lower_m"]
        )
        target = np.asarray(manifest["coordinate_contract"]["target_lwh_m"])
        if not np.allclose(bounds, target, rtol=0, atol=1e-4):
            raise RuntimeError(f"S3 finalizer asset bounds/LWH 错配: {sample}")
        assets[sample] = {
            "manifest": str(path),
            "manifest_sha256": sha256_file(path),
            "asset": manifest["asset"],
            "bounds_extent_error_max_m": float(np.max(np.abs(bounds - target))),
        }

    renders = []
    expected_render_names = [
        f"{sample}_view{index}"
        for sample in ("high_support_1view", "high_support_2view")
        for index in (0, 1)
    ]
    checkpoint_sha = config["streetgs"]["checkpoint_sha256"]
    for name in expected_render_names:
        path = args.run_dir / f"artifacts/renders/{name}/render_manifest.json"
        render = load_json(path)
        if render["status"] != "done":
            raise RuntimeError(f"S3 finalizer render 未完成: {name}")
        if render["checkpoint_sha256_before"] != checkpoint_sha or render[
            "checkpoint_sha256_after"
        ] != checkpoint_sha:
            raise RuntimeError(f"S3 finalizer checkpoint mutation: {name}")
        if min(int(value) for value in render["effect_pixels"].values()) < int(
            config["smoke"]["minimum_effect_pixels"]
        ):
            raise RuntimeError(f"S3 finalizer render effect 过小: {name}")
        if len({verify_spec(spec, f"{name} {variant}") for variant, spec in render["images"].items()}) != 3:
            raise RuntimeError(f"S3 finalizer render 三路输出不完整: {name}")
        for variant, spec in render["effect_masks"].items():
            verify_spec(spec, f"{name} {variant} effect")
        renders.append(
            {
                "name": name,
                "manifest": str(path),
                "manifest_sha256": sha256_file(path),
                "effect_pixels": render["effect_pixels"],
                "runtime": render["runtime"],
            }
        )

    evaluation_path = args.run_dir / "artifacts/evaluation/evaluation_summary.json"
    evaluation = load_json(evaluation_path)
    if evaluation["status"] != "done" or len(evaluation["rows"]) != 4:
        raise RuntimeError("S3 finalizer evaluation 不完整")
    for sample in ("high_support_1view", "high_support_2view"):
        if evaluation["aggregate"][sample]["view_count_evaluated"] != 2:
            raise RuntimeError(f"S3 finalizer {sample} 未覆盖两个真实视角")
    for panel in evaluation["panels"]:
        verify_spec(panel, "evaluation panel")

    audits = {
        "actor_identity_contract": True,
        "heldout_excluded": True,
        "official_source_and_weights_exact": True,
        "transitive_hf_models_exact_offline": True,
        "one_and_two_view_outputs_nonempty": True,
        "actor_asset_reload_exact": True,
        "actor_bounds_match_lwh": True,
        "trajectory_original_lateral_delete_four_runs": True,
        "observed_view_metrics_and_panels_complete": True,
        "d2_checkpoint_immutable": True,
        "resource_gates_passed": True,
        "generated_actor_provenance_complete": True,
    }
    selected_asset = {
        "asset_harvester_1view": "high_support_1view",
        "asset_harvester_2view": "high_support_2view",
        "native_streetgs": "native_streetgs",
    }[args.decision]
    summary = {
        "schema_version": "worldsim_v32_s3_final_v1",
        "task_id": config["task_id"],
        "status": "done",
        "run_id": args.run_dir.name,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "input_manifest": str(inputs_path),
        "input_manifest_sha256": sha256_file(inputs_path),
        "actor_identity": {
            "dataset_instance_id": expected_identity[0],
            "instance_token": expected_identity[1],
            "rigid_model_index": expected_identity[2],
            "status": "validated",
        },
        "environment": {
            "path": str(environment_path),
            "sha256": sha256_file(environment_path),
        },
        "inference": {
            "path": str(inference_path),
            "sha256": sha256_file(inference_path),
            "runtime": resource,
        },
        "assets": assets,
        "renders": renders,
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": sha256_file(evaluation_path),
            "aggregate": evaluation["aggregate"],
        },
        "decision": {
            "selected": selected_asset,
            "reason": args.decision_reason,
            "generated_backside_claim": "completeness/consistency only; no GT correctness",
        },
        "audits": audits,
    }
    summary_path = args.run_dir / "summary.json"
    atomic_json(summary_path, summary)
    status_path = args.run_dir / "status.json"
    status = load_json(status_path)
    status.update(
        {
            "status": "done",
            "stage": "complete",
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "selected": selected_asset,
        }
    )
    atomic_json(status_path, status)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
