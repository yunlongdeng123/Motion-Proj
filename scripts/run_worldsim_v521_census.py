#!/usr/bin/env python3
"""准备或汇总 V5.2.1 Discovery-only base badcase census。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from motion_proj.worldsim_v521.census import (
    CensusProtocol,
    assert_unique_keys,
    evaluate_discovery_view,
    load_metric_mask,
    load_metric_rgb,
    sha256_file,
    temporal_proxy_row,
    validate_discovery_record,
)
from motion_proj.worldsim_v521.protocol import atomic_json, atomic_jsonl, temporal_window_partition
from motion_proj.worldsim_v521.shadow_data import build_adgs_discovery_adapter, build_streetgs_shadow


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_git_audit(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    return {
        "path": str(path.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "status_porcelain": run("status", "--short"),
        "diff_sha256": __import__("hashlib").sha256(run("diff", "--binary").encode("utf-8")).hexdigest(),
    }


def load_contract(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_discovery_quality_read":
        raise RuntimeError("P2 config 未在 quality read 前冻结")
    if config["partition"]["allowed_quality_partition"] != "discovery":
        raise RuntimeError("P2 partition lock 漂移")
    if config["ranking_contract"]["scalar_composite_score"] != "forbidden":
        raise RuntimeError("禁止单一总分")
    return config


def verify_p1(config: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    p1 = Path(config["bindings"]["p1_run"])
    bindings = {
        "BASE_ASSET_REGISTRY.json": "base_asset_registry_sha256",
        "DISCOVERY_CONFIRMATION_FREEZE.json": "split_freeze_sha256",
        "MATCHED_FRAME_REGISTRY.jsonl": "matched_frame_registry_sha256",
    }
    for name, field in bindings.items():
        observed = sha256_file(p1 / name)
        if observed != config["bindings"][field]:
            raise RuntimeError(f"P1 binding SHA 漂移：{name}")
    records = read_jsonl(p1 / "MATCHED_FRAME_REGISTRY.jsonl")
    registry = json.loads((p1 / "BASE_ASSET_REGISTRY.json").read_text(encoding="utf-8"))
    return p1, records, registry


def scene_records(records: list[dict[str, Any]], scene: str, partition: str = "discovery") -> list[dict[str, Any]]:
    return sorted(
        [row for row in records if row["scene"] == scene and row["partition"] == partition],
        key=lambda row: (int(row["frame"]), int(row["camera"])),
    )


def find_asset(registry: Mapping[str, Any], base: str, scene: str) -> Mapping[str, Any]:
    hits = [
        row
        for row in registry["assets"]
        if row["base"] == base
        and row["scene"] == scene
        and row["state"] == "PRESENT_EXACT"
        and row.get("protocol") == "strict_mod5"
    ]
    if len(hits) != 1:
        raise RuntimeError(f"exact asset 非唯一：{base}/{scene}")
    return hits[0]


def prepare(config_path: Path, run_dir: Path) -> None:
    config = load_contract(config_path)
    _, records, registry = verify_p1(config)
    discovery = [row for row in records if row["partition"] == "discovery"]
    for row in discovery:
        validate_discovery_record(row)
    run_dir.mkdir(parents=True, exist_ok=False)
    staging = run_dir / "staging"
    shadow_root = staging / "streetgs_shadow"
    commands: list[dict[str, Any]] = []
    scene_rows = []
    for scene in sorted({row["scene"] for row in discovery}):
        rows = scene_records(discovery, scene)
        scene_index = int(rows[0]["scene_index"])
        source_scene = Path(rows[0]["target_path"]).parents[1]
        records_path = staging / "records" / f"{scene}.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_jsonl(records_path, rows)
        street_manifest = build_streetgs_shadow(
            source_scene=source_scene,
            shadow_root=shadow_root,
            scene_index=scene_index,
            discovery_records=rows,
        )
        adgs_adapter = staging / "adgs_adapter" / scene
        adgs_manifest = build_adgs_discovery_adapter(
            train_adapter=Path("/root/autodl-tmp/data/worldsim_v4/adgs_processed_v4/train") / scene,
            source_scene=source_scene,
            destination=adgs_adapter,
            discovery_records=rows,
        )
        street_asset = find_asset(registry, "streetgs", scene)
        adgs_asset = find_asset(registry, "adgs", scene)
        street_output = run_dir / "renders" / "streetgs" / scene
        adgs_output = run_dir / "renders" / "adgs" / scene
        commands.extend(
            [
                {
                    "base": "streetgs",
                    "scene": scene,
                    "argv": [
                        sys.executable,
                        "scripts/render_streetgs_v521.py",
                        "--checkpoint", street_asset["path"],
                        "--shadow-root", str(shadow_root),
                        "--records", str(records_path),
                        "--output", str(street_output),
                    ],
                },
                {
                    "base": "adgs",
                    "scene": scene,
                    "argv": [
                        sys.executable,
                        "scripts/render_adgs_v521.py",
                        "--adgs-python", "/root/autodl-tmp/envs/adgs/bin/python",
                        "--source-root", "/root/autodl-tmp/third_party/AD-GS",
                        "--model-source", str(Path(next(iter(adgs_asset["files"].values()))["path"]).parents[2]),
                        "--adapter", str(adgs_adapter),
                        "--records", str(records_path),
                        "--output", str(adgs_output),
                        "--config", "configs/worldsim_v4/adgs_nuscenes_v4.py",
                    ],
                },
            ]
        )
        scene_rows.append(
            {
                "scene": scene,
                "scene_index": scene_index,
                "discovery_views": len(rows),
                "source_scene": str(source_scene),
                "records": str(records_path),
                "streetgs_shadow_manifest": street_manifest,
                "adgs_adapter_manifest": {key: value for key, value in adgs_manifest.items() if key != "render_map"},
            }
        )
    atomic_json(run_dir / "PREPARE_MANIFEST.json", {
        "schema": "worldsim_v521_p2_prepare_v1",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "discovery_views": len(discovery),
        "confirmation_views_decoded": 0,
        "scenes": scene_rows,
        "source_audit": {
            "adgs": source_git_audit(Path("/root/autodl-tmp/third_party/AD-GS")),
        },
    })
    atomic_json(run_dir / "RENDER_COMMANDS.json", {"commands": commands, "serial_gpu_processes": 1})
    atomic_json(run_dir / "status.json", {"task_id": config["task_id"], "status": "prepared", "outcome": "p2_prepare_pass"})


def render(config_path: Path, run_dir: Path, only: str | None = None, scene: str | None = None) -> None:
    load_contract(config_path)
    payload = json.loads((run_dir / "RENDER_COMMANDS.json").read_text(encoding="utf-8"))
    selected = [row for row in payload["commands"] if (only is None or row["base"] == only) and (scene is None or row["scene"] == scene)]
    for row in selected:
        started = time.monotonic()
        completed = subprocess.run(row["argv"], cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        log = run_dir / "logs" / f"render_{row['base']}_{row['scene']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"renderer failed: {row['base']}/{row['scene']}，见 {log}")
        row["wall_seconds"] = time.monotonic() - started
    atomic_json(run_dir / "RENDER_EXECUTION.json", {"executed": selected, "gpu_processes": 1})


def prediction_maps(run_dir: Path) -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mapping: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    manifests, audits = [], []
    for base in ("streetgs", "adgs"):
        for scene_dir in sorted((run_dir / "renders" / base).iterdir()):
            rows = read_jsonl(scene_dir / "RENDER_MAP.jsonl")
            audit = json.loads((scene_dir / "RENDER_AUDIT.json").read_text(encoding="utf-8"))
            audits.append({"base": base, "scene": scene_dir.name, **audit})
            for row in rows:
                key = (base, row["scene"], int(row["frame"]), int(row["camera"]))
                if key in mapping:
                    raise RuntimeError(f"重复 prediction key：{key}")
                mapping[key] = row
                manifests.append({"base": base, **row})
    return mapping, manifests, audits


def evaluate(config_path: Path, run_dir: Path) -> None:
    import torch
    import lpips

    config = load_contract(config_path)
    _, records, _ = verify_p1(config)
    discovery = [row for row in records if row["partition"] == "discovery"]
    mapping, prediction_manifest, audits = prediction_maps(run_dir)
    expected = len(discovery) * 2
    if len(mapping) != expected:
        raise RuntimeError(f"prediction denominator {len(mapping)} != {expected}")
    model = lpips.LPIPS(net="alex", verbose=False).eval().cuda()
    protocol = CensusProtocol()
    base_rows, actor_rows = [], []
    decoded: dict[tuple[str, str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    by_lookup = {(row["scene"], int(row["frame"]), int(row["camera"])): row for row in discovery}
    for base in ("adgs", "streetgs"):
        for record in sorted(discovery, key=lambda row: (row["scene"], int(row["frame"]), int(row["camera"]))):
            key = (base, record["scene"], int(record["frame"]), int(record["camera"]))
            prediction = mapping[key]
            mask_path = Path(record["target_path"]).parents[1] / "dynamic_masks" / "all" / f"{int(record['frame']):03d}_{int(record['camera'])}.png"
            base_row, actor_row = evaluate_discovery_view(
                base=base,
                record=record,
                prediction_path=prediction["prediction_path"],
                dynamic_mask_path=mask_path,
                lpips_model=model,
                renderer_provenance={"render_audit": str(Path(prediction["prediction_path"]).parents[1] / "RENDER_AUDIT.json")},
                resource={"render_seconds": prediction.get("render_seconds")},
                protocol=protocol,
            )
            base_rows.append(base_row)
            actor_rows.append(actor_row)
            pred_array, _ = load_metric_rgb(prediction["prediction_path"], protocol)
            target_array, _ = load_metric_rgb(record["target_path"], protocol)
            dynamic, _ = load_metric_mask(mask_path, protocol)
            decoded[key] = (pred_array, target_array, dynamic)
    target_hash_pairs: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in base_rows:
        target_hash_pairs[(row["scene"], row["frame"], row["camera"])].add(row["target_sha256"])
    if any(len(values) != 1 for values in target_hash_pairs.values()):
        raise RuntimeError("同 view 跨 base target decoded hash 漂移")
    temporal_rows, partition_rows = [], []
    sample_rows = {}
    for row in records:
        sample_rows[(row["scene"], int(row["frame"]))] = row
    for scene in sorted({row["scene"] for row in records}):
        development_frames = sorted({int(row["frame"]) for row in records if row["scene"] == scene})
        for camera in (0, 1, 2):
            for earlier_frame, later_frame in zip(development_frames, development_frames[1:]):
                earlier_meta, later_meta = sample_rows[(scene, earlier_frame)], sample_rows[(scene, later_frame)]
                partition = temporal_window_partition([earlier_meta, later_meta])
                partition_rows.append({
                    "scene": scene, "camera": camera, "member_frames": [earlier_frame, later_frame], **partition,
                })
                if partition != {"status": "defined", "partition": "discovery"}:
                    continue
                earlier = by_lookup[(scene, earlier_frame, camera)]
                later = by_lookup[(scene, later_frame, camera)]
                for base in ("adgs", "streetgs"):
                    pa, ta, ma = decoded[(base, scene, earlier_frame, camera)]
                    pb, tb, mb = decoded[(base, scene, later_frame, camera)]
                    temporal_rows.append({
                        "base": base,
                        **temporal_proxy_row(
                            earlier=earlier, later=later,
                            earlier_prediction=pa, earlier_target=ta,
                            later_prediction=pb, later_target=tb,
                            earlier_dynamic=ma, later_dynamic=mb,
                            protocol=protocol,
                        ),
                    })
    assert_unique_keys(base_rows, ("base", "scene", "canonical_sample_index", "camera"))
    assert_unique_keys(actor_rows, ("base", "scene", "canonical_sample_index", "camera", "actor_token"))
    assert_unique_keys(temporal_rows, ("base", "scene", "camera", "window_id"))
    atomic_jsonl(run_dir / config["outputs"]["base_metrics"], base_rows)
    atomic_jsonl(run_dir / config["outputs"]["actor_metrics"], actor_rows)
    atomic_jsonl(run_dir / config["outputs"]["temporal_metrics"], temporal_rows)
    atomic_jsonl(run_dir / "TEMPORAL_WINDOW_PARTITION_AUDIT.jsonl", partition_rows)
    atomic_jsonl(run_dir / config["outputs"]["prediction_manifest"], prediction_manifest)
    atomic_json(run_dir / config["outputs"]["renderer_audit"], {"audits": audits})
    first = base_rows[0]
    finite_gate = all(
        result[metric] is None or np.isfinite(float(result[metric]))
        for result in first["metrics"].values() if isinstance(result, dict)
        for metric in ("psnr", "ssim", "lpips_alex") if metric in result
    )
    summary = {
        "schema": "worldsim_v521_p2_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "outcome": "p2_gate_pass",
        "base_rows": len(base_rows),
        "actor_rows": len(actor_rows),
        "temporal_rows": len(temporal_rows),
        "expected_base_rows": expected,
        "discovery_views": len(discovery),
        "confirmation_views": sum(row["partition"] == "confirmation" for row in records),
        "confirmation_prediction_pixels_decoded": 0,
        "confirmation_target_pixels_decoded": 0,
        "fresh_validation_test_kitti_pixels_decoded": 0,
        "checkpoint_hash_gate": all(audit.get("checkpoint_sha256_before") == audit.get("checkpoint_sha256_after") for audit in audits),
        "target_hash_cross_base_gate": True,
        "finite_metric_gate": finite_gate,
        "lpips_backbone": "alex",
        "geometry_status": "undefined_no_comparable_base_depth",
        "actor_instance_status": "undefined_no_instance_region",
        "temporal_status": "unwarped_temporal_proxy",
        "scalar_composite_score": False,
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch_cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "output_sha256": {},
    }
    for name in (config["outputs"]["base_metrics"], config["outputs"]["actor_metrics"], config["outputs"]["temporal_metrics"]):
        summary["output_sha256"][name] = sha256_file(run_dir / name)
    atomic_json(run_dir / config["outputs"]["summary"], summary)
    atomic_json(run_dir / "status.json", {"task_id": config["task_id"], "status": "done", "outcome": "p2_gate_pass"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v521/census_protocol_v1.yaml"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=("prepare", "render", "evaluate"), required=True)
    parser.add_argument("--base", choices=("streetgs", "adgs"))
    parser.add_argument("--scene")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.config, args.run_dir)
    elif args.phase == "render":
        render(args.config, args.run_dir, args.base, args.scene)
    else:
        evaluate(args.config, args.run_dir)


if __name__ == "__main__":
    main()
