#!/usr/bin/env python3
"""P9：taxonomy freeze 后一次性读取 internal Confirmation。"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from motion_proj.worldsim_v521.badcase_registry import build_leaderboards, build_registry, panel_union
from motion_proj.worldsim_v521.census import (
    CensusProtocol,
    assert_unique_keys,
    evaluate_discovery_view,
    load_metric_mask,
    load_metric_rgb,
    sha256_file,
    temporal_proxy_row,
)
from motion_proj.worldsim_v521.panels import build_view_panel
from motion_proj.worldsim_v521.protocol import atomic_json, atomic_jsonl, sha256_json, temporal_window_partition
from motion_proj.worldsim_v521.shadow_data import (
    build_adgs_discovery_adapter,
    build_streetgs_shadow,
    repair_adgs_zero_depth_links,
)


P1_RUN = Path("/root/autodl-tmp/runs/worldsim_v521/20260820T084604Z__p1-base-asset-census-s0-r001")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_asset(registry: Mapping[str, Any], base: str, scene: str) -> Mapping[str, Any]:
    rows = [
        row for row in registry["assets"]
        if row["base"] == base and row["scene"] == scene
        and row["state"] == "PRESENT_EXACT" and row.get("protocol") == "strict_mod5"
    ]
    if len(rows) != 1:
        raise RuntimeError(f"exact asset 非唯一：{base}/{scene}")
    return rows[0]


def prerequisite_audit(census_run: Path) -> dict[str, Any]:
    required = [
        "BADCASE_TAXONOMY_FREEZE.yaml", "BADCASE_REGISTRY.jsonl", "BADCASE_LEADERBOARDS.json",
        "P4_SUMMARY.json", "P5_LOCALIZATION.json", "M1_REAUDIT.json", "M2_REAUDIT.json", "M3_REAUDIT.json",
    ]
    missing = [name for name in required if not (census_run / name).is_file()]
    if missing:
        raise RuntimeError(f"P9 prerequisite 缺失：{missing}")
    return {name: sha256_file(census_run / name) for name in required}


def prepare(census_run: Path, run_dir: Path) -> None:
    census_run = census_run.resolve()
    prerequisites = prerequisite_audit(census_run)
    taxonomy = yaml.safe_load((census_run / "BADCASE_TAXONOMY_FREEZE.yaml").read_text(encoding="utf-8"))
    if taxonomy["status"] != "frozen_after_discovery_before_confirmation" or taxonomy["confirmation_refit"] is not False:
        raise RuntimeError("taxonomy 未冻结或允许 confirmation refit")
    records = read_jsonl(P1_RUN / "MATCHED_FRAME_REGISTRY.jsonl")
    confirmation = [row for row in records if row["partition"] == "confirmation"]
    registry = json.loads((P1_RUN / "BASE_ASSET_REGISTRY.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=False)
    # 在任何 Confirmation pixel decode 前先落盘所有模板/阈值绑定。
    atomic_json(
        run_dir / "CONFIRMATION_PROTOCOL_FREEZE.json",
        {
            "schema": "worldsim_v521_confirmation_protocol_v1",
            "task_id": "WS-V521-P9-BADCASE-CONFIRMATION-01",
            "status": "frozen_before_confirmation_pixel_decode",
            "census_run": str(census_run),
            "prerequisite_sha256": prerequisites,
            "taxonomy_sha256": prerequisites["BADCASE_TAXONOMY_FREEZE.yaml"],
            "thresholds": taxonomy["thresholds"],
            "predicates": taxonomy["predicates"],
            "minimum_support_pixels": taxonomy["minimum_support_pixels"],
            "leaderboard_k": taxonomy["leaderboard_k"],
            "refit_thresholds": False,
            "change_ranking_k": False,
            "change_failure_classes": False,
            "confirmation_views": len(confirmation),
            "fresh_validation_test_kitti": False,
        },
    )
    staging = run_dir / "staging"
    shadow_root = staging / "streetgs_shadow"
    commands, scene_manifest = [], []
    for scene in sorted({row["scene"] for row in confirmation}):
        rows = sorted(
            [row for row in confirmation if row["scene"] == scene],
            key=lambda row: (int(row["frame"]), int(row["camera"])),
        )
        scene_index = int(rows[0]["scene_index"])
        source_scene = Path(rows[0]["target_path"]).parents[1]
        records_path = staging / "records" / f"{scene}.jsonl"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_jsonl(records_path, rows)
        street_manifest = build_streetgs_shadow(
            source_scene=source_scene, shadow_root=shadow_root, scene_index=scene_index,
            discovery_records=rows, quality_partition="confirmation",
        )
        adapter = staging / "adgs_adapter" / scene
        adgs_manifest = build_adgs_discovery_adapter(
            train_adapter=Path("/root/autodl-tmp/data/worldsim_v4/adgs_processed_v4/train") / scene,
            source_scene=source_scene, destination=adapter, discovery_records=rows,
            quality_partition="confirmation",
        )
        street = find_asset(registry, "streetgs", scene)
        adgs = find_asset(registry, "adgs", scene)
        commands.extend(
            [
                {
                    "base": "streetgs", "scene": scene,
                    "argv": [
                        "/root/autodl-tmp/envs/drivestudio/bin/python", "scripts/render_streetgs_v521.py",
                        "--checkpoint", street["path"], "--shadow-root", str(shadow_root),
                        "--records", str(records_path), "--output", str(run_dir / "renders" / "streetgs" / scene),
                        "--partition", "confirmation",
                    ],
                },
                {
                    "base": "adgs", "scene": scene,
                    "argv": [
                        "/root/autodl-tmp/envs/motionproj/bin/python", "scripts/render_adgs_v521.py",
                        "--adgs-python", "/root/autodl-tmp/envs/adgs/bin/python",
                        "--source-root", "/root/autodl-tmp/third_party/AD-GS",
                        "--model-source", str(Path(next(iter(adgs["files"].values()))["path"]).parents[2]),
                        "--adapter", str(adapter), "--records", str(records_path),
                        "--output", str(run_dir / "renders" / "adgs" / scene),
                        "--config", "configs/worldsim_v4/adgs_nuscenes_v4.py", "--partition", "confirmation",
                    ],
                },
            ]
        )
        scene_manifest.append(
            {
                "scene": scene, "views": len(rows), "streetgs_shadow": street_manifest,
                "adgs_adapter": {key: value for key, value in adgs_manifest.items() if key != "render_map"},
            }
        )
    atomic_json(run_dir / "PREPARE_MANIFEST.json", {"confirmation_views": len(confirmation), "scenes": scene_manifest})
    atomic_json(run_dir / "RENDER_COMMANDS.json", {"commands": commands, "serial_gpu_processes": 1})
    atomic_json(run_dir / "status.json", {"status": "prepared", "outcome": "p9_prepare_pass"})


def render(run_dir: Path) -> None:
    freeze = json.loads((run_dir / "CONFIRMATION_PROTOCOL_FREEZE.json").read_text(encoding="utf-8"))
    if freeze["status"] != "frozen_before_confirmation_pixel_decode":
        raise RuntimeError("Confirmation freeze 缺失")
    rows = json.loads((run_dir / "RENDER_COMMANDS.json").read_text(encoding="utf-8"))["commands"]
    for row in rows:
        argv = list(row["argv"])
        output = Path(argv[argv.index("--output") + 1])
        if (output / "RENDER_MAP.jsonl").is_file() and (output / "RENDER_AUDIT.json").is_file():
            row["resume_status"] = "already_complete"
            continue
        if output.exists():
            raise RuntimeError(f"存在不完整 Confirmation render：{output}")
        if row["base"] == "adgs":
            adapter = Path(argv[argv.index("--adapter") + 1])
            row["zero_depth_link_repairs"] = repair_adgs_zero_depth_links(adapter)
        started = time.monotonic()
        completed = subprocess.run(argv, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        log = run_dir / "logs" / f"render_{row['base']}_{row['scene']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"Confirmation renderer failed：{row['base']}/{row['scene']}")
        row.update({"resume_status": "executed", "wall_seconds": time.monotonic() - started})
    atomic_json(run_dir / "RENDER_EXECUTION.json", {"commands": rows, "gpu_processes": 1})


def maps(run_dir: Path) -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], list[dict[str, Any]]]:
    result, audits = {}, []
    for base in ("adgs", "streetgs"):
        for scene in sorted((run_dir / "renders" / base).iterdir()):
            audit = json.loads((scene / "RENDER_AUDIT.json").read_text(encoding="utf-8"))
            audits.append({"base": base, "scene": scene.name, **audit})
            for row in read_jsonl(scene / "RENDER_MAP.jsonl"):
                result[(base, row["scene"], int(row["frame"]), int(row["camera"]))] = row
    return result, audits


def class_counts(rows: list[dict[str, Any]], base: str) -> Counter:
    return Counter(
        label for row in rows if row["base"] == base and row["entity_kind"] == "view"
        for label in row["failure_class"] if label != "B-MIXED"
    )


def evaluate(census_run: Path, run_dir: Path) -> None:
    import lpips
    import torch

    census_run = census_run.resolve()
    freeze = json.loads((run_dir / "CONFIRMATION_PROTOCOL_FREEZE.json").read_text(encoding="utf-8"))
    taxonomy = yaml.safe_load((census_run / "BADCASE_TAXONOMY_FREEZE.yaml").read_text(encoding="utf-8"))
    if sha256_file(census_run / "BADCASE_TAXONOMY_FREEZE.yaml") != freeze["taxonomy_sha256"]:
        raise RuntimeError("taxonomy 在 Confirmation prepare 后漂移")
    all_records = read_jsonl(P1_RUN / "MATCHED_FRAME_REGISTRY.jsonl")
    records = [row for row in all_records if row["partition"] == "confirmation"]
    prediction_map, audits = maps(run_dir)
    if len(prediction_map) != len(records) * 2:
        raise RuntimeError("Confirmation prediction denominator 不完整")
    model = lpips.LPIPS(net="alex", verbose=False).eval().cuda()
    protocol = CensusProtocol()
    base_rows, actor_rows, decoded = [], [], {}
    lookup = {(row["scene"], int(row["frame"]), int(row["camera"])): row for row in records}
    for base in ("adgs", "streetgs"):
        for record in sorted(records, key=lambda row: (row["scene"], int(row["frame"]), int(row["camera"]))):
            key = (base, record["scene"], int(record["frame"]), int(record["camera"]))
            prediction = prediction_map[key]
            mask_path = Path(record["target_path"]).parents[1] / "dynamic_masks" / "all" / f"{int(record['frame']):03d}_{int(record['camera'])}.png"
            base_row, actor_row = evaluate_discovery_view(
                base=base, record=record, prediction_path=prediction["prediction_path"],
                dynamic_mask_path=mask_path, lpips_model=model,
                renderer_provenance={"quality_partition": "confirmation"}, resource={},
                protocol=protocol, quality_partition="confirmation",
            )
            base_rows.append(base_row)
            actor_rows.append(actor_row)
            decoded[key] = (*load_metric_rgb(prediction["prediction_path"], protocol)[:1],
                            *load_metric_rgb(record["target_path"], protocol)[:1],
                            *load_metric_mask(mask_path, protocol)[:1])
    temporal_rows = []
    sample = {(row["scene"], int(row["frame"])): row for row in all_records}
    for scene in sorted({row["scene"] for row in all_records}):
        frames = sorted({int(row["frame"]) for row in all_records if row["scene"] == scene})
        for camera in (0, 1, 2):
            for fa, fb in zip(frames, frames[1:]):
                ra, rb = sample[(scene, fa)], sample[(scene, fb)]
                if temporal_window_partition([ra, rb]) != {"status": "defined", "partition": "confirmation"}:
                    continue
                for base in ("adgs", "streetgs"):
                    pa, ta, ma = decoded[(base, scene, fa, camera)]
                    pb, tb, mb = decoded[(base, scene, fb, camera)]
                    temporal_rows.append({
                        "base": base,
                        **temporal_proxy_row(
                            earlier=lookup[(scene, fa, camera)], later=lookup[(scene, fb, camera)],
                            earlier_prediction=pa, earlier_target=ta, later_prediction=pb, later_target=tb,
                            earlier_dynamic=ma, later_dynamic=mb, protocol=protocol,
                            quality_partition="confirmation",
                        ),
                    })
    minimums = taxonomy["minimum_support_pixels"]
    leaderboards = build_leaderboards(base_rows, temporal_rows, minimums)
    selected = panel_union(leaderboards, int(taxonomy["panel_union_limit"]))
    confirmation_registry = build_registry(
        base_rows, temporal_rows, taxonomy["thresholds"], minimums, selected,
        split_role="confirmation", evidence_tier="C",
    )
    discovery_registry = read_jsonl(census_run / "BADCASE_REGISTRY.jsonl")
    class_verdicts = {}
    for base in ("adgs", "streetgs"):
        dcounts, ccounts = class_counts(discovery_registry, base), class_counts(confirmation_registry, base)
        for label in sorted(set(dcounts) | set(ccounts)):
            scenes = {row["scene"] for row in confirmation_registry if row["base"] == base and label in row["failure_class"]}
            if dcounts[label] == 0:
                verdict = "not_applicable"
            elif ccounts[label] == 0:
                verdict = "hypothesis_not_confirmed"
            elif len(scenes) >= 2:
                verdict = "direction_confirmed"
            else:
                verdict = "insufficient"
            class_verdicts[f"{base}|{label}"] = {
                "verdict": verdict, "discovery_cases": dcounts[label], "confirmation_cases": ccounts[label],
                "confirmation_scenes": sorted(scenes),
            }
    for rows, is_confirmation in ((discovery_registry, False), (confirmation_registry, True)):
        for row in rows:
            primary = [label for label in row["failure_class"] if label != "B-MIXED"]
            verdicts = [class_verdicts[f"{row['base']}|{label}"]["verdict"] for label in primary]
            if verdicts and all(value == "direction_confirmed" for value in verdicts):
                row["confirmation_verdict"] = "direction_confirmed"
            elif "hypothesis_not_confirmed" in verdicts:
                row["confirmation_verdict"] = "hypothesis_not_confirmed"
            else:
                row["confirmation_verdict"] = "insufficient"
    verdict_payload = {
        "schema": "worldsim_v521_confirmation_verdict_v1",
        "task_id": "WS-V521-P9-BADCASE-CONFIRMATION-01",
        "status": "verdict_frozen_before_confirmation_panels",
        "class_verdicts": class_verdicts,
        "threshold_refit": False, "ranking_k_changed": False, "failure_predicate_changed": False,
        "discovery_denominator_per_base": len(read_jsonl(census_run / "BASE_CENSUS_METRICS.jsonl")) // 2,
        "confirmation_denominator_per_base": len(records),
    }
    atomic_json(run_dir / "P9_CONFIRMATION_VERDICT.json", verdict_payload)
    # verdict 已先落盘；此后才允许 Confirmation panel。
    row_lookup = {(row["base"], row["scene"], row["canonical_sample_index"], row["camera"]): row for row in base_rows}
    panel_rows = []
    for case in [row for row in confirmation_registry if row["selected_for_panel"] and row["classification_status"] == "labeled"]:
        key = (case["scene"], case["canonical_sample_index"], case["camera"])
        pair = {base: row_lookup[(base, *key)] for base in ("adgs", "streetgs")}
        if pair["adgs"]["target_sha256"] != pair["streetgs"]["target_sha256"]:
            raise RuntimeError(f"Confirmation matched target hash 漂移：{case['case_id']}")
        output_dir = run_dir / "panels" / case["case_id"]
        output_dir.mkdir(parents=True, exist_ok=False)
        panel = build_view_panel(
            target_path=pair["adgs"]["target_path"],
            prediction_paths={base: pair[base]["prediction_path"] for base in pair},
            dynamic_mask_path=pair["adgs"]["dynamic_mask_path"], output=output_dir / "panel.png",
        )
        metadata = {
            "schema": "worldsim_v521_badcase_panel_v1",
            "case_id": case["case_id"], "split_role": "confirmation",
            "classification_status": "labeled", "failure_axes": case["failure_axes"],
            "metric_row_sha256": sha256_json(case),
            "inputs": {
                "target": {
                    "path": pair["adgs"]["target_path"],
                    "source_file_sha256": pair["adgs"]["target_source_sha256"],
                    "decoded_metric_pixel_sha256": pair["adgs"]["target_sha256"],
                },
                "adgs": {"path": pair["adgs"]["prediction_path"], "sha256": pair["adgs"]["prediction_sha256"]},
                "streetgs": {"path": pair["streetgs"]["prediction_path"], "sha256": pair["streetgs"]["prediction_sha256"]},
                "dynamic_mask": {
                    "path": pair["adgs"]["dynamic_mask_path"],
                    "decoded_metric_pixel_sha256": pair["adgs"]["dynamic_mask_sha256"],
                },
            },
            **panel,
        }
        atomic_json(output_dir / "metadata.json", metadata)
        case["panel_path"] = panel["panel_path"]
        panel_rows.append(metadata)
    assert_unique_keys(base_rows, ("base", "scene", "canonical_sample_index", "camera"))
    atomic_jsonl(run_dir / "CONFIRMATION_BASE_CENSUS_METRICS.jsonl", base_rows)
    atomic_jsonl(run_dir / "CONFIRMATION_ACTOR_CENSUS_METRICS.jsonl", actor_rows)
    atomic_jsonl(run_dir / "CONFIRMATION_TEMPORAL_CENSUS_METRICS.jsonl", temporal_rows)
    atomic_json(run_dir / "CONFIRMATION_BADCASE_LEADERBOARDS.json", leaderboards)
    atomic_jsonl(run_dir / "CONFIRMATION_BADCASE_REGISTRY.jsonl", confirmation_registry)
    atomic_jsonl(run_dir / "CONFIRMATION_PANEL_REGISTRY.jsonl", panel_rows)
    atomic_jsonl(census_run / "BADCASE_REGISTRY.jsonl", discovery_registry)
    atomic_json(
        run_dir / "P9_SUMMARY.json",
        {
            "status": "done", "outcome": "p9_confirmation_complete",
            "base_rows": len(base_rows), "actor_rows": len(actor_rows), "temporal_rows": len(temporal_rows),
            "confirmation_registry_rows": len(confirmation_registry), "confirmation_panels": len(panel_rows),
            "checkpoint_hash_gate": all(a["checkpoint_sha256_before"] == a["checkpoint_sha256_after"] for a in audits),
            "threshold_refit": False, "fresh_validation_test_kitti_quality_read": False,
        },
    )
    atomic_json(run_dir / "status.json", {"status": "done", "outcome": "p9_confirmation_complete"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-run", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=("prepare", "render", "evaluate"), required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.census_run, args.run_dir)
    elif args.phase == "render":
        render(args.run_dir)
    else:
        evaluate(args.census_run, args.run_dir)


if __name__ == "__main__":
    main()
