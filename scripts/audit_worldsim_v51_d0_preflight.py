#!/usr/bin/env python3
"""Fail-closed preflight for the preregistered V5.1 D0 faithful port."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, V51_BRANCH, sha256_file


SCHEMA = "worldsim_v51_stage_d_progressive_preflight_v1"
TASK_ID = "WS-V51-M1-D-PROGRESSIVE-01"
SCENES = ("scene-0471", "scene-1087", "scene-0379")
THRESHOLDS = (0.9, 0.8, 0.7, 0.6, 0.5)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"YAML top level must be a mapping: {path}")
    return payload


def _verify_file(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ProtocolError(f"SHA drift for {label}: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA:
        raise ProtocolError("D0 preflight schema drift")
    if config.get("task_id") != TASK_ID or config.get("status") != "preregistered":
        raise ProtocolError("D0 preflight task/status drift")
    if int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("D0 seed drift")

    route = config["route"]
    if route.get("arm") != "D0" or route.get("invariant_baseline") != "U2_B3":
        raise ProtocolError("D0 arm or U2/B3 invariant drift")
    if route.get("previous_route_status") != "rejected":
        raise ProtocolError("rejected LUDVIG route was reopened")
    if route.get("next_route_on_rejection") != "SUPER_PRIMITIVE_OR_ANCHOR":
        raise ProtocolError("frozen M1 route order drift")

    paper = config["paper_source"]
    if paper["official_repository"] != "https://github.com/yd-yin/SAI3D.git":
        raise ProtocolError("SAI3D repository drift")
    if paper["license_inventory"] != {
        "explicit_license_file_present": False,
        "policy": "clean_room_reimplementation_from_paper_equations_no_upstream_code_copy",
    }:
        raise ProtocolError("SAI3D license policy drift")

    method = config["faithful_mechanism"]
    expected = {
        "node": "raw_gaussian",
        "node_change": False,
        "maximum_logical_distance": 2,
        "semantic_distribution": "l2_normalized_binary_vector_one_minus_p_and_p",
        "pair_similarity": "cosine",
        "logical_distance_decay": 0.5,
        "progressive_thresholds": list(THRESHOLDS),
        "high_confidence_actor_seed_minimum": 0.9,
        "high_confidence_background_seed_maximum": 0.1,
        "seed_source": "frozen_u2_b3_posterior",
        "global_one_shot_smoothing": False,
        "small_region_forced_merge": False,
        "parameter_search": False,
    }
    for name, value in expected.items():
        if method.get(name) != value:
            raise ProtocolError(f"D0 mechanism drift: {name}")

    scenes = config["historical_inputs"]
    if tuple(item["scene"] for item in scenes) != SCENES:
        raise ProtocolError("D0 H scene order drift")
    if tuple(int(item["expected_evaluation_view_count"]) for item in scenes) != (8, 1, 3):
        raise ProtocolError("D0 H evaluation denominator drift")

    gate = config["h_gate_after_operator_freeze"]
    if gate.get("comparator") != "U2_B3_G0":
        raise ProtocolError("D0 comparator drift")
    if gate.get("minimum_positive_boundary_f1_scenes") != 2:
        raise ProtocolError("D0 H BF1 support gate drift")
    if gate.get("minimum_scene_balanced_boundary_f1_delta_exclusive") != 0.0:
        raise ProtocolError("D0 H BF1 mean gate drift")
    if gate.get("minimum_scene_balanced_iou_delta") != 0.0:
        raise ProtocolError("D0 H IoU gate drift")
    if gate.get("maximum_scene_balanced_false_negative_semantic_mass_delta") != 0.02:
        raise ProtocolError("D0 H FN gate drift")
    if gate.get("fail_action") != (
        "reject_progressive_skip_d1_then_advance_super_primitive_or_anchor"
    ):
        raise ProtocolError("D0 fail action drift")

    locks = config["locks"]
    for name in (
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "validation_parameter_search",
        "test_quality_read",
        "test_parameter_search",
        "final_heldout_quality_read",
        "kitti_method_tuning",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"D0 quality lock drift: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 must remain pending")
    if config["preflight"].get("quality_read") is not False:
        raise ProtocolError("D0 preflight may not read quality")
    if config.get("failure_ledger_refs") != [
        "V51-F31",
        "V51-F32",
        "V51-F33",
        "V51-F34",
        "V51-F35",
    ]:
        raise ProtocolError("D0 failure ledger reference drift")


def audit(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    validate_config(config)
    inventory: list[dict[str, Any]] = []

    if _git(PROJECT, "branch", "--show-current") != V51_BRANCH:
        raise ProtocolError("D0 preflight must run on the V5.1 M1 branch")
    if config["preflight"].get("require_clean_worktree") is not True:
        raise ProtocolError("D0 clean-worktree requirement drift")
    if _git(PROJECT, "status", "--porcelain"):
        raise ProtocolError("D0 preflight requires a clean project worktree")

    for name, spec in config["project_bindings"].items():
        record = _verify_file(PROJECT / spec["path"], spec["sha256"], name)
        record["label"] = f"project_binding:{name}"
        inventory.append(record)

    stage_a = _load_yaml(PROJECT / config["project_bindings"]["stage_a_closeout"]["path"])
    if stage_a.get("stage_a_arms", {}).get("survivor") != "U2_B3":
        raise ProtocolError("Stage A survivor is no longer U2/B3")
    ludvig = _load_yaml(
        PROJECT / config["project_bindings"]["rejected_ludvig_freeze"]["path"]
    )
    if ludvig.get("status") != "rejected":
        raise ProtocolError("LUDVIG rejected freeze drift")

    upstream = config["paper_source"]
    checkout = Path(upstream["checkout"])
    if _git(checkout, "rev-parse", "HEAD") != upstream["commit"]:
        raise ProtocolError("SAI3D commit drift")
    if _git(checkout, "rev-parse", "HEAD^{tree}") != upstream["tree"]:
        raise ProtocolError("SAI3D tree drift")
    if _git(checkout, "status", "--porcelain"):
        raise ProtocolError("SAI3D checkout is dirty")
    license_files = sorted(
        path.name
        for path in checkout.iterdir()
        if path.is_file()
        and (path.name.upper().startswith("LICENSE") or path.name.upper().startswith("COPYING"))
    )
    if license_files or upstream["license_inventory"]["explicit_license_file_present"]:
        raise ProtocolError("SAI3D license inventory drift")
    for relative, expected in upstream["files"].items():
        record = _verify_file(checkout / relative, expected, f"upstream:{relative}")
        record["label"] = f"upstream:{relative}"
        inventory.append(record)

    scene_reports = []
    for scene in config["historical_inputs"]:
        graph_config_path = PROJECT / scene["v5_graph_config"]["path"]
        record = _verify_file(
            graph_config_path,
            scene["v5_graph_config"]["sha256"],
            f"{scene['scene']}:v5_graph_config",
        )
        record["label"] = f"{scene['scene']}:v5_graph_config"
        inventory.append(record)
        graph_config = _load_yaml(graph_config_path)
        graph = graph_config["graph"]
        if graph.get("candidate_k") != 6 or graph.get("base_model_consumed_by_graph") is not False:
            raise ProtocolError(f"frozen topology contract drift: {scene['scene']}")
        b3 = scene["b3_unary"]
        if graph_config["inputs"]["unary_b3_gaussians"] != b3:
            raise ProtocolError(f"B3 binding differs from frozen V5 config: {scene['scene']}")
        record = _verify_file(Path(b3["path"]), b3["sha256"], f"{scene['scene']}:B3")
        record["label"] = f"{scene['scene']}:B3"
        inventory.append(record)
        with np.load(b3["path"], allow_pickle=False) as table:
            gaussian_count = int(table["gaussian_id"].size)
            if gaussian_count != int(scene["expected_gaussian_count"]):
                raise ProtocolError(f"Gaussian count drift: {scene['scene']}")

        run = scene["v5_graph_run"]
        run_path = Path(run["path"])
        for relative, expected in run["files"].items():
            record = _verify_file(
                run_path / relative,
                expected,
                f"{scene['scene']}:graph_run:{relative}",
            )
            record["label"] = f"{scene['scene']}:graph_run:{relative}"
            inventory.append(record)
        status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "done":
            raise ProtocolError(f"frozen V5 graph terminal drift: {scene['scene']}")
        with np.load(run_path / "artifacts/graph/edges.npz", allow_pickle=False) as edges:
            edge_count = int(edges["source_gaussian_id"].size)
            if edge_count != gaussian_count * 6:
                raise ProtocolError(f"frozen KNN denominator drift: {scene['scene']}")
        scene_reports.append(
            {
                "scene": scene["scene"],
                "gaussian_count": gaussian_count,
                "directed_edge_count": edge_count,
                "evaluation_view_count": int(scene["expected_evaluation_view_count"]),
            }
        )

    return {
        "schema_version": "worldsim_v51_stage_d_progressive_preflight_report_v1",
        "task_id": TASK_ID,
        "status": "done",
        "conclusion": "d0_faithful_port_inputs_and_source_ready_without_quality_read",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_commit": _git(PROJECT, "rev-parse", "HEAD"),
        "source_tree": _git(PROJECT, "rev-parse", "HEAD^{tree}"),
        "upstream_commit": upstream["commit"],
        "upstream_tree": upstream["tree"],
        "upstream_explicit_license_file_present": False,
        "implementation_policy": upstream["license_inventory"]["policy"],
        "scenes": scene_reports,
        "inventory": inventory,
        "inventory_file_count": len(inventory),
        "quality_read": False,
        "screening_quality_read": False,
        "confirmation_quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "m2_status": "pending",
        "m3_status": "pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_d_progressive_preflight_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.config.resolve())
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
