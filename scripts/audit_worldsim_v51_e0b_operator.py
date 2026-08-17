#!/usr/bin/env python3
"""独立重放并审计 Stage E r021 的 E0b no-quality operator。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.progressive_propagation import (
    affinity_matrices_from_topologies,
    build_exact_logical_topologies,
    progressive_region_growing,
)
from motion_proj.worldsim_v51.protocol import load_yaml
from motion_proj.worldsim_v51.superprimitive_propagation import (
    aggregate_node_evidence,
    broadcast_node_result,
    quotient_directed_edges,
)
from scripts.run_worldsim_v51_d0_progressive_operator import load_observation_matrices
from scripts.run_worldsim_v51_e0b_same_propagation import validate_config


SCENES = ("scene-0471", "scene-1087", "scene-0379")


class AuditError(RuntimeError):
    """r021 的冻结 operator evidence 发生漂移。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as table:
        return {name: np.asarray(table[name]) for name in table.files}


def _assert_payload(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AuditError(f"字段漂移: {label}")
        for key in expected:
            _assert_payload(actual[key], expected[key], f"{label}/{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AuditError(f"列表漂移: {label}")
        for index, value in enumerate(expected):
            _assert_payload(actual[index], value, f"{label}/{index}")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise AuditError(f"值漂移: {label}")
        return
    if isinstance(expected, (int, float)):
        if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
            raise AuditError(f"数值漂移: {label}: {actual} != {expected}")
        return
    if actual != expected:
        raise AuditError(f"值漂移: {label}")


def _verify_inventory(run_dir: Path, manifest: dict[str, Any]) -> dict[str, int]:
    expected = {row["path"]: row for row in manifest["inventory"]}
    observed = {
        path.relative_to(run_dir).as_posix(): path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "status.json"}
    }
    if set(observed) != set(expected):
        raise AuditError("manifest inventory 覆盖范围漂移")
    total_bytes = 0
    for relative, path in observed.items():
        record = expected[relative]
        size = path.stat().st_size
        if size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise AuditError(f"manifest identity 漂移: {relative}")
        total_bytes += size
    return {"entry_count": len(observed), "bytes": total_bytes}


def _historical_inputs(e0a: dict[str, Any]) -> dict[str, dict[str, Any]]:
    e0a_config_path = PROJECT / e0a["source"]["config"]["path"]
    e0a_config = load_yaml(e0a_config_path)
    prereg_spec = e0a_config["frozen_stage_d_inputs"]["preregistration"]
    prereg_path = PROJECT / prereg_spec["path"]
    if _sha256(prereg_path) != prereg_spec["sha256"]:
        raise AuditError("Stage D preregistration 身份漂移")
    prereg = load_yaml(prereg_path)
    return {scene["scene"]: scene for scene in prereg["historical_inputs"]}


def _view_names(config: dict[str, Any]) -> list[str]:
    observation = config["observation_contract"]
    return [
        f"f{int(frame):03d}_c{int(camera)}.npz"
        for frame in observation["frames"]
        for camera in observation["cameras"]
    ]


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config, e0a, _ = validate_config(config_path)
    status = _json(run_dir / "status.json")
    summary = _json(run_dir / "summary.json")
    manifest = _json(run_dir / "manifest.json")
    conclusion = "e0b_fine_q50_same_propagation_sidecars_ready_without_quality_read"
    if status.get("status") != "done" or manifest.get("status") != "done":
        raise AuditError("r021 terminal 必须保持 done")
    if summary.get("status") != "done" or summary.get("conclusion") != conclusion:
        raise AuditError("r021 summary terminal 漂移")
    if status.get("conclusion") != conclusion:
        raise AuditError("r021 status conclusion 漂移")
    if (run_dir / "resolved_config.yaml").read_text(
        encoding="utf-8"
    ) != config_path.read_text(encoding="utf-8"):
        raise AuditError("r021 resolved config 不是 byte exact")
    inventory = _verify_inventory(run_dir, manifest)
    source_tree = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", f"{summary['source_commit']}^{{tree}}"],
        text=True,
    ).strip()
    if source_tree != summary["source_tree"] or status["source_commit"] != summary["source_commit"]:
        raise AuditError("r021 source commit/tree 漂移")

    historical = _historical_inputs(e0a)
    bindings = {scene["scene"]: scene for scene in config["scenes"]}
    view_names = _view_names(config)
    scene_audits = []
    for scene_index, scene_name in enumerate(SCENES):
        recorded = summary["scene_reports"][scene_index]
        persisted = _json(run_dir / f"artifacts/reports/{scene_name}.json")
        _assert_payload(persisted, recorded, f"{scene_name}/persisted_report")
        spec = historical[scene_name]
        binding = bindings[scene_name]
        b3_path = Path(spec["b3_unary"]["path"])
        if _sha256(b3_path) != spec["b3_unary"]["sha256"]:
            raise AuditError(f"B3 identity 漂移: {scene_name}")
        b3 = _npz(b3_path)
        gaussian_id = np.asarray(b3["gaussian_id"], dtype=np.int64)
        unary = np.asarray(b3["unary_posterior"], dtype=np.float32)
        gaussian_count = int(binding["gaussian_count"])
        if not np.array_equal(gaussian_id, np.arange(gaussian_count, dtype=np.int64)):
            raise AuditError(f"Gaussian 分母漂移: {scene_name}")

        node_path = Path(binding["node_sidecar"]["path"])
        d0_path = Path(binding["d0_sidecar"]["path"])
        if _sha256(node_path) != binding["node_sidecar"]["sha256"]:
            raise AuditError(f"node sidecar identity 漂移: {scene_name}")
        if _sha256(d0_path) != binding["d0_sidecar"]["sha256"]:
            raise AuditError(f"D0 sidecar identity 漂移: {scene_name}")
        node_source = _npz(node_path)
        d0 = _npz(d0_path)
        node_id = np.asarray(node_source["node_id"], dtype=np.int64)
        d0_posterior = np.asarray(d0["d0_posterior"], dtype=np.float32)
        if not np.array_equal(node_source["gaussian_id"], gaussian_id):
            raise AuditError(f"node Gaussian ID 漂移: {scene_name}")
        if not np.array_equal(d0["gaussian_id"], gaussian_id):
            raise AuditError(f"D0 Gaussian ID 漂移: {scene_name}")
        if int(node_source["node_count"]) != int(binding["node_count"]):
            raise AuditError(f"node count 漂移: {scene_name}")

        graph_config_path = PROJECT / spec["v5_graph_config"]["path"]
        if _sha256(graph_config_path) != spec["v5_graph_config"]["sha256"]:
            raise AuditError(f"graph config identity 漂移: {scene_name}")
        graph_config = load_yaml(graph_config_path)
        unary_manifest = graph_config["inputs"]["unary_manifest"]
        manifest_path = Path(unary_manifest["path"])
        probability, visibility, observation_reports = load_observation_matrices(
            unary_run=manifest_path.parent,
            unary_manifest_path=manifest_path,
            unary_manifest_sha256=unary_manifest["sha256"],
            gaussian_count=gaussian_count,
            view_names=view_names,
        )
        _assert_payload(observation_reports, recorded["observation_views"], f"{scene_name}/views")

        edge_path = Path(spec["v5_graph_run"]["path"]) / "artifacts/graph/edges.npz"
        if _sha256(edge_path) != spec["v5_graph_run"]["files"][
            "artifacts/graph/edges.npz"
        ]:
            raise AuditError(f"raw KNN identity 漂移: {scene_name}")
        edges = _npz(edge_path)
        node_unary, node_probability, node_visibility, aggregation = aggregate_node_evidence(
            unary, probability, visibility, node_id
        )
        quotient_source, quotient_target, quotient = quotient_directed_edges(
            np.asarray(edges["source_gaussian_id"], dtype=np.int64),
            np.asarray(edges["target_gaussian_id"], dtype=np.int64),
            node_id,
        )
        topologies = build_exact_logical_topologies(
            quotient_source,
            quotient_target,
            node_count=int(binding["node_count"]),
            maximum_logical_distance=int(config["method"]["maximum_logical_distance"]),
        )
        affinities, affinity_reports = affinity_matrices_from_topologies(
            topologies,
            node_probability,
            node_visibility,
            chunk_size=int(config["execution"]["affinity_chunk_size"]),
        )
        result = progressive_region_growing(
            node_unary,
            topologies=topologies,
            affinities=affinities,
            progressive_thresholds=config["method"]["progressive_thresholds"],
            logical_distance_decay=float(config["method"]["logical_distance_decay"]),
            actor_seed_minimum=float(config["method"]["actor_seed_minimum"]),
            background_seed_maximum=float(config["method"]["background_seed_maximum"]),
            unknown_probability=float(config["method"]["unknown_probability"]),
        )
        label, posterior, assignment = broadcast_node_result(
            node_id, result["labels"], result["posterior"], result["assignment_level"]
        )
        output_path = run_dir / f"artifacts/sidecars/{scene_name}.npz"
        output = _npz(output_path)
        if set(output) != {
            "assignment_level",
            "d0_raw_posterior",
            "e0b_label",
            "e0b_posterior",
            "gaussian_id",
            "node_id",
            "u2_b3_posterior",
        }:
            raise AuditError(f"E0b output fields 漂移: {scene_name}")
        expected_arrays = {
            "assignment_level": assignment,
            "d0_raw_posterior": d0_posterior,
            "e0b_label": label,
            "e0b_posterior": posterior,
            "gaussian_id": gaussian_id,
            "node_id": node_id,
            "u2_b3_posterior": unary,
        }
        for field, expected in expected_arrays.items():
            if not np.array_equal(output[field], expected):
                raise AuditError(f"E0b full replay array drift: {scene_name}/{field}")

        changed = output["e0b_posterior"] != output["d0_raw_posterior"]
        replayed = {
            "aggregation": aggregation,
            "quotient": quotient,
            "logical_neighbor_counts": [int(matrix.nnz) for matrix in topologies],
            "affinity_reports": affinity_reports,
            "propagation": result["report"],
            "gaussian_readout": {
                "changed_posterior_vs_d0_gaussian_count": int(np.count_nonzero(changed)),
                "changed_posterior_vs_d0_gaussian_fraction": float(
                    np.mean(changed, dtype=np.float64)
                ),
                "node_constant": True,
            },
        }
        for field, value in replayed.items():
            _assert_payload(value, recorded[field], f"{scene_name}/{field}")
        if output_path.stat().st_size != int(recorded["sidecar"]["bytes"]):
            raise AuditError(f"E0b output bytes 漂移: {scene_name}")
        if _sha256(output_path) != recorded["sidecar"]["sha256"]:
            raise AuditError(f"E0b output SHA 漂移: {scene_name}")
        scene_audits.append(
            {
                "scene": scene_name,
                "gaussian_count": gaussian_count,
                "node_count": int(binding["node_count"]),
                "quotient_directed_edge_count": quotient["quotient_directed_edge_count"],
                "changed_posterior_vs_d0_gaussian_count": int(np.count_nonzero(changed)),
                "full_operator_replay_exact": True,
                "node_constant_readout_exact": True,
            }
        )

    repeat0 = run_dir / "artifacts/repeatability/node_prefix_0.npz"
    repeat1 = run_dir / "artifacts/repeatability/node_prefix_1.npz"
    if repeat0.read_bytes() != repeat1.read_bytes():
        raise AuditError("repeatability prefix 不再 byte exact")
    if _sha256(repeat0) != summary["repeatability_probe"]["sha256"]:
        raise AuditError("repeatability prefix SHA 漂移")

    samples = [
        json.loads(line)
        for line in (run_dir / "artifacts/resource_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    valid = [row for row in samples if "monitor_error" not in row]
    resources = summary["resources"]
    if len(samples) != resources["sample_count"] or len(valid) != resources["sample_count"]:
        raise AuditError("resource sample 分母漂移")
    if max(int(row["gpu_used_mib"]) for row in valid) != resources["peak_gpu_used_mib"]:
        raise AuditError("GPU peak 复算漂移")
    if max(int(row["cgroup_memory_current_bytes"]) for row in valid) != resources[
        "peak_cgroup_memory_bytes"
    ]:
        raise AuditError("cgroup peak 复算漂移")
    if resources["memory_events_before"] != resources["memory_events_after"]:
        raise AuditError("memory events 漂移")
    for lock in (
        "quality_read",
        "h_quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "parameter_search",
        "e1_panogs_execution",
        "e2_ag2aussian_execution",
    ):
        if summary.get(lock) is not False:
            raise AuditError(f"禁止项漂移: {lock}")
    if summary.get("m2_status") != "pending" or summary.get("m3_status") != "pending":
        raise AuditError("M2/M3 status 漂移")
    events = [
        json.loads(line)["event"]
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if events != ["run_started", "scene_completed", "scene_completed", "scene_completed", "run_completed"]:
        raise AuditError("r021 event sequence 漂移")

    return {
        "status": "passed",
        "conclusion": "r021_e0b_full_operator_sidecars_and_resources_replayed_exact",
        "run": str(run_dir),
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "inventory": inventory,
        "scene_audits": scene_audits,
        "repeatability": summary["repeatability_probe"],
        "resources": resources,
        "locks": {
            "quality_read": False,
            "parameter_search": False,
            "e1_panogs_execution": False,
            "e2_ag2aussian_execution": False,
            "m2_status": "pending",
            "m3_status": "pending",
        },
        "hashes": {
            name: _sha256(run_dir / name)
            for name in (
                "summary.json",
                "status.json",
                "manifest.json",
                "events.jsonl",
                "artifacts/resources.json",
                "artifacts/resource_samples.jsonl",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_e_e0b_same_propagation_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.config.resolve(), args.run_dir.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
