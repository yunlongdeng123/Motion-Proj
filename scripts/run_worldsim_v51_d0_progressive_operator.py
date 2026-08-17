#!/usr/bin/env python3
"""Run the frozen V5.1 D0 operator on H inputs without reading quality."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v5.evidence_schema import atomic_save_npz
from motion_proj.worldsim_v51.progressive_propagation import (
    affinity_matrices_from_topologies,
    build_exact_logical_topologies,
    progressive_region_growing,
)
from motion_proj.worldsim_v51.protocol import (
    ProtocolError,
    V51_BRANCH,
    load_yaml,
    sha256_file,
)


SCHEMA = "worldsim_v51_stage_d_progressive_operator_v1"
TASK_ID = "WS-V51-M1-D-PROGRESSIVE-01"
SCENES = ("scene-0471", "scene-1087", "scene-0379")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "status.json"}
    ]


def _cgroup_value(name: str) -> int:
    return int((Path("/sys/fs/cgroup") / name).read_text(encoding="utf-8").strip())


def _nvidia_used_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [int(value.strip()) for value in output.splitlines() if value.strip()]
    if len(values) != 1:
        raise ProtocolError("D0 operator expects exactly one GPU")
    return values[0]


class ResourceMonitor:
    def __init__(self, interval_seconds: float):
        self.interval_seconds = float(interval_seconds)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(
                    {
                        "at_utc": _utc_now(),
                        "cgroup_memory_current_bytes": _cgroup_value("memory.current"),
                        "gpu_used_mib": _nvidia_used_mib(),
                    }
                )
            except Exception as error:
                self.samples.append(
                    {
                        "at_utc": _utc_now(),
                        "monitor_error": f"{type(error).__name__}: {error}",
                    }
                )
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != SCHEMA:
        raise ProtocolError("D0 operator schema drift")
    if config.get("task_id") != TASK_ID or config.get("status") != "running":
        raise ProtocolError("D0 operator task/status drift")
    if config.get("phase") != "d0_faithful_operator_without_quality_read":
        raise ProtocolError("D0 operator phase drift")
    if int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("D0 operator seed drift")

    freeze_spec = config["preflight_freeze"]
    freeze_path = PROJECT / freeze_spec["path"]
    if not freeze_path.is_file() or sha256_file(freeze_path) != freeze_spec["sha256"]:
        raise ProtocolError("D0 preflight freeze identity drift")
    freeze = load_yaml(freeze_path)
    if freeze.get("status") != freeze_spec["required_status"]:
        raise ProtocolError("D0 preflight terminal drift")
    if freeze["canonical_run"]["conclusion"] != freeze_spec["required_conclusion"]:
        raise ProtocolError("D0 preflight conclusion drift")

    method = config["method"]
    expected_method = {
        "arm": "D0",
        "invariant_baseline": "U2_B3",
        "node": "raw_gaussian",
        "maximum_logical_distance": 2,
        "semantic_distribution": "l2_normalized_binary_vector_one_minus_p_and_p",
        "pair_similarity": "cosine",
        "logical_distance_decay": 0.5,
        "progressive_thresholds": [0.9, 0.8, 0.7, 0.6, 0.5],
        "actor_seed_minimum": 0.9,
        "background_seed_maximum": 0.1,
        "unknown_probability": 0.5,
        "parameter_search": False,
        "upstream_code_import": False,
    }
    for name, expected in expected_method.items():
        if method.get(name) != expected:
            raise ProtocolError(f"D0 operator mechanism drift: {name}")

    observation = config["observation_contract"]
    if observation["frames"] != [0, 40, 80, 120, 160]:
        raise ProtocolError("D0 observation frame drift")
    if observation["cameras"] != [0, 1, 2]:
        raise ProtocolError("D0 observation camera drift")
    if observation.get("expected_view_count_per_scene") != 15:
        raise ProtocolError("D0 observation denominator drift")
    if observation.get("use_evaluation_views") is not False:
        raise ProtocolError("D0 operator cannot read evaluation views")
    if tuple(config["execution"]["scenes"]) != SCENES:
        raise ProtocolError("D0 H scene order drift")
    if config["execution"]["maximum_rounds_per_threshold"] is not None:
        raise ProtocolError("D0 must grow to a fixed point")

    locks = config["locks"]
    for name in (
        "parse_v5_quality_diagnostics",
        "render_evaluation_views",
        "h_quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "final_heldout_quality_read",
        "kitti_method_tuning",
        "parameter_search",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"D0 operator lock drift: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 must remain pending")
    return config, freeze


def _view_names(config: dict[str, Any]) -> list[str]:
    observation = config["observation_contract"]
    return [
        f"f{int(frame):03d}_c{int(camera)}.npz"
        for frame in observation["frames"]
        for camera in observation["cameras"]
    ]


def load_observation_matrices(
    *,
    unary_run: Path,
    unary_manifest_path: Path,
    unary_manifest_sha256: str,
    gaussian_count: int,
    view_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if sha256_file(unary_manifest_path) != unary_manifest_sha256:
        raise ProtocolError("D0 unary manifest identity drift")
    manifest = json.loads(unary_manifest_path.read_text(encoding="utf-8"))
    inventory = {record["path"]: record for record in manifest["inventory"]}
    probability = np.full((gaussian_count, len(view_names)), 0.5, dtype=np.float32)
    visibility = np.zeros_like(probability)
    reports = []
    for view_index, name in enumerate(view_names):
        relative = f"artifacts/observations/{name}"
        record = inventory.get(relative)
        path = unary_run / relative
        if record is None or not path.is_file():
            raise ProtocolError(f"D0 observation missing from manifest: {relative}")
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ProtocolError(f"D0 observation identity drift: {relative}")
        with np.load(path, allow_pickle=False) as table:
            gaussian_id = np.asarray(table["gaussian_id"], dtype=np.int64)
            if gaussian_id.ndim != 1 or np.unique(gaussian_id).size != gaussian_id.size:
                raise ProtocolError(f"D0 observation Gaussian ids are not unique: {relative}")
            if np.any(gaussian_id < 0) or np.any(gaussian_id >= gaussian_count):
                raise ProtocolError(f"D0 observation Gaussian id out of range: {relative}")
            sam_probability = np.asarray(table["sam_probability"], dtype=np.float32)
            observed_visibility = np.asarray(table["visibility"], dtype=np.float32)
            valid = (
                np.asarray(table["sam_probability_available"], dtype=bool)
                & np.asarray(table["mask_quality_accepted"], dtype=bool)
                & (np.asarray(table["reliability"], dtype=np.float32) > 0.0)
                & (observed_visibility > 0.0)
            )
            if not np.isfinite(sam_probability).all() or np.any(
                (sam_probability < 0.0) | (sam_probability > 1.0)
            ):
                raise ProtocolError(f"D0 observation probability drift: {relative}")
            if not np.isfinite(observed_visibility).all() or np.any(
                (observed_visibility < 0.0) | (observed_visibility > 1.0)
            ):
                raise ProtocolError(f"D0 observation visibility drift: {relative}")
            selected = gaussian_id[valid]
            probability[selected, view_index] = sam_probability[valid]
            visibility[selected, view_index] = observed_visibility[valid]
            reports.append(
                {
                    "view": name[:-4],
                    "manifest_sha256": record["sha256"],
                    "row_count": int(gaussian_id.size),
                    "valid_row_count": int(np.count_nonzero(valid)),
                }
            )
    return probability, visibility, reports


def _scene_input_map(freeze: dict[str, Any]) -> dict[str, dict[str, Any]]:
    prereg_path = PROJECT / freeze["source"]["preregistration"]["path"]
    if sha256_file(prereg_path) != freeze["source"]["preregistration"]["sha256"]:
        raise ProtocolError("D0 preregistration identity drift")
    prereg = load_yaml(prereg_path)
    return {scene["scene"]: scene for scene in prereg["historical_inputs"]}


def _prefix_repeatability_probe(
    unary: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    probability_by_view: np.ndarray,
    visibility_by_view: np.ndarray,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    probe = config["execution"]["repeatability_probe"]
    prefix_count = min(int(probe["node_prefix_count"]), unary.size)
    selected = (source < prefix_count) & (target < prefix_count)
    if not np.any(selected):
        raise ProtocolError("D0 repeatability prefix has no edges")
    payloads = []
    paths = []
    for repeat in range(2):
        topologies = build_exact_logical_topologies(
            source[selected],
            target[selected],
            node_count=prefix_count,
            maximum_logical_distance=int(config["method"]["maximum_logical_distance"]),
        )
        affinities, _ = affinity_matrices_from_topologies(
            topologies,
            probability_by_view[:prefix_count],
            visibility_by_view[:prefix_count],
            chunk_size=int(config["execution"]["affinity_chunk_size"]),
        )
        result = progressive_region_growing(
            unary[:prefix_count],
            topologies=topologies,
            affinities=affinities,
            progressive_thresholds=config["method"]["progressive_thresholds"],
            logical_distance_decay=float(config["method"]["logical_distance_decay"]),
            actor_seed_minimum=float(config["method"]["actor_seed_minimum"]),
            background_seed_maximum=float(config["method"]["background_seed_maximum"]),
            unknown_probability=float(config["method"]["unknown_probability"]),
        )
        payload = {
            "gaussian_id": np.arange(prefix_count, dtype=np.int64),
            "d0_label": result["labels"],
            "d0_posterior": result["posterior"],
            "assignment_level": result["assignment_level"],
        }
        path = run_dir / f"artifacts/repeatability/prefix_{repeat}.npz"
        atomic_save_npz(path, payload)
        payloads.append(payload)
        paths.append(path)
    for name in payloads[0]:
        if not np.array_equal(payloads[0][name], payloads[1][name]):
            raise ProtocolError(f"D0 repeatability array drift: {name}")
    if paths[0].read_bytes() != paths[1].read_bytes():
        raise ProtocolError("D0 repeatability NPZ is not byte exact")
    return {
        "node_prefix_count": prefix_count,
        "edge_count": int(np.count_nonzero(selected)),
        "byte_exact": True,
        "sha256": sha256_file(paths[0]),
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config, freeze = validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"D0 run directory already exists: {run_dir}")
    if _git("branch", "--show-current") != V51_BRANCH:
        raise ProtocolError("D0 operator must run on the V5.1 M1 branch")
    if _git("status", "--porcelain"):
        raise ProtocolError("D0 operator requires a clean worktree")

    run_dir.mkdir(parents=True)
    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    events = [{"event": "run_started", "at_utc": _utc_now(), "source_commit": source_commit}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(
        run_dir / "status.json",
        {"schema_version": "worldsim_v51_d0_operator_status_v1", "status": "running"},
    )

    monitor = ResourceMonitor(config["resources"]["monitor_interval_seconds"])
    start = time.perf_counter()
    memory_events_before = Path("/sys/fs/cgroup/memory.events").read_text(encoding="utf-8")
    monitor.start()
    try:
        inputs = _scene_input_map(freeze)
        view_names = _view_names(config)
        scene_reports = []
        repeatability_report = None
        for scene_name in SCENES:
            scene_start = time.perf_counter()
            spec = inputs[scene_name]
            b3_path = Path(spec["b3_unary"]["path"])
            if sha256_file(b3_path) != spec["b3_unary"]["sha256"]:
                raise ProtocolError(f"D0 B3 identity drift: {scene_name}")
            with np.load(b3_path, allow_pickle=False) as b3:
                gaussian_id = np.asarray(b3["gaussian_id"], dtype=np.int64).copy()
                unary = np.asarray(b3["unary_posterior"], dtype=np.float32).copy()
            gaussian_count = int(spec["expected_gaussian_count"])
            if gaussian_id.size != gaussian_count or not np.array_equal(
                gaussian_id, np.arange(gaussian_count, dtype=np.int64)
            ):
                raise ProtocolError(f"D0 Gaussian id denominator drift: {scene_name}")

            graph_config = load_yaml(PROJECT / spec["v5_graph_config"]["path"])
            unary_manifest = graph_config["inputs"]["unary_manifest"]
            unary_manifest_path = Path(unary_manifest["path"])
            unary_run = unary_manifest_path.parent
            probability_by_view, visibility_by_view, observation_reports = (
                load_observation_matrices(
                    unary_run=unary_run,
                    unary_manifest_path=unary_manifest_path,
                    unary_manifest_sha256=unary_manifest["sha256"],
                    gaussian_count=gaussian_count,
                    view_names=view_names,
                )
            )
            load_seconds = time.perf_counter() - scene_start

            edge_path = Path(spec["v5_graph_run"]["path"]) / "artifacts/graph/edges.npz"
            expected_edge_sha = spec["v5_graph_run"]["files"]["artifacts/graph/edges.npz"]
            if sha256_file(edge_path) != expected_edge_sha:
                raise ProtocolError(f"D0 edge identity drift: {scene_name}")
            with np.load(edge_path, allow_pickle=False) as edges:
                source = np.asarray(edges["source_gaussian_id"], dtype=np.int64).copy()
                target = np.asarray(edges["target_gaussian_id"], dtype=np.int64).copy()

            topology_start = time.perf_counter()
            topologies = build_exact_logical_topologies(
                source,
                target,
                node_count=gaussian_count,
                maximum_logical_distance=int(config["method"]["maximum_logical_distance"]),
            )
            topology_seconds = time.perf_counter() - topology_start
            affinity_start = time.perf_counter()
            affinities, affinity_reports = affinity_matrices_from_topologies(
                topologies,
                probability_by_view,
                visibility_by_view,
                chunk_size=int(config["execution"]["affinity_chunk_size"]),
            )
            affinity_seconds = time.perf_counter() - affinity_start

            if repeatability_report is None and config["execution"]["repeatability_probe"]["enabled"]:
                repeatability_report = _prefix_repeatability_probe(
                    unary,
                    source,
                    target,
                    probability_by_view,
                    visibility_by_view,
                    config,
                    run_dir,
                )

            propagation_start = time.perf_counter()
            result = progressive_region_growing(
                unary,
                topologies=topologies,
                affinities=affinities,
                progressive_thresholds=config["method"]["progressive_thresholds"],
                logical_distance_decay=float(config["method"]["logical_distance_decay"]),
                actor_seed_minimum=float(config["method"]["actor_seed_minimum"]),
                background_seed_maximum=float(config["method"]["background_seed_maximum"]),
                unknown_probability=float(config["method"]["unknown_probability"]),
                maximum_rounds_per_threshold=config["execution"][
                    "maximum_rounds_per_threshold"
                ],
            )
            propagation_seconds = time.perf_counter() - propagation_start
            output_path = run_dir / f"artifacts/sidecars/{scene_name}.npz"
            atomic_save_npz(
                output_path,
                {
                    "assignment_level": result["assignment_level"],
                    "d0_label": result["labels"],
                    "d0_posterior": result["posterior"],
                    "gaussian_id": gaussian_id,
                    "u2_b3_posterior": unary,
                },
            )
            report = {
                "scene": scene_name,
                "gaussian_count": gaussian_count,
                "directed_input_edge_count": int(source.size),
                "logical_neighbor_counts": [int(matrix.nnz) for matrix in topologies],
                "observation_views": observation_reports,
                "affinity_reports": affinity_reports,
                "propagation": result["report"],
                "seconds": {
                    "load": load_seconds,
                    "topology": topology_seconds,
                    "affinity": affinity_seconds,
                    "propagation": propagation_seconds,
                    "total": time.perf_counter() - scene_start,
                },
                "sidecar": {
                    "path": output_path.relative_to(run_dir).as_posix(),
                    "bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                },
                "quality_read": False,
            }
            if report["seconds"]["total"] > float(
                config["resources"]["maximum_wall_seconds_per_scene"]
            ):
                raise ProtocolError(f"D0 scene wall limit exceeded: {scene_name}")
            _write_json(run_dir / f"artifacts/reports/{scene_name}.json", report)
            scene_reports.append(report)
            del topologies, affinities, probability_by_view, visibility_by_view
            del source, target, unary, gaussian_id, result
            gc.collect()
            events.append({"event": "scene_completed", "at_utc": _utc_now(), "scene": scene_name})
            _write_jsonl(run_dir / "events.jsonl", events)

        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid_samples = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid_samples:
            raise ProtocolError("D0 resource monitor has no valid samples")
        resources = {
            "sample_count": len(monitor.samples),
            "monitor_error_count": len(monitor.samples) - len(valid_samples),
            "peak_cgroup_memory_bytes": max(
                int(row["cgroup_memory_current_bytes"]) for row in valid_samples
            ),
            "peak_gpu_used_mib": max(int(row["gpu_used_mib"]) for row in valid_samples),
            "memory_events_before": memory_events_before,
            "memory_events_after": Path("/sys/fs/cgroup/memory.events").read_text(
                encoding="utf-8"
            ),
            "total_wall_seconds": time.perf_counter() - start,
        }
        _write_json(run_dir / "artifacts/resources.json", resources)
        if resources["monitor_error_count"] != 0:
            raise ProtocolError("D0 resource monitor reported errors")
        if resources["peak_cgroup_memory_bytes"] > int(
            config["resources"]["maximum_cgroup_memory_bytes"]
        ):
            raise ProtocolError("D0 cgroup memory limit exceeded")
        if resources["peak_gpu_used_mib"] > int(
            config["resources"]["maximum_gpu_used_mib"]
        ):
            raise ProtocolError("D0 unexpected GPU usage")
        if resources["total_wall_seconds"] > float(
            config["resources"]["maximum_total_wall_seconds"]
        ):
            raise ProtocolError("D0 total wall limit exceeded")
        if resources["memory_events_before"] != resources["memory_events_after"]:
            raise ProtocolError("D0 cgroup memory events changed")

        summary = {
            "schema_version": "worldsim_v51_d0_operator_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "conclusion": config["success_conclusion"],
            "source_commit": source_commit,
            "source_tree": source_tree,
            "scene_reports": scene_reports,
            "repeatability_probe": repeatability_report,
            "resources": resources,
            "quality_read": False,
            "h_quality_read": False,
            "screening_quality_read": False,
            "confirmation_quality_read": False,
            "validation_quality_read": False,
            "test_quality_read": False,
            "m2_status": "pending",
            "m3_status": "pending",
        }
        _write_json(run_dir / "summary.json", summary)
        events.append({"event": "run_completed", "at_utc": _utc_now()})
        _write_jsonl(run_dir / "events.jsonl", events)
        manifest = {
            "schema_version": "worldsim_v51_d0_operator_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_d0_operator_status_v1",
                "task_id": TASK_ID,
                "status": "done",
                "conclusion": config["success_conclusion"],
                "source_commit": source_commit,
            },
        )
        return summary
    except BaseException as error:
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        events.append(
            {
                "event": "run_blocked",
                "at_utc": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_d0_operator_status_v1",
                "task_id": TASK_ID,
                "status": "blocked",
                "error": f"{type(error).__name__}: {error}",
                "source_commit": source_commit,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_d_progressive_operator_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
