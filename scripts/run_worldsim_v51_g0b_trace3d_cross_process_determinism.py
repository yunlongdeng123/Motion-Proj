#!/usr/bin/env python3
"""Test the exact Trace3D reverse-tracing extension across fresh processes."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.run_worldsim_v51_f0b_three_view_association_parity import _load_yaml, _verify, repository_source_identity
from scripts.run_worldsim_v51_h_uplift import _inventory, _utc_now, _write_json, _write_jsonl, _write_text


SCHEMA = "worldsim_v51_stage_g_g0b_trace3d_cross_process_determinism_v1"
TASK_ID = "WS-V51-M1-G-AMBIGUITY-01"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _validate_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("G0b Trace3D determinism config drift")
    auth = config["authorization"]["capability_freeze"]
    freeze = _load_yaml(_verify(PROJECT / auth["path"], auth["sha256"], "Trace3D capability freeze", int(auth["bytes"])))
    if (
        freeze.get("status") != auth["required_status"]
        or freeze["governance"].get("next_task") != auth["required_next_task"]
        or freeze["governance"].get("next_phase") != auth["required_next_phase"]
        or freeze["new_observation"].get("cross_process_alpha_exact") is not auth["required_cross_process_alpha_exact"]
    ):
        raise ProtocolError("G0b Trace3D determinism authorization drift")
    inherited = config["synthetic_probe"]
    _verify(PROJECT / inherited["inherited_exact_from"], inherited["inherited_config_sha256"], "G0a inherited config", int(inherited["inherited_config_bytes"]))
    if int(config["runtime"]["fresh_process_count"]) != 8 or config["decision"].get("threshold_search_allowed") is not False or config["decision"].get("source_patch_allowed") is not False:
        raise ProtocolError("G0b frozen process/decision drift")
    false_locks = (
        "network_access", "source_patch", "official_source_mutation", "real_checkpoint_read", "camera_metadata_read",
        "image_pixels_read", "mask_pixels_read", "quality_metrics_read", "training", "gaussian_mutation", "h_quality_read",
        "screening_quality_read", "confirmation_quality_read", "validation_quality_read", "test_quality_read", "kitti_method_tuning",
    )
    if any(config["locks"].get(name) is not False for name in false_locks):
        raise ProtocolError("G0b no-data/no-quality lock drift")
    return config


def _projection_matrix(torch: Any, device: str) -> Any:
    znear, zfar = 0.01, 100.0
    matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix.transpose(0, 1).contiguous()


def _child_probe(config: dict[str, Any]) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    package_root = Path(config["runtime"]["package_root"])
    sys.path.insert(0, str(package_root))
    torch = importlib.import_module("torch")
    raster = importlib.import_module("diff_id_rasterization")
    device = "cuda:0"
    view = torch.eye(4, dtype=torch.float32, device=device)
    projection = _projection_matrix(torch, device)
    operator = raster.GaussianRasterizer(raster_settings=raster.GaussianRasterizationSettings(
        image_height=32, image_width=32, tanfovx=1.0, tanfovy=1.0, bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0, viewmatrix=view, projmatrix=projection, sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device=device), prefiltered=False, debug=False, include_feature=False,
    ))
    means3d = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=device)
    means2d = torch.zeros_like(means3d)
    opacity = torch.tensor([[0.9]], dtype=torch.float32, device=device)
    scales = torch.tensor([[0.1, 0.1]], dtype=torch.float32, device=device)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    inputs = [means3d, means2d, opacity, scales, rotations, view, projection]
    before = [tensor.clone() for tensor in inputs]
    torch.cuda.reset_peak_memory_stats(0)

    def traced(label: int, alpha: bool) -> list[list[float]]:
        weights = torch.zeros((1, 2), dtype=torch.float32, device=device)
        mask = torch.full((32, 32), label, dtype=torch.int32, device=device)
        operator.trace(means3D=means3d, means2D=means2d, opacities=opacity, weights=weights, id_masks=mask,
                       num_class=1, alpha_w=alpha, scales=scales, rotations=rotations)
        torch.cuda.synchronize(0)
        return weights.cpu().tolist()

    row = {
        "background_hard": traced(0, False),
        "foreground_hard_first": traced(1, False),
        "foreground_hard_second": traced(1, False),
        "foreground_alpha_first": traced(1, True),
        "foreground_alpha_second": traced(1, True),
        "inputs_immutable": all(torch.equal(old, new) for old, new in zip(before, inputs)),
        "torch_peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
    }
    return row


def _vector_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    identity = repository_source_identity()
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "running", "source_commit": identity["commit"]})
    started = time.perf_counter()

    runtime = config["runtime"]
    extension = Path(runtime["package_root"]) / runtime["extension"]["path"]
    _verify(extension, runtime["extension"]["sha256"], "Trace3D extension", int(runtime["extension"]["bytes"]))
    source = config["official_source_hazard"]
    repo = Path(source["repository"])
    source_path = repo / source["path"]
    _verify(source_path, source["sha256"], "Trace3D id_trace source", int(source["bytes"]))
    if _git(repo, "rev-parse", "HEAD") != source["commit"] or _git(repo, "status", "--porcelain"):
        raise ProtocolError("G0b official source identity drift")
    text = source_path.read_text(encoding="utf-8")
    begin = text.index(source["function_begin"])
    end = text.index(source["function_end"], begin)
    function = text[begin:end]
    hazard = {
        "plain_global_weight_increment_count": function.count("weights[int(collected_id[j] * (num_class+1) + C)] +="),
        "atomic_add_count": function.count("atomicAdd"),
    }
    if hazard["plain_global_weight_increment_count"] != int(source["expected_plain_global_weight_increment_count"]) or hazard["atomic_add_count"] != int(source["expected_atomic_add_count"]):
        raise ProtocolError(f"G0b source hazard audit drift: {hazard}")
    _write_json(run_dir / "artifacts/source_hazard.json", hazard)

    child_env = os.environ.copy()
    child_env.update({"PYTHONPATH": runtime["package_root"], "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "0"})
    rows = []
    for index in range(int(runtime["fresh_process_count"])):
        command = [runtime["python"], str(Path(__file__).resolve()), "--config", str(config_path), "--child"]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, env=child_env)
        if completed.returncode != 0:
            raise ProtocolError(f"G0b child {index} failed: {completed.stderr[-2000:]}")
        row = json.loads(completed.stdout)
        row["process_index"] = index
        rows.append(row)
        _write_json(run_dir / f"artifacts/process_{index:02d}.json", row)

    background_values = [row["background_hard"] for row in rows]
    hard_values = [row[key] for row in rows for key in ("foreground_hard_first", "foreground_hard_second")]
    alpha_values = [row[key] for row in rows for key in ("foreground_alpha_first", "foreground_alpha_second")]
    unique_hard = sorted({_vector_key(value) for value in hard_values})
    unique_alpha = sorted({_vector_key(value) for value in alpha_values})
    expected_background = config["gates"]["background_hard_all_exact"]
    expected_foreground = config["gates"]["foreground_hard_expected"]
    alpha_scalars = [float(value[0][1]) for value in alpha_values]
    checks = {
        "all_processes_exit_zero": len(rows) == int(runtime["fresh_process_count"]),
        "background_hard_all_exact": all(value == expected_background for value in background_values),
        "foreground_hard_unique_vector_count": len(unique_hard) <= int(config["gates"]["foreground_hard_unique_vector_count_maximum"]),
        "foreground_hard_expected": all(value == expected_foreground for value in hard_values),
        "foreground_alpha_unique_vector_count": len(unique_alpha) <= int(config["gates"]["foreground_alpha_unique_vector_count_maximum"]),
        "alpha_all_finite_positive_and_not_above_corresponding_hard": all(math.isfinite(value) and 0 < value <= 1.0 for value in alpha_scalars),
        "every_process_input_tensors_bitwise_immutable": all(row["inputs_immutable"] is True for row in rows),
    }
    passed = all(checks.values())
    outcome = config["decision"]["pass_outcome"] if passed else config["decision"]["fail_outcome"]
    conclusion = "faithful_trace3d_reverse_tracing_operator_cross_process_determinism_pass" if passed else config["decision"]["fail_conclusion"]
    next_task = TASK_ID if passed else config["decision"]["fail_next_task"]
    next_action = config["decision"]["pass_next_action"] if passed else config["decision"]["fail_next_action"]
    forensic = {
        "schema_version": "worldsim_v51_g0b_trace3d_determinism_forensic_v1", "outcome": outcome,
        "process_count": len(rows), "checks": checks, "unique_hard_vectors": unique_hard, "unique_alpha_vectors": unique_alpha,
        "alpha_min": min(alpha_scalars), "alpha_max": max(alpha_scalars), "alpha_unique_count": len(unique_alpha),
        "source_hazard": hazard, "root_cause_proven": False,
        "threshold_search": False, "source_patch": False, "real_checkpoint_read": False, "quality_metrics_read": False,
    }
    _write_json(run_dir / "artifacts/forensic.json", forensic)
    resources = {
        "wall_seconds": time.perf_counter() - started,
        "cgroup_memory_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text().strip()),
        "maximum_child_torch_peak_gpu_memory_bytes": max(int(row["torch_peak_gpu_memory_bytes"]) for row in rows),
        "disk_free_after_bytes": shutil.disk_usage(run_dir).free,
    }
    limits = config["resources"]
    resource_checks = {
        "wall": resources["wall_seconds"] <= float(limits["maximum_wall_seconds"]),
        "cgroup": resources["cgroup_memory_current_bytes"] <= int(limits["maximum_cgroup_memory_bytes"]),
        "gpu": resources["maximum_child_torch_peak_gpu_memory_bytes"] <= int(limits["maximum_child_torch_peak_gpu_memory_bytes"]),
        "disk": resources["disk_free_after_bytes"] >= int(limits["minimum_disk_free_bytes_after"]),
    }
    if not all(resource_checks.values()):
        raise ProtocolError(f"G0b resource gate failed: {resource_checks}")
    _write_json(run_dir / "artifacts/resources.json", resources)
    summary = {
        "schema_version": "worldsim_v51_g0b_trace3d_determinism_summary_v1", "task_id": TASK_ID, "status": "done",
        "outcome": outcome, "conclusion": conclusion, "source_commit": identity["commit"], "source_tree": identity["tree"],
        "forensic": forensic, "resources": resources, "resource_checks": resource_checks,
        "network_access": False, "source_patch": False, "official_source_mutation": False,
        "real_checkpoint_read": False, "camera_metadata_read": False, "image_pixels_read": False, "mask_pixels_read": False,
        "quality_metrics_read": False, "training": False, "gaussian_mutation": False,
        "next_task": next_task, "next_action": next_action, "m2_status": "pending", "m3_status": "pending",
    }
    _write_json(run_dir / "summary.json", summary)
    events.append({"event": "run_completed", "at_utc": _utc_now(), "outcome": outcome})
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "manifest.json", {"task_id": TASK_ID, "status": "done", "inventory": _inventory(run_dir)})
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "done", "outcome": outcome, "conclusion": conclusion, "source_commit": identity["commit"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_g_g0b_trace3d_cross_process_determinism_v1.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.child:
        print(json.dumps(_child_probe(_validate_config(config_path)), sort_keys=True))
        return
    if args.run_dir is None:
        raise ProtocolError("--run-dir is required outside child mode")
    print(json.dumps(run(config_path, args.run_dir.resolve()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
