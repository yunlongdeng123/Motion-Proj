#!/usr/bin/env python3
"""Build the exact Trace3D ID rasterizer and run a synthetic capability probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.run_worldsim_v51_f0b_three_view_association_parity import _load_yaml, _verify, repository_source_identity
from scripts.run_worldsim_v51_h_uplift import _inventory, _utc_now, _write_json, _write_jsonl, _write_text


SCHEMA = "worldsim_v51_stage_g_g0a_trace3d_reverse_tracing_capability_v1"
TASK_ID = "WS-V51-M1-G-AMBIGUITY-01"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _command_output(command: list[str], env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(command, text=True, env=env).strip()


def _validate_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("G0a Trace3D capability config drift")
    auth = config["authorization"]["source_preflight_freeze"]
    freeze = _load_yaml(_verify(PROJECT / auth["path"], auth["sha256"], "Trace3D source freeze", int(auth["bytes"])))
    if (
        freeze.get("status") != auth["required_status"]
        or freeze["governance"].get("next_task") != auth["required_next_task"]
        or freeze["governance"].get("next_phase") != auth["required_next_phase"]
    ):
        raise ProtocolError("G0a Trace3D capability authorization drift")
    source = config["official_source"]
    if source.get("source_patch_allowed") is not False or source.get("source_execution_allowed") is not True:
        raise ProtocolError("G0a official-source execution boundary drift")
    false_locks = (
        "network_access", "official_source_mutation", "submodule_initialization", "model_download", "real_checkpoint_read",
        "camera_metadata_read", "image_pixels_read", "mask_pixels_read", "quality_metrics_read", "training", "gaussian_mutation",
        "h_quality_read", "screening_quality_read", "confirmation_quality_read", "validation_quality_read", "test_quality_read",
        "kitti_method_tuning",
    )
    if any(config["locks"].get(name) is not False for name in false_locks):
        raise ProtocolError("G0a no-data/no-quality lock drift")
    if config["synthetic_probe"].get("forbidden_claim") != "real_worldsim_adapter_or_quality_supported":
        raise ProtocolError("G0a synthetic interpretation drift")
    return config


def _tensor_identity(tensor: Any) -> dict[str, Any]:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
    }


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def _run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, text=True, env=env)


def _projection_matrix(torch: Any, tan_fovx: float, tan_fovy: float, znear: float, zfar: float, device: str) -> Any:
    matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)
    matrix[0, 0] = 1.0 / tan_fovx
    matrix[1, 1] = 1.0 / tan_fovy
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix.transpose(0, 1).contiguous()


def _synthetic_probe(config: dict[str, Any], package_root: Path) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(package_root))
    torch = importlib.import_module("torch")
    raster = importlib.import_module("diff_id_rasterization")
    module_path = Path(raster.__file__).resolve()
    try:
        module_path.relative_to(package_root.resolve())
    except ValueError as exc:
        raise ProtocolError(f"G0a imported package outside frozen target: {module_path}") from exc

    spec = config["synthetic_probe"]
    image = spec["image"]
    device = spec["device"]
    if torch.cuda.get_device_name(0) != spec["gpu_name"] or list(torch.cuda.get_device_capability(0)) != spec["compute_capability"]:
        raise ProtocolError("G0a synthetic GPU identity drift")
    torch.cuda.reset_peak_memory_stats(0)
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))

    height, width = int(image["height"]), int(image["width"])
    view = torch.eye(4, dtype=torch.float32, device=device)
    projection = _projection_matrix(torch, float(image["tan_fovx"]), float(image["tan_fovy"]), float(image["znear"]), float(image["zfar"]), device)
    settings = raster.GaussianRasterizationSettings(
        image_height=height, image_width=width, tanfovx=float(image["tan_fovx"]), tanfovy=float(image["tan_fovy"]),
        bg=torch.zeros(3, dtype=torch.float32, device=device), scale_modifier=1.0,
        viewmatrix=view, projmatrix=projection, sh_degree=0, campos=torch.zeros(3, dtype=torch.float32, device=device),
        prefiltered=False, debug=False, include_feature=False,
    )
    operator = raster.GaussianRasterizer(raster_settings=settings)
    means3d = torch.tensor(spec["gaussian"]["means3d"], dtype=torch.float32, device=device)
    means2d = torch.zeros_like(means3d)
    opacity = torch.tensor(spec["gaussian"]["opacity"], dtype=torch.float32, device=device)
    scales = torch.tensor(spec["gaussian"]["scales_2d"], dtype=torch.float32, device=device)
    rotations = torch.tensor(spec["gaussian"]["rotations_wxyz"], dtype=torch.float32, device=device)
    background_mask = torch.zeros((height, width), dtype=torch.int32, device=device)
    foreground_mask = torch.ones((height, width), dtype=torch.int32, device=device)
    inputs = {
        "means3d": means3d, "means2d": means2d, "opacity": opacity, "scales": scales, "rotations": rotations,
        "view": view, "projection": projection, "background_mask": background_mask, "foreground_mask": foreground_mask,
    }
    before = {name: _tensor_identity(tensor) for name, tensor in inputs.items()}

    def traced(mask: Any, alpha_weighted: bool) -> Any:
        weights = torch.zeros((1, 2), dtype=torch.float32, device=device)
        operator.trace(
            means3D=means3d, means2D=means2d, opacities=opacity, weights=weights, id_masks=mask,
            num_class=1, alpha_w=alpha_weighted, scales=scales, rotations=rotations,
        )
        torch.cuda.synchronize(0)
        return weights.detach().clone()

    background = traced(background_mask, False)
    foreground = traced(foreground_mask, False)
    foreground_repeat = traced(foreground_mask, False)
    foreground_alpha = traced(foreground_mask, True)
    after = {name: _tensor_identity(tensor) for name, tensor in inputs.items()}
    outputs = {
        "background_hard": background.cpu().tolist(), "foreground_hard": foreground.cpu().tolist(),
        "foreground_hard_repeat": foreground_repeat.cpu().tolist(), "foreground_alpha": foreground_alpha.cpu().tolist(),
    }
    checks = {
        "background_case_class0_positive": bool(background[0, 0].item() > 0),
        "background_case_class1_zero": bool(background[0, 1].item() == 0),
        "foreground_case_class0_zero": bool(foreground[0, 0].item() == 0),
        "foreground_case_class1_positive": bool(foreground[0, 1].item() > 0),
        "hard_count_repeat_bitwise_equal": bool(torch.equal(foreground, foreground_repeat)),
        "alpha_weighted_class1_finite_positive_and_not_above_hard_count": bool(torch.isfinite(foreground_alpha).all().item() and 0 < foreground_alpha[0, 1].item() <= foreground[0, 1].item()),
        "input_tensors_bitwise_immutable": before == after,
    }
    if not all(checks.values()):
        raise ProtocolError(f"G0a synthetic reverse-tracing gate failed: {checks}; outputs={outputs}")
    extension_paths = sorted(package_root.rglob("_C*.so"))
    if len(extension_paths) != 1:
        raise ProtocolError(f"G0a compiled extension denominator drift: {extension_paths}")
    return {
        "schema_version": "worldsim_v51_g0a_trace3d_synthetic_probe_v1", "status": "pass",
        "package_module": str(module_path),
        "extension": {"path": extension_paths[0].relative_to(package_root).as_posix(), "bytes": extension_paths[0].stat().st_size, "sha256": sha256_file(extension_paths[0])},
        "inputs_before": before, "inputs_after": after, "outputs": outputs, "checks": checks,
        "torch_peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "real_checkpoint_read": False, "camera_metadata_read": False, "image_pixels_read": False,
        "mask_pixels_read": False, "quality_metrics_read": False, "training": False, "gaussian_mutation": False,
        "interpretation": spec["purpose"], "forbidden_claim": spec["forbidden_claim"],
    }


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
    source = config["official_source"]
    build = config["build"]
    repo = Path(source["repository"])
    publish_target = Path(build["publish_target"])
    partial_target = publish_target.with_name(publish_target.name + build["atomic_partial_suffix"])
    if publish_target.exists() or partial_target.exists():
        raise ProtocolError(f"G0a refusing existing publish target/partial: {publish_target}")
    if _git(repo, "rev-parse", "HEAD") != source["commit"] or _git(repo, "rev-parse", "HEAD^{tree}") != source["tree"] or _git(repo, "status", "--porcelain"):
        raise ProtocolError("G0a official repository identity drift")
    ls_tree = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", source["commit"], source["subdirectory"]])
    if len(ls_tree.splitlines()) != int(source["tracked_entry_count"]) or hashlib.sha256(ls_tree).hexdigest() != source["git_ls_tree_sha256"]:
        raise ProtocolError("G0a official source subtree drift")

    python = build["python"]
    environment = {
        "python_version": _command_output([python, "-c", "import platform; print(platform.python_version())"]),
        "torch_version": _command_output([python, "-c", "import torch; print(torch.__version__)"]),
        "torch_cuda": _command_output([python, "-c", "import torch; print(torch.version.cuda)"]),
        "gpu_name": _command_output([python, "-c", "import torch; print(torch.cuda.get_device_name(0))"]),
        "compute_capability": json.loads(_command_output([python, "-c", "import json,torch; print(json.dumps(list(torch.cuda.get_device_capability(0))))"])),
        "nvcc_version": _command_output([str(Path(build["cuda_home"]) / "bin/nvcc"), "--version"]),
        "gcc_version": _command_output(["gcc", "-dumpfullversion"]),
        "ninja_version": _command_output([str(Path(python).parent / "ninja"), "--version"]),
    }
    if (
        not environment["python_version"].startswith(build["expected_python"] + ".")
        or environment["torch_version"] != build["expected_torch"]
        or environment["torch_cuda"] != build["expected_torch_cuda"]
        or f"release {build['expected_nvcc_release']}" not in environment["nvcc_version"]
        or environment["gcc_version"] != build["expected_gcc"]
        or not environment["ninja_version"].startswith(build["expected_ninja_prefix"])
        or environment["gpu_name"] != config["synthetic_probe"]["gpu_name"]
        or environment["compute_capability"] != config["synthetic_probe"]["compute_capability"]
    ):
        raise ProtocolError(f"G0a build environment drift: {environment}")
    _write_json(run_dir / "artifacts/environment.json", environment)
    _write_json(run_dir / "artifacts/source_identity.json", {
        "repository": str(repo), "commit": source["commit"], "tree": source["tree"], "subdirectory": source["subdirectory"],
        "tracked_entry_count": len(ls_tree.splitlines()), "git_ls_tree_sha256": hashlib.sha256(ls_tree).hexdigest(),
        "source_patch_applied": False, "source_repository_status_porcelain": "",
    })

    work = run_dir / "work"
    archive_path = work / "official_diff_id_source.tar"
    extracted = work / "source"
    wheel_dir = run_dir / "artifacts/wheels"
    work.mkdir()
    extracted.mkdir()
    wheel_dir.mkdir(parents=True)
    subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "--output", str(archive_path), source["commit"], source["subdirectory"]], check=True)
    with tarfile.open(archive_path, "r") as archive:
        archive.extractall(extracted)
    source_root = extracted / source["subdirectory"]
    build_log = run_dir / "artifacts/build.log"
    child_env = os.environ.copy()
    child_env.update({
        "CUDA_HOME": build["cuda_home"], "TORCH_CUDA_ARCH_LIST": build["torch_cuda_arch_list"],
        "MAX_JOBS": str(build["maximum_jobs"]), "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": str(Path(build["cuda_home"]) / "bin") + os.pathsep + str(Path(python).parent) + os.pathsep + child_env.get("PATH", ""),
    })
    wheel_command = [python, "-m", "pip", "wheel", *build["pip_flags"], "--wheel-dir", str(wheel_dir), str(source_root)]
    _run_logged(wheel_command, build_log, child_env)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ProtocolError(f"G0a wheel denominator drift: {wheels}")
    install_command = [python, "-m", "pip", "install", *build["install_flags"], "--target", str(partial_target), str(wheels[0])]
    _run_logged(install_command, build_log, child_env)
    os.replace(partial_target, publish_target)
    shutil.rmtree(work)
    build_report = {
        "schema_version": "worldsim_v51_g0a_trace3d_build_v1", "status": "pass",
        "wheel": {"path": wheels[0].relative_to(run_dir).as_posix(), "bytes": wheels[0].stat().st_size, "sha256": sha256_file(wheels[0])},
        "build_log": {"path": build_log.relative_to(run_dir).as_posix(), "bytes": build_log.stat().st_size, "sha256": sha256_file(build_log)},
        "publish_target": str(publish_target), "publish_inventory": _tree_inventory(publish_target),
        "source_patch_applied": False, "network_access": False, "submodules_initialized": False,
    }
    _write_json(run_dir / "artifacts/build_report.json", build_report)
    probe = _synthetic_probe(config, publish_target)
    _write_json(run_dir / "artifacts/probe.json", probe)
    if _git(repo, "rev-parse", "HEAD") != source["commit"] or _git(repo, "status", "--porcelain"):
        raise ProtocolError("G0a official repository mutated during build/probe")

    resources = {
        "wall_seconds": time.perf_counter() - started,
        "cgroup_memory_current_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text().strip()),
        "torch_peak_gpu_memory_bytes": int(probe["torch_peak_gpu_memory_bytes"]),
        "disk_free_after_bytes": shutil.disk_usage(run_dir).free,
    }
    limits = config["resources"]
    checks = {
        "wall": resources["wall_seconds"] <= float(limits["maximum_wall_seconds"]),
        "cgroup": resources["cgroup_memory_current_bytes"] <= int(limits["maximum_cgroup_memory_bytes"]),
        "gpu": resources["torch_peak_gpu_memory_bytes"] <= int(limits["maximum_torch_peak_gpu_memory_bytes"]),
        "disk": resources["disk_free_after_bytes"] >= int(limits["minimum_disk_free_bytes_after"]),
    }
    if not all(checks.values()):
        raise ProtocolError(f"G0a resource gate failed: {checks}")
    _write_json(run_dir / "artifacts/resources.json", resources)
    summary = {
        "schema_version": "worldsim_v51_g0a_trace3d_capability_summary_v1", "task_id": TASK_ID, "status": "done",
        "conclusion": config["decision"]["pass_conclusion"], "source_commit": identity["commit"], "source_tree": identity["tree"],
        "official_source": {"commit": source["commit"], "tree": source["tree"], "source_patch_applied": False},
        "environment": environment, "build": build_report, "probe": probe, "resources": resources, "resource_checks": checks,
        "network_access": False, "official_source_mutation": False, "submodules_initialized": False, "model_download": False,
        "real_checkpoint_read": False, "camera_metadata_read": False, "image_pixels_read": False, "mask_pixels_read": False,
        "quality_metrics_read": False, "training": False, "gaussian_mutation": False,
        "real_worldsim_adapter_supported": False, "quality_supported": False,
        "next_action": config["decision"]["next_action"], "m2_status": "pending", "m3_status": "pending",
    }
    _write_json(run_dir / "summary.json", summary)
    events.append({"event": "run_completed", "at_utc": _utc_now()})
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "manifest.json", {"task_id": TASK_ID, "status": "done", "inventory": _inventory(run_dir)})
    _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "done", "conclusion": summary["conclusion"], "source_commit": identity["commit"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_g_g0a_trace3d_reverse_tracing_capability_v1.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    try:
        result = run(args.config.resolve(), run_dir)
    except Exception as exc:
        if run_dir.is_dir():
            events_path = run_dir / "events.jsonl"
            events = []
            if events_path.is_file():
                events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
            events.append({"event": "run_blocked", "at_utc": _utc_now(), "error_type": type(exc).__name__, "error": str(exc)})
            _write_jsonl(events_path, events)
            _write_json(run_dir / "status.json", {"task_id": TASK_ID, "status": "blocked", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
