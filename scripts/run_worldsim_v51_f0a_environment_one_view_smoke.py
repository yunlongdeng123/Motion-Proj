#!/usr/bin/env python3
"""构建隔离 DEVA 环境并执行单视图资源与输出 schema smoke。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
from packaging.utils import canonicalize_name, parse_wheel_filename
from PIL import Image
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, sha256_file
from scripts.run_worldsim_v51_h_uplift import (
    ResourceMonitor,
    _git,
    _inventory,
    _nvidia_used_mib,
    _utc_now,
    _write_json,
    _write_jsonl,
    _write_text,
)


SCHEMAS = {
    "worldsim_v51_stage_f_f0a_environment_one_view_smoke_v1",
    "worldsim_v51_stage_f_f0a_environment_one_view_smoke_v2",
    "worldsim_v51_stage_f_f0a_environment_one_view_smoke_v3",
    "worldsim_v51_stage_f_f0a_environment_one_view_smoke_v4",
    "worldsim_v51_stage_f_f0a_environment_one_view_smoke_v5",
}
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"YAML root must be a mapping: {path}")
    return payload


def _verify(path: Path, digest: str, label: str, expected_bytes: int | None = None) -> Path:
    if not path.is_file() or sha256_file(path) != digest:
        raise ProtocolError(f"identity drift: {label}: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ProtocolError(f"byte drift: {label}: {path}")
    return path


def _matches_file(path: Path, digest: str, expected_bytes: int) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_bytes
        and sha256_file(path) == digest
    )


def _git_at(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def repository_source_identity(project: Path = PROJECT) -> dict[str, str]:
    return {
        "commit": _git(project, "rev-parse", "HEAD"),
        "tree": _git(project, "rev-parse", "HEAD^{tree}"),
    }


def _validate_config(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") not in SCHEMAS or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0a environment smoke config identity drift")
    if config.get("status") != "running" or int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("F0a environment smoke status or seed drift")
    authorization = config["authorization"]["asset_source_freeze"]
    freeze_path = _verify(
        PROJECT / authorization["path"], authorization["sha256"], "F0a asset/source freeze"
    )
    freeze = _load_yaml(freeze_path)
    if freeze.get("status") != authorization["required_status"]:
        raise ProtocolError("F0a asset/source freeze status drift")
    if freeze["canonical_run"].get("conclusion") != authorization["required_conclusion"]:
        raise ProtocolError("F0a asset/source conclusion drift")
    if freeze["governance"].get("next_phase") != authorization["required_next_phase"]:
        raise ProtocolError("F0a asset/source freeze did not unlock environment smoke")

    for name in ("gaussian_grouping", "grounded_segment_anything"):
        spec = config["sources"][name]
        root = Path(spec["path"])
        if _git_at(root, "rev-parse", "HEAD") != spec["commit"]:
            raise ProtocolError(f"source commit drift: {name}")
        if _git_at(root, "rev-parse", "HEAD^{tree}") != spec["tree"]:
            raise ProtocolError(f"source tree drift: {name}")
        if _git_at(root, "status", "--porcelain"):
            raise ProtocolError(f"source checkout not clean: {name}")
    for name, spec in config["assets"].items():
        _verify(Path(spec["path"]), spec["sha256"], name, int(spec["bytes"]))
    hidden = config.get("hidden_torchvision_assets")
    if hidden is not None:
        source_text = Path(hidden["upstream_source_file"]).read_text(encoding="utf-8")
        torch_home = Path(hidden["torch_home"])
        runtime_environment = config.get("runtime_environment", {})
        if runtime_environment.get("TORCH_HOME") != str(torch_home):
            raise ProtocolError("dedicated TORCH_HOME contract drift")
        if runtime_environment.get("PYTORCH_CUDA_ALLOC_CONF") != "max_split_size_mb:128":
            raise ProtocolError("F0a v4 CUDA allocator recovery drift")
        upstream_defaults = config["one_view"].get("upstream_defaults", {})
        if upstream_defaults != {
            "SAM_NUM_POINTS_PER_SIDE": 64,
            "SAM_NUM_POINTS_PER_BATCH": 64,
        }:
            raise ProtocolError("official SAM point grid or batch drift")
        batch_override = config["one_view"]["arguments"].get(
            "SAM_NUM_POINTS_PER_BATCH"
        )
        if config["schema_version"].endswith("_v4") and batch_override is not None:
            raise ProtocolError("F0a v4 must keep the official SAM batch default")
        if config["schema_version"].endswith("_v5") and batch_override != 32:
            raise ProtocolError("F0a v5 must be the preregistered batch-32 recovery")
        for name, spec in hidden["assets"].items():
            if spec["url"] not in source_text:
                raise ProtocolError(f"hidden torchvision URL drift: {name}")
            target = torch_home / "hub/checkpoints" / spec["filename"]
            source = Path(spec["source_cache_path"])
            digest = str(spec["sha256"])
            expected_bytes = int(spec["bytes"])
            if not (
                _matches_file(target, digest, expected_bytes)
                or _matches_file(source, digest, expected_bytes)
            ):
                raise ProtocolError(f"hidden torchvision asset unavailable: {name}")
    image = config["one_view"]["source_image"]
    _verify(Path(image["path"]), image["sha256"], "one-view image", int(image["bytes"]))
    if config["decision"].get("materialization_authorized") is not False:
        raise ProtocolError("one-view smoke must not authorize materialization")
    if config["decision"].get("identity_training_authorized") is not False:
        raise ProtocolError("one-view smoke must not authorize identity training")
    locks = config["locks"]
    if int(locks.get("image_pixels_decoded_count", -1)) != 1:
        raise ProtocolError("one-view decode denominator drift")
    if int(locks.get("output_mask_pixels_read_count", -1)) != 1:
        raise ProtocolError("one-view output denominator drift")
    for name, value in locks.items():
        if name in {"image_pixels_decoded_count", "output_mask_pixels_read_count"}:
            continue
        if name in {"m2_status", "m3_status"}:
            if value != "pending":
                raise ProtocolError(f"{name} must remain pending")
        elif value is not False:
            raise ProtocolError(f"F0a environment smoke lock drift: {name}")
    return config


def _wheel_records(wheelhouse: Path, package_specs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        (canonicalize_name(str(spec["distribution"])), str(spec["version"])): spec
        for spec in package_specs
    }
    records = []
    observed = set()
    for path in sorted(wheelhouse.glob("*.whl")):
        distribution, version, _, _ = parse_wheel_filename(path.name)
        key = (canonicalize_name(str(distribution)), str(version))
        if key not in wanted or key in observed:
            raise ProtocolError(f"unexpected or duplicate wheel: {path.name}")
        observed.add(key)
        records.append(
            {
                "distribution": wanted[key]["distribution"],
                "version": wanted[key]["version"],
                "import_name": wanted[key]["import_name"],
                "filename": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if observed != set(wanted):
        raise ProtocolError(f"wheel denominator drift: {observed} != {set(wanted)}")
    return sorted(records, key=lambda row: canonicalize_name(row["distribution"]))


def _acquire_wheels(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    wheelhouse = Path(config["wheelhouse"])
    packages = list(config["packages"])
    if not wheelhouse.exists():
        partial = Path(f"{wheelhouse}{config['partial_suffix']}")
        if partial.exists():
            raise ProtocolError(f"partial wheelhouse requires explicit recovery: {partial}")
        partial.mkdir(parents=True)
        command = [
            config["base_runtime"],
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--dest",
            str(partial),
            *[f"{row['distribution']}=={row['version']}" for row in packages],
        ]
        subprocess.run(command, check=True)
        _wheel_records(partial, packages)
        partial.replace(wheelhouse)
    return _wheel_records(wheelhouse, packages)


def _environment_import_report(runtime: Path, packages: list[Mapping[str, Any]]) -> dict[str, str]:
    pairs = {
        str(row["import_name"]): str(row["distribution"])
        for row in packages
    }
    script = (
        "import importlib,importlib.metadata,json; "
        f"pairs={json.dumps(pairs)}; "
        "[importlib.import_module(n) for n in pairs]; "
        "print(json.dumps({n: importlib.metadata.version(d) for n,d in pairs.items()},sort_keys=True))"
    )
    return json.loads(subprocess.check_output([str(runtime), "-c", script], text=True))


def _solver_smokes(runtime: Path) -> dict[str, Any]:
    gurobi_script = """
import json
import gurobipy as gp
from gurobipy import GRB
m = gp.Model('worldsim_v51_gurobi_smoke')
m.Params.OutputFlag = 0
x = m.addVar(vtype=GRB.BINARY, name='x')
m.setObjective(x, GRB.MAXIMIZE)
m.optimize()
print(json.dumps({'status': int(m.Status), 'solution': float(x.X), 'version': list(gp.gurobi.version())}, sort_keys=True))
"""
    pulp_script = """
import json
import pulp
m = pulp.LpProblem('worldsim_v51_pulp_smoke', pulp.LpMaximize)
x = pulp.LpVariable('x', cat=pulp.LpBinary)
m += x
status = m.solve(pulp.PULP_CBC_CMD(msg=0))
print(json.dumps({'status': int(status), 'status_name': pulp.LpStatus[status], 'solution': float(x.value())}, sort_keys=True))
"""
    gurobi = parse_last_json_line(
        subprocess.check_output([str(runtime), "-c", gurobi_script], text=True), "Gurobi"
    )
    pulp = parse_last_json_line(
        subprocess.check_output([str(runtime), "-c", pulp_script], text=True), "PuLP"
    )
    if gurobi["solution"] != 1.0 or gurobi["status"] != 2:
        raise ProtocolError(f"Gurobi tiny solver smoke failed: {gurobi}")
    if pulp["solution"] != 1.0 or pulp["status_name"] != "Optimal":
        raise ProtocolError(f"PuLP tiny solver smoke failed: {pulp}")
    return {"gurobi": gurobi, "pulp": pulp}


def parse_last_json_line(output: str, label: str) -> dict[str, Any]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ProtocolError(f"{label} solver emitted no stdout")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise ProtocolError(f"{label} solver terminal JSON drift: {lines[-1]!r}") from error
    if not isinstance(payload, dict):
        raise ProtocolError(f"{label} solver terminal payload must be a mapping")
    payload["stdout_prefix"] = lines[:-1]
    return payload


def _build_environment(config: Mapping[str, Any], wheels: list[Mapping[str, Any]]) -> dict[str, Any]:
    target = Path(config["target"])
    if not target.exists():
        partial = Path(f"{target}{config['partial_suffix']}")
        if partial.exists():
            raise ProtocolError(f"partial environment requires explicit recovery: {partial}")
        subprocess.run(
            [
                config["base_runtime"],
                "-m",
                "venv",
                "--system-site-packages",
                str(partial),
            ],
            check=True,
        )
        runtime = partial / "bin/python"
        subprocess.run(
            [
                str(runtime),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--find-links",
                str(config["wheelhouse"]),
                *[f"{row['distribution']}=={row['version']}" for row in config["packages"]],
            ],
            check=True,
        )
        site_packages = Path(
            subprocess.check_output(
                [str(runtime), "-c", "import site; print(site.getsitepackages()[0])"], text=True
            ).strip()
        )
        for index, source in enumerate(config["source_paths"]):
            (site_packages / f"worldsim_v51_source_{index}.pth").write_text(
                str(source) + "\n", encoding="utf-8"
            )
        partial.replace(target)
        acquisition = "created_then_atomic_publish"
    else:
        acquisition = "reused_existing_environment"
    runtime = target / "bin/python"
    if not runtime.is_file():
        raise ProtocolError("isolated environment runtime missing")
    imports = _environment_import_report(runtime, list(config["packages"]))
    for row in config["packages"]:
        observed = imports[row["import_name"]]
        if observed != row["version"]:
            raise ProtocolError(
                f"isolated import version drift: {row['import_name']}: {observed}"
            )
    source_imports = subprocess.check_output(
        [
            str(runtime),
            "-c",
            "import deva,segment_anything; print(deva.__file__); print(segment_anything.__file__)",
        ],
        text=True,
    ).splitlines()
    if not source_imports[0].startswith(config["source_paths"][0]):
        raise ProtocolError("DEVA source import drift")
    if not source_imports[1].startswith(config["source_paths"][1]):
        raise ProtocolError("Segment Anything source import drift")
    freeze = subprocess.check_output([str(runtime), "-m", "pip", "freeze", "--all"], text=True)
    return {
        "path": str(target),
        "runtime": str(runtime),
        "system_site_packages": True,
        "acquisition": acquisition,
        "wheel_records": wheels,
        "pinned_import_versions": imports,
        "source_import_paths": source_imports,
        "pip_freeze": freeze.splitlines(),
        "partial_path_absent_after_publish": not Path(f"{target}{config['partial_suffix']}").exists(),
    }


def _prepare_torch_hub_assets(config: Mapping[str, Any]) -> dict[str, Any]:
    torch_home = Path(config["torch_home"])
    checkpoint_dir = torch_home / "hub/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, spec in sorted(config["assets"].items()):
        source = Path(spec["source_cache_path"])
        target = checkpoint_dir / spec["filename"]
        digest = str(spec["sha256"])
        expected_bytes = int(spec["bytes"])
        partial = target.with_name(f"{target.name}{config['partial_suffix']}")
        if target.exists():
            _verify(target, digest, name, expected_bytes)
            acquisition = "reused_exact_canonical_asset"
        else:
            _verify(source, digest, f"{name} source cache", expected_bytes)
            if partial.exists():
                raise ProtocolError(f"partial hidden asset requires explicit recovery: {partial}")
            shutil.copyfile(source, partial)
            _verify(partial, digest, f"{name} partial", expected_bytes)
            partial.replace(target)
            acquisition = "copied_then_atomic_publish"
        records.append(
            {
                "name": name,
                "url": spec["url"],
                "filename": spec["filename"],
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "acquisition": acquisition,
                "partial_path_absent_after_publish": not partial.exists(),
            }
        )
    return {"torch_home": str(torch_home), "assets": records}


def _one_view_command(
    config: Mapping[str, Any], runtime: Path, input_dir: Path, output_dir: Path
) -> list[str]:
    one = config["one_view"]
    args = one["arguments"]
    command = [
        str(runtime),
        one["upstream_command"],
        "--model",
        config["assets"]["deva"]["path"],
        "--SAM_CHECKPOINT_PATH",
        config["assets"]["sam_vit_h"]["path"],
        "--chunk_size",
        str(args["chunk_size"]),
        "--img_path",
        str(input_dir),
        "--amp",
        "--temporal_setting",
        args["temporal_setting"],
        "--size",
        str(args["size"]),
        "--output",
        str(output_dir),
        "--use_short_id",
        "--suppress_small_objects",
        "--SAM_PRED_IOU_THRESHOLD",
        str(args["SAM_PRED_IOU_THRESHOLD"]),
    ]
    if "SAM_NUM_POINTS_PER_BATCH" in args:
        command.extend(
            [
                "--SAM_NUM_POINTS_PER_BATCH",
                str(args["SAM_NUM_POINTS_PER_BATCH"]),
            ]
        )
    return command


def _run_one_view(config: Mapping[str, Any], run_dir: Path, runtime: Path) -> dict[str, Any]:
    one = config["one_view"]
    input_dir = run_dir / "artifacts/one_view_input"
    output_dir = run_dir / "artifacts/one_view_output"
    input_dir.mkdir(parents=True)
    staged = input_dir / one["staging_filename"]
    staged.symlink_to(Path(one["source_image"]["path"]))
    deva_root = Path(config["sources"]["deva"]["path"])
    command = _one_view_command(config, runtime, input_dir, output_dir)
    stdout_path = run_dir / "artifacts/one_view_stdout.log"
    stderr_path = run_dir / "artifacts/one_view_stderr.log"
    subprocess_environment = os.environ.copy()
    subprocess_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    runtime_environment = config.get("runtime_environment", {})
    for name in ("TORCH_HOME", "PYTORCH_CUDA_ALLOC_CONF"):
        if name in runtime_environment:
            subprocess_environment[name] = str(runtime_environment[name])
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(
            command,
            cwd=deva_root,
            stdout=stdout,
            stderr=stderr,
            check=True,
            env=subprocess_environment,
        )
    contract = one["output_contract"]
    mask_path = output_dir / contract["mask"]
    metadata_path = output_dir / contract["metadata"]
    if not mask_path.is_file() or not metadata_path.is_file():
        raise ProtocolError("one-view upstream output contract missing")
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    if list(mask.shape) != [int(value) for value in contract["mask_shape_hw"]]:
        raise ProtocolError(f"one-view mask shape drift: {mask.shape}")
    if str(mask.dtype) != contract["mask_dtype"]:
        raise ProtocolError(f"one-view mask dtype drift: {mask.dtype}")
    labels, counts = np.unique(mask, return_counts=True)
    if int(labels.min()) < int(contract["label_minimum"]) or int(labels.max()) > int(
        contract["label_maximum"]
    ):
        raise ProtocolError("one-view short-ID range drift")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    annotations = metadata.get("annotations")
    if not isinstance(annotations, list) or len(annotations) != 1:
        raise ProtocolError("one-view pred.json denominator drift")
    return {
        "command": command,
        "source_image_sha256": one["source_image"]["sha256"],
        "staged_path": str(staged),
        "staged_is_symlink": staged.is_symlink(),
        "mask_path": str(mask_path),
        "mask_bytes": mask_path.stat().st_size,
        "mask_sha256": sha256_file(mask_path),
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "unique_label_histogram": {
            str(int(label)): int(count) for label, count in zip(labels, counts)
        },
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "annotation_count": len(annotations),
        "input_image_pixels_decoded_count": 1,
        "output_mask_pixels_read_count": 1,
        "association_capability_claim": False,
        "quality_claim": False,
        "runtime_environment": {
            "TORCH_HOME": subprocess_environment.get("TORCH_HOME"),
            "PYTORCH_CUDA_ALLOC_CONF": subprocess_environment.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
            "PYTHONDONTWRITEBYTECODE": subprocess_environment[
                "PYTHONDONTWRITEBYTECODE"
            ],
        },
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    config = _validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    identity = repository_source_identity()
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v51_f0a_environment_smoke_status_v1",
            "task_id": TASK_ID,
            "status": "running",
            "source_commit": identity["commit"],
        },
    )
    monitor = ResourceMonitor(float(config["resources"]["monitor_interval_seconds"]))
    nvidia_start = _nvidia_used_mib()
    if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
        raise ProtocolError(f"unexpected GPU use at start: {nvidia_start} MiB")
    started = time.perf_counter()
    monitor.start()
    try:
        environment_config = config["environment"]
        wheels = _acquire_wheels(environment_config)
        environment = _build_environment(environment_config, wheels)
        solvers = _solver_smokes(Path(environment["runtime"]))
        if "hidden_torchvision_assets" in config:
            hidden_assets = _prepare_torch_hub_assets(
                config["hidden_torchvision_assets"]
            )
            environment["hidden_torchvision_assets"] = hidden_assets
        one_view = _run_one_view(config, run_dir, Path(environment["runtime"]))
        environment["solver_smokes"] = solvers
        _write_json(run_dir / "artifacts/environment_lock.json", environment)
        _write_json(run_dir / "artifacts/one_view_report.json", one_view)
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid:
            raise ProtocolError("environment smoke resource monitor produced no valid sample")
        resources = {
            "nvidia_start_mib": nvidia_start,
            "nvidia_peak_mib": max(int(row["gpu_used_mib"]) for row in valid),
            "cgroup_memory_peak_bytes": max(
                int(row["cgroup_memory_current_bytes"]) for row in valid
            ),
            "sample_count": len(monitor.samples),
            "monitor_error_count": len(monitor.samples) - len(valid),
            "wall_seconds": time.perf_counter() - started,
            "disk_free_after_bytes": shutil.disk_usage(Path(environment_config["target"])).free,
        }
        _write_json(run_dir / "artifacts/resources.json", resources)
        ceilings = config["resources"]
        resource_checks = {
            "nvidia_peak": resources["nvidia_peak_mib"]
            <= int(ceilings["maximum_nvidia_peak_mib"]),
            "cgroup_memory_peak": resources["cgroup_memory_peak_bytes"]
            <= int(ceilings["maximum_cgroup_memory_bytes"]),
            "wall": resources["wall_seconds"] <= float(ceilings["maximum_wall_seconds"]),
            "disk_free_after": resources["disk_free_after_bytes"]
            >= int(ceilings["minimum_disk_free_bytes_after"]),
            "monitor": resources["monitor_error_count"] == 0,
        }
        if not all(resource_checks.values()):
            raise ProtocolError(f"environment smoke resource gate failed: {resource_checks}")
        conclusion = config["decision"]["expected_conclusion"]
        report = {
            "schema_version": "worldsim_v51_f0a_environment_smoke_report_v1",
            "task_id": TASK_ID,
            "status": "done",
            "conclusion": conclusion,
            "environment": environment,
            "one_view": one_view,
            "resource_checks": resource_checks,
            "materialization_authorized": False,
            "identity_training_authorized": False,
            "next_action": config["decision"]["next_action"],
            "quality_read": False,
            "parameter_search": False,
            "association_capability_claim": False,
            "m2_status": "pending",
            "m3_status": "pending",
        }
        _write_json(run_dir / "artifacts/environment_smoke_report.json", report)
        summary = {
            **report,
            "schema_version": "worldsim_v51_f0a_environment_smoke_summary_v1",
            "source_commit": identity["commit"],
            "source_tree": identity["tree"],
            "resources": resources,
            "h_quality_read": False,
            "screening_quality_read": False,
            "confirmation_quality_read": False,
            "validation_quality_read": False,
            "test_quality_read": False,
            "kitti_method_tuning": False,
            "f1_execution": False,
            "f2_execution": False,
        }
        _write_json(run_dir / "summary.json", summary)
        events.append({"event": "run_completed", "at_utc": _utc_now(), "status": "done"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "manifest.json",
            {
                "schema_version": "worldsim_v51_f0a_environment_smoke_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_f0a_environment_smoke_status_v1",
                "task_id": TASK_ID,
                "status": "done",
                "conclusion": conclusion,
                "source_commit": identity["commit"],
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
                "schema_version": "worldsim_v51_f0a_environment_smoke_status_v1",
                "task_id": TASK_ID,
                "status": "blocked",
                "error": f"{type(error).__name__}: {error}",
                "source_commit": identity["commit"],
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_f_f0a_environment_one_view_smoke_v5.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
