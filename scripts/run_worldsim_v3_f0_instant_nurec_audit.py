#!/usr/bin/env python
"""执行 F0 Instant NuRec 官方代码与本机能力的只读审计。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/f0_instant_nurec_audit_v1.yaml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"F0 protocol invalid: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": list(command),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
            "wall_seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": list(command),
            "exit_code": None,
            "stdout": error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or ""),
            "stderr": error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or ""),
            "timed_out": True,
            "wall_seconds": time.monotonic() - started,
        }
    except FileNotFoundError as error:
        return {
            "command": list(command),
            "exit_code": None,
            "stdout": "",
            "stderr": str(error),
            "timed_out": False,
            "wall_seconds": time.monotonic() - started,
        }


def validate_schema(protocol: Mapping[str, Any]) -> None:
    require(protocol["schema_version"] == 1, "schema_version")
    require(protocol["task_id"] == "WS-V3-F0-FEEDFORWARD-AUDIT-01", "task_id")
    require(protocol["profile_id"] == "F0-INSTANT-NUREC-AUDIT-v1", "profile_id")
    require(
        protocol["protocol_status"] == "frozen_before_formal_local_audit",
        "protocol_status",
    )
    require(protocol["seed"] == 0, "seed")

    authorization = protocol["authorization"]
    for key in (
        "official_source_read_only_audit_authorized",
        "local_environment_preflight_authorized",
        "official_focused_tests_authorized",
        "inference_smoke_authorized_only_if_all_prerequisites_pass",
    ):
        require(authorization[key], f"authorization {key}")
    for key in (
        "dependency_install_authorized",
        "weight_download_authorized",
        "gated_dataset_download_authorized",
        "training_authorized",
        "gpu_launch_when_any_prerequisite_fails_authorized",
        "f1_pilot_authorized",
    ):
        require(not authorization[key], f"authorization {key}")

    checkout = protocol["official_source_checkout"]
    require(len(checkout["repository_revision"]) == 40, "repository revision")
    require(len(checkout["repository_tree"]) == 40, "repository tree")
    require(checkout["clean_required"], "clean source checkout")
    require(len(checkout["files"]) == 16, "source file fingerprint count")
    require(all(len(value) == 64 for value in checkout["files"].values()), "source file SHA")

    license_contract = protocol["license_contract"]
    require(license_contract["source_code"] == "Apache-2.0", "source license")
    require(license_contract["model_weights"] == "NVIDIA Open Model License", "model license")
    require(license_contract["dataset_is_gated"], "dataset gated")
    require(license_contract["dataset_terms_acceptance_required"], "dataset terms")

    weights = protocol["checkpoint_provenance"]["supported_weights"]
    require([row["profile"] for row in weights] == ["pa-front", "pa-multiview", "pq-front"], "weight profiles")
    require(all(len(row["commit"]) == 40 for row in weights), "weight commit")
    require(all(len(row["sha256"]) == 64 for row in weights), "weight SHA")
    require(all(int(row["bytes"]) > 700_000_000 for row in weights), "weight bytes")

    research = protocol["research_model_contract"]
    require(all(research["output"].values()), "research-model outputs")
    require(research["claim_boundary"] == "paper_and_model_card_not_local_cli_evidence", "research claim boundary")

    cli = protocol["standalone_cli_contract"]
    require(cli["python"] == ">=3.11,<3.12", "Python contract")
    require(cli["input_format"] == "NCore V4 .json or .lst", "input contract")
    require(cli["camera_kernel"]["accepted_type"] == "FTheta", "camera kernel")
    require(not cli["camera_kernel"]["pinhole_or_generic_fisheye_accepted"], "camera restriction")
    require(not cli["reads"]["lidar"], "CLI LiDAR boundary")
    require(cli["exports"] == {
        "format": "PLY",
        "static_layer": True,
        "dynamic_layer": False,
        "sky_cubemap": False,
        "isp_affine": False,
        "actor_registry": False,
        "actor_trajectories": False,
        "depth_or_point_map": False,
    }, "CLI export boundary")
    require(cli["inference_requires_cuda"], "CUDA requirement")

    prerequisites = protocol["local_smoke_prerequisites"]
    require(prerequisites["all_required"], "all-prerequisites rule")
    require(prerequisites["inference_vram_minimum_mib"] == 30_720, "VRAM minimum")
    require(prerequisites["disk_free_minimum_bytes"] == 100_000_000_000, "disk minimum")

    formal = protocol["formal_audit"]
    require(len(formal["focused_tests"]) == 2, "focused test matrices")
    require(formal["inference_command_must_not_be_constructed_when_gate_fails"], "fail-closed inference")

    decision = protocol["f1_decision_contract"]
    require(decision["current_decision_if_cli_contract_matches"] == "conditional_not_unlocked", "F1 decision")
    require(decision["static_ply_alone_is_not_exact_streetgs_checkpoint"], "static PLY boundary")
    require(all(protocol["claim_boundary"].values()), "claim boundary")


def audit_source(protocol: Mapping[str, Any]) -> dict[str, Any]:
    checkout = protocol["official_source_checkout"]
    root = Path(checkout["path"])
    require(root.is_dir(), f"official checkout missing: {root}")

    head = run_command(["git", "rev-parse", "HEAD"], cwd=root)
    tree = run_command(["git", "rev-parse", "HEAD^{tree}"], cwd=root)
    status = run_command(["git", "status", "--porcelain"], cwd=root)
    require(head["exit_code"] == 0, "cannot read official checkout HEAD")
    require(tree["exit_code"] == 0, "cannot read official checkout tree")
    require(status["exit_code"] == 0, "cannot read official checkout status")

    files: dict[str, Any] = {}
    for relative, expected_sha in checkout["files"].items():
        path = root / relative
        files[relative] = {
            "exists": path.is_file(),
            "expected_sha256": expected_sha,
            "actual_sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }

    signatures: dict[str, Any] = {}
    for relative, expected_strings in protocol["static_code_signatures"].items():
        path = root / relative
        text = path.read_text(encoding="utf-8")
        signatures[relative] = {
            value: text.count(value) for value in expected_strings
        }

    python_files = sorted(root.glob("instant_nurec/**/*.py"))
    lidar_code_matches = []
    for path in python_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "lidar" in line.lower():
                lidar_code_matches.append({
                    "path": str(path.relative_to(root)),
                    "line": line_number,
                    "text": line.strip(),
                })

    exact = {
        "head": head["stdout"].strip() == checkout["repository_revision"],
        "tree": tree["stdout"].strip() == checkout["repository_tree"],
        "clean": status["stdout"] == "",
        "file_hashes": all(row["actual_sha256"] == row["expected_sha256"] for row in files.values()),
        "code_signatures": all(count > 0 for rows in signatures.values() for count in rows.values()),
        "no_lidar_reader_in_python_code": len(lidar_code_matches) == 0,
    }
    return {
        "checkout": str(root),
        "head": head["stdout"].strip(),
        "tree": tree["stdout"].strip(),
        "git_status_porcelain": status["stdout"],
        "files": files,
        "static_code_signatures": signatures,
        "lidar_code_matches": lidar_code_matches,
        "exact_checks": exact,
        "all_exact": all(exact.values()),
    }


def parse_nvidia_smi_gpu(result: Mapping[str, Any]) -> dict[str, Any]:
    if result["exit_code"] != 0 or not result["stdout"].strip():
        return {"available": False, "raw": result}
    rows = []
    for line in result["stdout"].strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append({
                "name": parts[0],
                "memory_total_mib": int(parts[1]),
                "driver_version": parts[2],
                "compute_capability": float(parts[3]),
            })
        except ValueError:
            continue
    return {"available": bool(rows), "gpus": rows, "raw": result}


def read_int_file(path: Path) -> int | str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else value


def read_system_memory_bytes() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def find_supported_weight(protocol: Mapping[str, Any], checkout: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    explicit = os.environ.get("INSTANT_NUREC_FULL_PT")
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(checkout / row["path"] for row in protocol["checkpoint_provenance"]["supported_weights"])
    observed = []
    supported = {
        (int(row["bytes"]), row["sha256"]): row["profile"]
        for row in protocol["checkpoint_provenance"]["supported_weights"]
    }
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            observed.append({"path": str(path), "exists": False})
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        observed.append({
            "path": str(path),
            "exists": True,
            "bytes": size,
            "sha256": digest,
            "matched_profile": supported.get((size, digest)),
        })
    return {
        "observed": observed,
        "exact_supported_weight_present": any(row.get("matched_profile") for row in observed),
    }


def collect_environment(protocol: Mapping[str, Any], source_audit: Mapping[str, Any]) -> dict[str, Any]:
    formal = protocol["formal_audit"]
    checkout = Path(protocol["official_source_checkout"]["path"])
    gpu_command = run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    gpu = parse_nvidia_smi_gpu(gpu_command)
    gpu_processes = run_command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    cli_help = run_command(
        [formal["current_python"], str(checkout / "run_inference.py"), "--help"],
        cwd=checkout,
        timeout=int(formal["cli_help_timeout_seconds"]),
    )
    disk = shutil.disk_usage(Path(formal["run_root"]).parent)
    weight = find_supported_weight(protocol, checkout)
    ncore_input = os.environ.get("INSTANT_NUREC_INPUT")
    ncore_path = Path(ncore_input) if ncore_input else None
    terms_value = os.environ.get("INSTANT_NUREC_DATASET_TERMS_ACCEPTED", "")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "current_python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
        },
        "python_3_11": shutil.which("python3.11"),
        "uv": shutil.which("uv"),
        "gpu": gpu,
        "gpu_compute_processes": gpu_processes,
        "system_memory_bytes": read_system_memory_bytes(),
        "cgroup": {
            "memory_max": read_int_file(Path("/sys/fs/cgroup/memory.max")),
            "memory_current": read_int_file(Path("/sys/fs/cgroup/memory.current")),
            "memory_events": Path("/sys/fs/cgroup/memory.events").read_text(encoding="utf-8") if Path("/sys/fs/cgroup/memory.events").is_file() else None,
        },
        "disk": {
            "path": str(Path(formal["run_root"]).parent),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "huggingface_token_present": any(bool(os.environ.get(name)) for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")),
        "weight": weight,
        "ncore_input": {
            "configured": bool(ncore_input),
            "path": ncore_input,
            "exists": bool(ncore_path and ncore_path.exists()),
            "suffix": ncore_path.suffix if ncore_path else None,
        },
        "dataset_terms_acceptance_recorded": terms_value.lower() in {"1", "true", "yes"},
        "cli_help": cli_help,
        "source_checkout_all_exact": bool(source_audit["all_exact"]),
    }


def evaluate_smoke_prerequisites(
    protocol: Mapping[str, Any], environment: Mapping[str, Any]
) -> dict[str, Any]:
    required = protocol["local_smoke_prerequisites"]
    gpu_rows = environment["gpu"].get("gpus", [])
    maximum_vram = max((row["memory_total_mib"] for row in gpu_rows), default=0)
    maximum_capability = max((row["compute_capability"] for row in gpu_rows), default=0.0)
    system_memory = environment["system_memory_bytes"] or 0
    checks = {
        "python_3_11_available": environment["python_3_11"] is not None,
        "uv_available": environment["uv"] is not None,
        "cuda_compute_capability_minimum": maximum_capability >= float(required["cuda_compute_capability_minimum"]),
        "inference_vram_minimum_mib": maximum_vram >= int(required["inference_vram_minimum_mib"]),
        "system_memory_minimum_bytes": system_memory >= int(required["system_memory_minimum_bytes"]),
        "disk_free_minimum_bytes": environment["disk"]["free_bytes"] >= int(required["disk_free_minimum_bytes"]),
        "exact_supported_weight_present": bool(environment["weight"]["exact_supported_weight_present"]),
        "licensed_ncore_v4_input_present": bool(
            environment["ncore_input"]["exists"]
            and environment["ncore_input"]["suffix"] in {".json", ".lst"}
        ),
        "dataset_terms_acceptance_recorded": bool(environment["dataset_terms_acceptance_recorded"]),
        "source_checkout_exact_and_clean": bool(environment["source_checkout_all_exact"]),
        "cli_help_success": environment["cli_help"]["exit_code"] == 0,
    }
    require(set(checks) == set(required) - {"all_required"}, "prerequisite key drift")
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
        "failed": failed,
        "all_passed": not failed,
        "inference_smoke_authorized": not failed,
        "inference_smoke_status": "authorized_not_yet_run" if not failed else "not_run_prerequisites_failed",
        "inference_command_constructed": False,
    }


def run_official_focused_tests(protocol: Mapping[str, Any]) -> dict[str, Any]:
    formal = protocol["formal_audit"]
    checkout = Path(protocol["official_source_checkout"]["path"])
    rows = []
    for test_paths in formal["focused_tests"]:
        command = [formal["current_python"], "-m", "pytest", "-q", *test_paths]
        rows.append(run_command(command, cwd=checkout, timeout=int(formal["test_timeout_seconds"])))
    return {
        "rows": rows,
        "all_passed": all(row["exit_code"] == 0 for row in rows),
        "failures_are_environment_preflight_not_upstream_quality_evidence": True,
    }


def build_capability_matrix(protocol: Mapping[str, Any]) -> dict[str, Any]:
    research = protocol["research_model_contract"]
    cli = protocol["standalone_cli_contract"]
    return {
        "research_model": {
            "evidence": "official paper_and_model_card",
            "input": research["input"],
            "output": research["output"],
        },
        "standalone_cli": {
            "evidence": "exact_official_repository_revision_and_code",
            "input_format": cli["input_format"],
            "profiles": cli["profiles"],
            "camera_kernel": cli["camera_kernel"],
            "reads": cli["reads"],
            "exports": cli["exports"],
            "ply_fields": cli["ply_fields"],
        },
        "material_differences": [
            "standalone_cli_exports_static_layer_ply_only",
            "standalone_cli_does_not_read_lidar",
            "standalone_cli_does_not_export_dynamic_layer_sky_or_isp",
            "standalone_cli_does_not_preserve_actor_registry_or_trajectories",
            "standalone_cli_camera_converter_accepts_ftheta_only",
            "web_demo_capability_is_not_local_cli_capability",
        ],
    }


def build_f1_decision(protocol: Mapping[str, Any], prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": "conditional_not_unlocked",
        "f1_launched": False,
        "f1_authorized": False,
        "reasons": [
            "formal_local_inference_smoke_not_completed",
            "standalone_cli_output_is_static_ply_only",
            "no_actor_registry_or_dynamic_asset_export",
            "no_exact_nuscenes_scene0230_to_ncore_v4_converter",
            "static_ply_is_not_an_exact_streetgs_checkpoint",
            *[f"local_prerequisite_failed:{name}" for name in prerequisites["failed"]],
        ],
        "future_narrow_pilot_boundary": "compatible_hardware_plus_licensed_ncore_input_plus_exact_converter_may_test_static_background_initialization_only",
        "dggt_fallback": "optional_not_required_for_r0",
        "does_not_claim_permanent_upstream_impossibility": True,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def artifact_fingerprint(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(run_dir)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def current_commit() -> str:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=PROJECT)
    require(result["exit_code"] == 0, "cannot resolve Motion-Proj source commit")
    return result["stdout"].strip()


def make_run_dir(protocol: Mapping[str, Any], explicit: str | None) -> Path:
    if explicit:
        run_dir = Path(explicit)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(protocol["formal_audit"]["run_root"]) / f"{timestamp}__{protocol['formal_audit']['run_slug']}"
    require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def execute(protocol_path: Path, run_dir_override: str | None = None) -> Path:
    started_wall = time.monotonic()
    protocol = OmegaConf.to_container(OmegaConf.load(protocol_path), resolve=True)
    validate_schema(protocol)
    run_dir = make_run_dir(protocol, run_dir_override)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    shutil.copyfile(protocol_path, run_dir / "protocol.yaml")

    source_audit = audit_source(protocol)
    require(source_audit["all_exact"], "official source checkout is not exact and clean")
    environment = collect_environment(protocol, source_audit)
    prerequisites = evaluate_smoke_prerequisites(protocol, environment)
    focused_tests = run_official_focused_tests(protocol)
    capabilities = build_capability_matrix(protocol)
    f1_decision = build_f1_decision(protocol, prerequisites)

    require(not prerequisites["inference_command_constructed"], "inference command was constructed")
    if not prerequisites["all_passed"]:
        require(
            prerequisites["inference_smoke_status"] == "not_run_prerequisites_failed",
            "inference smoke must fail closed",
        )

    values = {
        "official_source_audit.json": source_audit,
        "capability_matrix.json": capabilities,
        "environment_preflight.json": environment,
        "smoke_prerequisites.json": prerequisites,
        "official_test_preflight.json": focused_tests,
        "f1_decision.json": f1_decision,
    }
    for name, value in values.items():
        atomic_json(artifacts / name, value)

    resource = {
        "wall_seconds": time.monotonic() - started_wall,
        "torch_imported": "torch" in sys.modules,
        "gpu_inference_launched": False,
        "training_launched": False,
        "dependency_install_launched": False,
        "weight_or_dataset_download_launched": False,
        "disk_free_bytes_at_preflight": environment["disk"]["free_bytes"],
        "cgroup_memory_current_at_preflight": environment["cgroup"]["memory_current"],
        "gpu_compute_processes_at_preflight": environment["gpu_compute_processes"],
    }
    atomic_json(artifacts / "resource_audit.json", resource)

    summary = {
        "schema_version": 1,
        "task_id": protocol["task_id"],
        "profile_id": protocol["profile_id"],
        "seed": protocol["seed"],
        "source_commit": current_commit(),
        "protocol_sha256": sha256_file(protocol_path),
        "official_repository_revision": source_audit["head"],
        "official_repository_tree": source_audit["tree"],
        "official_source_exact_and_clean": source_audit["all_exact"],
        "standalone_cli_export": "static_layer_ply_only",
        "standalone_cli_dynamic_sky_isp_actor_registry_export": False,
        "standalone_cli_reads_lidar": False,
        "smoke_prerequisites": prerequisites,
        "inference_smoke": "not_run_prerequisites_failed" if not prerequisites["all_passed"] else "authorized_not_run_by_audit_runner",
        "official_focused_tests_all_passed": focused_tests["all_passed"],
        "official_test_failures_are_report_only": True,
        "f1_decision": f1_decision["decision"],
        "f1_launched": False,
        "audit_outcome": "done_local_inference_not_executable_on_current_host" if not prerequisites["all_passed"] else "done_prerequisites_passed_no_inference_launched",
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_json(run_dir / "summary.json", summary)

    fingerprint_paths = [run_dir / "protocol.yaml", *sorted(artifacts.glob("*.json")), run_dir / "summary.json"]
    manifest = {
        "schema_version": 1,
        "task_id": protocol["task_id"],
        "source_commit": summary["source_commit"],
        "protocol_sha256": summary["protocol_sha256"],
        "artifacts": [artifact_fingerprint(path, run_dir) for path in fingerprint_paths],
        "mutations": {
            "official_checkout": "none_read_only",
            "dependencies": "none",
            "weights": "none",
            "datasets": "none",
            "gpu": "none",
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(run_dir / "terminal.json", {"status": "done", "exit_code": 0})
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_dir = execute(args.protocol, args.run_dir)
    except Exception as error:
        print(f"F0 audit failed: {error}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
