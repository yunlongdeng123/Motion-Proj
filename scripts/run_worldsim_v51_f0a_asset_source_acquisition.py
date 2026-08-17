#!/usr/bin/env python3
"""获取并冻结 F0a 所需的官方源、权重与 train-only 输入身份。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.feature_sidecar import record_chain_sha256
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


SCHEMA = "worldsim_v51_stage_f_f0a_asset_source_acquisition_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
CONCLUSION = "f0a_assets_and_sources_frozen_environment_setup_required"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"YAML root must be a mapping: {path}")
    return payload


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ProtocolError(f"identity drift: {label}: {path}")
    return path


def _git_at(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def repository_source_identity(project: Path = PROJECT) -> dict[str, str]:
    return {
        "commit": _git(project, "rev-parse", "HEAD"),
        "tree": _git(project, "rev-parse", "HEAD^{tree}"),
    }


def select_train_only_records(
    records: Sequence[Mapping[str, Any]],
    scenes: Sequence[str],
    frames: Sequence[int],
    cameras: Sequence[int],
    role: str,
) -> list[dict[str, Any]]:
    wanted = {
        (scene, int(frame), int(camera))
        for scene in scenes
        for frame in frames
        for camera in cameras
    }
    selected_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in records:
        key = (str(raw["scene"]), int(raw["frame"]), int(raw["camera"]))
        if str(raw["role"]) != role or key not in wanted:
            continue
        if key in selected_by_key:
            raise ProtocolError(f"duplicate train-only image record: {key}")
        selected_by_key[key] = dict(raw)
    if set(selected_by_key) != wanted:
        missing = sorted(wanted - set(selected_by_key))
        raise ProtocolError(f"train-only image denominator drift: {missing[:8]}")
    return [
        selected_by_key[(scene, int(frame), int(camera))]
        for scene in scenes
        for frame in frames
        for camera in cameras
    ]


def _validate_config(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("F0a asset/source config identity drift")
    if config.get("status") != "running" or int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("F0a asset/source status or seed drift")

    authorization = config["authorization"]["f0_preflight_freeze"]
    freeze_path = _verify(
        PROJECT / authorization["path"], authorization["sha256"], "F0 preflight freeze"
    )
    freeze = _load_yaml(freeze_path)
    if freeze.get("status") != authorization["required_status"]:
        raise ProtocolError("F0 preflight terminal drift")
    if freeze["canonical_run"].get("conclusion") != authorization["required_conclusion"]:
        raise ProtocolError("F0 preflight conclusion drift")
    if freeze["governance"].get("next_phase") != authorization["required_next_phase"]:
        raise ProtocolError("F0 preflight did not unlock F0a preregistration")

    source = config["official_source"]
    grouping = source["gaussian_grouping"]
    grouping_root = Path(grouping["path"])
    if _git_at(grouping_root, "rev-parse", "HEAD") != grouping["commit"]:
        raise ProtocolError("Gaussian Grouping commit drift")
    if _git_at(grouping_root, "rev-parse", "HEAD^{tree}") != grouping["tree"]:
        raise ProtocolError("Gaussian Grouping tree drift")
    if _git_at(grouping_root, "status", "--porcelain"):
        raise ProtocolError("Gaussian Grouping repository is not clean")
    _verify(grouping_root / "docs/install.md", grouping["install_doc_sha256"], "install doc")

    deva = source["vendored_deva"]
    deva_root = Path(deva["path"])
    _verify(deva_root / "LICENSE.md", deva["license_sha256"], "DEVA license")
    for relative, digest in deva["files"].items():
        _verify(deva_root / relative, digest, f"DEVA/{relative}")
    download_script = (deva_root / "scripts/download_models.sh").read_text(encoding="utf-8")
    for asset in config["assets"]["records"]:
        if asset["url"] not in download_script:
            raise ProtocolError(f"asset URL not declared by upstream: {asset['name']}")
        if int(asset["expected_bytes"]) <= 0 or asset.get("expected_sha256") is not None:
            raise ProtocolError(f"asset first-acquisition contract drift: {asset['name']}")

    input_config = config["train_only_input"]
    _verify(
        PROJECT / input_config["input_freeze"]["path"],
        input_config["input_freeze"]["sha256"],
        "Stage B input freeze",
    )
    manifest_path = _verify(
        Path(input_config["image_manifest"]["path"]),
        input_config["image_manifest"]["sha256"],
        "image manifest",
    )
    manifest = _read_json(manifest_path)
    if manifest.get("record_chain_sha256") != input_config["image_manifest"][
        "record_chain_sha256"
    ]:
        raise ProtocolError("image manifest record-chain drift")
    if record_chain_sha256(manifest["records"]) != manifest["record_chain_sha256"]:
        raise ProtocolError("image manifest record-chain replay drift")

    if config["decision"].get("materialization_authorized") is not False:
        raise ProtocolError("asset acquisition must not authorize materialization")
    if config["decision"].get("identity_training_authorized") is not False:
        raise ProtocolError("asset acquisition must not authorize identity training")
    for name, value in config["locks"].items():
        if name in {"m2_status", "m3_status"}:
            if value != "pending":
                raise ProtocolError(f"{name} must remain pending")
        elif value is not False:
            raise ProtocolError(f"F0a lock drift: {name}")
    return config


def _clone_exact_source(spec: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(spec["target"])
    if not target.exists():
        partial = Path(f"{target}.partial")
        if partial.exists():
            raise ProtocolError(f"partial source checkout requires explicit recovery: {partial}")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                f"--filter={spec['clone_filter']}",
                "--no-checkout",
                spec["url"],
                str(partial),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(partial), "checkout", "--detach", spec["commit"]],
            check=True,
        )
        partial.replace(target)
        acquisition = "cloned"
    else:
        acquisition = "reused_existing_exact"
    commit = _git_at(target, "rev-parse", "HEAD")
    tree = _git_at(target, "rev-parse", "HEAD^{tree}")
    if commit != spec["commit"] or _git_at(target, "status", "--porcelain"):
        raise ProtocolError("Grounded Segment Anything checkout identity drift")
    subtree = target / spec["required_subtree"]
    if not subtree.is_dir():
        raise ProtocolError("Grounded Segment Anything required subtree missing")
    license_candidates = [target / "LICENSE", target / "LICENSE.md"]
    licenses = [path for path in license_candidates if path.is_file()]
    if len(licenses) != 1:
        raise ProtocolError("Grounded Segment Anything license file ambiguity")
    license_text = licenses[0].read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ProtocolError("Grounded Segment Anything license declaration drift")
    return {
        "url": spec["url"],
        "path": str(target),
        "commit": commit,
        "tree": tree,
        "clean": True,
        "required_subtree": spec["required_subtree"],
        "license": spec["expected_license"],
        "license_path": str(licenses[0]),
        "license_sha256": sha256_file(licenses[0]),
        "acquisition": acquisition,
    }


def _download_asset(
    asset_dir: Path,
    partial_suffix: str,
    curl_arguments: Sequence[str],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    target = asset_dir / spec["filename"]
    expected_bytes = int(spec["expected_bytes"])
    asset_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != expected_bytes:
            raise ProtocolError(f"existing asset byte drift: {target}")
        acquisition = "reused_existing_exact_bytes"
    else:
        partial = Path(f"{target}{partial_suffix}")
        if partial.exists() and partial.stat().st_size > expected_bytes:
            raise ProtocolError(f"oversized partial asset: {partial}")
        subprocess.run(
            [
                "curl",
                *[str(value) for value in curl_arguments],
                "--output",
                str(partial),
                spec["url"],
            ],
            check=True,
        )
        if not partial.is_file() or partial.stat().st_size != expected_bytes:
            observed = partial.stat().st_size if partial.exists() else -1
            raise ProtocolError(
                f"downloaded asset byte drift: {spec['name']}: {observed} != {expected_bytes}"
            )
        partial.replace(target)
        acquisition = "downloaded_then_atomic_publish"
    return {
        "name": spec["name"],
        "url": spec["url"],
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "acquisition": acquisition,
        "partial_path_absent_after_publish": not Path(f"{target}{partial_suffix}").exists(),
    }


def _environment_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    present = {
        name: importlib.util.find_spec(name) is not None
        for name in config["modules_expected_present"]
    }
    absent = {
        name: importlib.util.find_spec(name) is None
        for name in config["modules_expected_absent_before_isolated_setup"]
    }
    if not all(present.values()) or not all(absent.values()):
        raise ProtocolError(
            f"environment precondition drift: present={present}, absent={absent}"
        )
    return {
        "runtime": config["runtime"],
        "modules_expected_present": present,
        "modules_expected_absent": absent,
        "environment_mutated": False,
        "next_environment_policy": config["next_environment_policy"],
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
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
            "schema_version": "worldsim_v51_f0a_asset_source_status_v1",
            "task_id": TASK_ID,
            "status": "running",
            "source_commit": identity["commit"],
        },
    )
    monitor = ResourceMonitor(float(config["resources"]["monitor_interval_seconds"]))
    started = time.perf_counter()
    nvidia_start = _nvidia_used_mib()
    if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
        raise ProtocolError(f"unexpected GPU use at start: {nvidia_start} MiB")
    free_before = shutil.disk_usage(Path(config["assets"]["directory"]).parent).free
    if free_before < int(config["resources"]["minimum_free_bytes_before"]):
        raise ProtocolError(f"insufficient disk before asset acquisition: {free_before}")
    monitor.start()
    try:
        source_report = _clone_exact_source(
            config["official_source"]["grounded_segment_anything_fork"]
        )
        asset_config = config["assets"]
        assets = [
            _download_asset(
                Path(asset_config["directory"]),
                asset_config["atomic_partial_suffix"],
                asset_config["curl_arguments"],
                spec,
            )
            for spec in asset_config["records"]
        ]

        input_config = config["train_only_input"]
        source_manifest = _read_json(Path(input_config["image_manifest"]["path"]))
        selected = select_train_only_records(
            source_manifest["records"],
            input_config["scenes"],
            input_config["frames"],
            input_config["cameras"],
            input_config["role"],
        )
        if len(selected) != int(input_config["expected_view_count"]):
            raise ProtocolError("selected train-only image count drift")
        expected_size = [int(value) for value in input_config["expected_image_size_wh"]]
        for row in selected:
            path = Path(row["path"])
            if [int(row["width"]), int(row["height"])] != expected_size:
                raise ProtocolError(f"train-only image dimension drift: {path}")
            if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                raise ProtocolError(f"train-only image identity drift: {path}")
        selected_manifest = {
            "schema_version": "worldsim_v51_f0a_train_only_image_manifest_v1",
            "task_id": TASK_ID,
            "order": input_config["order"],
            "record_count": len(selected),
            "total_bytes": sum(int(row["bytes"]) for row in selected),
            "record_chain_sha256": record_chain_sha256(selected),
            "records": selected,
            "image_pixels_decoded": False,
        }
        _write_json(run_dir / "artifacts/train_only_image_manifest.json", selected_manifest)
        environment = _environment_probe(config["environment_probe"])
        report = {
            "schema_version": "worldsim_v51_f0a_asset_source_report_v1",
            "task_id": TASK_ID,
            "status": "done",
            "conclusion": config["decision"]["expected_conclusion"],
            "grounded_segment_anything": source_report,
            "assets": assets,
            "train_only_input": {
                "record_count": selected_manifest["record_count"],
                "total_bytes": selected_manifest["total_bytes"],
                "record_chain_sha256": selected_manifest["record_chain_sha256"],
                "scene_count": len(input_config["scenes"]),
                "views_per_scene": len(input_config["frames"]) * len(input_config["cameras"]),
                "image_pixels_decoded": False,
                "stage_images": False,
            },
            "environment": environment,
            "asset_ready": True,
            "source_ready": True,
            "materialization_authorized": False,
            "identity_training_authorized": False,
            "next_action": config["decision"]["next_action"],
            "quality_read": False,
            "sam_execution": False,
            "deva_execution": False,
            "m2_status": "pending",
            "m3_status": "pending",
        }
        _write_json(run_dir / "artifacts/asset_source_report.json", report)
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid:
            raise ProtocolError("F0a resource monitor produced no valid sample")
        resources = {
            "nvidia_start_mib": nvidia_start,
            "nvidia_peak_mib": max(int(row["gpu_used_mib"]) for row in valid),
            "cgroup_memory_peak_bytes": max(
                int(row["cgroup_memory_current_bytes"]) for row in valid
            ),
            "sample_count": len(monitor.samples),
            "monitor_error_count": len(monitor.samples) - len(valid),
            "wall_seconds": time.perf_counter() - started,
            "disk_free_before_bytes": free_before,
            "disk_free_after_bytes": shutil.disk_usage(Path(asset_config["directory"])).free,
        }
        _write_json(run_dir / "artifacts/resources.json", resources)
        ceilings = config["resources"]
        resource_checks = {
            "nvidia_peak": resources["nvidia_peak_mib"]
            <= int(ceilings["maximum_nvidia_peak_mib"]),
            "cgroup_memory_peak": resources["cgroup_memory_peak_bytes"]
            <= int(ceilings["maximum_cgroup_memory_bytes"]),
            "wall": resources["wall_seconds"] <= float(ceilings["maximum_wall_seconds"]),
            "monitor": resources["monitor_error_count"] == 0,
        }
        if not all(resource_checks.values()):
            raise ProtocolError(f"F0a resource gate failed: {resource_checks}")
        summary = {
            **report,
            "schema_version": "worldsim_v51_f0a_asset_source_summary_v1",
            "source_commit": identity["commit"],
            "source_tree": identity["tree"],
            "resources": resources,
            "resource_checks": resource_checks,
            "parameter_search": False,
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
                "schema_version": "worldsim_v51_f0a_asset_source_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_f0a_asset_source_status_v1",
                "task_id": TASK_ID,
                "status": "done",
                "conclusion": config["decision"]["expected_conclusion"],
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
                "schema_version": "worldsim_v51_f0a_asset_source_status_v1",
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
        default=PROJECT / "configs/worldsim_v51/stage_f_f0a_asset_source_acquisition_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
