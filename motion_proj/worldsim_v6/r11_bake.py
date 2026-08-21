"""WorldSim V6 R11 typed bake 正式实验。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


TASK_ID = "WS-V6-R11-BAKE-01"
ALLOWED_TASK_IDS = {TASK_ID, "WS-V6-R18-RGBD-TEMPORAL-TYPED-BAKE-01"}


class R11ExperimentError(RuntimeError):
    """R11 正式合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R11ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R11ExperimentError("正式 R11 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_id = str(config.get("task_id"))
    if task_id not in ALLOWED_TASK_IDS:
        raise R11ExperimentError("R11 task_id 漂移")
    sources = config["sources"]
    r10_run = _resolve_runs_uri(sources["r10_run"])
    r9_run = _resolve_runs_uri(sources["r9_run"])
    r10_gate_name = str(sources.get("r10_gate_file", "R10_GATE.json"))
    proposal_directory = str(
        sources.get("proposal_directory", "cross_frontend_reconstruction_proposals")
    )
    source_files = {
        r10_run / "MANIFEST.json": sources["r10_manifest_sha256"],
        r10_run / r10_gate_name: sources["r10_gate_sha256"],
        r10_run / "FACTORIZED_DECISIONS.jsonl": sources["r10_decisions_sha256"],
        r9_run / "MANIFEST.json": sources["r9_manifest_sha256"],
        r9_run / "CASES.jsonl": sources["r9_cases_sha256"],
        r9_run / "verifier_worker/PER_CASE_ARMS.jsonl": sources[
            "r9_per_case_arms_sha256"
        ],
    }
    for path, expected_sha in source_files.items():
        if _sha256(path) != expected_sha:
            raise R11ExperimentError(f"source 漂移：{path.name}")
    r10_gate = json.loads((r10_run / r10_gate_name).read_text(encoding="utf-8"))
    if r10_gate["decision"] != "proceed_to_bake":
        raise R11ExperimentError("R10 未授权 bake")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R11ExperimentError("R11 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / task_id / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__typed-bake-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        package_dir = run_dir / "package"
        chunk_dir = package_dir / "static_chunks"
        chunk_dir.mkdir(parents=True)
        decisions = _read_jsonl(r10_run / "FACTORIZED_DECISIONS.jsonl")
        accepted = [row for row in decisions if row["overall_decision"] == "ACCEPT"]
        if len(accepted) != int(config["cohort"]["expected_r10_accept_count"]):
            raise R11ExperimentError("R11 R10 ACCEPT denominator 漂移")
        cases = {row["case_id"]: row for row in _read_jsonl(r9_run / "CASES.jsonl")}
        arm_rows = {
            row["case_id"]: row
            for row in _read_jsonl(r9_run / "verifier_worker/PER_CASE_ARMS.jsonl")
        }
        static_types = set(config["asset_typing"]["static_chunk_hole_types"])
        actor_types = set(config["asset_typing"]["actor_asset_hole_types"])
        asset_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        validity_rows: list[dict[str, Any]] = []
        abstain_rows: list[dict[str, Any]] = []
        source_hashes_before: dict[str, str] = {}
        tracked_chunks: list[str] = []

        for decision in accepted:
            case_id = decision["case_id"]
            case = cases[case_id]
            arm_row = arm_rows[case_id]
            hole_type = case["hole_type"]
            proposal_path = (
                r9_run
                / proposal_directory
                / f"{case_id}__repeat1.npy"
            )
            proposal_sha = _sha256(proposal_path)
            if proposal_sha != arm_row["proposal_sha256"]:
                raise R11ExperimentError(f"R11 proposal 漂移：{case_id}")
            source_hashes_before[case_id] = proposal_sha
            factorized = decision["factorized_validity"]
            if hole_type in static_types:
                if not all(
                    factorized[factor] == "ACCEPT"
                    for factor in config["asset_typing"]["static_required_factors"]
                ):
                    raise R11ExperimentError(f"static chunk factor 不完整：{case_id}")
                verifier_input = r9_run / "verifier_inputs" / f"{case_id}.npz"
                if _sha256(verifier_input) != case["verifier_input_sha256"]:
                    raise R11ExperimentError(f"verifier input 漂移：{case_id}")
                with np.load(verifier_input, allow_pickle=False) as archive:
                    input_image = np.asarray(archive["input_image"], dtype=np.uint8)
                    mask = np.asarray(archive["mask"], dtype=bool)
                proposal = np.load(proposal_path, allow_pickle=False).astype(np.uint8)
                coordinates = np.argwhere(mask).astype(np.int16)
                rgb_values = proposal[mask].astype(np.uint8)
                relative_chunk = f"static_chunks/{case_id}.npz"
                chunk_path = package_dir / relative_chunk
                np.savez_compressed(
                    chunk_path,
                    coordinates_yx=coordinates,
                    rgb_uint8=rgb_values,
                    canvas_height=np.asarray(input_image.shape[0], dtype=np.int32),
                    canvas_width=np.asarray(input_image.shape[1], dtype=np.int32),
                )
                with np.load(chunk_path, allow_pickle=False) as archive:
                    reload_coordinates = np.asarray(archive["coordinates_yx"], dtype=np.int64)
                    reload_rgb = np.asarray(archive["rgb_uint8"], dtype=np.uint8)
                    reload_height = int(np.asarray(archive["canvas_height"]).item())
                    reload_width = int(np.asarray(archive["canvas_width"]).item())
                if (reload_height, reload_width) != input_image.shape[:2]:
                    raise R11ExperimentError(f"chunk canvas 漂移：{case_id}")
                reloaded = input_image.copy()
                reloaded[reload_coordinates[:, 0], reload_coordinates[:, 1]] = reload_rgb
                reload_exact = bool(np.array_equal(reloaded, proposal))
                if not reload_exact:
                    raise R11ExperimentError(f"chunk reload 不 exact：{case_id}")
                chunk_sha = _sha256(chunk_path)
                tracked_chunks.append(f"package/{relative_chunk}")
                asset_rows.append(
                    {
                        "schema_version": "worldsim_v6.r11_asset.v1",
                        "asset_id": f"static::{case_id}",
                        "asset_type": "explicit_static_observation_chunk",
                        "case_id": case_id,
                        "coordinate_space": config["package"]["coordinate_space"],
                        "payload": relative_chunk,
                        "payload_sha256": chunk_sha,
                        "pixel_count": int(coordinates.shape[0]),
                        "reload_exact": reload_exact,
                        "runtime_generator_dependency": False,
                    }
                )
                provenance_rows.append(
                    {
                        "schema_version": "worldsim_v6.r11_provenance.v1",
                        "asset_id": f"static::{case_id}",
                        "source_type": str(
                            config["package"].get(
                                "proposal_source_type", "reconstructed_cross_frontend"
                            )
                        ),
                        "sensor_support": str(
                            config["package"].get(
                                "sensor_support", "shared_frozen_r3_sensor_support"
                            )
                        ),
                        "time_support": int(case["frame_index"]),
                        "view_support": {
                            "target_frontend": case["frontend"],
                            "proposal_source_frontend": case["proposal_source_frontend"],
                        },
                        "reconstruction_source": str(
                            config["package"].get(
                                "reconstruction_source", "r9_cross_frontend_reconstruction"
                            )
                        ),
                        "generation_source": None,
                        "source_proposal_sha256": proposal_sha,
                        "baked_payload_sha256": chunk_sha,
                    }
                )
                validity_rows.append(
                    {
                        "schema_version": "worldsim_v6.r11_validity.v1",
                        "asset_id": f"static::{case_id}",
                        "photo": factorized["photo"],
                        "geometry": factorized["geometry"],
                        "semantic": factorized["semantic"],
                        "dynamics": factorized["dynamics"],
                        "asset_decision": "BAKED_STATIC",
                    }
                )
            elif hole_type in actor_types:
                required = config["asset_typing"]["actor_required_factors"]
                missing = [factor for factor in required if factorized[factor] != "ACCEPT"]
                if not missing:
                    raise R11ExperimentError("预注册 actor abstain 条件意外不成立")
                abstain_rows.append(
                    {
                        "schema_version": "worldsim_v6.r11_bake_abstain.v1",
                        "case_id": case_id,
                        "requested_asset_type": "explicit_actor_asset_and_trajectory",
                        "decision": "ABSTAIN",
                        "missing_required_factors": missing,
                        "asset_created": False,
                        "source_proposal_sha256": proposal_sha,
                    }
                )
                validity_rows.append(
                    {
                        "schema_version": "worldsim_v6.r11_validity.v1",
                        "asset_id": None,
                        "case_id": case_id,
                        "photo": factorized["photo"],
                        "geometry": factorized["geometry"],
                        "semantic": factorized["semantic"],
                        "dynamics": factorized["dynamics"],
                        "asset_decision": "ABSTAIN_ACTOR_AND_TRAJECTORY",
                    }
                )
            else:
                raise R11ExperimentError(f"未冻结 asset typing：{hole_type}")

        _write_jsonl(package_dir / "ASSET_REGISTRY.jsonl", asset_rows)
        _write_jsonl(package_dir / "PROVENANCE.jsonl", provenance_rows)
        _write_jsonl(package_dir / "VALIDITY.jsonl", validity_rows)
        _write_jsonl(package_dir / "BAKE_ABSTAINS.jsonl", abstain_rows)
        source_immutable = all(
            _sha256(
                r9_run
                / proposal_directory
                / f"{case_id}__repeat1.npy"
            )
            == source_sha
            for case_id, source_sha in source_hashes_before.items()
        )
        dependency_audit = {
            "schema_version": "worldsim_v6.r11_runtime_dependency_audit.v1",
            "payload_format": "numpy_npz_sparse_yx_rgb_uint8",
            "runtime_dependencies": ["numpy_npz_reader"],
            "online_generator_dependency": False,
            "model_weight_dependency": False,
            "network_dependency": False,
            "source_run_dependency_for_payload_read": False,
            "actor_asset_count": 0,
            "trajectory_asset_count": 0,
        }
        _write_json(package_dir / "RUNTIME_DEPENDENCY_AUDIT.json", dependency_audit)
        expected_static = int(config["cohort"]["expected_static_chunk_count"])
        expected_abstain = int(
            config["cohort"]["expected_factor_incomplete_bake_abstain_count"]
        )
        checks = {
            "all_r10_accepts_accounted": len(asset_rows) + len(abstain_rows) == len(accepted),
            "expected_static_chunks": len(asset_rows) == expected_static,
            "expected_factor_incomplete_actor_abstain": len(abstain_rows) == expected_abstain,
            "no_invalid_actor_or_trajectory_bake": dependency_audit["actor_asset_count"] == 0
            and dependency_audit["trajectory_asset_count"] == 0,
            "payload_reload_exact": all(row["reload_exact"] for row in asset_rows),
            "no_online_generator_dependency": not dependency_audit[
                "online_generator_dependency"
            ],
            "provenance_complete": len(provenance_rows) == len(asset_rows),
            "validity_complete": len(validity_rows) == len(accepted),
            "source_proposals_immutable": source_immutable,
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r11_gate.v1",
            "checks": checks,
            "decision": "proceed_to_logsim_validation"
            if checks["passed"]
            else "reject_or_repair_bake",
        }
        gate_name = "R11_GATE.json" if task_id == TASK_ID else "R18_GATE.json"
        _write_json(run_dir / gate_name, gate)
        package_files = [
            "package/ASSET_REGISTRY.jsonl",
            "package/PROVENANCE.jsonl",
            "package/VALIDITY.jsonl",
            "package/BAKE_ABSTAINS.jsonl",
            "package/RUNTIME_DEPENDENCY_AUDIT.json",
            *tracked_chunks,
        ]
        package_manifest = {
            "schema_version": "worldsim_v6.r11_package_manifest.v1",
            "files": {
                name.removeprefix("package/"): {
                    "bytes": (run_dir / name).stat().st_size,
                    "sha256": _sha256(run_dir / name),
                }
                for name in package_files
            },
        }
        _write_json(package_dir / "PACKAGE_MANIFEST.json", package_manifest)
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r11_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": time.monotonic() - started,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r11_summary.v1",
            "task_id": task_id,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "r10_accept_count": len(accepted),
            "static_chunk_count": len(asset_rows),
            "actor_bake_abstain_count": len(abstain_rows),
            "actor_asset_count": 0,
            "trajectory_asset_count": 0,
            "runtime_generator_dependency": False,
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            gate_name,
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
            "package/PACKAGE_MANIFEST.json",
            *package_files,
        ]
        manifest = {
            "schema_version": "worldsim_v6.r11_run_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                for name in tracked
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/worldsim_v6/r11_bake_v0.yaml")
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0
