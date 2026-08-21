"""WorldSim V6 R30 三因子通过后的 typed actor appearance bake。"""

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


TASK_ID = "WS-V6-R30-TYPED-ACTOR-BAKE-01"


class R30ExperimentError(RuntimeError):
    """R30 正式合同失败。"""


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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R30ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R30ExperimentError("正式 R30 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R30ExperimentError("R30 task_id 漂移")
    sources = config["sources"]
    r24 = _resolve_runs_uri(sources["r24_run"])
    r29 = _resolve_runs_uri(sources["r29_run"])
    source_files = {
        r24 / "MANIFEST.json": sources["r24_manifest_sha256"],
        r24 / "CASES.jsonl": sources["r24_cases_sha256"],
        r24 / "verifier_worker/PER_CASE_ARMS.jsonl": sources[
            "r24_per_case_arms_sha256"
        ],
        r29 / "MANIFEST.json": sources["r29_manifest_sha256"],
        r29 / "R29_GATE.json": sources["r29_gate_sha256"],
        r29 / "ACTOR_FACTORIZED_DECISIONS.jsonl": sources["r29_decisions_sha256"],
    }
    for path, expected in source_files.items():
        if _sha256(path) != expected:
            raise R30ExperimentError(f"冻结输入漂移：{path}")
    r29_gate = json.loads((r29 / "R29_GATE.json").read_text(encoding="utf-8"))
    if not r29_gate.get("checks", {}).get("passed"):
        raise R30ExperimentError("R29 未授权 typed actor bake")
    if r29_gate.get("decision") != "proceed_to_typed_actor_bake":
        raise R30ExperimentError("R29 bake decision 漂移")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R30ExperimentError("R30 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__typed-actor-bake-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        package_dir = run_dir / "package"
        actor_dir = package_dir / "actor_assets"
        actor_dir.mkdir(parents=True)
        decisions = _read_jsonl(r29 / "ACTOR_FACTORIZED_DECISIONS.jsonl")
        accepted = [row for row in decisions if row["overall_decision"] == "ACCEPT"]
        expected = int(config["cohort"]["expected_actor_accept_count"])
        if len(accepted) != expected:
            raise R30ExperimentError("R30 accepted actor denominator 漂移")
        cases = {row["case_id"]: row for row in _read_jsonl(r24 / "CASES.jsonl")}
        arm_rows = {
            row["case_id"]: row
            for row in _read_jsonl(r24 / "verifier_worker/PER_CASE_ARMS.jsonl")
        }
        proposal_directory = str(sources["proposal_directory"])
        asset_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        validity_rows: list[dict[str, Any]] = []
        trajectory_rows: list[dict[str, Any]] = []
        tracked_payloads: list[str] = []
        source_hashes: dict[str, str] = {}

        for decision in accepted:
            case_id = decision["case_id"]
            case = cases[case_id]
            arm_row = arm_rows[case_id]
            factors = decision["factorized_validity"]
            if case["hole_type"] != "actor_removal_hole":
                raise R30ExperimentError("R30 非 actor hole 意外 ACCEPT")
            if not all(factors[name] == "ACCEPT" for name in ("photo", "geometry", "semantic")):
                raise R30ExperimentError("R30 actor appearance factor 不完整")
            if factors["dynamics"] != "ABSTAIN":
                raise R30ExperimentError("R30 dynamics 状态漂移")
            proposal_path = r24 / proposal_directory / f"{case_id}__repeat1.npy"
            proposal_sha = _sha256(proposal_path)
            if proposal_sha != arm_row["proposal_sha256"]:
                raise R30ExperimentError("R30 proposal 漂移")
            source_hashes[case_id] = proposal_sha
            verifier_input = r24 / "verifier_inputs" / f"{case_id}.npz"
            if _sha256(verifier_input) != case["verifier_input_sha256"]:
                raise R30ExperimentError("R30 verifier input 漂移")
            with np.load(verifier_input, allow_pickle=False) as archive:
                input_image = np.asarray(archive["input_image"], dtype=np.uint8)
                mask = np.asarray(archive["mask"], dtype=bool)
            proposal = np.load(proposal_path, allow_pickle=False).astype(np.uint8)
            coordinates = np.argwhere(mask).astype(np.int16)
            rgb_values = proposal[mask].astype(np.uint8)
            y0, x0 = coordinates.min(axis=0).tolist()
            y1, x1 = coordinates.max(axis=0).tolist()
            relative_payload = f"actor_assets/{case_id}.npz"
            payload_path = package_dir / relative_payload
            np.savez_compressed(
                payload_path,
                coordinates_yx=coordinates,
                rgb_uint8=rgb_values,
                canvas_height=np.asarray(input_image.shape[0], dtype=np.int32),
                canvas_width=np.asarray(input_image.shape[1], dtype=np.int32),
                bbox_yxyx=np.asarray([y0, x0, y1 + 1, x1 + 1], dtype=np.int32),
            )
            with np.load(payload_path, allow_pickle=False) as archive:
                reload_coordinates = np.asarray(archive["coordinates_yx"], dtype=np.int64)
                reload_rgb = np.asarray(archive["rgb_uint8"], dtype=np.uint8)
                reload_height = int(np.asarray(archive["canvas_height"]).item())
                reload_width = int(np.asarray(archive["canvas_width"]).item())
            reloaded = input_image.copy()
            reloaded[reload_coordinates[:, 0], reload_coordinates[:, 1]] = reload_rgb
            reload_exact = bool(
                (reload_height, reload_width) == input_image.shape[:2]
                and np.array_equal(reloaded, proposal)
            )
            if not reload_exact:
                raise R30ExperimentError("R30 actor payload reload 不 exact")
            payload_sha = _sha256(payload_path)
            tracked_payloads.append(f"package/{relative_payload}")
            asset_id = f"actor_appearance::{case_id}"
            asset_rows.append(
                {
                    "schema_version": "worldsim_v6.r30_actor_asset.v1",
                    "asset_id": asset_id,
                    "asset_type": "explicit_actor_observation_plane_appearance",
                    "case_id": case_id,
                    "actor_identity_binding": "case_scoped_unresolved",
                    "coordinate_space": config["package"]["coordinate_space"],
                    "payload": relative_payload,
                    "payload_sha256": payload_sha,
                    "pixel_count": int(coordinates.shape[0]),
                    "bbox_yxyx": [y0, x0, y1 + 1, x1 + 1],
                    "reload_exact": True,
                    "runtime_generator_dependency": False,
                }
            )
            provenance_rows.append(
                {
                    "schema_version": "worldsim_v6.r30_provenance.v1",
                    "asset_id": asset_id,
                    "source_type": "same_time_cross_frontend_reconstruction",
                    "scene": case["scene"],
                    "frame_index": int(case["frame_index"]),
                    "target_frontend": case["frontend"],
                    "proposal_source_frontend": case["proposal_source_frontend"],
                    "source_proposal_sha256": proposal_sha,
                    "baked_payload_sha256": payload_sha,
                    "generation_source": None,
                }
            )
            validity_rows.append(
                {
                    "schema_version": "worldsim_v6.r30_validity.v1",
                    "asset_id": asset_id,
                    "photo": factors["photo"],
                    "geometry": factors["geometry"],
                    "semantic": factors["semantic"],
                    "dynamics": factors["dynamics"],
                    "appearance_decision": "BAKED_ACTOR_APPEARANCE",
                    "trajectory_decision": "ABSTAIN",
                }
            )
            trajectory_rows.append(
                {
                    "schema_version": "worldsim_v6.r30_trajectory_abstain.v1",
                    "asset_id": asset_id,
                    "decision": "ABSTAIN",
                    "reason": "no_independent_dynamics_or_trajectory_verifier",
                    "trajectory_created": False,
                }
            )

        _write_jsonl(package_dir / "ACTOR_ASSET_REGISTRY.jsonl", asset_rows)
        _write_jsonl(package_dir / "PROVENANCE.jsonl", provenance_rows)
        _write_jsonl(package_dir / "VALIDITY.jsonl", validity_rows)
        _write_jsonl(package_dir / "TRAJECTORY_ABSTAINS.jsonl", trajectory_rows)
        dependency = {
            "schema_version": "worldsim_v6.r30_runtime_dependency_audit.v1",
            "payload_format": "numpy_npz_sparse_yx_rgb_uint8",
            "runtime_dependencies": ["numpy_npz_reader"],
            "online_generator_dependency": False,
            "model_weight_dependency": False,
            "network_dependency": False,
            "source_run_dependency_for_payload_read": False,
            "actor_appearance_asset_count": len(asset_rows),
            "trajectory_asset_count": 0,
        }
        _write_json(package_dir / "RUNTIME_DEPENDENCY_AUDIT.json", dependency)
        source_immutable = all(
            _sha256(r24 / proposal_directory / f"{case_id}__repeat1.npy") == sha
            for case_id, sha in source_hashes.items()
        )
        wall_seconds = time.monotonic() - started
        checks = {
            "all_r29_accepts_accounted": len(asset_rows) == len(accepted),
            "expected_actor_appearance_assets": len(asset_rows) == expected,
            "photo_geometry_semantic_all_accept": all(
                all(row[name] == "ACCEPT" for name in ("photo", "geometry", "semantic"))
                for row in validity_rows
            ),
            "dynamics_and_trajectory_abstain": len(trajectory_rows) == expected
            and all(row["decision"] == "ABSTAIN" for row in trajectory_rows),
            "no_trajectory_asset_created": dependency["trajectory_asset_count"] == 0,
            "payload_reload_exact": all(row["reload_exact"] for row in asset_rows),
            "no_online_generator_dependency": not dependency["online_generator_dependency"],
            "provenance_complete": len(provenance_rows) == len(asset_rows),
            "validity_complete": len(validity_rows) == len(asset_rows),
            "source_proposals_immutable": source_immutable,
            "wall_within_budget": wall_seconds
            <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R30_GATE.json",
            {
                "schema_version": "worldsim_v6.r30_gate.v1",
                "checks": checks,
                "decision": "proceed_to_actor_logsim_validation"
                if checks["passed"]
                else "reject_or_repair_actor_bake",
            },
        )
        package_files = [
            "package/ACTOR_ASSET_REGISTRY.jsonl",
            "package/PROVENANCE.jsonl",
            "package/VALIDITY.jsonl",
            "package/TRAJECTORY_ABSTAINS.jsonl",
            "package/RUNTIME_DEPENDENCY_AUDIT.json",
            *tracked_payloads,
        ]
        _write_json(
            package_dir / "PACKAGE_MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r30_package_manifest.v1",
                "files": {
                    name.removeprefix("package/"): {
                        "bytes": (run_dir / name).stat().st_size,
                        "sha256": _sha256(run_dir / name),
                    }
                    for name in package_files
                },
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r30_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r30_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "actor_appearance_asset_count": len(asset_rows),
            "trajectory_asset_count": 0,
            "trajectory_abstain_count": len(trajectory_rows),
            "runtime_generator_dependency": False,
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R30_GATE.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
            "package/PACKAGE_MANIFEST.json",
            *package_files,
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r30_manifest.v1",
                "files": {
                    name: {
                        "bytes": (run_dir / name).stat().st_size,
                        "sha256": _sha256(run_dir / name),
                    }
                    for name in tracked
                },
            },
        )
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
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r30_typed_actor_bake_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0
