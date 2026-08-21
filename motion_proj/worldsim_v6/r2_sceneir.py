"""WorldSim V6 R2 SceneIR v0 的正式、无训练表示实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from motion_proj.worldsim_v6.capabilities import LogicalURIResolver, load_local_capabilities
from motion_proj.worldsim_v6.sceneir import verify_sceneir, write_sceneir
from motion_proj.worldsim_v6.sceneir_adapters import (
    native_recondrive_view,
    native_streetgs_views,
    recondrive_to_sceneir,
    streetgs_to_sceneir,
)
from motion_proj.worldsim_v6.sceneir_render import compare_views, render_view


TASK_ID = "WS-V6-R2-SCENEIR-V0-01"


class R2ExperimentError(RuntimeError):
    """R2 正式实验合同失败。"""


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, dict[int, dict[str, Any]], dict[str, Any]]:
    fixture = config["recondrive_conformance_fixture"]
    count = int(fixture["primitive_count"])
    rng = np.random.default_rng(int(config["seed"]))
    quaternions = rng.normal(size=(count, 4)).astype(np.float32)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    outputs = {
        "xyz": rng.normal(size=(count, 3)).astype(np.float32),
        "rot_maps": quaternions,
        "scale_maps": (0.01 + rng.random((count, 3)) * 0.1).astype(np.float32),
        "opacity_maps": rng.random((count, 1)).astype(np.float32),
        "sh_maps": rng.normal(size=(count, 3, 25)).astype(np.float32),
        "forward_flow": rng.normal(scale=0.5, size=(count, 3)).astype(np.float32),
    }
    assignments = np.full(count, -1, dtype=np.int64)
    assignments[np.arange(count) % 5 == 1] = 0
    assignments[np.arange(count) % 5 == 2] = 1
    timestamps = fixture["timestamps_us"]
    poses = {
        0: {
            "class": "vehicle",
            "translation_m": np.asarray([[1.0, 0.0, 0.0], [1.05, 0.0, 0.0]], dtype=np.float32),
            "rotation_wxyz": np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "visibility": np.asarray([True, True]),
        },
        1: {
            "class": "vehicle",
            "translation_m": np.asarray([[-2.0, 1.0, 0.0], [-2.0, 1.1, 0.0]], dtype=np.float32),
            "rotation_wxyz": np.asarray([[1.0, 0.0, 0.0, 0.0], [0.9998477, 0.0, 0.0, 0.0174524]], dtype=np.float32),
            "visibility": np.asarray([True, True]),
        },
    }
    camera = {
        "camera_model": "pinhole",
        "resolution_px": [518, 280],
        "intrinsics_3x3": [[400.0, 0.0, 259.0], [0.0, 400.0, 140.0], [0.0, 0.0, 1.0]],
    }
    return outputs, assignments, poses, camera


def _compare_recondrive(
    package: Path,
    outputs: Mapping[str, np.ndarray],
    assignments: np.ndarray,
    timestamp_us: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    native = native_recondrive_view(outputs)
    actual = render_view(package, timestamp_us)
    comparisons = {}
    for group, mask in (("static", assignments == -1), ("dynamic", assignments >= 0)):
        expected = {name: value[mask] for name, value in native.items()}
        order = np.argsort(expected["source_indices"], kind="stable")
        expected = {name: value[order] for name, value in expected.items()}
        comparisons[group] = compare_views(expected, actual[group], atol=atol, rtol=rtol)
    return {"passed": all(value["passed"] for value in comparisons.values()), "groups": comparisons}


def _fresh_reload(repo_root: Path, package: Path) -> dict[str, Any]:
    expression = (
        "import json,sys; from pathlib import Path; "
        "from motion_proj.worldsim_v6.sceneir import verify_sceneir; "
        "print(json.dumps(verify_sceneir(Path(sys.argv[1])),sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", expression, str(package)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise R2ExperimentError(f"fresh-process reload 失败：{completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def run_experiment(
    repo_root: Path,
    config_path: Path,
    local_manifest_path: Path,
    run_root: Path,
) -> Path:
    """执行一次真实 StreetGS + ReconDrive schema fixture 的 R2 表示实验。"""
    if _git(repo_root, "status", "--porcelain"):
        raise R2ExperimentError("正式 R2 run 禁止使用 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "worldsim_v6.sceneir_v0_experiment.v1" or config.get("task_id") != TASK_ID:
        raise R2ExperimentError("R2 config schema/task 漂移")
    local = load_local_capabilities(local_manifest_path)
    resolver = LogicalURIResolver(local)
    matrix_path = resolver.resolve(config["sources"]["streetgs_matrix"])
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    scene = config["sources"]["streetgs_scene"]
    checkpoint_path = Path(matrix["baselines"]["streetgs"]["checkpoints"][scene]["path"])
    if not checkpoint_path.is_file():
        raise R2ExperimentError("StreetGS reference checkpoint 缺失")
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != config["sources"]["streetgs_checkpoint_sha256"]:
        raise R2ExperimentError("StreetGS checkpoint hash 漂移")
    recondrive_path = resolver.resolve(config["sources"]["recondrive_repo"])
    recondrive_commit = _git(recondrive_path, "rev-parse", "HEAD")
    if recondrive_commit != config["sources"]["recondrive_version"]:
        raise R2ExperimentError("ReconDrive checkout commit 漂移")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__sceneir-v0-s{config['seed']}-r1"
    if run_dir.exists():
        raise R2ExperimentError(f"run 已存在：{run_dir}")
    run_dir.mkdir(parents=True)
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        street_document, street_arrays = streetgs_to_sceneir(
            checkpoint,
            source_sha256=checkpoint_sha,
            source_uri="checkpoint://streetgs_reference",
            reconstructor_version=config["sources"]["streetgs_version"],
            seed=int(config["seed"]),
        )
        street_package = run_dir / "sceneir_packages/streetgs_scene0230"
        write_sceneir(street_package, street_document, street_arrays)
        timestamp_index = int(config["equivalence"]["timestamp_index"])
        timestamp_us = timestamp_index * 100_000
        street_native = native_streetgs_views(checkpoint, timestamp_index)
        street_actual = render_view(street_package, timestamp_us)
        atol = float(config["equivalence"]["absolute_tolerance"])
        rtol = float(config["equivalence"]["relative_tolerance"])
        street_comparison = {
            group: compare_views(street_native[group], street_actual[group], atol=atol, rtol=rtol)
            for group in ("static", "dynamic")
        }
        street_passed = all(value["passed"] for value in street_comparison.values())

        outputs, assignments, poses, camera = _fixture(config)
        recon_document, recon_arrays = recondrive_to_sceneir(
            outputs,
            assignments,
            poses,
            timestamps_us=config["recondrive_conformance_fixture"]["timestamps_us"],
            source_sha256=hashlib.sha256(_git(recondrive_path, "rev-parse", "HEAD").encode()).hexdigest(),
            source_uri="third_party://recondrive",
            reconstructor_version=recondrive_commit,
            camera=camera,
            seed=int(config["seed"]),
        )
        recon_package = run_dir / "sceneir_packages/recondrive_schema_fixture"
        write_sceneir(recon_package, recon_document, recon_arrays)
        recon_comparison = _compare_recondrive(
            recon_package,
            outputs,
            assignments,
            int(config["recondrive_conformance_fixture"]["timestamps_us"][0]),
            atol,
            rtol,
        )
        equivalence = {
            "schema_version": "worldsim_v6.sceneir_equivalence.v1",
            "streetgs": {"passed": street_passed, "timestamp_index": timestamp_index, "groups": street_comparison},
            "recondrive_schema_fixture": recon_comparison,
            "passed": bool(street_passed and recon_comparison["passed"]),
        }
        _write_json(run_dir / "REPRESENTATION_EQUIVALENCE.json", equivalence)
        reloads = {
            "streetgs": _fresh_reload(repo_root, street_package),
            "recondrive_schema_fixture": _fresh_reload(repo_root, recon_package),
        }
        _write_json(run_dir / "FRESH_PROCESS_RELOAD.json", reloads)
        summary = {
            "schema_version": "worldsim_v6.r2_summary.v1",
            "task_id": TASK_ID,
            "status": "done" if equivalence["passed"] else "rejected",
            "source_commit": source_commit,
            "source_dirty": False,
            "streetgs_checkpoint_sha256": checkpoint_sha,
            "recondrive_commit": recondrive_commit,
            "representation_equivalence_passed": equivalence["passed"],
            "fresh_process_reload_passed": True,
            "training_started": False,
            "model_inference_started": False,
            "quality_data_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "REPRESENTATION_EQUIVALENCE.json",
            "FRESH_PROCESS_RELOAD.json",
            "SUMMARY.json",
            "sceneir_packages/streetgs_scene0230/MANIFEST.json",
            "sceneir_packages/recondrive_schema_fixture/MANIFEST.json",
        ]
        manifest = {
            "schema_version": "worldsim_v6.r2_run_manifest.v1",
            "files": {
                relative: {
                    "bytes": (run_dir / relative).stat().st_size,
                    "sha256": _sha256(run_dir / relative),
                }
                for relative in tracked
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        terminal = {
            "task_id": TASK_ID,
            "status": summary["status"],
            "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        _write_json(run_dir / "TERMINAL_STATUS.json", terminal)
        if not equivalence["passed"]:
            raise R2ExperimentError(f"representation equivalence rejected：{run_dir}")
        return run_dir
    except Exception as error:
        if not (run_dir / "TERMINAL_STATUS.json").exists():
            _write_json(
                run_dir / "TERMINAL_STATUS.json",
                {
                    "task_id": TASK_ID,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                },
            )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/sceneir_v0.yaml"))
    parser.add_argument("--local-manifest", type=Path, default=Path(".local/worldsim_v6/capabilities.local.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args(argv)
    run_dir = run_experiment(
        args.repo_root.resolve(),
        (args.repo_root / args.config).resolve() if not args.config.is_absolute() else args.config,
        (args.repo_root / args.local_manifest).resolve() if not args.local_manifest.is_absolute() else args.local_manifest,
        args.run_root.resolve(),
    )
    print(run_dir)
    return 0
