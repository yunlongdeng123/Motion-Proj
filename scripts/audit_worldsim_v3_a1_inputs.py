#!/usr/bin/env python
"""Formal A1 audit of native calibration, metadata, and LiDAR provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


SCENES = {
    "scene-0230": {
        "scene_index": 179,
        "source_run": "20260805T171624Z__scene0230-reuse-eval-s0-r1",
        "actor_run": "20260805T173900Z__scene0230-actor-metrics-s0-r1",
    },
    "scene-0242": {
        "scene_index": 191,
        "source_run": "20260805T171914Z__scene0242-reuse-eval-s0-r1",
        "actor_run": "20260805T174100Z__scene0242-actor-metrics-s0-r1",
    },
    "scene-0255": {
        "scene_index": 204,
        "source_run": "20260805T162355Z__scene0255-native30k-s0-r1",
        "actor_run": "20260805T174300Z__scene0255-actor-metrics-s0-r1",
    },
}
TASK_ROOT = Path(
    "/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01"
)
TIMING_FIELDS = {
    "exposure",
    "exposure_time",
    "readout_direction",
    "readout_time",
    "rolling_shutter",
    "row_time",
    "row_timing",
    "shutter",
}


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*command: str, cwd: Path) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def stats(values: torch.Tensor) -> dict[str, float | int | None]:
    array = values.detach().double().reshape(-1).cpu().numpy()
    if not array.size:
        return {"count": 0, "min": None, "mean": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    first = value[..., :3]
    second = value[..., 3:]
    b1 = torch.nn.functional.normalize(first, dim=-1)
    b2 = torch.nn.functional.normalize(
        second - (b1 * second).sum(dim=-1, keepdim=True) * b1, dim=-1
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def rotation_angles_deg(matrix: torch.Tensor) -> torch.Tensor:
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def decode_affine(state: dict[str, torch.Tensor], embeddings: torch.Tensor) -> torch.Tensor:
    hidden = torch.relu(
        embeddings @ state["decoder.0.weight"].T + state["decoder.0.bias"]
    )
    output = hidden @ state["decoder.2.weight"].T + state["decoder.2.bias"]
    return output.reshape(-1, 3, 4)


def calibration_audit(
    checkpoint: Path, test_indices: list[int]
) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu")
    models = payload["models"]
    affine = models["Affine"]
    pose = models["CamPose"]
    image_count = int(affine["embedding.weight"].shape[0])
    test_mask = torch.zeros(image_count, dtype=torch.bool)
    test_mask[torch.tensor(test_indices, dtype=torch.long)] = True
    train_mask = ~test_mask

    train_affine = decode_affine(affine, affine["embedding.weight"][train_mask])
    fallback_affine = decode_affine(
        affine, affine["embedding.weight"].mean(dim=0, keepdim=True)
    )
    matrix_residual = train_affine[:, :, :3]
    shifts = train_affine[:, :, 3]
    fallback_matrix = fallback_affine[0, :, :3]
    fallback_shift = fallback_affine[0, :, 3]

    raw_pose = pose["embeds.weight"]
    translation = raw_pose[:, :3]
    identity = pose["identity"].reshape(1, 6)
    rotation = rotation_6d_to_matrix(raw_pose[:, 3:] + identity)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "step": int(payload["step"]),
        "full_image_count": image_count,
        "train_image_count": int(train_mask.sum()),
        "heldout_image_count": int(test_mask.sum()),
        "native_affine": {
            "parameter_count": int(sum(value.numel() for value in affine.values())),
            "embedding_shape": list(affine["embedding.weight"].shape),
            "train_matrix_residual_frobenius": stats(
                torch.linalg.matrix_norm(matrix_residual)
            ),
            "train_shift_l2": stats(torch.linalg.vector_norm(shifts, dim=-1)),
            "heldout_policy": "decode_mean_of_all_image_embeddings",
            "heldout_fallback_matrix_residual_frobenius": float(
                torch.linalg.matrix_norm(fallback_matrix)
            ),
            "heldout_fallback_shift_l2": float(
                torch.linalg.vector_norm(fallback_shift)
            ),
        },
        "native_cam_pose": {
            "parameter_count": int(sum(value.numel() for value in pose.values())),
            "embedding_shape": list(raw_pose.shape),
            "train_translation_l2_m": stats(
                torch.linalg.vector_norm(translation[train_mask], dim=-1)
            ),
            "train_rotation_angle_deg": stats(
                rotation_angles_deg(rotation[train_mask])
            ),
            "heldout_translation_l2_m": stats(
                torch.linalg.vector_norm(translation[test_mask], dim=-1)
            ),
            "heldout_rotation_angle_deg": stats(
                rotation_angles_deg(rotation[test_mask])
            ),
            "amplitude_bound": None,
            "temporal_smoothness": None,
        },
    }
    del payload, models, affine, pose
    return result


def processed_audit(root: Path) -> dict[str, object]:
    directories = {
        path.name: sum(1 for child in path.rglob("*") if child.is_file())
        for path in root.iterdir()
        if path.is_dir()
    }
    timing_files = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and any(field in path.name.lower() for field in TIMING_FIELDS)
    ]
    lidar_files = sorted((root / "lidar").glob("*.bin"))
    lidar_points = [path.stat().st_size // (4 * 4) for path in lidar_files]
    return {
        "root": str(root),
        "directories": directories,
        "top_level_names": sorted(path.name for path in root.iterdir()),
        "timing_like_files": timing_files,
        "lidar_frame_count": len(lidar_files),
        "lidar_points_per_frame": {
            "min": min(lidar_points),
            "mean": float(np.mean(lidar_points)),
            "max": max(lidar_points),
            "storage_contract": "float32 Nx4",
        },
    }


def markdown_report(result: dict[str, object]) -> str:
    lines = [
        "# WorldSim V3 A1 input and native-calibration audit",
        "",
        f"Rolling shutter: **{result['rolling_shutter']['status']}**.",
        "",
        result["rolling_shutter"]["reason"],
        "",
        "| scene | native Affine params | native CamPose params | pose train translation p95 / rotation p95 | processed LiDAR frames |",
        "|---|---:|---:|---:|---:|",
    ]
    for scene, row in result["scenes"].items():
        calibration = row["calibration"]
        pose = calibration["native_cam_pose"]
        lines.append(
            f"| {scene} | {calibration['native_affine']['parameter_count']:,} "
            f"| {pose['parameter_count']:,} "
            f"| {pose['train_translation_l2_m']['p95']:.4f} m / "
            f"{pose['train_rotation_angle_deg']['p95']:.4f}° "
            f"| {row['processed']['lidar_frame_count']} |"
        )
    lines.extend(
        [
            "",
            "Decisions:",
            "",
            "- C0 deletes native Affine and CamPose from the model config.",
            "- C2 replaces only native per-image Affine with camera + continuous-time factorization.",
            "- C3 adds bounded axis-angle pose residuals and per-camera temporal smoothness on C2.",
            "- LiDAR runtime inputs will be hashed by the A1 trainer; ancestry after densification remains unsupported.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--drivestudio-root", type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    (args.run_dir / "artifacts").mkdir(parents=True)
    (args.run_dir / "source_snapshot").mkdir()
    atomic_json(args.run_dir / "terminal.json", {"status": "running", "failure": None})
    source_file = args.project_root / "scripts/audit_worldsim_v3_a1_inputs.py"
    shutil.copy2(source_file, args.run_dir / "source_snapshot" / source_file.name)
    manifest = {
        "schema_version": 1,
        "task_id": "WS-V3-A1-CALIBRATION-01",
        "component": "input/native-calibration/LiDAR provenance audit",
        "project_commit": command_output(
            "git", "rev-parse", "HEAD", cwd=args.project_root
        ),
        "project_status": command_output(
            "git", "status", "--short", cwd=args.project_root
        ).splitlines(),
        "metadata_root": str(args.metadata_root),
        "processed_root": str(args.processed_root),
        "drivestudio_root": str(args.drivestudio_root),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)
    try:
        sample_data = json.loads(
            (args.metadata_root / "sample_data.json").read_text(encoding="utf-8")
        )
        calibrated_sensor = json.loads(
            (args.metadata_root / "calibrated_sensor.json").read_text(
                encoding="utf-8"
            )
        )
        raw_fields = sorted({key for row in sample_data for key in row})
        calibration_fields = sorted({key for row in calibrated_sensor for key in row})
        timing_present = sorted(TIMING_FIELDS.intersection(raw_fields + calibration_fields))

        scene_results: dict[str, object] = {}
        for scene, spec in SCENES.items():
            source_dir = TASK_ROOT / spec["source_run"]
            actor_dir = TASK_ROOT / spec["actor_run"]
            source = json.loads((source_dir / "summary.json").read_text())
            actor = json.loads((actor_dir / "summary.json").read_text())[
                "actor_metrics"
            ]
            checkpoint = Path(source["checkpoint"]["checkpoint"])
            config = OmegaConf.load(checkpoint.parent / "config.yaml")
            raw_manifest_path = args.manifest_root / f"{scene}_raw_manifest.json"
            raw_manifest = json.loads(raw_manifest_path.read_text())
            processed_root = args.processed_root / str(spec["scene_index"])
            scene_results[scene] = {
                "scene_index": spec["scene_index"],
                "source_run": str(source_dir),
                "actor_run": str(actor_dir),
                "raw_manifest": str(raw_manifest_path),
                "raw_manifest_sha256": sha256_file(raw_manifest_path),
                "raw_sensor_counts": raw_manifest["sensor_counts"],
                "raw_camera_timestamp_only": True,
                "processed": processed_audit(processed_root),
                "native_config": {
                    "Affine": OmegaConf.to_container(config.model.Affine, resolve=True),
                    "CamPose": OmegaConf.to_container(config.model.CamPose, resolve=True),
                    "background_init": OmegaConf.to_container(
                        config.model.Background.init, resolve=True
                    ),
                    "rigid_init": OmegaConf.to_container(
                        config.model.RigidNodes.init, resolve=True
                    ),
                },
                "calibration": calibration_audit(
                    checkpoint, actor["heldout_split"]["test_full_image_indices"]
                ),
            }
        result = {
            "status": "done",
            "raw_schema": {
                "sample_data_fields": raw_fields,
                "calibrated_sensor_fields": calibration_fields,
                "timing_fields_present": timing_present,
            },
            "rolling_shutter": {
                "status": "not_supported",
                "reason": (
                    "nuScenes sample_data provides frame timestamps only; raw calibrated_sensor and "
                    "all three processed scene trees contain no exposure, readout direction/time, "
                    "row timing, shutter model, or equivalent metadata"
                ),
            },
            "scenes": scene_results,
            "implementation_contract": {
                "c0-off": "delete model.Affine and model.CamPose",
                "c1-native": "A0 native checkpoint and parameterization",
                "c2-factorized-isp": "camera embedding plus continuous normalized-time affine; no exposure claim",
                "c3-bounded-pose": "C2 plus <=0.15m, <=2deg pose residual and temporal regularization",
                "c4-rolling-shutter": "not_supported",
            },
            "lidar_provenance": {
                "source_files": {
                    "driving_dataset.py": sha256_file(
                        args.drivestudio_root / "datasets/driving_dataset.py"
                    ),
                    "scene_graph.py": sha256_file(
                        args.drivestudio_root / "models/trainers/scene_graph.py"
                    ),
                },
                "current_checkpoint_limitation": (
                    "native checkpoints do not retain sampled LiDAR indices or post-densification ancestry"
                ),
                "a1_action": (
                    "WorldSimV3Trainer hashes exact background and per-instance runtime initialization tensors"
                ),
            },
        }
        audit_path = args.run_dir / "artifacts" / "a1_input_audit.json"
        atomic_json(audit_path, result)
        report_path = args.run_dir / "artifacts" / "a1_input_audit.md"
        report_path.write_text(markdown_report(result), encoding="utf-8")
        summary = {
            "status": "done",
            "audit": str(audit_path),
            "report": str(report_path),
            "rolling_shutter": result["rolling_shutter"],
            "scenes": list(scene_results),
        }
        atomic_json(args.run_dir / "summary.json", summary)
        atomic_json(args.run_dir / "terminal.json", {"status": "done", "failure": None})
        print(json.dumps(summary, indent=2, sort_keys=True))
    except BaseException as error:
        atomic_json(
            args.run_dir / "terminal.json",
            {
                "status": "blocked",
                "failure": {
                    "code": "A1_INPUT_AUDIT_FAILED",
                    "detail": f"{type(error).__name__}: {error}",
                },
            },
        )
        raise


if __name__ == "__main__":
    main()
