#!/usr/bin/env python
"""Run the first real, conservative A3 R1 S-B/T0 paired engineering smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf
import torch


PROJECT = Path("/root/autodl-tmp/motion_proj")
DEFAULT_DRIVESTUDIO = Path(
    "/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1"
)
DEFAULT_D2_RUN = Path(
    "/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/"
    "20260809T113230Z__a2-d2-formal30k-s0-r1"
)
DEFAULT_D2_WORK = (
    DEFAULT_D2_RUN
    / "work_dirs/worldsim_v3_a2/"
    "scene0230_a2_d2_boundary_residual_formal_s0_i30000"
)
DEFAULT_SIDECAR_MANIFEST = Path(
    "/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/"
    "20260809T133911Z__a3-sb-sidecar-s0-r3/sidecar/manifest.json"
)
SELECTED_CHECKPOINT_STEP = 30_000
UNIT_ARRAY_KEYS = {
    "source_actor_footprint",
    "edited_actor_footprint",
    "affected_pixel_mask",
    "rgb_loss_mask",
    "geometry_loss_mask",
    "depth_render_expected",
    "depth_render_expected_valid",
    "depth_surface_first_hit",
    "depth_surface_first_hit_valid",
    "depth_lidar_measured",
    "depth_lidar_measured_valid",
}
MASK_KEYS = {
    "source_actor_footprint",
    "edited_actor_footprint",
    "affected_pixel_mask",
    "rgb_loss_mask",
    "geometry_loss_mask",
    "depth_render_expected_valid",
    "depth_surface_first_hit_valid",
    "depth_lidar_measured_valid",
}
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def command_output(*command: str, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def to_device(values: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in values.items()
    }


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def named_parameter_fingerprints(
    trainer: Any, *, mutable_rows: torch.Tensor, mutable_fields: set[str]
) -> dict[str, dict[str, str]]:
    immutable: dict[str, str] = {}
    mutable_outside: dict[str, str] = {}
    mutable_inside: dict[str, str] = {}
    rows = mutable_rows.bool()
    for class_name, model in trainer.models.items():
        for parameter_name, parameter in model.named_parameters():
            name = f"{class_name}.{parameter_name}"
            if name in mutable_fields:
                device_rows = rows.to(parameter.device)
                mutable_outside[name] = tensor_sha256(parameter[~device_rows])
                mutable_inside[name] = tensor_sha256(parameter[device_rows])
            else:
                immutable[name] = tensor_sha256(parameter)
    return {
        "immutable": immutable,
        "mutable_outside": mutable_outside,
        "mutable_inside": mutable_inside,
    }


def model_tensor_layout(trainer: Any) -> dict[str, dict[str, Any]]:
    layout: dict[str, dict[str, Any]] = {}
    for class_name, model in trainer.models.items():
        for key, value in model.state_dict().items():
            if torch.is_tensor(value):
                layout[f"{class_name}.{key}"] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
    return layout


def rigid_contract_sha256(rigid: Any) -> str:
    fields = (
        "_means",
        "_scales",
        "_quats",
        "_features_dc",
        "_features_rest",
        "_opacities",
        "point_ids",
        "instances_size",
        "instances_fv",
        "instances_trans",
        "instances_quats",
    )
    digest = hashlib.sha256()
    for name in fields:
        digest.update(name.encode())
        digest.update(tensor_sha256(getattr(rigid, name)).encode())
    return digest.hexdigest()


def ordered_unit_records(
    protocol: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records = manifest["evidence"]["units"]
    by_key = {(row["role"], row["edit"]): row for row in records}
    expected = [
        (role, edit)
        for role in protocol["paired_design"]["actor_roles"]
        for edit in protocol["paired_design"]["edits"]
    ]
    if set(by_key) != set(expected) or len(records) != len(expected):
        raise RuntimeError("A3 sidecar unit Cartesian product drift")
    ordered = [by_key[key] for key in expected]
    heldout = set(protocol["paired_design"]["heldout"]["frames"])
    for record in ordered:
        role = record["role"]
        actor = protocol["paired_design"]["actors"][role]
        if (
            record["heldout"] is not False
            or int(record["frame"]) in heldout
            or record["instance_token"] != actor["instance_token"]
            or int(record["rigid_model_index"]) != int(actor["rigid_model_index"])
        ):
            raise RuntimeError(f"A3 unit provenance drift: {role}/{record['edit']}")
    return ordered


def load_and_validate_unit(
    sidecar_root: Path, record: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    path = sidecar_root / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"A3 unit SHA drift: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != UNIT_ARRAY_KEYS:
            raise RuntimeError(f"A3 unit array schema drift: {path}")
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise RuntimeError(f"A3 unit raster shape drift: {path}")
    if any(arrays[key].dtype != np.bool_ for key in MASK_KEYS):
        raise RuntimeError(f"A3 unit mask dtype drift: {path}")
    source = arrays["source_actor_footprint"]
    edited = arrays["edited_actor_footprint"]
    affected = arrays["affected_pixel_mask"]
    rgb = arrays["rgb_loss_mask"]
    geometry = arrays["geometry_loss_mask"]
    measured_valid = arrays["depth_lidar_measured_valid"]
    if (
        rgb.any()
        or not geometry.any()
        or np.any(geometry & ~affected)
        or np.any(geometry & source)
        or np.any(geometry & edited)
        or np.any(geometry & ~measured_valid)
        or int(geometry.sum()) != int(record["s_b_t0_geometry_pixels"])
    ):
        raise RuntimeError(f"A3 S-B/T0 unit semantics drift: {path}")
    for depth_key, valid_key in (
        ("depth_render_expected", "depth_render_expected_valid"),
        ("depth_surface_first_hit", "depth_surface_first_hit_valid"),
        ("depth_lidar_measured", "depth_lidar_measured_valid"),
    ):
        if not np.isfinite(arrays[depth_key][arrays[valid_key]]).all():
            raise RuntimeError(f"A3 valid typed depth is non-finite: {depth_key}")
    return arrays


def apply_unit_edit(rigid: Any, record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    from scripts.run_dr_v2_m4_pilot import move_actor_local_y

    model_index = int(record["rigid_model_index"])
    snapshot = {
        "instances_trans": rigid.instances_trans.detach().clone(),
        "instances_fv": rigid.instances_fv.detach().clone(),
    }
    if record["edit"] == "lateral":
        moved = move_actor_local_y(rigid, model_index, 1.0)
        if moved <= 0:
            raise RuntimeError("A3 lateral edit moved no valid trajectory frames")
    elif record["edit"] == "delete":
        with torch.no_grad():
            rigid.instances_fv[:, model_index] = False
    else:
        raise RuntimeError(f"unknown A3 edit: {record['edit']}")
    return snapshot


def restore_unit_edit(rigid: Any, snapshot: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        rigid.instances_trans.copy_(snapshot["instances_trans"])
        rigid.instances_fv.copy_(snapshot["instances_fv"])


def cgroup_memory_current() -> int | None:
    path = Path("/sys/fs/cgroup/memory.current")
    return int(path.read_text().strip()) if path.is_file() else None


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT / "configs/worldsim_v3/a3_local_refine_protocol_v1.yaml",
    )
    parser.add_argument("--sidecar-manifest", type=Path, default=DEFAULT_SIDECAR_MANIFEST)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_D2_WORK / "config.yaml")
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_D2_WORK / "checkpoint_final.pth",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_D2_RUN
        / "artifacts/evaluations/fixed-d2-boundary-residual/actor_registry.json",
    )
    parser.add_argument("--drivestudio-root", type=Path, default=DEFAULT_DRIVESTUDIO)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)

    from motion_proj.worldsim_v3.local_refinement import (
        load_a3_refinement_sidecar,
        validate_a3_protocol,
    )
    from scripts.materialize_worldsim_v3_a3_config import materialize_r1_config

    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_a3_protocol(protocol)
    manifest = json.loads(args.sidecar_manifest.read_text(encoding="utf-8"))
    units = ordered_unit_records(protocol, manifest)
    if len(units) != 4:
        raise RuntimeError("A3 first paired gate requires exactly four units")
    input_hashes = {
        "protocol": sha256_file(args.protocol),
        "source_config": sha256_file(args.source_config),
        "source_checkpoint": sha256_file(args.source_checkpoint),
        "registry": sha256_file(args.registry),
        "sidecar_manifest": sha256_file(args.sidecar_manifest),
    }
    dependencies = protocol["depends_on"]
    expected_hashes = {
        "protocol": manifest["protocol_sha256"],
        "source_config": dependencies["selected_checkpoint_config_sha256"],
        "source_checkpoint": dependencies["selected_checkpoint_sha256"],
        "registry": dependencies["selected_actor_registry_sha256"],
        "sidecar_manifest": input_hashes["sidecar_manifest"],
    }
    if input_hashes != expected_hashes:
        raise RuntimeError(f"A3 paired smoke immutable input drift: {input_hashes!r}")
    sidecar_root = args.sidecar_manifest.parent
    unit_arrays = [load_and_validate_unit(sidecar_root, record) for record in units]

    args.run_dir.mkdir(parents=True)
    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("artifacts", "logs", "source_snapshot"):
        (args.run_dir / name).mkdir()
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "failure": None},
    )
    started = time.monotonic()
    resolved_config = materialize_r1_config(
        OmegaConf.load(args.source_config),
        protocol,
        manifest,
        protocol_sha256=input_hashes["protocol"],
        source_config_sha256=input_hashes["source_config"],
        checkpoint_sha256=input_hashes["source_checkpoint"],
        sidecar_manifest_path=str(args.sidecar_manifest.resolve()),
        optimizer_steps=len(units),
    )
    resolved_path = args.run_dir / "resolved.yaml"
    OmegaConf.save(config=resolved_config, f=resolved_path)
    source_files = (
        PROJECT / "scripts/run_worldsim_v3_a3_s_b_paired_smoke.py",
        PROJECT / "scripts/materialize_worldsim_v3_a3_config.py",
        PROJECT / "motion_proj/worldsim_v3/local_refinement.py",
        PROJECT / "compatibility/DriveStudio-WorldSim-A3-R1-local-refine-v1.patch",
        args.protocol,
    )
    source_hashes = {}
    for source in source_files:
        relative = source.relative_to(PROJECT)
        destination = args.run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha256_file(source)
    project_status = command_output("git", "status", "--short", cwd=PROJECT).splitlines()
    run_manifest = {
        "schema_version": 1,
        "task_id": protocol["task_id"],
        "component": "A3 R1 conservative S-B/T0 paired engineering smoke",
        "formal_training_authorized": False,
        "numeric_protocol_frozen": False,
        "seed": int(protocol["paired_design"]["seed"]),
        "optimizer_steps": len(units),
        "unit_order": [f"{row['role']}::{row['edit']}" for row in units],
        "input_hashes": input_hashes,
        "source_hashes": source_hashes,
        "project_commit": command_output("git", "rev-parse", "HEAD", cwd=PROJECT),
        "project_status": project_status,
    }
    atomic_json(args.run_dir / "manifest.json", run_manifest)

    sys.path.insert(0, str(args.drivestudio_root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    device = torch.device(args.device)
    torch.manual_seed(int(protocol["paired_design"]["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(protocol["paired_design"]["seed"]))
        torch.cuda.reset_peak_memory_stats(device)
    dataset = DrivingDataset(data_cfg=resolved_config.data)
    trainer = import_str(resolved_config.trainer.type)(
        **resolved_config.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=resolved_config.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(str(args.source_checkpoint), load_only_model=True)
    if int(trainer.step) != SELECTED_CHECKPOINT_STEP:
        raise RuntimeError(f"A3 source checkpoint step drift: {trainer.step}")
    trainer.initialize_optimizer()
    guard = trainer.a3_local_refinement_guard
    if guard is None:
        raise RuntimeError("A3 local refinement guard was not initialized")
    loaded_sidecar = load_a3_refinement_sidecar(
        args.sidecar_manifest,
        protocol_sha256=input_hashes["protocol"],
        checkpoint_sha256=input_hashes["source_checkpoint"],
        device=device,
    )
    if not torch.equal(guard.mutable_rows, loaded_sidecar.mutable_rows):
        raise RuntimeError("A3 trainer/sidecar mutable rows drift")
    initial_layout = model_tensor_layout(trainer)
    initial_parameters = named_parameter_fingerprints(
        trainer,
        mutable_rows=guard.mutable_rows,
        mutable_fields=guard.mutable_fields,
    )
    rigid = trainer.models["RigidNodes"]
    initial_rigid_sha256 = rigid_contract_sha256(rigid)
    per_unit = []
    memory_samples = [cgroup_memory_current()]

    for offset, (record, arrays) in enumerate(zip(units, unit_arrays), start=1):
        step = SELECTED_CHECKPOINT_STEP + offset
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()
        image_infos, camera_infos = dataset.full_image_set.get_image(
            int(record["image_index"]), camera_downscale=1.0
        )
        dataset_depth = image_infos["lidar_depth_map"].detach().cpu().numpy()
        if not np.array_equal(dataset_depth, arrays["depth_lidar_measured"]):
            raise RuntimeError(
                f"A3 T0 dataset/sidecar depth drift: {record['role']}/{record['edit']}"
            )
        image_infos["lidar_depth_map"] = torch.from_numpy(
            arrays["depth_lidar_measured"]
        )
        image_infos["a3_paired_loss_authorized"] = torch.tensor(True)
        image_infos["a3_rgb_loss_mask"] = torch.from_numpy(arrays["rgb_loss_mask"])
        image_infos["a3_geometry_loss_mask"] = torch.from_numpy(
            arrays["geometry_loss_mask"]
        )
        image_infos = to_device(image_infos, device)
        camera_infos = to_device(camera_infos, device)
        edit_snapshot = apply_unit_edit(rigid, record)
        edited_rigid_sha256 = rigid_contract_sha256(rigid)
        try:
            outputs = trainer(image_infos, camera_infos)
            trainer.update_visibility_filter()
            loss_dict = trainer.compute_losses(outputs, image_infos, camera_infos)
            if set(loss_dict) - {
                "rgb_loss",
                "ssim_loss",
                "sky_loss_opacity",
                "depth_loss",
            }:
                raise RuntimeError(f"A3 unauthorized loss terms: {sorted(loss_dict)}")
            loss_values = {key: float(value.detach().item()) for key, value in loss_dict.items()}
            if not all(np.isfinite(value) for value in loss_values.values()):
                raise RuntimeError(f"A3 non-finite loss: {loss_values}")
            trainer.backward(loss_dict)
            trainer.postprocess_per_train_step(step=step)
        finally:
            restore_unit_edit(rigid, edit_snapshot)
        restored_rigid_sha256 = rigid_contract_sha256(rigid)
        if restored_rigid_sha256 != initial_rigid_sha256:
            raise RuntimeError("A3 target/non-target Rigid state failed exact restore")
        gradient_audit = trainer._a3_last_gradient_audit
        exactness_audit = trainer._a3_last_exactness_audit
        if not gradient_audit["pass"] or not exactness_audit["pass"]:
            raise RuntimeError("A3 per-unit guard audit failed")
        per_unit.append(
            {
                "step": step,
                "role": record["role"],
                "edit": record["edit"],
                "frame": int(record["frame"]),
                "camera": int(record["camera"]),
                "image_index": int(record["image_index"]),
                "geometry_pixels": int(arrays["geometry_loss_mask"].sum()),
                "unit_sha256": record["sha256"],
                "edited_rigid_sha256": edited_rigid_sha256,
                "restored_rigid_sha256": restored_rigid_sha256,
                "losses": loss_values,
                "learning_rates": {
                    group["name"]: float(group["lr"])
                    for group in trainer.optimizer.param_groups
                },
                "gradient_audit": gradient_audit,
                "exactness_audit": exactness_audit,
            }
        )
        memory_samples.append(cgroup_memory_current())

    final_layout = model_tensor_layout(trainer)
    final_parameters = named_parameter_fingerprints(
        trainer,
        mutable_rows=guard.mutable_rows,
        mutable_fields=guard.mutable_fields,
    )
    immutable_exact = initial_parameters["immutable"] == final_parameters["immutable"]
    outside_exact = (
        initial_parameters["mutable_outside"]
        == final_parameters["mutable_outside"]
    )
    changed_inside = {
        name: initial_parameters["mutable_inside"][name]
        != final_parameters["mutable_inside"][name]
        for name in sorted(guard.mutable_fields)
    }
    rigid_exact = rigid_contract_sha256(rigid) == initial_rigid_sha256
    layout_exact = initial_layout == final_layout
    all_step_exact = all(row["exactness_audit"]["pass"] for row in per_unit)
    if not (
        immutable_exact
        and outside_exact
        and rigid_exact
        and layout_exact
        and all_step_exact
        and any(changed_inside.values())
    ):
        raise RuntimeError("A3 final exactness/movement gate failed")

    checkpoint_dir = args.run_dir / "artifacts" / "r1_checkpoint"
    checkpoint_dir.mkdir()
    trainer.save_checkpoint(
        log_dir=str(checkpoint_dir), save_only_model=True, is_final=True
    )
    checkpoint_path = checkpoint_dir / "checkpoint_final.pth"
    duration = time.monotonic() - started
    summary = {
        "status": "done",
        "task_id": protocol["task_id"],
        "component": "A3 R1 conservative S-B/T0 paired engineering smoke",
        "formal_training_authorized": False,
        "numeric_protocol_frozen": False,
        "evidence_tier": "real_scene_paired_engineering_only_not_quality_evidence",
        "scene": protocol["paired_design"]["scene"],
        "seed": int(protocol["paired_design"]["seed"]),
        "optimizer_steps": len(units),
        "support": {
            "mode": manifest["evidence"]["support_mode"],
            "s_a_rgb_pixels": 0,
            "s_b_t0_geometry_pixels": sum(
                int(row["s_b_t0_geometry_pixels"]) for row in units
            ),
            "mutable_background_rows": int(guard.mutable_rows.sum().item()),
            "affected_background_rows": int(loaded_sidecar.affected_rows.sum().item()),
            "s_c_abstain_rows": int(
                (loaded_sidecar.affected_rows & ~loaded_sidecar.mutable_rows).sum().item()
            ),
            "heldout_excluded": True,
        },
        "per_unit": per_unit,
        "audits": {
            "immutable_parameters_exact": immutable_exact,
            "mutable_fields_outside_rows_exact": outside_exact,
            "mutable_fields_changed_inside": changed_inside,
            "rigid_registry_trajectory_actor_exact": rigid_exact,
            "checkpoint_tensor_shape_dtype_order_exact": layout_exact,
            "all_per_step_parameter_and_adam_audits_exact": all_step_exact,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "step": int(trainer.step),
        },
        "resources": {
            "wall_time_seconds": duration,
            "peak_gpu_memory_mib": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else None
            ),
            "peak_cgroup_memory_bytes_sampled": max(
                value for value in memory_samples if value is not None
            )
            if any(value is not None for value in memory_samples)
            else None,
        },
        "input_hashes": input_hashes,
        "resolved_config_sha256": sha256_file(resolved_path),
        "project_commit": run_manifest["project_commit"],
    }
    atomic_json(args.run_dir / "summary.json", summary)
    run_manifest["status"] = "done"
    run_manifest["checkpoint_sha256"] = summary["checkpoint"]["sha256"]
    run_manifest["summary_sha256"] = sha256_file(args.run_dir / "summary.json")
    atomic_json(args.run_dir / "manifest.json", run_manifest, replace=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "done", "failure": None},
        replace=True,
    )
    _TERMINAL_FINAL = True
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and not _TERMINAL_FINAL:
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "failure": {
                        "code": "A3_R1_S_B_PAIRED_SMOKE_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
