#!/usr/bin/env python
"""执行 A4-P5 单次只读 checkpoint reload 与 actor 索引审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf
import torch


PROJECT = Path("/root/autodl-tmp/motion_proj")
DRIVESTUDIO = Path("/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p5_registry_resume_protocol_v1.yaml"
_ACTIVE_RUN_DIR: Path | None = None


from scripts.eval_worldsim_v3_a3_r1_heldout import load_model_checkpoint_read_only
from scripts.run_worldsim_v3_a4_p0_profile import ResourceSampler
from scripts.run_worldsim_v3_a4_p5_registry import (
    atomic_json,
    cgroup_memory_events,
    nvidia_compute_rows,
    sha256_file,
    write_stage,
)
from scripts.validate_worldsim_v3_a4_p5_registry_resume_protocol import (
    validate_inputs,
    validate_schema,
)


def index_sha256(indices: Sequence[int]) -> str:
    encoded = json.dumps(
        [int(value) for value in indices], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_actor_indices(
    point_ids: Sequence[int], actors: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for actor in actors:
        model_index = int(actor["rigid_model_index"])
        indices = [
            index for index, point_id in enumerate(point_ids) if int(point_id) == model_index
        ]
        expected = actor["checkpoint_tensor_slice"]
        actual_hash = index_sha256(indices)
        expected_count = int(expected["gaussian_count"])
        expected_hash = expected["flat_indices_sha256"]
        rows.append(
            {
                "rigid_model_index": model_index,
                "instance_token": actor["instance_token"],
                "expected_availability": actor["availability"],
                "actual_availability": (
                    "available" if indices else "unavailable_empty_checkpoint_slice"
                ),
                "expected_gaussian_count": expected_count,
                "actual_gaussian_count": len(indices),
                "expected_flat_indices_sha256": expected_hash,
                "actual_flat_indices_sha256": actual_hash,
                "exact": len(indices) == expected_count and actual_hash == expected_hash,
            }
        )
    return rows


def runtime_rigid_point_ids(rigid: Any) -> list[int]:
    """读取 DriveStudio RigidNodes 的运行时 actor 索引。"""
    # checkpoint 键为 points_ids，但 load_state_dict 后运行时属性名是 point_ids。
    if not hasattr(rigid, "point_ids"):
        raise RuntimeError("A4-P5 RigidNodes runtime point_ids attribute missing")
    return rigid.point_ids[:, 0].detach().cpu().tolist()


def main() -> None:
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    _ACTIVE_RUN_DIR = args.run_dir
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    validate_inputs(protocol)
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("input_audit", "registry_materialize"):
        path = args.run_dir / "stages" / f"{name}.json"
        if sha256_file(path) != manifest["stage_hashes"][name]:
            raise RuntimeError(f"A4-P5 completed stage hash drift: {name}")
    gpu_rows = nvidia_compute_rows()
    if gpu_rows:
        raise RuntimeError(f"A4-P5 reload GPU preflight not idle: {gpu_rows}")
    sys.path.insert(0, str(DRIVESTUDIO))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    selected = protocol["selected_asset"]
    checkpoint = Path(selected["checkpoint"]["path"])
    actor_registry_path = Path(selected["actor_registry"]["path"])
    source_registry = json.loads(actor_registry_path.read_text(encoding="utf-8"))
    config = OmegaConf.load(selected["source_config"]["path"])
    device = torch.device(protocol["reload_smoke"]["device"])
    torch.cuda.set_device(device)
    torch.empty((), device=device)
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    events_before = cgroup_memory_events()
    with ResourceSampler(os.getpid()) as sampler:
        started = time.perf_counter()
        prepare_started = time.perf_counter()
        dataset = DrivingDataset(data_cfg=config.data)
        prepare_seconds = time.perf_counter() - prepare_started
        trainer_started = time.perf_counter()
        trainer = import_str(config.trainer.type)(
            **config.trainer,
            num_timesteps=dataset.num_img_timesteps,
            model_config=config.model,
            num_train_images=len(dataset.train_image_set),
            num_full_images=len(dataset.full_image_set),
            test_set_indices=dataset.test_timesteps,
            scene_aabb=dataset.get_aabb().reshape(2, 3),
            device=device,
        )
        trainer_construction_seconds = time.perf_counter() - trainer_started
        if hasattr(trainer, "optimizer"):
            raise RuntimeError("A4-P5 read-only reload constructed optimizer")
        torch.cuda.synchronize(device)
        load_started = time.perf_counter()
        load_model_checkpoint_read_only(trainer, checkpoint, device)
        torch.cuda.synchronize(device)
        checkpoint_load_seconds = time.perf_counter() - load_started
        trainer.set_eval()
        model_counts = {
            name: int(model._means.shape[0])
            for name, model in trainer.models.items()
            if hasattr(model, "_means")
        }
        rigid = trainer.models["RigidNodes"]
        point_ids = runtime_rigid_point_ids(rigid)
        actor_rows = audit_actor_indices(point_ids, source_registry["actors"])
        total_seconds = time.perf_counter() - started
        peak_allocated = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        peak_reserved = float(torch.cuda.max_memory_reserved(device) / (1024**2))
        no_optimizer = not hasattr(trainer, "optimizer")
        del trainer, dataset, rigid, point_ids
        torch.cuda.empty_cache()
    sampler_summary = sampler.summary()
    if (
        sampler_summary["sampling_errors"]
        or sampler_summary["peak_nvidia_process_memory_mib_sampled"] is None
        or sampler_summary["peak_cgroup_memory_bytes_sampled"] is None
    ):
        raise RuntimeError(f"A4-P5 resource sampling incomplete: {sampler_summary}")
    stage = {
        "status": "done",
        "stage": "reload_smoke",
        "dataset_prepare_seconds": prepare_seconds,
        "trainer_construction_seconds": trainer_construction_seconds,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "reload_total_seconds": total_seconds,
        "checkpoint_load_count": 1,
        "checkpoint_loader": "cpu_staged_load_model_checkpoint_read_only",
        "filesystem_cache": protocol["reload_smoke"]["filesystem_cache"],
        "render_count": 0,
        "model_gaussian_counts": model_counts,
        "actor_rows": actor_rows,
        "all_actor_indices_exact": all(row["exact"] for row in actor_rows),
        "unavailable_actor_remains_explicitly_empty": all(
            row["actual_gaussian_count"] == 0
            for row in actor_rows
            if row["expected_availability"] == "unavailable_empty_checkpoint_slice"
        ),
        "no_optimizer_constructed_or_step_executed": no_optimizer,
        "checkpoint_sha256_after_reload": sha256_file(checkpoint),
        "actor_registry_sha256_after_reload": sha256_file(actor_registry_path),
        "peak_torch_allocated_mib": peak_allocated,
        "peak_torch_reserved_mib": peak_reserved,
        "cgroup_memory_events_before_reload": events_before,
        "cgroup_memory_events_after_reload": cgroup_memory_events(),
        "minimum_rerun_unit": "reload_smoke_and_downstream",
        **sampler_summary,
    }
    write_stage(args.run_dir, manifest, "reload_smoke", stage)
    manifest["reload_complete"] = True
    atomic_json(manifest_path, manifest, replace=True)
    print(json.dumps({"status": "reload_complete", "run_dir": str(args.run_dir)}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.exists():
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "failure": {
                        "code": "A4_P5_RELOAD_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
