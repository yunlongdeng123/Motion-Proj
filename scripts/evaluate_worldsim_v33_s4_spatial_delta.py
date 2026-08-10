#!/usr/bin/env python3
"""用真实 DriveStudio renderer 验收 V3.3 S4 空间 delta 与精确回滚。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

_ACTIVE_RUN_DIR: Path | None = None

from motion_proj.worldsim_v33.roadpatch import load_patch_delta  # noqa: E402
from motion_proj.worldsim_v33.spatial_delta import (  # noqa: E402
    PACKAGE_SCHEMA_VERSION,
    atomic_json,
    load_actor_insert_delta,
    load_erase_delta,
    sha256_file,
    temporary_spatial_composition,
    validate_stack_manifest,
)
from scripts.lift_worldsim_v32_semantics import build_runtime  # noqa: E402
from scripts.run_worldsim_v32_s2_3dgic import render_snapshot  # noqa: E402


def verify(path: str | Path, expected: str, role: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{role} 不存在: {source}")
    actual = sha256_file(source)
    if actual != expected:
        raise RuntimeError(f"{role} SHA 漂移: expected={expected} actual={actual}")
    return {"path": str(source), "sha256": actual, "bytes": source.stat().st_size}


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def effect_mask(
    left: np.ndarray, right: np.ndarray, *, threshold: int, dilation: int
) -> np.ndarray:
    effect = np.max(
        np.abs(left.astype(np.int16) - right.astype(np.int16)), axis=2
    ) > int(threshold)
    return binary_dilation(effect, iterations=int(dilation)) if dilation else effect


def mean_l1(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        raise ValueError("L1 mask 为空")
    return float(
        np.abs(left.astype(np.float32) - right.astype(np.float32))[selected].mean()
    )


def cgroup_value(name: str) -> int | None:
    path = Path("/sys/fs/cgroup") / name
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    return None if raw == "max" else int(raw)


def memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.is_file():
        return {}
    return {
        key: int(value)
        for key, value in (
            line.split() for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def make_panel(path: Path, images: Mapping[str, np.ndarray], title: str) -> None:
    names = ["base_only", "erase", "erase_background", "actor_override", "full"]
    width, height = images[names[0]].shape[1], images[names[0]].shape[0]
    header = 30
    panel = Image.new("RGB", (width * len(names), height + header), "white")
    draw = ImageDraw.Draw(panel)
    for index, name in enumerate(names):
        panel.paste(Image.fromarray(images[name]), (index * width, header))
        draw.text((index * width + 5, 7), name, fill="black")
    draw.text((width * len(names) - 240, 7), title, fill="black")
    panel.save(path)


def load_package(
    manifest_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    verify(manifest_path, expected_sha256, "S4 package manifest")
    package = json.loads(manifest_path.read_text(encoding="utf-8"))
    if package.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise RuntimeError("S4 package schema version 漂移")
    root = manifest_path.parent
    for row in package["inventory"]:
        verify(root / row["path"], row["sha256"], f"package:{row['path']}")
    for stack in package["stacks"]:
        stack_path = root / stack["path"]
        verify(stack_path, stack["sha256"], f"stack:{stack['path']}")
        validate_stack_manifest(json.loads(stack_path.read_text(encoding="utf-8")))
    actor_id = int(package["actor"]["dataset_instance_id"])
    erase = load_erase_delta(
        root / f"deltas/delete_actor_{actor_id}/erase_indices.npz"
    )
    background = load_patch_delta(
        root / f"deltas/delete_actor_{actor_id}/background_patch.npz"
    )
    actor = load_actor_insert_delta(
        root / f"deltas/actor_override_{actor_id}/actor_override.npz"
    )
    return package, erase, background, actor


def render_rgb(
    trainer: Any, dataset: Any, *, frame: int, camera: int, device: torch.device
) -> np.ndarray:
    snapshot = render_snapshot(
        trainer=trainer,
        dataset=dataset,
        frame=frame,
        camera_id=camera,
        device=device,
    )
    return np.asarray(snapshot["rgb"], dtype=np.uint8)


def main() -> int:
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "worldsim_v33_s4_spatial_delta_v1":
        raise ValueError("S4 config schema version 漂移")
    run_dir = args.run_dir.resolve()
    _ACTIVE_RUN_DIR = run_dir
    if run_dir.exists():
        raise FileExistsError(run_dir)
    render_root = run_dir / "artifacts/renders"
    panel_root = run_dir / "artifacts/panels"
    render_root.mkdir(parents=True)
    panel_root.mkdir(parents=True)
    atomic_json(
        run_dir / "status.json",
        {
            "state": "running",
            "task_id": config["task_id"],
            "stage": "real_renderer_evaluation",
            "started_unix": started,
        },
    )
    before_events = memory_events()
    cgroup_samples = [cgroup_value("memory.current")]
    verified = {
        name: verify(spec["path"], spec["sha256"], name)
        for name, spec in config["inputs"].items()
    }
    package, erase, background, actor = load_package(
        args.package_manifest.resolve(), args.package_manifest_sha256
    )
    if package["base"]["checkpoint"]["sha256"] != verified["checkpoint"]["sha256"]:
        raise RuntimeError("package base checkpoint 与 S4 config 错配")
    if package["base"]["actor_registry"]["sha256"] != verified["actor_registry"]["sha256"]:
        raise RuntimeError("package actor registry 与 S4 config 错配")
    with np.load(config["inputs"]["target_mask"]["path"], allow_pickle=False) as payload:
        target_mask = payload["binary"].astype(bool)
    if not target_mask.any():
        raise RuntimeError("S4 target mask 为空")
    if not torch.cuda.is_available():
        raise RuntimeError("S4 real renderer evaluation 需要 CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    runtime_config = dict(config)
    runtime_config["inputs"] = {
        name: spec["path"] for name, spec in config["inputs"].items()
    }
    dataset, trainer = build_runtime(runtime_config, device)
    if hasattr(trainer, "optimizer"):
        raise RuntimeError("S4 只读 evaluator 禁止 optimizer")
    trainer.set_eval()
    cgroup_samples.append(cgroup_value("memory.current"))

    evaluation = config["evaluation"]
    views = [
        tuple(int(value) for value in evaluation["edit_target_view"]),
        *[
            tuple(int(value) for value in row)
            for row in evaluation["development_views"]
        ],
        *[
            tuple(int(value) for value in row)
            for row in evaluation["heldout_confirmation_views"]
        ],
    ]
    if len(set(views)) != len(views):
        raise RuntimeError("S4 evaluation views 重复")
    threshold = int(evaluation["effect_threshold_uint8"])
    dilation = int(evaluation["effect_dilation_pixels"])
    rows: list[dict[str, Any]] = []
    rollback_checks: list[dict[str, Any]] = []
    full_target_sha: str | None = None
    target_metrics: dict[str, Any] | None = None

    stacks = {
        "erase": (None, None),
        "erase_background": (background, None),
        "actor_override": (None, actor),
        "full": (background, actor),
    }
    for frame, camera in views:
        view_dir = render_root / f"f{frame:03d}_c{camera}"
        view_dir.mkdir()
        base = render_rgb(
            trainer, dataset, frame=frame, camera=camera, device=device
        )
        base_sha = array_sha256(base)
        images: dict[str, np.ndarray] = {"base_only": base}
        imageio.imwrite(view_dir / "base_only.png", base)
        audits: dict[str, Any] = {}
        for stack_name, (background_value, actor_value) in stacks.items():
            with temporary_spatial_composition(
                trainer.models,
                erase_delta=erase,
                background_delta=background_value,
                actor_delta=actor_value,
            ) as audit:
                image = render_rgb(
                    trainer, dataset, frame=frame, camera=camera, device=device
                )
                audits[stack_name] = audit
            rollback = render_rgb(
                trainer, dataset, frame=frame, camera=camera, device=device
            )
            rollback_sha = array_sha256(rollback)
            exact = bool(np.array_equal(rollback, base) and rollback_sha == base_sha)
            rollback_checks.append(
                {
                    "frame": frame,
                    "camera_id": camera,
                    "stack": stack_name,
                    "base_render_sha256": base_sha,
                    "rollback_render_sha256": rollback_sha,
                    "exact": exact,
                }
            )
            if not exact:
                raise RuntimeError(
                    f"S4 rollback render 不精确: f{frame} c{camera} {stack_name}"
                )
            images[stack_name] = image
            imageio.imwrite(view_dir / f"{stack_name}.png", image)
            cgroup_samples.append(cgroup_value("memory.current"))

        erase_effect = effect_mask(
            base, images["erase"], threshold=threshold, dilation=dilation
        )
        background_effect = effect_mask(
            images["erase"],
            images["erase_background"],
            threshold=threshold,
            dilation=dilation,
        )
        actor_effect = effect_mask(
            images["erase"],
            images["actor_override"],
            threshold=threshold,
            dilation=dilation,
        )
        full_effect = effect_mask(
            base, images["full"], threshold=threshold, dilation=dilation
        )
        row = {
            "frame": frame,
            "camera_id": camera,
            "base_render_sha256": base_sha,
            "render_sha256": {
                name: array_sha256(image) for name, image in images.items()
            },
            "effect_pixels": {
                "erase": int(erase_effect.sum()),
                "insert_background": int(background_effect.sum()),
                "insert_actor": int(actor_effect.sum()),
                "full": int(full_effect.sum()),
            },
            "audits": audits,
        }
        rows.append(row)
        if (frame, camera) == tuple(evaluation["edit_target_view"]):
            outside = ~binary_dilation(target_mask, iterations=8)
            target_metrics = {
                "erase_effect_pixels": int(erase_effect.sum()),
                "background_effect_pixels": int(background_effect.sum()),
                "actor_effect_pixels": int(actor_effect.sum()),
                "full_effect_pixels": int(full_effect.sum()),
                "erase_mask_coverage": float(
                    np.sum(erase_effect & target_mask) / max(int(target_mask.sum()), 1)
                ),
                "actor_mask_coverage": float(
                    np.sum(actor_effect & target_mask) / max(int(target_mask.sum()), 1)
                ),
                "full_outside_target_l1_uint8": mean_l1(
                    base, images["full"], outside
                ),
            }
            full_target_sha = array_sha256(images["full"])
        make_panel(
            panel_root / f"f{frame:03d}_c{camera}.png",
            images,
            f"f{frame:03d} c{camera}",
        )
        print(
            f"S4 f{frame:03d} c{camera}: "
            f"erase={row['effect_pixels']['erase']} "
            f"bg={row['effect_pixels']['insert_background']} "
            f"actor={row['effect_pixels']['insert_actor']}",
            flush=True,
        )

    if target_metrics is None or full_target_sha is None:
        raise RuntimeError("S4 edit target view 未执行")
    target_frame, target_camera = (
        int(value) for value in evaluation["edit_target_view"]
    )
    base_replay = render_rgb(
        trainer,
        dataset,
        frame=target_frame,
        camera=target_camera,
        device=device,
    )
    base_replay_sha = array_sha256(base_replay)
    with temporary_spatial_composition(
        trainer.models,
        erase_delta=erase,
        background_delta=background,
        actor_delta=actor,
    ):
        full_replay = render_rgb(
            trainer,
            dataset,
            frame=target_frame,
            camera=target_camera,
            device=device,
        )
    rollback_replay = render_rgb(
        trainer,
        dataset,
        frame=target_frame,
        camera=target_camera,
        device=device,
    )
    deterministic_replay = array_sha256(full_replay) == full_target_sha
    exact_replay_rollback = bool(
        np.array_equal(rollback_replay, base_replay)
        and array_sha256(rollback_replay) == base_replay_sha
    )

    gates_cfg = config["gates"]
    aggregate_effect = {
        name: int(sum(row["effect_pixels"][name] for row in rows))
        for name in ("erase", "insert_background", "insert_actor", "full")
    }
    functional_gates = {
        "all_stack_rollbacks_render_exact": all(
            row["exact"] for row in rollback_checks
        ),
        "full_stack_deterministic_replay": deterministic_replay,
        "replay_rollback_render_exact": exact_replay_rollback,
        "erase_independently_effective": aggregate_effect["erase"] > 0,
        "background_independently_effective": aggregate_effect["insert_background"] > 0,
        "actor_independently_effective": aggregate_effect["insert_actor"] > 0,
        "edit_target_erase_effect": target_metrics["erase_effect_pixels"]
        >= int(gates_cfg["minimum_edit_target_erase_effect_pixels"]),
        "edit_target_background_effect": target_metrics["background_effect_pixels"]
        >= int(gates_cfg["minimum_edit_target_background_effect_pixels"]),
        "edit_target_actor_effect": target_metrics["actor_effect_pixels"]
        >= int(gates_cfg["minimum_edit_target_actor_effect_pixels"]),
        "edit_target_mask_coverage": min(
            target_metrics["erase_mask_coverage"],
            target_metrics["actor_mask_coverage"],
        )
        >= float(gates_cfg["minimum_edit_target_mask_coverage"]),
        "outside_target_preserved": target_metrics["full_outside_target_l1_uint8"]
        <= float(gates_cfg["maximum_outside_target_l1_uint8"]),
        "base_rows_never_deleted": all(
            audit["base_rows_deleted"] == 0
            for row in rows
            for audit in row["audits"].values()
        ),
        "erased_opacity_exact_zero": all(
            audit["effective_erased_opacity_nonzero"] == 0
            for row in rows
            for audit in row["audits"].values()
        ),
        "duplicate_insert_indices_zero": all(
            audit["duplicate_insert_indices"] == 0
            for row in rows
            for audit in row["audits"].values()
        ),
    }
    checkpoint_after = sha256_file(config["inputs"]["checkpoint"]["path"])
    registry_after = sha256_file(config["inputs"]["actor_registry"]["path"])
    functional_gates["checkpoint_immutable"] = (
        checkpoint_after == verified["checkpoint"]["sha256"]
    )
    functional_gates["actor_registry_immutable"] = (
        registry_after == verified["actor_registry"]["sha256"]
    )
    torch.cuda.synchronize(device)
    elapsed = time.time() - started
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)
    after_events = memory_events()
    oom_delta = int(after_events.get("oom", 0) - before_events.get("oom", 0))
    oom_kill_delta = int(
        after_events.get("oom_kill", 0) - before_events.get("oom_kill", 0)
    )
    run_bytes = directory_bytes(run_dir)
    resource_gates = {
        "peak_cuda_reserved_mib": peak_reserved_mib
        <= float(gates_cfg["maximum_peak_cuda_reserved_mib"]),
        "wall_seconds": elapsed <= float(gates_cfg["maximum_wall_seconds"]),
        "run_bytes": run_bytes <= int(gates_cfg["maximum_run_bytes"]),
        "oom_events_delta": oom_delta
        <= int(gates_cfg["maximum_oom_events_delta"]),
        "oom_kill_events_delta": oom_kill_delta
        <= int(gates_cfg["maximum_oom_kill_events_delta"]),
    }
    accepted = all(functional_gates.values()) and all(resource_gates.values())

    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir()
    for source in (
        args.config.resolve(),
        Path(__file__).resolve(),
        PROJECT / "motion_proj/worldsim_v33/spatial_delta.py",
    ):
        shutil.copy2(source, source_snapshot / source.name)
    atomic_json(run_dir / "artifacts/metrics.json", rows)
    decision = {
        "accepted": accepted,
        "functional_gates": functional_gates,
        "resource_gates": resource_gates,
        "target_metrics": target_metrics,
        "aggregate_effect_pixels": aggregate_effect,
        "deterministic_full_replay": {
            "first_sha256": full_target_sha,
            "replay_sha256": array_sha256(full_replay),
            "exact": deterministic_replay,
        },
    }
    decision_path = run_dir / "artifacts/decision.json"
    atomic_json(decision_path, decision)
    summary = {
        "schema_version": "worldsim_v33_s4_spatial_delta_evaluation_v1",
        "task_id": config["task_id"],
        "state": "completed" if accepted else "rejected",
        "views": [[frame, camera] for frame, camera in views],
        "heldout_used_for_selection": False,
        "optimization_performed": False,
        "package_manifest": verify(
            args.package_manifest,
            args.package_manifest_sha256,
            "S4 package manifest final",
        ),
        "verified_inputs": verified,
        "rows": rows,
        "rollback_checks": rollback_checks,
        "decision": decision,
        "decision_sha256": sha256_file(decision_path),
        "resources": {
            "elapsed_seconds": elapsed,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_cuda_reserved_mib": peak_reserved_mib,
            "cgroup_memory_samples_bytes": cgroup_samples,
            "run_bytes_before_summary": run_bytes,
            "oom_events_delta": oom_delta,
            "oom_kill_events_delta": oom_kill_delta,
        },
        "checkpoint_sha256_before": verified["checkpoint"]["sha256"],
        "checkpoint_sha256_after": checkpoint_after,
        "actor_registry_sha256_before": verified["actor_registry"]["sha256"],
        "actor_registry_sha256_after": registry_after,
        "source_snapshot": {
            path.name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(source_snapshot.iterdir())
        },
    }
    summary_path = run_dir / "summary.json"
    atomic_json(summary_path, summary)
    atomic_json(
        run_dir / "status.json",
        {
            "state": summary["state"],
            "task_id": config["task_id"],
            "stage": "real_renderer_evaluation",
            "summary_sha256": sha256_file(summary_path),
            "decision_sha256": sha256_file(decision_path),
            "completed_unix": time.time(),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.is_dir():
            atomic_json(
                _ACTIVE_RUN_DIR / "status.json",
                {
                    "state": "failed",
                    "stage": "real_renderer_evaluation",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_unix": time.time(),
                },
            )
        raise
