#!/usr/bin/env python3
"""Freeze Gaussian Grouping sources and audit F0 train-only adapter prerequisites."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
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


SCHEMA = "worldsim_v51_stage_f_f0_source_preflight_v1"
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
SCENES = ("scene-0471", "scene-1087", "scene-0379")
IDENTITY_FIELD_MARKERS = ("instance", "identity", "object_id", "mask_id", "class_id")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"YAML root must be a mapping: {path}")
    return payload


def _verify(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ProtocolError(f"identity drift: {label}: {path}")
    return path


def _assert_tokens(path: Path, tokens: Sequence[str], label: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [token for token in tokens if token not in text]
    if missing:
        raise ProtocolError(f"upstream semantic token drift: {label}: {missing}")


def _git_at(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def validate_config(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != SCHEMA:
        raise ProtocolError("F0 source preflight schema drift")
    if config.get("task_id") != TASK_ID or config.get("status") != "running":
        raise ProtocolError("F0 source preflight task/status drift")
    if config.get("phase") != "f0_faithful_source_and_frozen_base_adapter_preflight":
        raise ProtocolError("F0 source preflight phase drift")
    if int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("F0 source preflight seed drift")

    authorization = config["authorization"]
    plan_spec = authorization["normative_plan"]
    _verify(PROJECT / plan_spec["path"], plan_spec["sha256"], "normative plan")
    freeze_spec = authorization["stage_e_rejection_freeze"]
    freeze_path = _verify(
        PROJECT / freeze_spec["path"], freeze_spec["sha256"], "Stage E rejection freeze"
    )
    freeze = _load_yaml(freeze_path)
    canonical = freeze["canonical_run"]
    governance = freeze["governance"]
    if freeze.get("status") != freeze_spec["required_status"]:
        raise ProtocolError("Stage E rejection status drift")
    if canonical.get("conclusion") != freeze_spec["required_conclusion"]:
        raise ProtocolError("Stage E rejection conclusion drift")
    if governance.get("next_task") != freeze_spec["required_next_task"]:
        raise ProtocolError("Stage E route did not unlock normative Stage F task")

    paper = config["official_source"]["paper"]
    paper_path = _verify(Path(paper["path"]), paper["sha256"], "official paper")
    if paper_path.stat().st_size != int(paper["bytes"]):
        raise ProtocolError("official paper byte count drift")
    repository = config["official_source"]["repository"]
    repository_path = Path(repository["path"])
    if _git_at(repository_path, "rev-parse", "HEAD") != repository["commit"]:
        raise ProtocolError("official repository commit drift")
    if _git_at(repository_path, "rev-parse", "HEAD^{tree}") != repository["tree"]:
        raise ProtocolError("official repository tree drift")
    if _git_at(repository_path, "status", "--porcelain"):
        raise ProtocolError("official repository is not clean")
    for relative, expected in repository["files"].items():
        _verify(repository_path / relative, expected, f"official source/{relative}")
    license_text = (repository_path / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ProtocolError("official repository license declaration drift")

    _assert_tokens(
        repository_path / "train.py",
        (
            "torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)",
            "torch.nn.CrossEntropyLoss(reduction='none')",
            "loss_obj = loss_obj / torch.log(torch.tensor(num_classes))",
            "loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d",
            "torch.optim.Adam(classifier.parameters(), lr=5e-4)",
        ),
        "train.py",
    )
    _assert_tokens(
        repository_path / "scene/gaussian_model.py",
        ("self.num_objects = 16", "self._objects_dc = nn.Parameter", '"name": "obj_dc"'),
        "scene/gaussian_model.py",
    )
    _assert_tokens(
        repository_path / "utils/loss_utils.py",
        (
            "torch.cdist(sample_features, features)",
            "dists.topk(k, largest=False)",
            "return lambda_val * normalized_loss",
        ),
        "utils/loss_utils.py",
    )
    _assert_tokens(
        repository_path / "script/prepare_pseudo_label.sh",
        (
            "demo/demo_automatic.py",
            "--temporal_setting semionline",
            "--use_short_id",
            "--SAM_PRED_IOU_THRESHOLD 0.7",
        ),
        "script/prepare_pseudo_label.sh",
    )

    mechanism = config["faithful_mechanism"]
    expected_mechanism = {
        "mask_generation": "sam_everything_mode",
        "mask_association": "deva_semionline_cross_view_consistent_short_ids",
        "identity_encoding_dimension": 16,
        "identity_sh_degree": 0,
        "differentiable_rendering": "alpha_composited_identity_encoding",
        "classifier": "shared_linear_1x1_identity_to_scene_mask_count",
        "identity_loss": "pixel_cross_entropy_divided_by_log_scene_class_count",
    }
    for name, expected in expected_mechanism.items():
        if mechanism.get(name) != expected:
            raise ProtocolError(f"faithful mechanism drift: {name}")
    regularization = mechanism["spatial_regularization"]
    expected_regularization = {
        "center_metric": "euclidean",
        "neighbor_count": 5,
        "sampled_points": 1000,
        "maximum_candidate_points": 300000,
        "interval_steps": 2,
        "divergence": "forward_kl_sample_to_neighbor",
        "includes_self_as_code_topk_semantics": True,
        "weight": 2.0,
    }
    if regularization != expected_regularization:
        raise ProtocolError("faithful 3D regularization contract drift")
    adaptation = config["frozen_base_adaptation"]
    for lock in (
        "freeze_geometry",
        "freeze_appearance",
        "freeze_opacity",
        "freeze_dynamic_actor_poses",
        "learn_identity_encoding_only",
        "learn_shared_classifier_only",
        "preserve_upstream_mask_association",
        "preserve_upstream_identity_dimension",
        "preserve_upstream_2d_loss",
        "preserve_upstream_3d_regularization",
    ):
        if adaptation.get(lock) is not True:
            raise ProtocolError(f"frozen-base faithful adaptation drift: {lock}")
    forbidden = set(adaptation["forbidden"])
    required_forbidden = {
        "bayesian_initialization",
        "dino_graph",
        "anchor",
        "unknown_innovation",
        "binary_actor_union_as_instance_id_substitute",
        "evaluation_target_as_training_label",
        "geometry_or_appearance_finetuning",
    }
    if forbidden != required_forbidden:
        raise ProtocolError("F0 forbidden-mechanism set drift")
    return config


def summarize_instance_metadata(
    instances_info: Mapping[str, Any],
    frame_instances: Mapping[str, Sequence[int]],
    train_frames: Sequence[int],
) -> dict[str, Any]:
    active_by_frame = {
        str(frame): sorted({int(value) for value in frame_instances.get(str(frame), [])})
        for frame in train_frames
    }
    active_union = sorted({value for values in active_by_frame.values() for value in values})
    appearances = Counter(value for values in active_by_frame.values() for value in values)
    missing = [value for value in active_union if str(value) not in instances_info]
    if missing:
        raise ProtocolError(f"frame_instances references missing instance metadata: {missing[:8]}")
    stable_tokens = [str(instances_info[str(value)].get("id", "")) for value in active_union]
    if any(not value for value in stable_tokens) or len(set(stable_tokens)) != len(stable_tokens):
        raise ProtocolError("instance metadata stable token contract drift")
    classes = Counter(
        str(instances_info[str(value)].get("class_name", "")) for value in active_union
    )
    return {
        "metadata_instance_count": len(instances_info),
        "train_frame_active_counts": {
            frame: len(values) for frame, values in active_by_frame.items()
        },
        "train_frame_active_union_count": len(active_union),
        "train_frame_repeated_instance_count": sum(count >= 2 for count in appearances.values()),
        "stable_instance_tokens_present_unique": True,
        "active_class_counts": dict(sorted(classes.items())),
    }


def observation_schema_report(paths: Sequence[Path]) -> dict[str, Any]:
    field_sets = []
    for path in paths:
        if not path.is_file():
            raise ProtocolError(f"train-only observation missing: {path}")
        with np.load(path, allow_pickle=False) as table:
            field_sets.append(tuple(sorted(table.files)))
    unique = sorted({fields for fields in field_sets})
    all_fields = sorted({field for fields in unique for field in fields})
    identity_fields = sorted(
        field
        for field in all_fields
        if any(marker in field.lower() for marker in IDENTITY_FIELD_MARKERS)
    )
    return {
        "view_count": len(paths),
        "unique_schema_count": len(unique),
        "schemas": [list(fields) for fields in unique],
        "identity_label_fields": identity_fields,
        "associated_instance_identity_labels_present": bool(identity_fields),
    }


def differentiable_identity_render_smoke(device: str, channels: int, image_size: Sequence[int]) -> dict[str, Any]:
    from gsplat import rasterization

    torch.manual_seed(20260814)
    target = torch.device(device)
    means = torch.tensor(
        [[-0.25, -0.10, 3.0], [0.20, -0.10, 3.1], [-0.05, 0.25, 3.2]],
        dtype=torch.float32,
        device=target,
    )
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 3, dtype=torch.float32, device=target
    )
    scales = torch.full((3, 3), 0.12, dtype=torch.float32, device=target)
    opacities = torch.full((3,), 0.8, dtype=torch.float32, device=target)
    identity = torch.nn.Parameter(
        torch.linspace(-0.5, 0.5, 3 * channels, device=target).reshape(3, channels)
    )
    viewmats = torch.eye(4, dtype=torch.float32, device=target)[None]
    height, width = (int(image_size[0]), int(image_size[1]))
    intrinsics = torch.tensor(
        [[40.0, 0.0, width / 2.0], [0.0, 40.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=target,
    )[None]
    renders, alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=identity,
        viewmats=viewmats,
        Ks=intrinsics,
        width=width,
        height=height,
        near_plane=0.01,
        far_plane=100.0,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        render_mode="RGB",
        rasterize_mode="classic",
    )
    if tuple(renders.shape) != (1, height, width, channels):
        raise ProtocolError(f"16D identity render shape drift: {tuple(renders.shape)}")
    if not bool((alphas > 0).any()):
        raise ProtocolError("identity renderer smoke has no visible alpha")
    loss = renders.square().mean()
    loss.backward()
    gradient = identity.grad
    if gradient is None or not torch.isfinite(gradient).all() or not bool((gradient != 0).any()):
        raise ProtocolError("identity renderer did not produce finite nonzero gradient")
    if any(value.requires_grad or value.grad is not None for value in (means, quats, scales, opacities)):
        raise ProtocolError("identity renderer smoke mutated frozen-base gradient boundary")
    return {
        "render_shape": list(renders.shape),
        "alpha_positive_pixel_count": int(torch.count_nonzero(alphas > 0).item()),
        "identity_gradient_finite": True,
        "identity_gradient_nonzero_count": int(torch.count_nonzero(gradient).item()),
        "base_geometry_requires_grad": False,
        "base_geometry_gradients_absent": True,
        "loss": float(loss.detach().cpu()),
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v51_f0_source_preflight_status_v1",
            "task_id": TASK_ID,
            "status": "running",
            "source_commit": source_commit,
        },
    )
    monitor = ResourceMonitor(float(config["resources"]["monitor_interval_seconds"]))
    started = time.perf_counter()
    nvidia_start = _nvidia_used_mib()
    if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
        raise ProtocolError(f"unexpected GPU use at start: {nvidia_start} MiB")
    monitor.start()
    try:
        source = config["official_source"]
        source_report = {
            "paper": {
                "path": source["paper"]["path"],
                "bytes": int(source["paper"]["bytes"]),
                "sha256": source["paper"]["sha256"],
                "page_count_verified_by_local_pdf_read": int(source["paper"]["page_count"]),
            },
            "repository": {
                "path": source["repository"]["path"],
                "commit": source["repository"]["commit"],
                "tree": source["repository"]["tree"],
                "clean": True,
                "license": source["repository"]["license"],
                "file_count": len(source["repository"]["files"]),
                "file_hashes_exact": True,
                "semantic_tokens_exact": True,
            },
            "source_audit_pass": True,
        }

        contract = config["train_only_contract"]
        train_frames = [int(value) for value in contract["frames"]]
        cameras = [int(value) for value in contract["cameras"]]
        scene_reports = []
        for scene in contract["scenes"]:
            graph_path = _verify(
                PROJECT / scene["graph_config"]["path"],
                scene["graph_config"]["sha256"],
                f"{scene['scene']}/graph config",
            )
            graph = _load_yaml(graph_path)
            if graph["scene"]["name"] != scene["scene"] or int(graph["scene"]["index"]) != int(
                scene["index"]
            ):
                raise ProtocolError(f"graph scene identity drift: {scene['scene']}")
            unary_manifest = graph["inputs"]["unary_manifest"]
            manifest_path = _verify(
                Path(unary_manifest["path"]),
                unary_manifest["sha256"],
                f"{scene['scene']}/unary manifest",
            )
            processed = Path(scene["processed_scene"])
            instances_path = _verify(
                processed / "instances/instances_info.json",
                scene["instances_info_sha256"],
                f"{scene['scene']}/instances_info",
            )
            frame_instances_path = _verify(
                processed / "instances/frame_instances.json",
                scene["frame_instances_sha256"],
                f"{scene['scene']}/frame_instances",
            )
            metadata = summarize_instance_metadata(
                _read_json(instances_path), _read_json(frame_instances_path), train_frames
            )
            image_paths = []
            observation_paths = []
            for frame in train_frames:
                for camera in cameras:
                    matches = sorted((processed / "images").glob(f"{frame:03d}_{camera}.*"))
                    if len(matches) != 1:
                        raise ProtocolError(
                            f"train image denominator drift: {scene['scene']}/{frame}/{camera}"
                        )
                    image_paths.append(matches[0])
                    observation_paths.append(
                        manifest_path.parent
                        / f"artifacts/observations/f{frame:03d}_c{camera}.npz"
                    )
            observation = observation_schema_report(observation_paths)
            if len(image_paths) != int(contract["expected_views_per_scene"]):
                raise ProtocolError(f"train view count drift: {scene['scene']}")
            if observation["view_count"] != int(contract["expected_views_per_scene"]):
                raise ProtocolError(f"observation view count drift: {scene['scene']}")
            checkpoint = graph["inputs"]["formal_checkpoint"]
            checkpoint_path = _verify(
                Path(checkpoint["path"]), checkpoint["sha256"], f"{scene['scene']}/checkpoint"
            )
            scene_reports.append(
                {
                    "scene": scene["scene"],
                    "index": int(scene["index"]),
                    "train_image_count": len(image_paths),
                    "train_image_pixels_read": False,
                    "train_mask_pixels_read": False,
                    "checkpoint": {
                        "bytes": checkpoint_path.stat().st_size,
                        "sha256": checkpoint["sha256"],
                        "loaded": False,
                    },
                    "metadata": metadata,
                    "observation_schema": observation,
                }
            )

        existing = config["asset_probe"]["existing_non_substitute"]["sam2_hiera_large"]
        _verify(Path(existing["path"]), existing["sha256"], "existing SAM2 non-substitute")
        upstream_assets = {
            name: {
                "path": spec["path"],
                "present": Path(spec["path"]).is_file(),
                "expected_present_before_preflight": bool(
                    spec["expected_present_before_preflight"]
                ),
            }
            for name, spec in config["asset_probe"]["upstream_required"].items()
        }
        for name, row in upstream_assets.items():
            if row["present"] != row["expected_present_before_preflight"]:
                raise ProtocolError(f"upstream asset preflight expectation drift: {name}")
        training_input_ready = all(
            report["observation_schema"]["associated_instance_identity_labels_present"]
            for report in scene_reports
        )
        if training_input_ready:
            raise ProtocolError("expected current binary evidence gap disappeared before preregistration")

        torch.cuda.reset_peak_memory_stats(torch.device(config["adapter_probe"]["device"]))
        adapter = differentiable_identity_render_smoke(
            config["adapter_probe"]["device"],
            int(config["adapter_probe"]["identity_channels"]),
            config["adapter_probe"]["image_size"],
        )
        conclusion = config["decision"]["expected_preflight_conclusion"]
        report = {
            "schema_version": "worldsim_v51_f0_source_preflight_report_v1",
            "task_id": TASK_ID,
            "status": "done",
            "conclusion": conclusion,
            "source": source_report,
            "faithful_mechanism": config["faithful_mechanism"],
            "frozen_base_adaptation": config["frozen_base_adaptation"],
            "scenes": scene_reports,
            "adapter": adapter,
            "upstream_assets": upstream_assets,
            "existing_sam2_is_faithful_substitute": False,
            "source_audit_pass": True,
            "adapter_capability_pass": True,
            "current_training_input_ready": False,
            "identity_training_authorized": False,
            "next_action": config["decision"]["next_action"],
            "quality_read": False,
            "parameter_search": False,
            "deva_execution": False,
            "sam_execution": False,
            "m2_status": "pending",
            "m3_status": "pending",
        }
        _write_json(run_dir / "artifacts/source_preflight_report.json", report)
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid_samples = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid_samples:
            raise ProtocolError("F0 source preflight resource monitor produced no valid sample")
        resources = {
            "nvidia_start_mib": nvidia_start,
            "nvidia_peak_mib": max(int(row["gpu_used_mib"]) for row in valid_samples),
            "torch_allocated_peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "torch_reserved_peak_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "cgroup_memory_peak_bytes": max(
                int(row["cgroup_memory_current_bytes"]) for row in valid_samples
            ),
            "sample_count": len(monitor.samples),
            "monitor_error_count": len(monitor.samples) - len(valid_samples),
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json(run_dir / "artifacts/resources.json", resources)
        ceilings = config["resources"]
        resource_checks = {
            "nvidia_peak": resources["nvidia_peak_mib"]
            <= int(ceilings["maximum_nvidia_peak_mib"]),
            "torch_allocated_peak": resources["torch_allocated_peak_mib"]
            <= float(ceilings["maximum_torch_allocated_peak_mib"]),
            "torch_reserved_peak": resources["torch_reserved_peak_mib"]
            <= float(ceilings["maximum_torch_reserved_peak_mib"]),
            "cgroup_memory_peak": resources["cgroup_memory_peak_bytes"]
            <= int(ceilings["maximum_cgroup_memory_bytes"]),
            "wall": resources["wall_seconds"] <= float(ceilings["maximum_wall_seconds"]),
            "monitor": resources["monitor_error_count"] == 0,
        }
        if not all(resource_checks.values()):
            raise ProtocolError(f"F0 source preflight resource gate failed: {resource_checks}")
        summary = {
            **report,
            "schema_version": "worldsim_v51_f0_source_preflight_summary_v1",
            "source_commit": source_commit,
            "source_tree": source_tree,
            "resources": resources,
            "resource_checks": resource_checks,
            "h_quality_read": False,
            "screening_quality_read": False,
            "confirmation_quality_read": False,
            "validation_quality_read": False,
            "test_quality_read": False,
            "f1_execution": False,
            "f2_execution": False,
        }
        _write_json(run_dir / "summary.json", summary)
        events.append({"event": "run_completed", "at_utc": _utc_now(), "status": "done"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "manifest.json",
            {
                "schema_version": "worldsim_v51_f0_source_preflight_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_f0_source_preflight_status_v1",
                "task_id": TASK_ID,
                "status": "done",
                "conclusion": conclusion,
                "source_commit": source_commit,
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
                "schema_version": "worldsim_v51_f0_source_preflight_status_v1",
                "task_id": TASK_ID,
                "status": "blocked",
                "error": f"{type(error).__name__}: {error}",
                "source_commit": source_commit,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_f_f0_source_preflight_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
