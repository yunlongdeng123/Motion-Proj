#!/usr/bin/env python3
"""独立重放并审计 Stage F F0 r025 源与适配器预检。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
TASK_ID = "WS-V51-M1-F-IDENTITY-EMBEDDING-01"
CONFIG_SCHEMA = "worldsim_v51_stage_f_f0_source_preflight_v1"
CONCLUSION = "f0_source_adapter_preflight_done_input_materialization_required"
IDENTITY_FIELD_MARKERS = ("instance", "identity", "object_id", "mask_id", "class_id")


class AuditError(RuntimeError):
    """r025 冻结证据不再满足登记合同时抛出。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"YAML root must be a mapping: {path}")
    return payload


def _verify(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise AuditError(f"identity drift: {label}: {path}")
    return path


def _assert_payload(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AuditError(f"payload field drift: {label}")
        for key, value in expected.items():
            _assert_payload(actual[key], value, f"{label}/{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AuditError(f"payload list drift: {label}")
        for index, value in enumerate(expected):
            _assert_payload(actual[index], value, f"{label}/{index}")
        return
    if isinstance(expected, float):
        if not np.isclose(float(actual), expected, rtol=0.0, atol=1e-12):
            raise AuditError(f"payload float drift: {label}: {actual} != {expected}")
        return
    if actual != expected:
        raise AuditError(f"payload value drift: {label}: {actual!r} != {expected!r}")


def _verify_inventory(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, int]:
    expected = {row["path"]: row for row in manifest["inventory"]}
    observed = {
        path.relative_to(run_dir).as_posix(): path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "status.json"}
    }
    if set(observed) != set(expected):
        raise AuditError("run manifest inventory coverage drift")
    total_bytes = 0
    for relative, path in observed.items():
        record = expected[relative]
        size = path.stat().st_size
        if size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise AuditError(f"run manifest identity drift: {relative}")
        total_bytes += size
    return {"entry_count": len(observed), "bytes": total_bytes}


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def _source_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    official = config["official_source"]
    paper = official["paper"]
    paper_path = _verify(Path(paper["path"]), paper["sha256"], "official paper")
    if paper_path.stat().st_size != int(paper["bytes"]):
        raise AuditError("official paper byte count drift")

    repository = official["repository"]
    root = Path(repository["path"])
    if _git(root, "rev-parse", "HEAD") != repository["commit"]:
        raise AuditError("official repository commit drift")
    if _git(root, "rev-parse", "HEAD^{tree}") != repository["tree"]:
        raise AuditError("official repository tree drift")
    if _git(root, "status", "--porcelain"):
        raise AuditError("official repository is not clean")
    for relative, digest in repository["files"].items():
        _verify(root / relative, digest, f"official source/{relative}")

    semantic_tokens = {
        "train.py": (
            "torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)",
            "loss_obj = loss_obj / torch.log(torch.tensor(num_classes))",
            "loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d",
            "torch.optim.Adam(classifier.parameters(), lr=5e-4)",
        ),
        "scene/gaussian_model.py": (
            "self.num_objects = 16",
            "self._objects_dc = nn.Parameter",
        ),
        "utils/loss_utils.py": (
            "torch.cdist(sample_features, features)",
            "dists.topk(k, largest=False)",
            "return lambda_val * normalized_loss",
        ),
        "script/prepare_pseudo_label.sh": (
            "demo/demo_automatic.py",
            "--temporal_setting semionline",
            "--use_short_id",
        ),
    }
    for relative, tokens in semantic_tokens.items():
        text = (root / relative).read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in text]
        if missing:
            raise AuditError(f"official source semantic drift: {relative}: {missing}")
    return {
        "paper_sha256": paper["sha256"],
        "paper_bytes": paper_path.stat().st_size,
        "repository_commit": repository["commit"],
        "repository_tree": repository["tree"],
        "repository_file_count": len(repository["files"]),
        "semantic_tokens_exact": True,
    }


def _metadata_summary(
    instances: Mapping[str, Any],
    frame_instances: Mapping[str, Sequence[int]],
    frames: Sequence[int],
) -> dict[str, Any]:
    active_by_frame = {
        str(frame): sorted({int(value) for value in frame_instances.get(str(frame), [])})
        for frame in frames
    }
    union = sorted({value for values in active_by_frame.values() for value in values})
    appearances = Counter(value for values in active_by_frame.values() for value in values)
    if any(str(value) not in instances for value in union):
        raise AuditError("frame_instances references missing instance metadata")
    tokens = [str(instances[str(value)].get("id", "")) for value in union]
    if any(not token for token in tokens) or len(set(tokens)) != len(tokens):
        raise AuditError("instance metadata stable-token drift")
    classes = Counter(str(instances[str(value)].get("class_name", "")) for value in union)
    return {
        "metadata_instance_count": len(instances),
        "train_frame_active_counts": {
            frame: len(values) for frame, values in active_by_frame.items()
        },
        "train_frame_active_union_count": len(union),
        "train_frame_repeated_instance_count": sum(count >= 2 for count in appearances.values()),
        "stable_instance_tokens_present_unique": True,
        "active_class_counts": dict(sorted(classes.items())),
    }


def _observation_schema(paths: Sequence[Path]) -> dict[str, Any]:
    schemas = []
    for path in paths:
        if not path.is_file():
            raise AuditError(f"train-only observation missing: {path}")
        with np.load(path, allow_pickle=False) as table:
            schemas.append(tuple(sorted(table.files)))
    unique = sorted(set(schemas))
    fields = sorted({name for schema in unique for name in schema})
    identity_fields = sorted(
        name
        for name in fields
        if any(marker in name.lower() for marker in IDENTITY_FIELD_MARKERS)
    )
    return {
        "view_count": len(paths),
        "unique_schema_count": len(unique),
        "schemas": [list(schema) for schema in unique],
        "identity_label_fields": identity_fields,
        "associated_instance_identity_labels_present": bool(identity_fields),
    }


def _scene_audit(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = config["train_only_contract"]
    frames = [int(value) for value in contract["frames"]]
    cameras = [int(value) for value in contract["cameras"]]
    reports = []
    for scene in contract["scenes"]:
        graph_path = _verify(
            PROJECT / scene["graph_config"]["path"],
            scene["graph_config"]["sha256"],
            f"{scene['scene']}/graph config",
        )
        graph = _yaml(graph_path)
        if graph["scene"]["name"] != scene["scene"] or int(graph["scene"]["index"]) != int(
            scene["index"]
        ):
            raise AuditError(f"graph scene identity drift: {scene['scene']}")
        unary = graph["inputs"]["unary_manifest"]
        unary_path = _verify(Path(unary["path"]), unary["sha256"], f"{scene['scene']}/unary")
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
        metadata = _metadata_summary(
            _json(instances_path), _json(frame_instances_path), frames
        )
        images = []
        observations = []
        for frame in frames:
            for camera in cameras:
                matches = sorted((processed / "images").glob(f"{frame:03d}_{camera}.*"))
                if len(matches) != 1:
                    raise AuditError(f"train image denominator drift: {scene['scene']}/{frame}/{camera}")
                images.append(matches[0])
                observations.append(
                    unary_path.parent / f"artifacts/observations/f{frame:03d}_c{camera}.npz"
                )
        expected_views = int(contract["expected_views_per_scene"])
        if len(images) != expected_views:
            raise AuditError(f"train image count drift: {scene['scene']}")
        observation = _observation_schema(observations)
        if observation["view_count"] != expected_views:
            raise AuditError(f"observation denominator drift: {scene['scene']}")
        checkpoint = graph["inputs"]["formal_checkpoint"]
        checkpoint_path = _verify(
            Path(checkpoint["path"]), checkpoint["sha256"], f"{scene['scene']}/checkpoint"
        )
        reports.append(
            {
                "scene": scene["scene"],
                "index": int(scene["index"]),
                "train_image_count": len(images),
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
    return reports


def _adapter_smoke(device: str, channels: int, image_size: Sequence[int]) -> dict[str, Any]:
    from gsplat import rasterization

    torch.manual_seed(20260814)
    target = torch.device(device)
    torch.cuda.set_device(target)
    means = torch.tensor(
        [[-0.25, -0.10, 3.0], [0.20, -0.10, 3.1], [-0.05, 0.25, 3.2]],
        dtype=torch.float32,
        device=target,
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=torch.float32, device=target)
    scales = torch.full((3, 3), 0.12, dtype=torch.float32, device=target)
    opacities = torch.full((3,), 0.8, dtype=torch.float32, device=target)
    identity = torch.nn.Parameter(
        torch.linspace(-0.5, 0.5, 3 * channels, device=target).reshape(3, channels)
    )
    height, width = (int(image_size[0]), int(image_size[1]))
    viewmats = torch.eye(4, dtype=torch.float32, device=target)[None]
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
    loss = renders.square().mean()
    loss.backward()
    gradient = identity.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise AuditError("adapter identity gradient is absent or non-finite")
    return {
        "render_shape": list(renders.shape),
        "alpha_positive_pixel_count": int(torch.count_nonzero(alphas > 0).item()),
        "identity_gradient_finite": True,
        "identity_gradient_nonzero_count": int(torch.count_nonzero(gradient).item()),
        "base_geometry_requires_grad": False,
        "base_geometry_gradients_absent": all(
            value.grad is None for value in (means, quats, scales, opacities)
        ),
        "loss": float(loss.detach().cpu()),
    }


def _resource_audit(
    config: Mapping[str, Any], run_dir: Path, summary: Mapping[str, Any]
) -> dict[str, Any]:
    samples = [
        json.loads(line)
        for line in (run_dir / "artifacts/resource_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    valid = [row for row in samples if "monitor_error" not in row]
    resources = summary["resources"]
    if not valid or len(samples) != int(resources["sample_count"]):
        raise AuditError("resource sample denominator drift")
    if len(samples) - len(valid) != int(resources["monitor_error_count"]):
        raise AuditError("resource monitor error count drift")
    if max(int(row["gpu_used_mib"]) for row in valid) != int(resources["nvidia_peak_mib"]):
        raise AuditError("sampled NVIDIA peak drift")
    if max(int(row["cgroup_memory_current_bytes"]) for row in valid) != int(
        resources["cgroup_memory_peak_bytes"]
    ):
        raise AuditError("sampled cgroup peak drift")
    ceilings = config["resources"]
    checks = {
        "nvidia_peak": resources["nvidia_peak_mib"] <= int(ceilings["maximum_nvidia_peak_mib"]),
        "torch_allocated_peak": resources["torch_allocated_peak_mib"]
        <= float(ceilings["maximum_torch_allocated_peak_mib"]),
        "torch_reserved_peak": resources["torch_reserved_peak_mib"]
        <= float(ceilings["maximum_torch_reserved_peak_mib"]),
        "cgroup_memory_peak": resources["cgroup_memory_peak_bytes"]
        <= int(ceilings["maximum_cgroup_memory_bytes"]),
        "wall": resources["wall_seconds"] <= float(ceilings["maximum_wall_seconds"]),
        "monitor": resources["monitor_error_count"] == 0,
    }
    _assert_payload(summary["resource_checks"], checks, "resource checks")
    if not all(checks.values()):
        raise AuditError("r025 resource gate no longer passes")
    return {"sample_count": len(samples), "monitor_error_count": len(samples) - len(valid)}


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _yaml(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("task_id") != TASK_ID:
        raise AuditError("F0 config identity drift")
    status = _json(run_dir / "status.json")
    manifest = _json(run_dir / "manifest.json")
    summary = _json(run_dir / "summary.json")
    report = _json(run_dir / "artifacts/source_preflight_report.json")
    for label, payload in (("status", status), ("manifest", manifest), ("summary", summary), ("report", report)):
        if payload.get("status") != "done" or payload.get("task_id") != TASK_ID:
            raise AuditError(f"r025 {label} terminal drift")
    if status.get("conclusion") != CONCLUSION or summary.get("conclusion") != CONCLUSION:
        raise AuditError("r025 conclusion drift")
    if report.get("conclusion") != CONCLUSION:
        raise AuditError("r025 report conclusion drift")
    if (run_dir / "resolved_config.yaml").read_text(encoding="utf-8") != config_path.read_text(
        encoding="utf-8"
    ):
        raise AuditError("r025 resolved config is not byte-exact")

    inventory = _verify_inventory(run_dir, manifest)
    if inventory != {"entry_count": 6, "bytes": 29935}:
        raise AuditError(f"r025 inventory denominator drift: {inventory}")
    source_tree = _git(PROJECT, "rev-parse", f"{summary['source_commit']}^{{tree}}")
    if summary["source_commit"] != status["source_commit"] or source_tree != summary["source_tree"]:
        raise AuditError("r025 project source identity drift")

    for key, value in report.items():
        if key == "schema_version":
            continue
        if key not in summary:
            raise AuditError(f"summary omits report field: {key}")
        _assert_payload(summary[key], value, f"summary/report/{key}")

    source = _source_audit(config)
    scenes = _scene_audit(config)
    _assert_payload(report["scenes"], scenes, "scene replay")
    if any(row["observation_schema"]["associated_instance_identity_labels_present"] for row in scenes):
        raise AuditError("binary observation unexpectedly became an instance-identity label")

    assets = {}
    for name, spec in config["asset_probe"]["upstream_required"].items():
        present = Path(spec["path"]).is_file()
        if present != bool(spec["expected_present_before_preflight"]):
            raise AuditError(f"upstream asset presence drift: {name}")
        assets[name] = {
            "path": spec["path"],
            "present": present,
            "expected_present_before_preflight": bool(spec["expected_present_before_preflight"]),
        }
    _assert_payload(report["upstream_assets"], assets, "upstream assets")
    existing = config["asset_probe"]["existing_non_substitute"]["sam2_hiera_large"]
    _verify(Path(existing["path"]), existing["sha256"], "SAM2 non-substitute")

    adapter_config = config["adapter_probe"]
    adapter = _adapter_smoke(
        adapter_config["device"],
        int(adapter_config["identity_channels"]),
        adapter_config["image_size"],
    )
    _assert_payload(report["adapter"], adapter, "adapter replay")
    resources = _resource_audit(config, run_dir, summary)

    false_locks = (
        "quality_read",
        "parameter_search",
        "deva_execution",
        "sam_execution",
        "identity_training_authorized",
        "current_training_input_ready",
        "h_quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "f1_execution",
        "f2_execution",
    )
    for lock in false_locks:
        if summary.get(lock) is not False:
            raise AuditError(f"quality/training lock drift: {lock}")
    if summary.get("m2_status") != "pending" or summary.get("m3_status") != "pending":
        raise AuditError("M2/M3 lock drift")
    return {
        "schema_version": "worldsim_v51_f0_source_preflight_audit_v1",
        "task_id": TASK_ID,
        "run": str(run_dir),
        "status": "pass",
        "conclusion": CONCLUSION,
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "inventory": inventory,
        "source": source,
        "scene_count": len(scenes),
        "view_count": sum(row["train_image_count"] for row in scenes),
        "all_identity_label_fields_absent": True,
        "checkpoints_hash_verified_not_loaded": True,
        "upstream_assets_absent": True,
        "sam2_is_not_faithful_substitute": True,
        "adapter": adapter,
        "resources": resources,
        "quality_and_training_locks_preserved": True,
        "m2_status": "pending",
        "m3_status": "pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
