#!/usr/bin/env python3
"""正式审计 V5.1 Stage B B0/B1 与 lazy bilinear 纯算子。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as functional
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.feature_uplift import (
    LUDVIG_UPSTREAM_COMMIT,
    sample_patch_grid_bilinear,
    uplift_b0_b1,
)
from motion_proj.worldsim_v51.protocol import (
    ProtocolError,
    V51_BRANCH,
    load_yaml,
    sha256_file,
)


def _git(project: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), *args], text=True
    ).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "status.json"}:
            continue
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != "worldsim_v51_stage_b_operator_parity_v1":
        raise ProtocolError("Stage B operator parity schema 漂移")
    if config.get("task_id") != "WS-V51-M1-B-LUDVIG-UPLIFT-01":
        raise ProtocolError("Stage B operator parity task 漂移")
    if config.get("status") != "running" or config.get("seed") != 20260814:
        raise ProtocolError("Stage B operator parity status/seed 漂移")

    source = config["ludvig_source"]
    source_path = Path(source["path"])
    if _git(source_path, "remote", "get-url", "origin") != source["repository"]:
        raise ProtocolError("LUDVIG source origin 漂移")
    if _git(source_path, "rev-parse", "HEAD") != source["commit"]:
        raise ProtocolError("LUDVIG source commit 漂移")
    if _git(source_path, "rev-parse", "HEAD^{tree}") != source["tree"]:
        raise ProtocolError("LUDVIG source tree 漂移")
    if _git(source_path, "status", "--short") != source["expected_git_status"]:
        raise ProtocolError("LUDVIG source worktree 非 clean")
    for relative, expected in (
        ("LICENSE.txt", source["license_sha256"]),
        ("NOTICE.txt", source["notice_sha256"]),
        ("utils/solver.py", source["solver_sha256"]),
        (
            "gaussiansplatting/submodules/diff-gaussian-rasterization/"
            "cuda_rasterizer/apply_weights.cu",
            source["apply_weights_cuda_sha256"],
        ),
    ):
        if sha256_file(source_path / relative) != expected:
            raise ProtocolError(f"LUDVIG source file SHA 漂移: {relative}")
    if source.get("license") != "non-commercial":
        raise ProtocolError("LUDVIG license contract 漂移")
    if source.get("vendored_into_project") is not False:
        raise ProtocolError("LUDVIG source 不得 vendor 到项目")

    freeze_spec = config["resource_freeze"]
    freeze_path = PROJECT / freeze_spec["path"]
    if sha256_file(freeze_path) != freeze_spec["sha256"]:
        raise ProtocolError("DINO resource freeze binding 漂移")
    resource_freeze = load_yaml(freeze_path)
    if resource_freeze.get("status") != freeze_spec["required_status"]:
        raise ProtocolError("DINO resource freeze status 漂移")
    if resource_freeze["locks"]["quality_read"] is not False:
        raise ProtocolError("DINO resource freeze quality lock 漂移")

    operator = config["operator"]
    expected_operator = {
        "b0": "view_saturated_current_intersection_lift",
        "b1": "normalized_renderer_transpose",
        "minimum_intersection_contribution": 1e-4,
        "minimum_gaussian_view_mass": 1e-3,
        "epsilon": 1e-8,
        "accumulator_dtype": "float64",
        "output_dtype": "float32",
        "optional_pruning": False,
        "gaussian_row_order_immutable": True,
        "canonical_row_sort_for_chunk_order_invariance": True,
    }
    for name, expected in expected_operator.items():
        if operator.get(name) != expected:
            raise ProtocolError(f"operator contract 漂移: {name}")
    if source["commit"] != LUDVIG_UPSTREAM_COMMIT:
        raise ProtocolError("本地 operator provenance commit 漂移")

    synthetic = config["synthetic"]
    expected_synthetic = {
        "gaussian_count": 5,
        "view_count": 2,
        "pixel_count_per_view": 24,
        "feature_dimension": 3,
        "patch_grid_shape_chw": [3, 2, 3],
        "dense_image_shape_hw": [4, 6],
        "maximum_feature_absolute_error": 1e-6,
        "maximum_lazy_bilinear_absolute_error": 2e-6,
        "order_invariance": "bit_exact",
    }
    for name, expected in expected_synthetic.items():
        if synthetic.get(name) != expected:
            raise ProtocolError(f"synthetic contract 漂移: {name}")

    locks = config["locks"]
    for name in (
        "dino_model_load",
        "pca_fit",
        "feature_sidecar_persist",
        "renderer_start",
        "real_image_feature_read",
        "method_quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"operator parity lock 漂移: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 必须保持 pending")
    return config, resource_freeze


def _synthetic_payload(config: dict[str, Any]) -> dict[str, np.ndarray]:
    synthetic = config["synthetic"]
    rng = np.random.default_rng(int(config["seed"]))
    view_count = int(synthetic["view_count"])
    gaussian_count = int(synthetic["gaussian_count"])
    pixel_count = int(synthetic["pixel_count_per_view"])
    dimension = int(synthetic["feature_dimension"])
    features = rng.normal(size=(view_count, pixel_count, dimension)).astype(
        np.float32
    )
    weights = rng.uniform(0.01, 0.35, size=(view_count, gaussian_count, pixel_count))
    weights[1, 0] *= 0.04
    weights[0, 3] = 0.0
    weights[0, 3, :5] = 1e-4
    weights[:, 4] = 0.0
    weights[0, 4, :6] = 9e-5
    weights[1, 2, 0] = 1e-4
    views, gids, pixels = np.indices(weights.shape)
    return {
        "gaussian_id": gids.reshape(-1).astype(np.int64),
        "view_id": views.reshape(-1).astype(np.int64),
        "pixel_id": pixels.reshape(-1).astype(np.int64),
        "contribution_weight": weights.reshape(-1).astype(np.float64),
        "pixel_features": np.broadcast_to(
            features[:, None, :, :],
            (view_count, gaussian_count, pixel_count, dimension),
        )
        .reshape(-1, dimension)
        .copy(),
        "dense_features": features,
        "dense_weights": weights,
    }


def _independent_dense_reference(
    payload: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    weights = payload["dense_weights"]
    features = payload["dense_features"].astype(np.float64)
    gaussian_count = weights.shape[1]
    dimension = features.shape[2]
    b0 = np.zeros((gaussian_count, dimension), dtype=np.float64)
    b1 = np.zeros_like(b0)
    intersection_floor = float(
        config["operator"]["minimum_intersection_contribution"]
    )
    view_floor = float(config["operator"]["minimum_gaussian_view_mass"])
    epsilon = float(config["operator"]["epsilon"])
    for gaussian in range(gaussian_count):
        b0_num = np.zeros(dimension, dtype=np.float64)
        b0_den = 0.0
        b1_num = np.zeros(dimension, dtype=np.float64)
        b1_den = 0.0
        for view in range(weights.shape[0]):
            numerator = np.zeros(dimension, dtype=np.float64)
            mass = 0.0
            for pixel in range(weights.shape[2]):
                weight = float(weights[view, gaussian, pixel])
                if weight >= intersection_floor:
                    numerator += weight * features[view, pixel]
                    mass += weight
            if mass >= view_floor:
                saturation = -np.expm1(-mass)
                b0_num += saturation * (numerator / (mass + epsilon))
                b0_den += saturation
                b1_num += numerator
                b1_den += mass
        b0[gaussian] = b0_num / (b0_den + epsilon)
        b1[gaussian] = b1_num / (b1_den + epsilon)
    return b0.astype(np.float32), b1.astype(np.float32)


def execute_parity(config: dict[str, Any]) -> dict[str, Any]:
    payload = _synthetic_payload(config)
    operator = config["operator"]
    kwargs = {
        "gaussian_count": int(config["synthetic"]["gaussian_count"]),
        "minimum_intersection_contribution": float(
            operator["minimum_intersection_contribution"]
        ),
        "minimum_gaussian_view_mass": float(
            operator["minimum_gaussian_view_mass"]
        ),
        "epsilon": float(operator["epsilon"]),
    }
    observed = uplift_b0_b1(
        gaussian_id=payload["gaussian_id"],
        view_id=payload["view_id"],
        contribution_weight=payload["contribution_weight"],
        pixel_features=payload["pixel_features"],
        **kwargs,
    )
    reference_b0, reference_b1 = _independent_dense_reference(payload, config)
    b0_error = float(np.max(np.abs(observed["b0_feature"] - reference_b0)))
    b1_error = float(np.max(np.abs(observed["b1_feature"] - reference_b1)))

    permutation = np.random.default_rng(int(config["seed"]) + 1).permutation(
        payload["gaussian_id"].size
    )
    permuted = uplift_b0_b1(
        gaussian_id=payload["gaussian_id"][permutation],
        view_id=payload["view_id"][permutation],
        contribution_weight=payload["contribution_weight"][permutation],
        pixel_features=payload["pixel_features"][permutation],
        **kwargs,
    )
    order_exact = all(
        np.array_equal(observed[name], permuted[name])
        for name in (
            "b0_feature",
            "b1_feature",
            "b0_denominator",
            "b1_denominator",
            "supported_view_count",
        )
    )

    constant = np.asarray([1.5, -2.0, 0.25], dtype=np.float32)
    constant_result = uplift_b0_b1(
        gaussian_id=payload["gaussian_id"],
        view_id=payload["view_id"],
        contribution_weight=payload["contribution_weight"],
        pixel_features=np.broadcast_to(
            constant, payload["pixel_features"].shape
        ).copy(),
        **kwargs,
    )
    covered = constant_result["b1_denominator"] > 0.0
    constant_error = float(
        max(
            np.max(np.abs(constant_result["b0_feature"][covered] - constant)),
            np.max(np.abs(constant_result["b1_feature"][covered] - constant)),
        )
    )
    zero_denominator_exact = bool(
        np.array_equal(observed["b0_feature"][4], np.zeros(3, dtype=np.float32))
        and np.array_equal(
            observed["b1_feature"][4], np.zeros(3, dtype=np.float32)
        )
        and observed["b0_denominator"][4] == 0.0
        and observed["b1_denominator"][4] == 0.0
    )
    arm_difference_l2 = float(
        np.linalg.norm(observed["b1_feature"] - observed["b0_feature"])
    )

    rng = np.random.default_rng(int(config["seed"]) + 2)
    patch_grid = rng.normal(size=(3, 2, 3)).astype(np.float32)
    image_height, image_width = config["synthetic"]["dense_image_shape_hw"]
    pixel_id = np.arange(image_height * image_width, dtype=np.int64)
    lazy = sample_patch_grid_bilinear(
        patch_grid,
        pixel_id,
        image_height=int(image_height),
        image_width=int(image_width),
    )
    dense = (
        functional.interpolate(
            torch.from_numpy(patch_grid).unsqueeze(0),
            size=(int(image_height), int(image_width)),
            mode="bilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 0)
        .reshape(-1, 3)
        .numpy()
    )
    lazy_error = float(np.max(np.abs(lazy - dense)))

    maximum_feature_error = float(
        config["synthetic"]["maximum_feature_absolute_error"]
    )
    maximum_lazy_error = float(
        config["synthetic"]["maximum_lazy_bilinear_absolute_error"]
    )
    checks = {
        "independent_dense_reference": max(b0_error, b1_error)
        <= maximum_feature_error,
        "repeated_gaussian_and_pixel_intersections": observed["report"][
            "input_intersection_count"
        ]
        > observed["report"]["gaussian_view_count_before_mass_floor"],
        "below_intersection_floor": observed["report"][
            "supported_intersection_count"
        ]
        < observed["report"]["input_intersection_count"],
        "below_gaussian_view_mass_floor": observed["report"][
            "dropped_gaussian_view_count"
        ]
        >= 1,
        "zero_denominator_gaussian": zero_denominator_exact,
        "constant_feature_conservation": constant_error <= maximum_feature_error,
        "row_and_chunk_order_invariance": order_exact,
        "lazy_bilinear_dense_align_corners_false": lazy_error <= maximum_lazy_error,
        "b0_b1_non_alias": arm_difference_l2 > 1e-4,
        "output_dtype_float32": observed["b0_feature"].dtype == np.float32
        and observed["b1_feature"].dtype == np.float32,
        "optional_pruning_disabled": observed["report"]["optional_pruning"]
        is False,
    }
    required = set(config["synthetic"]["required_cases"])
    if not required.issubset(checks) or not all(checks[name] for name in required):
        raise ProtocolError(f"Stage B operator parity required case 失败: {checks}")
    if not all(checks.values()):
        raise ProtocolError(f"Stage B operator parity 附加检查失败: {checks}")
    return {
        "checks": checks,
        "b0_max_absolute_error": b0_error,
        "b1_max_absolute_error": b1_error,
        "constant_feature_max_absolute_error": constant_error,
        "lazy_bilinear_max_absolute_error": lazy_error,
        "b0_b1_difference_l2": arm_difference_l2,
        "operator_report": observed["report"],
        "b0_feature_sha256": _array_sha256(observed["b0_feature"]),
        "b1_feature_sha256": _array_sha256(observed["b1_feature"]),
        "lazy_feature_sha256": _array_sha256(lazy),
    }


def _array_sha256(value: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    branch = _git(PROJECT, "branch", "--show-current")
    head = _git(PROJECT, "rev-parse", "HEAD")
    if branch != V51_BRANCH:
        raise ProtocolError(f"必须在 {V51_BRANCH} 执行，当前为 {branch}")
    if _git(PROJECT, "status", "--short"):
        raise ProtocolError("Stage B operator parity 要求 clean project worktree")
    config, resource_freeze = validate_config(config_path)
    _write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(
            {"operator_parity": config, "resource_freeze": resource_freeze},
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    checkpoint = Path(config["checkpoint_immutability"]["path"])
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != config["checkpoint_immutability"]["sha256"]:
        raise ProtocolError("operator parity 前 checkpoint SHA 漂移")
    started = time.monotonic()
    report = execute_parity(config)
    duration = time.monotonic() - started
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise ProtocolError("operator parity 前后 checkpoint SHA 不一致")
    _write_json(run_dir / "artifacts/parity_report.json", report)
    _write_jsonl(
        run_dir / "metrics.jsonl",
        [
            {"metric": name, "value": value}
            for name, value in (
                ("b0_max_absolute_error", report["b0_max_absolute_error"]),
                ("b1_max_absolute_error", report["b1_max_absolute_error"]),
                (
                    "constant_feature_max_absolute_error",
                    report["constant_feature_max_absolute_error"],
                ),
                (
                    "lazy_bilinear_max_absolute_error",
                    report["lazy_bilinear_max_absolute_error"],
                ),
                ("b0_b1_difference_l2", report["b0_b1_difference_l2"]),
                ("duration_seconds", duration),
            )
        ],
    )
    summary = {
        "schema_version": "worldsim_v51_stage_b_operator_parity_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "conclusion": "synthetic_b0_b1_and_lazy_bilinear_operator_parity_passed",
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "config_sha256": sha256_file(config_path),
        "ludvig_source_commit": config["ludvig_source"]["commit"],
        "ludvig_source_tree": config["ludvig_source"]["tree"],
        "ludvig_license": config["ludvig_source"]["license"],
        "ludvig_vendored_into_project": False,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": True,
        "duration_seconds": duration,
        "report": report,
        "dino_model_load": False,
        "pca_fit": False,
        "feature_sidecar_persisted": False,
        "renderer_started": False,
        "real_image_feature_read": False,
        "method_quality_read": False,
        "screening_quality_read": False,
        "confirmation_quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "none",
        "created_at_utc": _utc_now(),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "fingerprint.json",
        {
            "schema_version": "worldsim_v51_stage_b_operator_parity_fingerprint_v1",
            "task_id": config["task_id"],
            "source_commit": head,
            "source_branch": branch,
            "config_sha256": summary["config_sha256"],
            "ludvig_source_commit": summary["ludvig_source_commit"],
            "ludvig_source_tree": summary["ludvig_source_tree"],
            "checkpoint_sha256": checkpoint_before,
            "seed": int(config["seed"]),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_b_operator_parity_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    try:
        summary = run(args.config.resolve(), run_dir)
        events.append({"event": "run_done", "at_utc": _utc_now()})
        _write_jsonl(run_dir / "events.jsonl", events)
        manifest = {
            "schema_version": "worldsim_v51_stage_b_operator_parity_manifest_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_stage_b_operator_parity_status_v1",
                "task_id": summary["task_id"],
                "status": "done",
                "source_commit": summary["source_commit"],
                "summary_sha256": sha256_file(run_dir / "summary.json"),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "finished_at_utc": _utc_now(),
            },
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": _utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_stage_b_operator_parity_status_v1",
                "task_id": "WS-V51-M1-B-LUDVIG-UPLIFT-01",
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
