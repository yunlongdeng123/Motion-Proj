#!/usr/bin/env python3
"""Materialize GPU-free adapter inputs for the V7.5 paired-edit pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_proj.cfbench.geometry import pose_at, shifted_pose  # noqa: E402
from motion_proj.cfbench.registry import load_registry, support_for_case  # noqa: E402
from motion_proj.cfbench.schema import validate_manifest  # noqa: E402


def _ego_poses(scene_root: Path) -> dict[int, np.ndarray]:
    return {
        int(path.stem): np.loadtxt(path)
        for path in sorted((scene_root / "lidar_pose").glob("*.txt"))
    }


def _relative_state(origin: np.ndarray, pose: np.ndarray) -> list[float]:
    relative = np.linalg.inv(origin) @ pose
    heading = math.atan2(float(relative[1, 0]), float(relative[0, 0]))
    return [float(relative[0, 3]), float(relative[1, 3]), heading]


def _trajectory(
    case: dict[str, Any], poses: dict[int, np.ndarray], branch: str
) -> list[list[float]]:
    event = int(case["anchor"]["event_frame"])
    rollout = int(case["anchor"]["rollout_frames"])
    sample_offsets = np.linspace(0.0, float(rollout - 1), 8)
    origin = poses[event]
    family = case["intervention"]["family"]
    control = case["intervention"][branch]
    rows: list[list[float]] = []
    for offset in sample_offsets:
        source_offset = float(offset)
        if branch == "counterfactual" and family == "actor_speed_change":
            source_offset *= float(control["speed_scale"])
        pose = pose_at(poses, event + source_offset)
        if pose is None:
            raise RuntimeError(
                f"{case['case_id']}/{branch}: missing ego pose at {event + source_offset:.3f}"
            )
        if branch == "counterfactual" and family == "actor_lateral_relocation":
            progress = float(np.clip(offset / max(1, rollout - 1), 0.0, 1.0))
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            pose = shifted_pose(
                pose,
                [0.0, float(control["lateral_offset_m"]) * smooth_progress, 0.0],
            )
        rows.append(_relative_state(origin, pose))
    return rows


def _set_resim_paths(config: dict[str, Any], runtime_root: Path, data_json: Path) -> None:
    config["args"]["load"] = str(runtime_root / "transformer")
    config["args"]["valid_data"] = [str(data_json)]
    config["data"]["params"]["n_subset"] = None
    config["data"]["params"]["ind_subset"] = None
    embeddings = config["model"]["conditioner_config"]["params"]["emb_models"]
    embeddings[0]["params"]["model_dir"] = str(runtime_root / "t5-v1_1-xxl")
    config["model"]["first_stage_config"]["params"]["ckpt_path"] = str(
        runtime_root / "vae" / "3d-vae.pt"
    )


def _materialize_resim(
    cases: list[dict[str, Any]],
    model: dict[str, Any],
    output_root: Path,
    template_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for case in cases:
        support = support_for_case(model, case)
        if support == "unsupported":
            continue
        if case["target"]["role"] != "ego":
            raise RuntimeError(f"ReSim adapter received non-ego case {case['case_id']}")
        scene_root = Path(case["dataset"]["root"]) / case["dataset"]["scene_id"]
        poses = _ego_poses(scene_root)
        event = int(case["anchor"]["event_frame"])
        pre = int(case["anchor"]["pre_frames"])
        rollout = int(case["anchor"]["rollout_frames"])
        image_paths = [
            str(
                Path(case["dataset"]["scene_id"])
                / "images"
                / f"{frame:03d}_0.jpg"
            )
            for frame in range(event - pre, event + rollout)
        ]
        case_dir = output_root / "resim" / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        branch_rows: dict[str, Any] = {}
        for branch in ("factual", "counterfactual"):
            data_json = case_dir / f"{branch}.json"
            payload = {
                "meta": {"data_root": str(Path(case["dataset"]["root"]))},
                "clips": [
                    {
                        "img_seq": image_paths,
                        "cmd": "Moving_Forward",
                        "traj_fut": _trajectory(case, poses, branch),
                        "lidar_pc_token": f"{case['case_id']}:{branch}",
                    }
                ],
            }
            data_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            config = json.loads(json.dumps(template))
            _set_resim_paths(config, runtime_root, data_json.resolve())
            config_path = case_dir / f"{branch}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            branch_rows[branch] = {
                "data": str(data_json.resolve()),
                "config": str(config_path.resolve()),
            }
        rows.append(
            {
                "case_id": case["case_id"],
                "support": support,
                "branches": branch_rows,
            }
        )
    return {
        "model_id": "resim",
        "case_count": len(rows),
        "model_execution_run": False,
        "entrypoint": "cd sat && bash inference_custom.sh <branch-config>",
        "cases": rows,
    }


def _generic_adapter_plan(
    cases: list[dict[str, Any]], model: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    for case in cases:
        support = support_for_case(model, case)
        if support == "unsupported":
            continue
        rows.append(
            {
                "case_id": case["case_id"],
                "support": support,
                "family": case["intervention"]["family"],
                "target_role": case["target"]["role"],
            }
        )
    return {
        "model_id": model["model_id"],
        "role": model["role"],
        "case_count": len(rows),
        "model_execution_run": False,
        "entrypoint": model.get("entrypoint"),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resim-template", type=Path, required=True)
    parser.add_argument("--resim-runtime", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    registry = load_registry(args.registry)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_rows = []
    for model in registry["models"]:
        if model["model_id"] == "resim":
            row = _materialize_resim(
                manifest["cases"],
                model,
                output_root,
                args.resim_template,
                args.resim_runtime.resolve(),
            )
        else:
            row = _generic_adapter_plan(manifest["cases"], model)
        model_rows.append(row)
    index = {
        "schema_version": "worldsim_v75_cfbench_adapter_inputs_v1",
        "gpu_inference_run": False,
        "models": model_rows,
    }
    index_path = output_root / "adapter-index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({row["model_id"]: row["case_count"] for row in model_rows}, indent=2))


if __name__ == "__main__":
    main()
