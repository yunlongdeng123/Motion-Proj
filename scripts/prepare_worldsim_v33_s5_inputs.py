#!/usr/bin/env python3
"""从 canonical S4 renders 与冻结 SAM2 masks 准备 S5 输入和 gate。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v33.semantic_gate import (  # noqa: E402
    build_semantic_gate,
    validate_semantic_gate,
)
from motion_proj.worldsim_v33.spatial_delta import (  # noqa: E402
    atomic_json,
    atomic_save_npz,
    sha256_file,
)


def verify(spec: Mapping[str, Any], role: str) -> dict[str, Any]:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{role} 不存在: {path}")
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise RuntimeError(f"{role} SHA 漂移: expected={spec['sha256']} actual={actual}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def find_mask_row(
    manifests: list[dict[str, Any]], *, role: str, frame: int, camera: int
) -> dict[str, Any]:
    rows = [
        row
        for manifest in manifests
        for row in manifest["masks"]
        if row["role"] == role
        and int(row["frame"]) == int(frame)
        and int(row["camera_id"]) == int(camera)
    ]
    if len(rows) != 1 or not rows[0].get("accepted"):
        raise RuntimeError(
            f"S5 mask 未唯一 accepted: role={role} frame={frame} camera={camera}"
        )
    return rows[0]


def copy_png(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"path": str(target), "sha256": sha256_file(target), "bytes": target.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "worldsim_v33_s5_semantic_gate_v1":
        raise ValueError("S5 config schema version 漂移")
    output = args.run_dir / "artifacts/inputs"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    verified = {
        name: verify(spec, name) for name, spec in config["inputs"].items()
    }
    s4 = json.loads(Path(config["inputs"]["s4_summary"]["path"]).read_text())
    if s4["state"] != "completed" or not s4["decision"]["accepted"]:
        raise RuntimeError("S5 输入的 S4 canonical decision 未 accepted")
    s4_status = json.loads(Path(config["inputs"]["s4_status"]["path"]).read_text())
    if s4_status["state"] != "completed":
        raise RuntimeError("S5 输入的 S4 terminal 非 completed")
    train_manifest = json.loads(
        Path(config["inputs"]["train_mask_manifest"]["path"]).read_text()
    )
    heldout_manifest = json.loads(
        Path(config["inputs"]["heldout_mask_manifest"]["path"]).read_text()
    )
    manifests = [train_manifest, heldout_manifest]
    if not train_manifest.get("heldout_excluded"):
        raise RuntimeError("S5 train SAM manifest 未排除 heldout")
    if not heldout_manifest.get("optimization_forbidden"):
        raise RuntimeError("S5 heldout SAM manifest 未禁止 optimization")

    target_view = tuple(int(value) for value in config["views"]["edit_target"])
    development = {
        tuple(int(value) for value in row) for row in config["views"]["development"]
    }
    heldout = {
        tuple(int(value) for value in row)
        for row in config["views"]["heldout_confirmation"]
    }
    views = [target_view, *sorted(development), *sorted(heldout)]
    s4_rows = {
        (int(row["frame"]), int(row["camera_id"])): row for row in s4["rows"]
    }
    s4_render_root = Path(config["inputs"]["s4_summary"]["path"]).parent / "artifacts/renders"
    gate_cfg = config["semantic_gate"]
    records = []
    for frame, camera in views:
        if (frame, camera) not in s4_rows:
            raise RuntimeError(f"S4 summary 缺 S5 view: {(frame, camera)}")
        phase = (
            "development"
            if (frame, camera) == target_view or (frame, camera) in development
            else "heldout_confirmation"
        )
        mask_row = find_mask_row(
            manifests,
            role=config["actor"]["role"],
            frame=frame,
            camera=camera,
        )
        verify(
            {"path": mask_row["mask"], "sha256": mask_row["mask_sha256"]},
            f"mask f{frame} c{camera}",
        )
        verify(
            {
                "path": mask_row["source_image"],
                "sha256": mask_row["source_image_sha256"],
            },
            f"source image f{frame} c{camera}",
        )
        view_dir = output / f"f{frame:03d}_c{camera}"
        view_dir.mkdir()
        images: dict[str, np.ndarray] = {}
        image_records = {}
        for stack in ("base_only", "erase", "erase_background", "actor_override", "full"):
            source = s4_render_root / f"f{frame:03d}_c{camera}/{stack}.png"
            value = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
            expected = s4_rows[(frame, camera)]["render_sha256"][stack]
            if array_sha256(value) != expected:
                raise RuntimeError(f"S4 render array SHA 漂移: f{frame} c{camera} {stack}")
            target = view_dir / f"{stack}.png"
            image_records[stack] = copy_png(source, target)
            image_records[stack]["array_sha256"] = expected
            images[stack] = value

        with np.load(mask_row["mask"], allow_pickle=False) as payload:
            target_mask = payload["binary"].astype(bool)
        if target_mask.shape != images["full"].shape[:2] or not target_mask.any():
            raise RuntimeError("S5 target mask shape/内容漂移")
        source_image = Image.open(mask_row["source_image"]).convert("RGB")
        source_image = source_image.resize(
            (images["full"].shape[1], images["full"].shape[0]),
            Image.Resampling.LANCZOS,
        )
        reference_path = view_dir / "reference_rgb.png"
        source_image.save(reference_path)
        mask_path = view_dir / "target_mask.png"
        Image.fromarray(target_mask.astype(np.uint8) * 255).save(mask_path)

        difference = np.max(
            np.abs(
                images["actor_override"].astype(np.int16)
                - images["erase"].astype(np.int16)
            ),
            axis=2,
        ) > int(gate_cfg["effect_threshold_uint8"])
        footprint = binary_dilation(
            difference, iterations=int(gate_cfg["effect_dilation_pixels"])
        )
        regions = build_semantic_gate(
            footprint,
            **{
                name: gate_cfg[name]
                for name in (
                    "boundary_inner_pixels",
                    "boundary_outer_pixels",
                    "contact_depth_pixels",
                    "contact_side_pixels",
                    "shadow_depth_pixels",
                    "shadow_side_pixels",
                    "boundary_weight",
                    "contact_weight",
                    "shadow_weight",
                )
            },
        )
        validate_semantic_gate(regions)
        gate_path = view_dir / "semantic_gate.npz"
        atomic_save_npz(gate_path, regions)
        ys, xs = np.nonzero(target_mask)
        prompt_box = [
            max(0, int(xs.min()) - 2),
            max(0, int(ys.min()) - 2),
            min(target_mask.shape[1] - 1, int(xs.max()) + 2),
            min(target_mask.shape[0] - 1, int(ys.max()) + 2),
        ]
        records.append(
            {
                "frame": frame,
                "camera_id": camera,
                "phase": phase,
                "images": image_records,
                "reference_rgb": {
                    "path": str(reference_path),
                    "sha256": sha256_file(reference_path),
                },
                "target_mask": {
                    "path": str(mask_path),
                    "sha256": sha256_file(mask_path),
                    "source_npz": mask_row["mask"],
                    "source_npz_sha256": mask_row["mask_sha256"],
                    "pixels": int(target_mask.sum()),
                },
                "semantic_gate": {
                    "path": str(gate_path),
                    "sha256": sha256_file(gate_path),
                    "actor_footprint_pixels": int(footprint.sum()),
                    "allowed_pixels": int(np.asarray(regions["allowed"]).sum()),
                },
                "sam_prompt_box_xyxy": prompt_box,
                "source_mask_manifest": (
                    verified["heldout_mask_manifest"]
                    if phase == "heldout_confirmation"
                    else verified["train_mask_manifest"]
                ),
            }
        )
    manifest = {
        "schema_version": "worldsim_v33_s5_input_manifest_v1",
        "task_id": config["task_id"],
        "config_sha256": sha256_file(args.config),
        "verified_inputs": verified,
        "records": records,
        "development_views": [
            [row["frame"], row["camera_id"]]
            for row in records
            if row["phase"] == "development"
        ],
        "heldout_confirmation_views": [
            [row["frame"], row["camera_id"]]
            for row in records
            if row["phase"] == "heldout_confirmation"
        ],
        "heldout_read_for_selection": False,
        "optimization_performed": False,
    }
    manifest_path = output / "input_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "done", "manifest_sha256": sha256_file(manifest_path), "views": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
