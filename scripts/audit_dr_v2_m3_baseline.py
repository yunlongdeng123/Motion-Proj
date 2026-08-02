#!/usr/bin/env python
"""Emit the M3 DriveStudio/StreetGS baseline readiness contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml


ALLOWED_STATUS = {"available", "missing", "incompatible"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str, cwd: Path | None = None) -> tuple[int, str, str]:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def item(status: str, evidence, detail: str) -> dict:
    if status not in ALLOWED_STATUS:
        raise ValueError(status)
    return {"status": status, "evidence": evidence, "detail": detail}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--upstream-root", type=Path, default=Path("/root/autodl-tmp/third_party/drivestudio"))
    parser.add_argument("--environment", type=Path, default=Path("/root/autodl-tmp/envs/drivestudio"))
    parser.add_argument("--processed-root", type=Path, default=Path("/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval"))
    parser.add_argument("--scene-index", type=int, default=179)
    parser.add_argument("--scene-name", default="scene-0230")
    args = parser.parse_args()

    root = args.upstream_root
    git_rc, commit, git_error = command("git", "rev-parse", "HEAD", cwd=root)
    origin_rc, origin, origin_error = command("git", "remote", "get-url", "origin", cwd=root)
    status_rc, git_status, _ = command("git", "status", "--short", cwd=root)
    license_candidates = [root / "LICENSE", root / "LICENSE.md", root / "COPYING"]
    license_path = next((path for path in license_candidates if path.is_file()), None)

    python = args.environment / "bin" / "python"
    env_rc, env_stdout, env_stderr = command(
        str(python),
        "-c",
        "import torch,gsplat,nvdiffrast,nuscenes,cv2,omegaconf,wandb; "
        "print(torch.__version__, torch.version.cuda, torch.cuda.is_available())",
    ) if python.is_file() else (127, "", "python missing")

    dataset_config = root / "configs" / "datasets" / "nuscenes" / "3cams.yaml"
    method_config = root / "configs" / "streetgs.yaml"
    data_config = yaml.safe_load(dataset_config.read_text()) if dataset_config.is_file() else {}
    cameras = data_config.get("data", {}).get("pixel_source", {}).get("cameras")
    scene_root = args.processed_root / f"{args.scene_index:03d}"
    images = sorted((scene_root / "images").glob("*.jpg")) if scene_root.is_dir() else []
    sky_masks = sorted((scene_root / "sky_masks").glob("*.png")) if scene_root.is_dir() else []
    instances_info = scene_root / "instances" / "instances_info.json"

    rigid_source = root / "models" / "nodes" / "rigid.py"
    rigid_text = rigid_source.read_text() if rigid_source.is_file() else ""
    train_source = root / "tools" / "train.py"
    eval_source = root / "tools" / "eval.py"
    checkpoint = args.checkpoint
    checkpoint_ok = checkpoint is not None and checkpoint.is_file() and checkpoint.stat().st_size > 0

    entries = {
        "official_live_repository": item(
            "available" if git_rc == 0 and origin_rc == 0 else "missing",
            {"path": str(root), "commit": commit or None, "origin": origin or None, "clean": status_rc == 0 and not git_status, "error": git_error or origin_error or None},
            "Live git checkout; ZIP snapshots are not used.",
        ),
        "license": item(
            "available" if license_path else "missing",
            {"path": str(license_path) if license_path else None, "sha256": sha256_file(license_path) if license_path else None},
            "Upstream license file.",
        ),
        "environment": item(
            "available" if env_rc == 0 and env_stdout.endswith("True") else "incompatible" if python.is_file() else "missing",
            {"python": str(python), "smoke_stdout": env_stdout, "smoke_stderr": env_stderr, "return_code": env_rc},
            "Imports native renderer dependencies and verifies CUDA.",
        ),
        "native_nuscenes_three_camera_config": item(
            "available" if dataset_config.is_file() and cameras == [0, 1, 2] and method_config.is_file() else "incompatible",
            {"dataset_config": str(dataset_config), "method_config": str(method_config), "cameras": cameras},
            "Frozen StreetGS method config with the native nuScenes three-camera dataset config.",
        ),
        "processed_scene": item(
            "available" if images and instances_info.is_file() else "missing",
            {"path": str(scene_root), "image_count": len(images), "instances_info": instances_info.is_file()},
            f"DriveStudio processed input for {args.scene_name} / numeric scene {args.scene_index}.",
        ),
        "required_sky_masks": item(
            "available" if images and len(sky_masks) == len([path for path in images if int(path.stem.rsplit('_', 1)[1]) in {0, 1, 2}]) else "missing",
            {"path": str(scene_root / 'sky_masks'), "mask_count": len(sky_masks), "required_camera_image_count": len([path for path in images if int(path.stem.rsplit('_', 1)[1]) in {0, 1, 2}])},
            "Sky masks are required by the native data config for cameras 0/1/2.",
        ),
        "actor_aware_checkpoint": item(
            "available" if checkpoint_ok else "missing",
            {"path": str(checkpoint) if checkpoint else None, "bytes": checkpoint.stat().st_size if checkpoint_ok else 0, "sha256": sha256_file(checkpoint) if checkpoint_ok else None},
            "scene-0230 checkpoint containing RigidNodes.",
        ),
        "actor_index_mapping": item(
            "available" if "for id_in_model, (id_in_dataset, v) in enumerate(instance_pts_dict.items())" in rigid_text and "points_ids" in rigid_text else "incompatible",
            {"source": str(rigid_source)},
            "Upstream initialization order and checkpoint point-id selector expose dataset-column→model-index→tensor-slice provenance.",
        ),
        "original_render_command": item(
            "available" if eval_source.is_file() and "--resume_from" in eval_source.read_text() else "missing",
            {"command": "python tools/eval.py --resume_from <checkpoint>"},
            "Native checkpoint render/evaluation entrypoint.",
        ),
        "remove_transform_render_capability": item(
            "available" if "def remove_instances" in rigid_text and "instances_trans" in rigid_text and train_source.is_file() else "incompatible",
            {"remove_api": "RigidNodes.remove_instances", "transform_tensor": "RigidNodes.instances_trans", "render_api": "MultiTrainer.forward"},
            "Low-level native actor mutation and renderer interfaces used by the V2 adapter.",
        ),
        "legacy_mini_path_override": item(
            "available" if dataset_config.is_file() and "args_from_cli" in train_source.read_text() else "incompatible",
            {"default_data_root": data_config.get("data", {}).get("data_root"), "override": f"data.data_root={args.processed_root}", "scene_override": f"data.scene_idx={args.scene_index}"},
            "The upstream mini default is explicit and safely overridden through the native OmegaConf CLI.",
        ),
    }
    payload = {
        "schema_version": 1,
        "baseline": "DriveStudio/StreetGS actor-aware native baseline",
        "scene_name": args.scene_name,
        "scene_index": args.scene_index,
        "selected_path": "path_1_existing_checkpoint" if checkpoint_ok else "path_2_official_training",
        "items": entries,
        "status_counts": {
            status: sum(entry["status"] == status for entry in entries.values())
            for status in sorted(ALLOWED_STATUS)
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "selected_path": payload["selected_path"], "status_counts": payload["status_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
