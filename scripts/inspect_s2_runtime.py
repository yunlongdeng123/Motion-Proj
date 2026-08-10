#!/usr/bin/env python
"""只读检查 S2 所需的 DriveStudio 相机、渲染与 checkpoint schema。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.lift_worldsim_v32_semantics import build_runtime
from scripts.eval_worldsim_v3_a3_r1_heldout import get_view_data


def shape(value):
    return list(value.shape) if hasattr(value, "shape") else str(type(value))


def main() -> None:
    config = yaml.safe_load(
        (PROJECT / "configs/worldsim_v32/s2_3dgic_v1.yaml").read_text(encoding="utf-8")
    )
    torch.cuda.set_device(0)
    torch.cuda.init()
    device = torch.device("cuda:0")
    dataset, trainer = build_runtime(config, device)
    image_infos, camera_infos, gt, measured, egocar, image_index = get_view_data(
        dataset, 91, 1, device
    )
    with torch.inference_mode():
        outputs = trainer(image_infos, camera_infos)
    camera = trainer.process_camera(camera_infos, image_infos["img_idx"].flatten()[0])
    background = trainer.models["Background"]
    rigid = trainer.models["RigidNodes"]
    masks = {}
    for name in ("high_support", "boundary_support"):
        path = Path(config["targets"][name]["mask"])
        with np.load(path) as payload:
            masks[name] = {key: shape(payload[key]) for key in payload.files}
    result = {
        "image_index": image_index,
        "num_cams": dataset.pixel_source.num_cams,
        "image_infos": {key: shape(value) for key, value in image_infos.items()},
        "camera_infos": {key: shape(value) for key, value in camera_infos.items()},
        "outputs": {key: shape(value) for key, value in outputs.items()},
        "processed_camera": {
            "camtoworlds": shape(camera.camtoworlds),
            "Ks": shape(camera.Ks),
            "H": shape(camera.H),
            "W": shape(camera.W),
        },
        "background_count": int(background.num_points),
        "background_ancestry": background._a2_ancestry.summary(),
        "rigid_count": int(rigid.num_instances),
        "rigid_visibility_shape": shape(rigid.instances_fv),
        "groundtruth": shape(gt),
        "measured_depth": shape(measured),
        "egocar": shape(egocar),
        "masks": masks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
