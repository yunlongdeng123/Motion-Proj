"""DriveStudio trainer adapter for WorldSim V3 calibration experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from models.trainers.scene_graph import MultiTrainer


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class WorldSimV3Trainer(MultiTrainer):
    """Add bounded-pose losses and exact initialization provenance capture."""

    def compute_losses(self, outputs, image_infos, cam_infos):
        losses = super().compute_losses(outputs, image_infos, cam_infos)
        camera_model = self.models.get("CamPose")
        if camera_model is not None and hasattr(camera_model, "compute_regularization"):
            for name, value in camera_model.compute_regularization().items():
                losses[f"worldsim_cam_pose_{name}"] = value
        return losses

    def init_gaussians_from_dataset(self, dataset) -> None:
        destination_value = os.environ.get("WORLDSIM_V3_INIT_PROVENANCE")
        if not destination_value:
            return super().init_gaussians_from_dataset(dataset)

        destination = Path(destination_value)
        seed = int(os.environ.get("WORLDSIM_V3_INIT_SEED", "0"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        captured: dict[str, object] = {
            "schema_version": 1,
            "truth_tier": "exact_runtime_initialization_inputs",
            "rng_reset": {
                "seed": seed,
                "python": True,
                "numpy": True,
                "torch_cpu": True,
                "torch_cuda_all": bool(torch.cuda.is_available()),
                "location": "immediately_before_gaussian_initialization",
            },
            "background_lidar_sample": None,
            "instance_lidar_samples": {},
        }
        original_lidar = dataset.get_lidar_samples
        original_objects = dataset.get_init_objects

        def capture_lidar(*args, **kwargs):
            points, colors, times = original_lidar(*args, **kwargs)
            captured["background_lidar_sample"] = {
                "point_count": int(points.shape[0]),
                "points_sha256": _tensor_sha256(points),
                "colors_sha256": _tensor_sha256(colors) if colors is not None else None,
                "times_sha256": _tensor_sha256(times) if times is not None else None,
            }
            return points, colors, times

        def capture_objects(*args, **kwargs):
            result = original_objects(*args, **kwargs)
            captured["instance_lidar_samples"] = {
                str(instance_id): {
                    "node_type": row["node_type"],
                    "point_count": int(row["pts"].shape[0]),
                    "points_sha256": _tensor_sha256(row["pts"]),
                    "colors_sha256": _tensor_sha256(row["colors"]),
                    "visible_frame_count": int(row["frame_info"].sum().item()),
                }
                for instance_id, row in sorted(result.items())
            }
            return result

        dataset.get_lidar_samples = capture_lidar
        dataset.get_init_objects = capture_objects
        try:
            super().init_gaussians_from_dataset(dataset)
        finally:
            dataset.get_lidar_samples = original_lidar
            dataset.get_init_objects = original_objects

        captured["initialized_gaussians"] = {
            name: int(model.num_points)
            for name, model in self.models.items()
            if hasattr(model, "num_points")
        }
        captured["limitations"] = [
            "background post-box-filter LiDAR identity is not retained separately from random near/far seeds",
            "post-densification checkpoints do not retain initialization ancestry per Gaussian",
        ]
        _atomic_json(destination, captured)
