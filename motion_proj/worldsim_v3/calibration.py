"""WorldSim V3 calibration modules for controlled A1 ablations."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn


def _first_scalar(value: Tensor) -> Tensor:
    if not torch.is_tensor(value) or value.numel() == 0:
        raise ValueError("expected a non-empty tensor")
    return value.reshape(-1)[0]


def _bounded_vector(raw: Tensor, maximum_norm: float) -> Tensor:
    if maximum_norm <= 0:
        raise ValueError("maximum_norm must be positive")
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    scale = maximum_norm * torch.tanh(norm) / torch.clamp(norm, min=1e-12)
    return raw * scale


def axis_angle_to_matrix(rotation_vectors: Tensor) -> Tensor:
    """Convert axis-angle vectors to rotation matrices with stable sinc terms."""

    if rotation_vectors.shape[-1] != 3:
        raise ValueError("axis-angle vectors must end in dimension three")
    x, y, z = rotation_vectors.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1
    ).reshape(rotation_vectors.shape[:-1] + (3, 3))
    theta = torch.linalg.vector_norm(rotation_vectors, dim=-1)
    first = torch.sinc(theta / math.pi)
    second = 0.5 * torch.sinc(theta / (2.0 * math.pi)).square()
    identity = torch.eye(
        3, dtype=rotation_vectors.dtype, device=rotation_vectors.device
    ).expand(rotation_vectors.shape[:-1] + (3, 3))
    return identity + first[..., None, None] * skew + second[..., None, None] * (
        skew @ skew
    )


class FactorizedAffineTransform(nn.Module):
    """Camera identity plus continuous-time RGB affine residual.

    Unlike DriveStudio's native per-image embedding, this module evaluates the same
    continuous function on held-out timesteps. No exposure metadata is claimed.
    """

    def __init__(
        self,
        class_name: str,
        n: int,
        num_cameras: int = 3,
        camera_embedding_dim: int = 4,
        time_embedding_dim: int = 8,
        num_time_frequencies: int = 2,
        base_mlp_layer_width: int = 64,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        if n <= 0 or num_cameras <= 0 or n % num_cameras != 0:
            raise ValueError("n must be positive and divisible by num_cameras")
        if num_time_frequencies < 0:
            raise ValueError("num_time_frequencies must be non-negative")
        self.class_prefix = class_name + "#"
        self.device = device
        self.num_images = int(n)
        self.num_cameras = int(num_cameras)
        self.num_frames = int(n // num_cameras)
        self.num_time_frequencies = int(num_time_frequencies)
        self.camera_embedding = nn.Embedding(num_cameras, camera_embedding_dim)
        time_feature_dim = 1 + 2 * num_time_frequencies
        self.time_encoder = nn.Sequential(
            nn.Linear(time_feature_dim, time_embedding_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(camera_embedding_dim + time_embedding_dim, base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(base_mlp_layer_width, 12),
        )
        self.in_test_set = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.camera_embedding.weight)
        nn.init.xavier_uniform_(self.time_encoder[0].weight)
        nn.init.zeros_(self.time_encoder[0].bias)
        nn.init.xavier_uniform_(self.decoder[0].weight)
        nn.init.zeros_(self.decoder[0].bias)
        nn.init.zeros_(self.decoder[2].weight)
        nn.init.zeros_(self.decoder[2].bias)

    def _time_features(self, normalized_time: Tensor) -> Tensor:
        values = [normalized_time]
        for frequency in range(self.num_time_frequencies):
            angle = normalized_time * (2.0 ** frequency) * (2.0 * math.pi)
            values.extend((torch.sin(angle), torch.cos(angle)))
        return torch.stack(values, dim=-1)

    def forward(self, image_infos: dict[str, Tensor]) -> Tensor:
        image_index = image_infos.get("img_idx")
        if image_index is None:
            camera_embedding = self.camera_embedding.weight.mean(dim=0)
            normalized_time = torch.tensor(
                0.5, device=camera_embedding.device, dtype=camera_embedding.dtype
            )
        else:
            scalar_index = _first_scalar(image_index).long()
            camera_index = torch.remainder(scalar_index, self.num_cameras)
            camera_embedding = self.camera_embedding(camera_index)
            if "normed_time" in image_infos:
                normalized_time = _first_scalar(image_infos["normed_time"]).to(
                    dtype=camera_embedding.dtype
                )
            else:
                frame_index = torch.div(
                    scalar_index, self.num_cameras, rounding_mode="floor"
                )
                denominator = max(self.num_frames - 1, 1)
                normalized_time = frame_index.to(camera_embedding.dtype) / denominator
        time_embedding = self.time_encoder(self._time_features(normalized_time))
        residual = self.decoder(torch.cat((camera_embedding, time_embedding), dim=-1))
        affine = residual.reshape(3, 4)
        identity = torch.eye(3, dtype=affine.dtype, device=affine.device)
        return torch.cat((affine[:, :3] + identity, affine[:, 3:]), dim=-1)

    def get_param_groups(self) -> dict[str, object]:
        return {self.class_prefix + "all": self.parameters()}


class BoundedCameraOptModule(nn.Module):
    """Per-image pose residual with explicit norm bounds and temporal priors."""

    def __init__(
        self,
        class_name: str,
        n: int,
        num_cameras: int = 3,
        max_translation_m: float = 0.15,
        max_rotation_deg: float = 2.0,
        translation_prior_weight: float = 1e-4,
        rotation_prior_weight: float = 1e-4,
        temporal_smoothness_weight: float = 1e-3,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        if n <= 0 or num_cameras <= 0 or n % num_cameras != 0:
            raise ValueError("n must be positive and divisible by num_cameras")
        if max_rotation_deg <= 0:
            raise ValueError("max_rotation_deg must be positive")
        self.class_prefix = class_name + "#"
        self.device = device
        self.num_images = int(n)
        self.num_cameras = int(num_cameras)
        self.num_frames = int(n // num_cameras)
        self.max_translation_m = float(max_translation_m)
        self.max_rotation_rad = math.radians(float(max_rotation_deg))
        self.translation_prior_weight = float(translation_prior_weight)
        self.rotation_prior_weight = float(rotation_prior_weight)
        self.temporal_smoothness_weight = float(temporal_smoothness_weight)
        self.embeds = nn.Embedding(n, 6)
        nn.init.zeros_(self.embeds.weight)

    def bounded_residuals(self) -> tuple[Tensor, Tensor]:
        raw = self.embeds.weight
        translation = _bounded_vector(raw[:, :3], self.max_translation_m)
        rotation = _bounded_vector(raw[:, 3:], self.max_rotation_rad)
        return translation, rotation

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        if camtoworlds.shape[:-2] != embed_ids.shape:
            raise ValueError("camera batch shape and embed_ids shape must match")
        raw = self.embeds(embed_ids)
        translation = _bounded_vector(raw[..., :3], self.max_translation_m)
        rotation_vector = _bounded_vector(raw[..., 3:], self.max_rotation_rad)
        rotation = axis_angle_to_matrix(rotation_vector)
        transform = torch.eye(
            4, dtype=camtoworlds.dtype, device=camtoworlds.device
        ).expand(camtoworlds.shape[:-2] + (4, 4)).clone()
        transform[..., :3, :3] = rotation.to(camtoworlds.dtype)
        transform[..., :3, 3] = translation.to(camtoworlds.dtype)
        return camtoworlds @ transform

    def compute_regularization(self) -> dict[str, Tensor]:
        translation, rotation = self.bounded_residuals()
        translation_prior = translation.square().sum(dim=-1).mean() / (
            self.max_translation_m**2
        )
        rotation_prior = rotation.square().sum(dim=-1).mean() / (
            self.max_rotation_rad**2
        )
        translation_grid = translation.reshape(self.num_frames, self.num_cameras, 3)
        rotation_grid = rotation.reshape(self.num_frames, self.num_cameras, 3)
        if self.num_frames > 1:
            translation_delta = translation_grid[1:] - translation_grid[:-1]
            rotation_delta = rotation_grid[1:] - rotation_grid[:-1]
            temporal = (
                translation_delta.square().sum(dim=-1).mean()
                / (self.max_translation_m**2)
                + rotation_delta.square().sum(dim=-1).mean()
                / (self.max_rotation_rad**2)
            )
        else:
            temporal = translation_prior.new_zeros(())
        return {
            "translation_prior": self.translation_prior_weight * translation_prior,
            "rotation_prior": self.rotation_prior_weight * rotation_prior,
            "temporal_smoothness": self.temporal_smoothness_weight * temporal,
        }

    def get_param_groups(self) -> dict[str, object]:
        return {self.class_prefix + "all": self.parameters()}
