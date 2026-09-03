"""FREE/OCCUPIED/UNKNOWN 连续证据质量。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EvidentialQueries:
    masses: np.ndarray
    opportunity_count: np.ndarray
    free_count: np.ndarray
    occupied_count: np.ndarray

    @property
    def free(self) -> np.ndarray:
        return self.masses[:, 0]

    @property
    def occupied(self) -> np.ndarray:
        return self.masses[:, 1]

    @property
    def unknown(self) -> np.ndarray:
        return self.masses[:, 2]


def build_evidential_queries(
    query_points: np.ndarray,
    ray_origins: np.ndarray,
    ray_endpoints: np.ndarray,
    *,
    beam_radius_m: float,
    endpoint_radius_m: float,
    device: torch.device,
    query_chunk_size: int = 512,
) -> EvidentialQueries:
    """按可用射线机会归一化，未穿过的空间保持 UNKNOWN。"""
    queries = torch.as_tensor(query_points, dtype=torch.float32, device=device).reshape(-1, 3)
    origins = torch.as_tensor(ray_origins, dtype=torch.float32, device=device).reshape(-1, 3)
    endpoints = torch.as_tensor(ray_endpoints, dtype=torch.float32, device=device).reshape(-1, 3)
    if len(origins) != len(endpoints):
        raise ValueError("ray_origins 与 ray_endpoints 数量不一致")
    ray_vectors = endpoints - origins
    ray_depth = torch.linalg.vector_norm(ray_vectors, dim=1).clamp_min(1.0e-6)
    directions = ray_vectors / ray_depth[:, None]
    opportunities, free_counts, occupied_counts = [], [], []
    with torch.inference_mode():
        for start in range(0, len(queries), int(query_chunk_size)):
            chunk = queries[start : start + query_chunk_size]
            relative = chunk[:, None, :] - origins[None, :, :]
            depth = torch.sum(relative * directions[None, :, :], dim=-1)
            lateral = torch.linalg.vector_norm(
                relative - depth[:, :, None] * directions[None, :, :], dim=-1
            )
            opportunity = (depth > 0.0) & (lateral <= float(beam_radius_m))
            endpoint_distance = torch.linalg.vector_norm(
                chunk[:, None, :] - endpoints[None, :, :], dim=-1
            )
            occupied = opportunity & (endpoint_distance <= float(endpoint_radius_m))
            free = opportunity & (depth < ray_depth[None, :] - float(endpoint_radius_m))
            opportunities.append(opportunity.sum(dim=1).cpu())
            free_counts.append(free.sum(dim=1).cpu())
            occupied_counts.append(occupied.sum(dim=1).cpu())
    opportunity = torch.cat(opportunities).numpy().astype(np.int32) if opportunities else np.empty(0, dtype=np.int32)
    free_count = torch.cat(free_counts).numpy().astype(np.int32) if free_counts else np.empty(0, dtype=np.int32)
    occupied_count = torch.cat(occupied_counts).numpy().astype(np.int32) if occupied_counts else np.empty(0, dtype=np.int32)
    denominator = np.maximum(opportunity.astype(np.float32), 1.0)
    free_mass = free_count.astype(np.float32) / denominator
    occupied_mass = occupied_count.astype(np.float32) / denominator
    known_mass = np.clip(free_mass + occupied_mass, 0.0, 1.0)
    unknown_mass = np.where(opportunity > 0, 1.0 - known_mass, 1.0).astype(np.float32)
    masses = np.column_stack([free_mass, occupied_mass, unknown_mass]).astype(np.float32)
    masses /= np.maximum(masses.sum(axis=1, keepdims=True), 1.0e-6)
    return EvidentialQueries(masses, opportunity, free_count, occupied_count)


def evidential_cross_entropy(predicted_logits: torch.Tensor, target_masses: torch.Tensor) -> torch.Tensor:
    if predicted_logits.shape != target_masses.shape or predicted_logits.shape[-1] != 3:
        raise ValueError("evidential logits/target 必须同为 [N,3]")
    return -(target_masses * torch.log_softmax(predicted_logits, dim=-1)).sum(dim=-1).mean()
