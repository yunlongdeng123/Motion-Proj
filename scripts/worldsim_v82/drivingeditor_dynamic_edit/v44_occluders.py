"""Measured foreground SAM requests and conservative external-mask certification.

All UVs/masks use the declared source image pixel bins [i,i+1). Pixel/depth
thresholds are explicit. No predictor, geometry mutation, depth interpolation,
box opacity, actor-specific rule or target-mask operation is implemented here.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OccluderConfig:
    max_pixel_gap: float = 12.0
    max_depth_gap: float = 0.5
    relative_depth_gap: float = 0.02
    min_cluster_pixels: int = 3
    max_positives: int = 4
    max_negatives: int = 4
    negative_exclusion_pixels: float = 2.0
    max_proposals: int = 16
    min_score: float = 0.8
    min_semantic_pixels: int = 4
    min_semantic_fraction: float = 0.5

    def __post_init__(self):
        for key in ("max_pixel_gap", "negative_exclusion_pixels"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("max_depth_gap", "relative_depth_gap"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("min_cluster_pixels", "max_positives", "max_negatives", "max_proposals", "min_semantic_pixels"):
            value = getattr(self, key)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not all(np.isfinite(x) and 0 <= x <= 1 for x in (self.min_score, self.min_semantic_fraction)):
            raise ValueError("score and semantic fraction must lie in [0,1]")


def _shape(shape):
    if len(shape) != 2 or any(isinstance(x, bool) or int(x) != x or x < 1 for x in shape):
        raise ValueError("image_shape must be positive integer (height,width)")
    return tuple(int(x) for x in shape)


def _uv(value, shape):
    points = np.asarray(value, dtype=float)
    if points.size == 0:
        points = points.reshape(0, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("UV arrays must have shape (N,2)")
    valid = (np.isfinite(points).all(1) & (points[:, 0] >= 0) & (points[:, 0] < shape[1])
             & (points[:, 1] >= 0) & (points[:, 1] < shape[0]))
    return points, valid


def _semantic(value, shape):
    if value is None:
        return None
    support = np.asarray(value)
    if support.shape != shape or support.dtype.kind != "b":
        raise ValueError("semantic_nonvehicle must be a boolean source-image map")
    return support


def _buckets(points, scale):
    keys = np.floor(points / scale).astype(np.int64)
    buckets = {}
    for i, key in enumerate(keys):
        buckets.setdefault(tuple(key), []).append(i)
    return keys, buckets


def _spread(points, limit):
    if len(points) == 0:
        return np.array([], dtype=int)
    chosen = [int(np.argmin(((points - points.mean(0)) ** 2).sum(1)))]
    nearest = ((points - points[chosen[0]]) ** 2).sum(1)
    while len(chosen) < min(limit, len(points)):
        nearest[chosen] = -1
        i = int(np.argmax(nearest)); chosen.append(i)
        nearest = np.minimum(nearest, ((points - points[i]) ** 2).sum(1))
    return np.asarray(chosen, dtype=int)


def propose_occluders(foreground_uv, foreground_depth, own_supported_uv, image_shape, *,
                      semantic_nonvehicle=None, config=None):
    """Cluster actual returns; select real, dispersed points with source indices.

    Own samples within negative_exclusion_pixels of measured foreground samples
    are retained as projected-overlap evidence, rather than contradictory SAM
    negatives. An independently certified image mask may hide that observation;
    its underlying own sources are always retained. Sparse clusters stay unknown.
    """
    cfg = OccluderConfig() if config is None else config; shape = _shape(image_shape)
    original, valid = _uv(foreground_uv, shape); depth = np.asarray(foreground_depth, float)
    if depth.shape != (len(original),):
        raise ValueError("foreground_depth must have shape (N,)")
    valid &= np.isfinite(depth) & (depth > 0); sources = np.flatnonzero(valid)
    points, z = original[valid], depth[valid]
    own, own_valid = _uv(own_supported_uv, shape); preserved_own = list(range(len(own)))
    own_sources = np.flatnonzero(own_valid); own = own[own_valid]
    sem = _semantic(semantic_nonvehicle, shape); keys, buckets = _buckets(points, cfg.max_pixel_gap)
    seen = np.zeros(len(points), bool); groups = []
    for seed in range(len(points)):
        if seen[seed]: continue
        queue = [seed]; seen[seed] = True; group = []
        while queue:
            i = queue.pop(); group.append(i)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    ids = np.asarray(buckets.get(tuple(keys[i] + [dx, dy]), []), dtype=int)
                    distance = ((points[ids] - points[i]) ** 2).sum(1)
                    gap = np.maximum(cfg.max_depth_gap, cfg.relative_depth_gap * np.minimum(z[ids], z[i]))
                    take = ids[(~seen[ids]) & (distance <= cfg.max_pixel_gap ** 2) & (abs(z[ids] - z[i]) <= gap)]
                    seen[take] = True; queue.extend(take.tolist())
        groups.append(np.asarray(group, dtype=int))
    proposals, unknown = [], []
    for group in groups:
        _, unique = np.unique(np.floor(points[group]).astype(int), axis=0, return_index=True)
        members = group[unique]
        if len(members) < cfg.min_cluster_pixels or len(proposals) >= cfg.max_proposals:
            unknown.append({"reason": "sparse_measurements" if len(members) < cfg.min_cluster_pixels else "proposal_limit", "source_indices": sources[group].tolist()}); continue
        positive = members[_spread(points[members], cfg.max_positives)]
        overlap = np.zeros(len(own), bool); scale = cfg.negative_exclusion_pixels
        _, near_buckets = _buckets(points[group], scale)
        for i, point in enumerate(own):
            key = np.floor(point / scale).astype(int)
            ids = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in near_buckets.get(tuple(key + [dx, dy]), [])]
            overlap[i] = bool(ids) and bool((((points[group[np.asarray(ids)]] - point) ** 2).sum(1) <= scale ** 2).any())
        candidates = np.flatnonzero(~overlap)
        _, unique_own = np.unique(np.floor(own[candidates]).astype(int), axis=0, return_index=True)
        candidates = candidates[unique_own]; negative = candidates[_spread(own[candidates], cfg.max_negatives)]
        sample = np.floor(points[members]).astype(int)
        proposals.append({"status": "proposal", "image_shape": shape, "measured_source_pixels": len(members),
                          "measured_uv": points[members].tolist(),
                          "source_indices": sources[group].tolist(),
                          "positive_indices": sources[positive].tolist(), "positive_uv": points[positive].tolist(),
                          "positive_depth": z[positive].tolist(),
                          "negative_indices": own_sources[negative].tolist(), "negative_uv": own[negative].tolist(),
                          "overlapping_own_indices": own_sources[overlap].tolist(), "overlapping_own_uv": own[overlap].tolist(),
                          "preserved_own_indices": preserved_own.copy(),
                          "semantic_point_support": None if sem is None else int(sem[sample[:, 1], sample[:, 0]].sum())})
    if not len(points): unknown.append({"reason": "no_valid_foreground_measurements", "source_indices": []})
    return {"proposals": proposals, "unknown": unknown, "invalid_foreground_indices": np.flatnonzero(~valid).tolist()}


def certify_occluder(proposal, mask, score, *, semantic_nonvehicle=None, config=None):
    """Certify an external image mask; return accepted_occluder or unknown.

    Every measured positive must be covered; no selected own negative may be
    covered. Semantic support is required inside this independent image mask.
    Acceptance can mark overlapping own observations hidden, without removing
    or reclassifying their underlying measured sources or any 3-D geometry.
    """
    cfg = OccluderConfig() if config is None else config; metrics = {}
    def unknown(reason):
        return {"status": "unknown", "reason": reason, "accepted_occluder": None, "metrics": metrics,
                "preserved_own_indices": list(proposal.get("preserved_own_indices", [])), "hidden_own_indices": []}
    if (proposal.get("status") != "proposal" or not proposal.get("positive_indices")
            or proposal.get("measured_source_pixels", 0) < cfg.min_cluster_pixels):
        return unknown("no_measured_proposal")
    shape = _shape(proposal["image_shape"]); arr = np.asarray(mask)
    if arr.shape != shape or not np.isin(arr, [0, 1, 255]).all():
        return unknown("invalid_source_mask")
    arr = arr.astype(bool)
    score_array = np.asarray(score)
    if score_array.ndim or score_array.dtype.kind not in "fiu": return unknown("insufficient_predictor_score")
    score = float(score_array)
    if not np.isfinite(score) or score < cfg.min_score or score > 1:
        return unknown("insufficient_predictor_score")
    positive, pv = _uv(proposal["positive_uv"], shape); negative, nv = _uv(proposal["negative_uv"], shape)
    if not pv.all() or not nv.all() or not len(positive): return unknown("invalid_measured_prompts")
    pos = np.floor(positive).astype(int); neg = np.floor(negative).astype(int)
    metrics.update(positive_covered=int(arr[pos[:, 1], pos[:, 0]].sum()), positive_count=len(pos),
                   own_negative_covered=int(arr[neg[:, 1], neg[:, 0]].sum()))
    if metrics["positive_covered"] != len(pos): return unknown("measured_positive_not_covered")
    if metrics["own_negative_covered"]: return unknown("own_negative_conflict")
    if 'measured_uv' in proposal:
        measured, valid = _uv(proposal['measured_uv'], shape)
        if not valid.all(): return unknown('invalid_measured_cluster')
        pixels = np.unique(np.floor(measured).astype(int), axis=0)
        hit = int(arr[pixels[:, 1], pixels[:, 0]].sum())
        metrics['covered_measured_source_pixels'] = hit
        if hit < cfg.min_cluster_pixels: return unknown('insufficient_measured_mask_support')
    try: sem = _semantic(semantic_nonvehicle, shape)
    except ValueError: return unknown("invalid_semantic_support")
    if sem is None: return unknown("semantic_support_unknown")
    count = int((arr & sem).sum()); fraction = count / max(1, int(arr.sum()))
    metrics.update(nonvehicle_pixels=count, nonvehicle_fraction=fraction, predictor_score=float(score))
    if count < cfg.min_semantic_pixels or fraction < cfg.min_semantic_fraction: return unknown("insufficient_nonvehicle_support")
    overlap, valid = _uv(proposal["overlapping_own_uv"], shape); uv = np.floor(overlap[valid]).astype(int)
    hidden = np.asarray(proposal["overlapping_own_indices"], int)[valid][arr[uv[:, 1], uv[:, 0]]].tolist()
    return {"status": "accepted_occluder", "reason": "independent_image_mask_certified", "accepted_occluder": arr.copy(),
            "metrics": metrics, "preserved_own_indices": proposal["preserved_own_indices"].copy(), "hidden_own_indices": hidden}
