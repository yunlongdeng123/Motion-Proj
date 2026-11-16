"""Conservative evidence from actual projected returns; no opaque-box occlusion.

UV coordinates use source pixel bins [i, i+1). ``image_shape`` is the actual
source (height, width); ``output_shape`` explicitly selects the query grid.
All depths must share one convention, e.g. camera-forward X in metres. Query
volume/surface maps are already expressed on that output grid. ``own_source``
marks real source returns belonging to the query, not filled pixels in its OBB.
Other OBBs, semantic classes, neighbourhood minima and empty space never supply
measurements. No depth interpolation or morphological expansion is performed.
"""
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class Evidence(IntEnum):
    UNKNOWN = 0
    SUPPORTED = 1
    MIXED = 2
    MEASURED_FOREGROUND = 3


@dataclass(frozen=True)
class VisibilityConfig:
    foreground_margin: float = 0.5
    min_foreground_pixels: int = 2
    min_foreground_returns: int = 2

    def __post_init__(self):
        if not np.isfinite(self.foreground_margin) or self.foreground_margin < 0:
            raise ValueError("foreground_margin must be finite and nonnegative")
        for name in ("min_foreground_pixels", "min_foreground_returns"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _shape(value, name):
    if len(value) != 2 or any(isinstance(x, bool) or int(x) != x or x < 1 for x in value):
        raise ValueError(f"{name} must contain positive integer (height, width)")
    return tuple(int(x) for x in value)


def _grid(value, shape, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(shape, arr.item(), dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must be scalar or have output_shape {shape}")
    return arr


def visibility_evidence(projected_uv, depth, own_source, image_shape, *,
                        output_shape=None, own_near=None, own_far=None,
                        expected_depth=None, surface_owner=None, query_owner=None,
                        unknown_owner=None, semantic_hint=None, config=None):
    """Return evidence maps and counts without altering an input mask.

    Volume near/far or expected surface depth is required. NaN/invalid query
    depth produces no foreground claim; measurements behind a volume remain
    UNKNOWN unless their actual source is owned by the query. A finite surface
    depth takes precedence over volume near depth. Ownership is stronger than
    approximate query geometry: an own return never becomes a foreground veto.

    Optional surface_owner contains measured-return identities (not projected
    other-box IDs). Known identities override own_source; unknown_owner leaves
    it unchanged. IDs are opaque and only compared with query_owner.

    Counts ending in ``pixels`` deduplicate (output cell, native source pixel).
    Raw-return counts are separate. MIXED always wins over a foreground claim,
    even with only one own return. Sparse foreground remains UNKNOWN. Semantic
    hints are returned for diagnostics and cannot change any evidence state.
    """
    cfg = VisibilityConfig() if config is None else config
    src = _shape(image_shape, "image_shape")
    dst = src if output_shape is None else _shape(output_shape, "output_shape")
    uv = np.asarray(projected_uv, dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)
    own = np.asarray(own_source)
    if uv.ndim != 2 or uv.shape[1] != 2 or z.shape != (len(uv),):
        raise ValueError("projected_uv must be (N,2) and depth must be (N,)")
    if own.shape != z.shape or own.dtype.kind != "b":
        raise ValueError("own_source must be a boolean (N,) measured-source membership")
    own = own.copy()
    if surface_owner is not None:
        owners = np.asarray(surface_owner, dtype=object)
        if owners.shape != z.shape or query_owner is None or np.ndim(query_owner):
            raise ValueError("surface_owner needs (N,) identities and a scalar query_owner")
        known = owners != unknown_owner
        own[known] = owners[known] == query_owner
    if (own_near is None) != (own_far is None):
        raise ValueError("own_near and own_far must be supplied together")
    if own_near is None and expected_depth is None:
        raise ValueError("supply volume near/far or expected_depth")
    reference = np.full(dst, np.nan)
    if own_near is not None:
        near = _grid(own_near, dst, "own_near")
        far = _grid(own_far, dst, "own_far")
        good = np.isfinite(near) & np.isfinite(far) & (near > 0) & (far >= near)
        reference[good] = near[good]
    if expected_depth is not None:
        expected = _grid(expected_depth, dst, "expected_depth")
        good = np.isfinite(expected) & (expected > 0)
        reference[good] = expected[good]
    hint = None if semantic_hint is None else np.asarray(semantic_hint)
    if hint is not None and hint.shape != dst:
        raise ValueError("semantic_hint must have output_shape")
    valid = (np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
             & (uv[:, 0] >= 0) & (uv[:, 0] < src[1])
             & (uv[:, 1] >= 0) & (uv[:, 1] < src[0]))
    points, values, owned = uv[valid], z[valid], own[valid]
    native_xy = np.floor(points).astype(np.int64)
    query_xy = np.floor(points * [dst[1] / src[1], dst[0] / src[0]]).astype(np.int64)
    cell = query_xy[:, 1] * dst[1] + query_xy[:, 0]
    native = native_xy[:, 1] * src[1] + native_xy[:, 0]
    ref = reference.ravel()[cell]
    front = ~owned & np.isfinite(ref) & (values < ref - cfg.foreground_margin)
    size, native_size = int(np.prod(dst)), int(np.prod(src))

    def returns(selection):
        return np.bincount(cell[selection], minlength=size).astype(np.int64).reshape(dst)

    def pixels(selection):
        pairs = np.unique(cell[selection] * native_size + native[selection])
        return np.bincount(pairs // native_size, minlength=size).astype(np.int64).reshape(dst)

    all_points = np.ones(len(cell), dtype=bool)
    measured_returns, measured_pixels = returns(all_points), pixels(all_points)
    own_returns, own_pixels = returns(owned), pixels(owned)
    foreground_returns, foreground_pixels = returns(front), pixels(front)
    supported = own_pixels > 0
    mixed = supported & (foreground_pixels > 0)
    sufficient = ((foreground_pixels >= cfg.min_foreground_pixels)
                  & (foreground_returns >= cfg.min_foreground_returns))
    conflict = sufficient & ~supported
    state = np.full(dst, Evidence.UNKNOWN, dtype=np.uint8)
    state[supported] = Evidence.SUPPORTED
    state[mixed] = Evidence.MIXED
    state[conflict] = Evidence.MEASURED_FOREGROUND
    return {"state": state, "conflict": conflict,
            "measured_pixels": measured_pixels, "measured_returns": measured_returns,
            "own_pixels": own_pixels, "own_returns": own_returns,
            "foreground_pixels": foreground_pixels, "foreground_returns": foreground_returns,
            "unassigned_pixels": pixels(~owned & ~front),
            "semantic_hint": None if hint is None else hint.copy(),
            "counts": {"state_pixels": {s.name: int((state == s).sum()) for s in Evidence},
                       "valid_returns": int(valid.sum()), "rejected_returns": int((~valid).sum()),
                       "unique_native_pixels": int(len(np.unique(native))),
                       "own_returns": int(owned.sum()), "foreground_returns": int(front.sum())},
            "image_shape": src, "output_shape": dst}
