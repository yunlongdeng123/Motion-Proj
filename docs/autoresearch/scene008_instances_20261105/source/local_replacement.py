"""Persistent row retirement with checkpoint-bound and view-paired evidence.

The inherited road gate is part of every comparison. Newly measured static
retirement coefficients are shared across cameras, timestamps and edits.
Historical and projected candidate rows alone have no authority to retire.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


RAW_IDS = tuple(range(89, 138))
CAMPAIGN = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
NATIVE_REL = Path("scene008_a37_only_B10k_20261103_r1/formal10k")
ROAD_REL = Path("scene008_a37_only_B10k_road_20261103_r3")
C2_REL = Path("scene008_a37_C2_20261103_r1")
VEHICLE_REL = Path("scene008_a37_C2_vehicle_20261104_r1")
HISTORICAL_REL = Path("scene008_static_origin_diag_20261104_r1/evidence/attribute_historical_native_10000")
ROAD = 1
HISTORICAL_C2 = 2
HISTORICAL_RAW113 = 4
HISTORICAL_OTHER_CAR = 8
PROJECTED_SUPPORT = 16
# At unit opacity, the native per-pixel alpha cutoff of 1/255 bounds support
# at this Mahalanobis radius. Actual opacity and native tiles can only narrow it.
SUPPORT_SIGMAS = float(np.sqrt(2 * np.log(255)))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return dict(arrays)


def _publish_new(temporary, path):
    # Publish without replacing a concurrently created artifact.
    os.link(temporary, path)
    temporary.unlink()


def _row_ids(ids, count):
    ids = np.asarray(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("row ids must be a one-dimensional integer array")
    if np.any((ids < 0) | (ids >= count)) or len(np.unique(ids)) != len(ids):
        raise ValueError("row ids must be unique and refer to the bound static checkpoint")
    return ids.astype(np.int64, copy=False)


def load_ledger(path, static_count=None, checkpoint_sha256=None, road_checkpoint_sha256=None):
    ledger = _read_npz(path)
    count = int(ledger["static_count"])
    if count < 1 or (static_count is not None and static_count != count):
        raise ValueError("replacement ledger static row count mismatch")
    if checkpoint_sha256 is not None and ledger["checkpoint_sha256"].item() != checkpoint_sha256:
        raise ValueError("replacement ledger static checkpoint fingerprint mismatch")
    if road_checkpoint_sha256 is not None and ledger["road_checkpoint_sha256"].item() != road_checkpoint_sha256:
        raise ValueError("replacement ledger road checkpoint fingerprint mismatch")
    if tuple(ledger["raw_ids"]) != RAW_IDS:
        raise ValueError("replacement ledger must be bound to the complete Scene008 evaluation span")
    ids = _row_ids(ledger["ids"], count)
    if np.any(ids[1:] <= ids[:-1]):
        raise ValueError("replacement ledger ids must be in native row order")
    for name in ("road_keep", "max_retirement"):
        value = ledger[name]
        if value.shape != ids.shape or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError(f"{name} must have one finite coefficient in [0,1] per row")
    return ledger


class PersistentReplacement(torch.nn.Module):
    """Opacity-only transform applied before rows enter the common raster."""

    def __init__(self, ledger, static_count=None):
        super().__init__()
        self.static_count = int(ledger["static_count"])
        if static_count is not None and static_count != self.static_count:
            raise ValueError("replacement row count mismatch")
        ids = _row_ids(ledger["ids"], self.static_count)
        road_keep = np.asarray(ledger["road_keep"])
        cap = np.asarray(ledger["max_retirement"])
        for name, values in (("road_keep", road_keep), ("max_retirement", cap)):
            if values.shape != ids.shape or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"invalid {name} array")
        road = road_keep != 1
        eligible = cap > 0
        self.register_buffer("road_ids", torch.as_tensor(ids[road]))
        self.register_buffer("road_keep", torch.as_tensor(road_keep[road], dtype=torch.float32))
        self.register_buffer("ids", torch.as_tensor(ids[eligible]))
        self.register_buffer("max_retirement", torch.as_tensor(cap[eligible], dtype=torch.float32))
        # Exact identity initialization with a usable derivative at the endpoint.
        self.retirement = torch.nn.Parameter(torch.zeros(len(self.ids)))
        self.dynamic_retirement = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("dynamic_retired", torch.tensor(False))

    def keep(self):
        return 1.0 - self.max_retirement * self.retirement.clamp(0, 1)

    def apply(self, kw, enabled=True):
        opacity = kw["opacities"]
        if opacity.shape != (self.static_count, 1):
            raise ValueError("replacement must precede raster row filtering and merging")
        if self.road_ids.numel():
            opacity = opacity.index_copy(0, self.road_ids, opacity[self.road_ids] * self.road_keep[:, None])
        if enabled and self.ids.numel():
            opacity = opacity.index_copy(0, self.ids, opacity[self.ids] * self.keep()[:, None])
        return {**kw, "opacities": opacity}

    def apply_dynamic(self, kw, enabled=True):
        if not enabled:
            return kw
        keep = (1.0 - self.dynamic_retirement.clamp(0, 1)) * (~self.dynamic_retired)
        return {**kw, "opacities": kw["opacities"] * keep}

    @torch.no_grad()
    def retire_dynamic(self):
        """Latch zero after paired takeover acceptance; caller records evaluation."""
        self.dynamic_retired.fill_(True)

    @torch.no_grad()
    def project_parameters(self):
        self.retirement.clamp_(0, 1)
        self.dynamic_retirement.clamp_(0, 1)

    def regularizer(self):
        return (1.0 - self.keep()).mean() if self.ids.numel() else self.retirement.sum()

    def verify_state_binding(self, state):
        for name in ("ids", "road_ids", "road_keep", "max_retirement"):
            expected = getattr(self, name)
            actual = state[name]
            if actual.dtype != expected.dtype or not torch.equal(actual.to(expected.device), expected):
                raise ValueError(f"replacement checkpoint binding mismatch: {name}")


def frame_attribution(marker, static_marker, domains):
    """Return exact native alpha*T per row and domain from marker gradients.

    The joint render must have appended one zero-valued feature per domain on
    static rows and zeros on every other component. Keeping the marker sparse
    is possible: scatter candidate markers into static rows before tagging.
    """
    if marker.ndim != 2 or static_marker.ndim != 3 or static_marker.shape[0] != marker.shape[1]:
        raise ValueError("marker and raster feature channels must match")
    if len(domains) != marker.shape[1] or any(d.shape != static_marker.shape[1:] or d.dtype != torch.bool for d in domains):
        raise ValueError("attribution domains must be booleans at render resolution")
    total = sum(static_marker[i][domain].sum() for i, domain in enumerate(domains))
    result, = torch.autograd.grad(total, marker, only_inputs=True)
    if not torch.isfinite(result).all() or (result < -1e-6).any():
        raise ValueError("invalid row marker contributions")
    return result.detach().clamp_min(0)


class ReplacementAttribution:
    """Collect current evidence before permitting any additional row retirement."""

    def __init__(self, ledger, projections, raw_ids=RAW_IDS):
        self.ledger = ledger
        self.projections = projections
        self.raw_ids = tuple(raw_ids)
        for name in ("raw_ids", "checkpoint_sha256", "observation_fingerprint"):
            if name in ledger and (name not in projections or not np.array_equal(ledger[name], projections[name])):
                raise ValueError(f"projection evidence binding mismatch: {name}")
        count = len(ledger["ids"])
        self.fit = np.zeros((count, 2), dtype=np.float64)
        self.guard = np.zeros(count, dtype=np.float64)
        self.worst_guard = np.zeros(count, dtype=np.float32)
        self.frames = set()

    def add(self, raw, include_a37, scores):
        """Scores are in ledger order and columns are a6, a37 and guard.

        a6/a37 domains may use the fixed verified observation masks or vetted
        proposal readout domains. Support/depth witnesses are applied per view
        before scores contribute. Guard contributions are counted in every view.
        """
        key = (raw, bool(include_a37))
        if raw not in self.raw_ids or key in self.frames:
            raise ValueError(f"unexpected or repeated attribution view {key}")
        scores = np.asarray(scores)
        if scores.shape != (len(self.fit), 3) or not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("invalid attribution score array")
        ids = self.projections[f"ids_{raw:03d}"]
        flags = self.projections[f"flags_{raw:03d}"]
        indices = np.searchsorted(self.ledger["ids"], ids)
        if (indices >= len(self.fit)).any() or not np.array_equal(self.ledger["ids"][indices], ids):
            raise ValueError(f"projection row ids are outside ledger at raw{raw}")
        for instance in range(2):
            if instance == 1 and not include_a37:
                continue
            source_bits = (1 << instance) | (4 if instance == 0 and not include_a37 else 0)
            fit = indices[(flags & source_bits) != 0]
            self.fit[fit, instance] += scores[fit, instance]
        self.guard += scores[:, 2]
        self.worst_guard = np.maximum(self.worst_guard, scores[:, 2])
        self.frames.add(key)

    def finish(self, *, min_fit=1e-4, max_guard_share=0.10, max_view_guard=0.25):
        required = {(raw, state) for raw in self.raw_ids for state in (True, False)}
        if self.frames != required:
            raise ValueError("both edit states of every requested view must be attributed before fitting")
        fit = self.fit.sum(1)
        positive = fit > min_fit
        mixed = positive & ((self.guard > max_guard_share * fit) | (self.worst_guard > max_view_guard))
        allowed = positive & ~mixed
        return {
            **self.ledger,
            "current_fit": self.fit,
            "current_guard": self.guard,
            "current_worst_guard": self.worst_guard,
            "mixed": mixed,
            "guard_limits": np.asarray((min_fit, max_guard_share, max_view_guard), dtype=np.float64),
            "max_retirement": allowed.astype(np.float32),
            "attributed_raw_ids": np.asarray(self.raw_ids, dtype=np.int16),
        }


def _bound_identity(identity):
    for name, (size, stamp) in identity.items():
        stat = Path(name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, stamp):
            raise ValueError(f"checkpoint identity mismatch: {name}")


def _quaternion_basis(quaternion):
    w, x, y, z = (quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)).T
    return np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ], axis=1).reshape(-1, 3, 3)


def _integral(mask):
    return np.pad(mask, ((1, 0), (1, 0))).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)


def _intersects(integral, rectangles):
    x0, y0, x1, y1 = rectangles.T
    return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) > 0


def _screen_bounds(xyz_camera, rad, K, shape):
    """Conservative axis bounds of projected Gaussian support."""
    height, width = shape
    z = xyz_camera[:, 2]
    rad_xy, rad_z = rad[:, :2], rad[:, -1]
    projected = xyz_camera[:, :2] / z[:, None].clip(1e-8)
    radius_uv = (rad_xy + np.abs(projected) * rad_z[:, None]) / (z - rad_z).clip(1e-8)[:, None]
    fx_fy = np.diag(K)[:2]
    uv = projected * fx_fy + K[:2, 2]
    radius_px = radius_uv * fx_fy + 2  # native 0.3px² low pass and pixel rounding
    lo, hi = uv - radius_px, uv + radius_px
    # If the support intersects the camera plane, its projection is unbounded.
    crossing = z <= rad_z
    lo[crossing], hi[crossing] = 0, (width, height)
    lower = np.floor(lo).clip(0, (width, height)).astype(np.int32)
    upper = (np.ceil(hi) + 1).clip(0, (width, height)).astype(np.int32)
    return np.c_[lower, upper]


def _depth_supported(rect, z_m, depth, domain):
    supported = np.zeros(len(rect), dtype=bool)
    valid = domain & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return supported
    candidates = np.flatnonzero(
        (z_m <= depth[valid].max() + .30) & _intersects(_integral(valid), rect)
    )
    # Bin row center depth to share integral images. The low bin edge gives
    # conservative admission within .30 m of the measured surface.
    quantized = np.floor(z_m[candidates] * 4).astype(np.int32)
    for band in np.unique(quantized):
        within = candidates[quantized == band]
        witnesses = valid & (depth >= band * .25 - .30)
        supported[within] = _intersects(_integral(witnesses), rect[within])
    return supported


def _reproject_reveal(frame, local_xyz, scale):
    """Same-object fused LiDAR support hidden by a37 in the current camera."""
    w2c = frame["model_to_camera"]
    K = frame["K"]
    t = frame["model_from_local"][0]
    camera = (local_xyz @ t[:3, :3].T + t[:3, 3]) @ w2c[:3, :3].T + w2c[:3, 3]
    camera = camera[camera[:, 2] > .01]
    pix = np.floor(camera[:, :2] / camera[:, 2:3] * np.diag(K)[:2] + K[:2, 2]).astype(np.int32)
    height, width = frame["visible"].shape[1:]
    inside = ((pix >= 0) & (pix < [width, height])).all(1)
    x, y = pix[inside].T
    distance = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(distance, (y, x), camera[inside, 2] / scale)
    reveal = frame["occluded"][0] & frame["visible"][1] & np.isfinite(distance)
    if "scene_depth_m" in frame:
        foreground_depth = frame["scene_depth_m"]
        reliable_fg = np.isfinite(foreground_depth) & (foreground_depth > 0) & frame["depth_valid"][1]
        reveal &= ~reliable_fg | (distance > foreground_depth + .10)
    return distance, reveal


def _historical_view(c2, raw):
    # Used only for a preliminary CPU projection before the new bank is ready.
    view = _read_npz(c2 / "inputs/a6_views" / f"{raw:03d}.npz")
    labels = view["actor_labels"]
    depth = np.stack((view["measured_a6_depth_m"], view["camera_depth_m"]))
    return {
        "K": view["K"],
        "model_to_camera": view["model_to_camera"],
        "visible": np.stack((view["known_visible_silhouette_mask"], (labels == 5) & ~view["uncertain_boundary_mask"])),
        "depth_m": depth,
        "depth_valid": np.isfinite(depth) & (depth > 0),
    }


def prepare_ledger(campaign, output, observations=None):
    """Bind historical evidence, and scan the real time-varying native checkpoint on CPU."""
    torch.set_num_threads(1)
    native = campaign / NATIVE_REL
    road = campaign / ROAD_REL
    c2 = campaign / C2_REL
    vehicle = campaign / VEHICLE_REL
    historical = campaign / HISTORICAL_REL
    static_file = native / "chkpnt10000.pth"
    road_file = road / "formal/checkpoint001000.pt"
    parent, step = torch.load(str(static_file), map_location="cpu", mmap=True, weights_only=False)
    if step != 10000 or len(parent) != 18:
        raise ValueError("unexpected native static checkpoint")
    road_state = torch.load(str(road_file), map_location="cpu", weights_only=False)
    if road_state["step"] != 1000:
        raise ValueError("unexpected frozen road checkpoint")
    _bound_identity(road_state["parent_identity"])
    if str(static_file) not in road_state["parent_identity"]:
        raise ValueError("road checkpoint does not bind the requested native static")
    evidence_paths = {
        "road_contributors": road / "inputs/contributors.npz",
        "c2_pvd_contributors": c2 / "inputs/actor_contributors_pvd.npz",
        "raw113_native_attribution": historical / "row_attribution.npz",
        "other_vehicle_protection": vehicle / "inputs/retained_static_protection.npz",
    }
    road_rows = _read_npz(evidence_paths["road_contributors"])
    if not np.array_equal(road_rows["ids"], road_state["repair"]["ids"].numpy()):
        raise ValueError("road contributor row order differs from frozen gate")
    c2_rows = _read_npz(evidence_paths["c2_pvd_contributors"])
    if json.loads(c2_rows["parent_identity_json"].item()) != json.loads(json.dumps(road_state["parent_identity"])):
        raise ValueError("historical C2 row IDs refer to a different native checkpoint")
    raw113 = _read_npz(evidence_paths["raw113_native_attribution"])
    raw113_summary = json.loads((historical / "summary.json").read_text())
    if raw113_summary["model"] != str(native) or not raw113_summary["exact"]:
        raise ValueError("raw113 historical evidence does not bind the native checkpoint")
    other = _read_npz(evidence_paths["other_vehicle_protection"])
    xyz, sigma, basis = parent[1].detach().numpy(), parent[4].detach().exp().numpy(), _quaternion_basis(parent[5].detach().numpy())
    radius = SUPPORT_SIGMAS * sigma.max(1)
    native_alpha = parent[6].detach().sigmoid().numpy().ravel()
    center_t, sigma_t, velocity = parent[7].detach().numpy(), parent[8].detach().exp().numpy(), parent[9].detach().numpy()
    angular_speed = 2 * np.pi / float(parent[16])
    count = len(xyz)
    flags = np.zeros(count, dtype=np.uint8)
    road_keep = np.ones(count, dtype=np.float32)
    road_ids = _row_ids(road_rows["ids"], count)
    road_keep[road_ids] = road_state["repair"]["gate_raw"].sigmoid().detach().numpy()
    flags[road_ids] |= ROAD
    flags[_row_ids(c2_rows["ids"], count)] |= HISTORICAL_C2
    flags[_row_ids(raw113["ids"][raw113["fit"] > 1e-6], count)] |= HISTORICAL_RAW113
    flags[_row_ids(other["ids"], count)] |= HISTORICAL_OTHER_CAR

    bank = None
    a6_xyz = None
    if observations is not None:
        from object_observations import ObservationBank
        bank = ObservationBank(observations)
        if tuple(bank.raw_ids) != RAW_IDS:
            raise ValueError("observation cache does not contain the fixed 49 views")
        a6_xyz = bank.asset(6)["xyz_local_m"]
    geom = _read_npz(c2 / "inputs/a6_geometry_minimal.npz")
    meter_to_model = np.linalg.norm(geom["model_from_world"][:3, :3], axis=0).mean()
    timestamps = json.loads((native / "actual_scene_binding.json").read_text())["timestamps"]
    if len(timestamps) != len(RAW_IDS):
        raise ValueError("native frame timestamps do not cover the fixed 49 views")
    projections = {"raw_ids": np.asarray(RAW_IDS, dtype=np.int16)}
    if bank is not None:
        projections["observation_fingerprint"] = np.asarray(sha256(observations / "receipt.json"))
    per_view = []
    for raw, timestamp in zip(RAW_IDS, timestamps):
        frame = bank.frame(raw) if bank is not None else _historical_view(c2, raw)
        if bank is not None and not np.isclose(frame["model_time"], timestamp, atol=1e-7, rtol=0):
            raise ValueError(f"raw{raw}: observation timestamp differs from the native frame")
        domains = [(frame["visible"][i] & frame["depth_valid"][i], frame["depth_m"][i], 1 << i)
                   for i in range(2)]
        if bank is not None:
            reveal_depth, reveal = _reproject_reveal(frame, a6_xyz, meter_to_model)
            domains.append((reveal, reveal_depth, 4))
            projections[f"reveal_{raw:03d}"] = reveal
            projections[f"reveal_z_{raw:03d}"] = np.where(reveal, reveal_depth, 0).astype(np.float32)
        union = np.logical_or.reduce([domain for domain, _, _ in domains])
        max_depth = max((depth[domain].max(initial=0) for domain, depth, _ in domains))
        marginal = np.exp(-.5 * ((timestamp - center_t) / sigma_t)**2).ravel()
        # Match the native temporal cull and drop rows beneath raster alpha threshold.
        live = np.flatnonzero((marginal > .05) & (native_alpha * marginal >= 1/255))
        means = xyz[live] + velocity[live] * (np.sin((timestamp - center_t[live]) * angular_speed) / angular_speed)
        w2c = frame["model_to_camera"].astype(np.float32)
        K = frame["K"].astype(np.float32)
        means_camera = means @ w2c[:3, :3].T + w2c[:3, 3]
        z_m = means_camera[:, 2] / meter_to_model
        near = (means_camera[:, 2] > .01) & (z_m <= max_depth + .30)
        live, means_camera, z_m = live[near], means_camera[near], z_m[near]
        # Cheap enclosing sphere, then directional covariance for surviving rows.
        sphere = _screen_bounds(means_camera, radius[live, None], K, union.shape)
        relevant = _intersects(_integral(union), sphere)
        live, means_camera, z_m = live[relevant], means_camera[relevant], z_m[relevant]
        axis_camera = w2c[:3, :3] @ basis[live]
        radii = SUPPORT_SIGMAS * np.linalg.norm(axis_camera * sigma[live, None, :], axis=2)
        rectangles = _screen_bounds(means_camera, radii, K, union.shape)
        support = np.zeros(len(live), dtype=np.uint8)
        for domain, depth, bit in domains:
            support[_depth_supported(rectangles, z_m, depth, domain)] |= bit
        ids = live[support > 0]
        flags[ids] |= PROJECTED_SUPPORT
        projections[f"ids_{raw:03d}"] = ids.astype(np.int32)
        projections[f"flags_{raw:03d}"] = support[support > 0]
        per_view.append({
            "raw": raw,
            "a6_projected_rows": int(((support & 1) != 0).sum()),
            "a37_projected_rows": int(((support & 2) != 0).sum()),
            "a6_reveal_projected_rows": int(((support & 4) != 0).sum()),
            "a6_reveal_lidar_pixels": int(domains[2][0].sum()) if bank is not None else None,
        })
        if raw in (89, 105, 113, 130, 137):
            print(json.dumps(per_view[-1]), flush=True)

    # Other-car historical protection is a source bit. New, paired guard
    # attribution determines whether those same rows still have mixed support.
    ids = np.flatnonzero(flags)
    fingerprint = sha256(static_file)
    _bound_identity(road_state["parent_identity"])
    projections["checkpoint_sha256"] = np.asarray(fingerprint)
    ledger = dict(
        raw_ids=np.asarray(RAW_IDS, dtype=np.int16),
        ids=ids, source_flags=flags[ids], road_keep=road_keep[ids],
        max_retirement=np.zeros(len(ids), dtype=np.float32),
        static_count=np.asarray(count),
        checkpoint_sha256=np.asarray(fingerprint),
        road_checkpoint_sha256=np.asarray(sha256(road_file)),
    )
    if bank is not None:
        ledger["observation_fingerprint"] = projections["observation_fingerprint"]
    projection_path = output.with_name(output.stem + "_projections.npz")
    receipt_path = output.with_suffix(".json")
    for path in (output, projection_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"preserve existing replacement evidence: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, arrays in ((output, ledger), (projection_path, projections)):
        temporary = path.with_suffix(".tmp")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        _publish_new(temporary, path)
    summary = {
        "static_checkpoint": str(static_file),
        "checkpoint_sha256": fingerprint,
        "road_checkpoint": str(road_file),
        "road_checkpoint_sha256": ledger["road_checkpoint_sha256"].item(),
        "observation_bank": str(observations) if observations is not None else None,
        "projection_status": "current_observation_bank" if bank is not None else "historical_reviewed_masks_provisional",
        "projection": "Native time-varying centers and conservative projected covariance support at the native alpha cutoff, in front of raw LiDAR surface witnesses",
        "candidate_source_flags": {"fixed_road_gate": ROAD, "historical_C2": HISTORICAL_C2,
                                   "historical_native_raw113": HISTORICAL_RAW113,
                                   "historical_other_vehicle": HISTORICAL_OTHER_CAR,
                                   "projected_support": PROJECTED_SUPPORT},
        "historical_sources": {name: {"path": str(path), "sha256": sha256(path)}
                               for name, path in evidence_paths.items()},
        "static_rows": count,
        "ledger_rows": len(ids),
        "historical_exact_C2_rows": len(c2_rows["ids"]),
        "historical_exact_raw113_positive_rows": int(((flags & HISTORICAL_RAW113) != 0).sum()),
        "historical_other_vehicle_protected_rows": len(other["ids"]),
        "fixed_road_rows": len(road_ids),
        "projection_candidate_rows": int(((flags & PROJECTED_SUPPORT) != 0).sum()),
        "new_retirement_authorized_rows": 0,
        "frames": per_view,
    }
    temporary = receipt_path.with_suffix(".receipt.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _publish_new(temporary, receipt_path)
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2), flush=True)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare",))
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    args = parser.parse_args()
    prepare_ledger(args.campaign, args.output, args.observations)


if __name__ == "__main__":
    main()
