"""Camera-aligned geometric hints from legal LiDAR and a local road plane.

Physical-core returns are discarded before z-buffering or fitting. Plane
continuations are inferred hints, never observations. The image and semantic
inputs must be the actual corrupted observation, not the clean training target.
"""
from __future__ import annotations
import time
import cv2
import numpy as np


def camera_calibration(observation):
    intrinsic = np.asarray(observation['camera_intrinsic_Brown'], np.float64)
    fx, fy, cx, cy = intrinsic[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], np.float64)
    distortion = intrinsic[4:9]
    ego = np.asarray(observation['ego_world'], np.float64).reshape(4, 4)
    capture = np.asarray(observation['camera_pose_world'], np.float64).reshape(4, 4)
    extrinsic = np.asarray(observation['camera_extrinsic'], np.float64).reshape(4, 4)
    camera_from_ego = np.linalg.solve(capture @ extrinsic, ego)
    waymo_to_cv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0.]], np.float64)
    cv_from_ego = np.eye(4)
    cv_from_ego[:3] = waymo_to_cv @ camera_from_ego[:3]
    return K, distortion, cv_from_ego


def project_points(points_ego, observation):
    points = np.asarray(points_ego, np.float64)[:, :3]
    K, distortion, transform = camera_calibration(observation)
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    valid = np.isfinite(camera).all(1) & (camera[:, 2] > .5)
    uv = np.full((len(points), 2), -1e6, np.float64)
    if valid.any():
        uv[valid] = cv2.projectPoints(camera[valid], np.zeros(3), np.zeros(3), K, distortion)[0].reshape(-1, 2)
    return uv, camera[:, 2], valid


def _fit_plane(points, seed=20261115, tolerance=.15):
    """RANSAC followed by robust z=ax+by+c fitting in the ego frame."""
    points = np.asarray(points, np.float64)
    if len(points) < 60:
        return None, dict(ready=False, reason='fewer than 60 legal visible-road returns', points=len(points))
    rng = np.random.default_rng(seed)
    sample = points if len(points) <= 12000 else points[rng.choice(len(points), 12000, replace=False)]
    A = np.c_[sample[:, :2], np.ones(len(sample))]
    best = None
    for _ in range(96):
        ids = rng.choice(len(sample), 3, replace=False)
        if abs(np.linalg.det(A[ids])) < .01:
            continue
        coefficients = np.linalg.solve(A[ids], sample[ids, 2])
        if np.linalg.norm(coefficients[:2]) > .4:
            continue
        residual = abs(A @ coefficients-sample[:, 2])
        inliers = residual < tolerance
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = (score, coefficients)
    if best is None or best[0] < 60:
        return None, dict(ready=False, reason='no supported low-slope road plane', points=len(points))
    coefficients = best[1]
    for _ in range(5):
        residual = A @ coefficients-sample[:, 2]
        weights = 1/(1+(residual/.08)**2)
        coefficients = np.linalg.lstsq(A*weights[:, None]**.5, sample[:, 2]*weights**.5, rcond=None)[0]
    residual = abs(A @ coefficients-sample[:, 2])
    inliers = residual < tolerance
    inlier_fraction = float(inliers.mean())
    ready = bool(inliers.sum() >= 60 and inlier_fraction >= .5 and np.linalg.norm(coefficients[:2]) <= .4)
    return coefficients if ready else None, dict(ready=ready, points=len(points), fit_points=len(sample),
        coefficients_z_equals_ax_by_c=coefficients.tolist(), inlier_fraction=inlier_fraction,
        inlier_points=int(inliers.sum()), inlier_median_m=float(np.median(residual[inliers])) if inliers.any() else None,
        tolerance_m=tolerance, seed=seed)


def _rays_ego(pixels_xy, observation):
    K, distortion, transform = camera_calibration(observation)
    # Waymo intrinsics and projected points use integer camera pixel centers.
    xy = cv2.undistortPoints(np.asarray(pixels_xy, np.float64).reshape(-1, 1, 2), K, distortion).reshape(-1, 2)
    direction_cv = np.c_[xy, np.ones(len(xy))]
    inverse = np.linalg.inv(transform)
    return inverse[:3, 3], direction_cv @ inverse[:3, :3].T


def _retained_box_in_front(origin, rays, ground_t, boxes, selected_track_id):
    hit = np.zeros(len(rays), bool)
    for box in boxes:
        if selected_track_id is not None and box.get('id') == selected_track_id:
            continue
        center = np.asarray(box['center'], np.float64)
        dims = np.asarray(box['dimensions'], np.float64)
        if 'ego_from_actor' in box:
            transform = np.asarray(box['ego_from_actor'], np.float64).reshape(4, 4)
            R = transform[:3, :3]
            center = transform[:3, 3]
        else:
            heading = float(box['heading'])
            c, s = np.cos(heading), np.sin(heading)
            # Row-vector ego -> box-local rotation.
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        local_o = (origin-center) @ R
        local_d = rays @ R
        parallel = abs(local_d) < 1e-10
        outside = (parallel & (abs(local_o) > dims/2)).any(1)
        safe = np.where(parallel, 1, local_d)
        t0 = (-dims/2-local_o)/safe
        t1 = (dims/2-local_o)/safe
        lo = np.where(parallel, -np.inf, np.minimum(t0, t1)).max(1)
        hi = np.where(parallel, np.inf, np.maximum(t0, t1)).min(1)
        hit |= ~outside & (hi >= np.maximum(lo, .05)) & (np.maximum(lo, .05) < ground_t-.1)
    return hit


def boxes_in_ego(observation):
    """Accept raw ego boxes or the bank's world-from-actor schema."""
    if 'boxes_ego' in observation:
        return observation['boxes_ego']
    ego = np.asarray(observation['ego_world'], np.float64).reshape(4, 4)
    result = []
    for box in observation.get('boxes', []):
        if 'world_from_actor' not in box or 'dimensions' not in box:
            continue
        transform = np.linalg.solve(ego, np.asarray(box['world_from_actor'], np.float64).reshape(4, 4))
        result.append(dict(id=box.get('id', box.get('track')), center=transform[:3, 3].tolist(),
                           dimensions=box['dimensions'], ego_from_actor=transform.tolist()))
    return result


def build_depth_hints(points_ego, observation, *, physical_core, visible_road,
                      foreground_possible=None, road_support=None, selected_track_id=None,
                      inverse_depth_scale_m=20.):
    tick = time.perf_counter()
    core = np.asarray(physical_core, bool)
    shape = core.shape
    road = np.asarray(visible_road, bool) & ~core
    foreground = np.zeros(shape, bool) if foreground_possible is None else np.asarray(foreground_possible, bool)
    road &= ~foreground
    points = np.asarray(points_ego, np.float64)[:, :3]
    uv, depth, front = project_points(points, observation)
    ij = np.floor(np.clip(uv+.5, -1e5, 1e5)).astype(np.int64)
    valid = front & (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    ids = np.flatnonzero(valid)
    removed = core[ij[ids, 1], ij[ids, 0]]
    ids = ids[~removed]
    # Crucial: no physically hidden return enters the nearest-depth buffer.
    flat_index = ij[ids, 1]*shape[1]+ij[ids, 0]
    zbuffer = np.full(shape[0]*shape[1], np.inf, np.float64)
    np.minimum.at(zbuffer, flat_index, depth[ids])
    nearest = depth[ids] <= zbuffer[flat_index]+.02
    visible_ids = ids[nearest]
    road_ids = visible_ids[road[ij[visible_ids, 1], ij[visible_ids, 0]]]
    plane, plane_receipt = _fit_plane(points[road_ids])
    observed = np.isfinite(zbuffer.reshape(shape))
    measured_depth = np.where(observed, zbuffer.reshape(shape), 0).astype(np.float32)
    inferred_depth = np.zeros(shape, np.float32)
    inferred_valid = np.zeros(shape, bool)
    retained_reveal = np.zeros(shape, bool)
    if plane is not None:
        support = (road | core) if road_support is None else np.asarray(road_support, bool).copy()
        support &= ~foreground | core
        yy, xx = np.where(support)
        origin, rays = _rays_ego(np.c_[xx, yy], observation)
        normal = np.array([-plane[0], -plane[1], 1.])
        denominator = rays @ normal
        t = (plane[2]-origin @ normal)/np.where(abs(denominator) > 1e-8, denominator, np.nan)
        good = np.isfinite(t) & (t > .5) & (t < 150)
        core_indices = np.flatnonzero(core[yy, xx] & good)
        if len(core_indices):
            hit = _retained_box_in_front(origin, rays[core_indices], t[core_indices],
                                         boxes_in_ego(observation), selected_track_id)
            retained_reveal[yy[core_indices[hit]], xx[core_indices[hit]]] = True
            good[core_indices[hit]] = False
        inferred_valid[yy[good], xx[good]] = True
        # Rays have CV forward component one, so t is camera z (not range).
        inferred_depth[yy[good], xx[good]] = t[good].astype(np.float32)
    combined = inferred_depth.copy()
    combined[observed] = measured_depth[observed]
    confidence = inferred_valid.astype(np.float32)*.25
    confidence[observed] = 1.
    normalized = np.zeros(shape, np.float32)
    valid_any = confidence > 0
    # Unlike clipped inverse depth, this retains metric distinctions below 20m.
    normalized[valid_any] = inverse_depth_scale_m/(inverse_depth_scale_m+combined[valid_any])
    assert not (observed & core).any()
    assert np.isfinite(normalized).all()
    return dict(normalized_depth=normalized, depth_confidence=confidence, depth_m=combined,
        depth_observed=observed, depth_inferred=inferred_valid & ~observed,
        depth_retained_reveal=retained_reveal,
        receipt=dict(schema='legal_camera_depth.v1', input_returns=len(points),
            projected_returns=int(valid.sum()), core_returns_discarded=int(removed.sum()),
            observed_depth_pixels=int(observed.sum()), observed_core_pixels=0,
            inferred_plane_pixels=int((inferred_valid & ~observed).sum()),
            inferred_core_pixels=int((inferred_valid & core).sum()),
            core_retained_box_reveal_pixels=int(retained_reveal.sum()),
            inverse_depth_scale_m=inverse_depth_scale_m,
            normalization='scale_m / (scale_m + camera_z); unavailable is 0',
            depth_confidence_codes={'no_hint': 0, 'inferred_plane': .25, 'observed_return': 1},
            plane=plane_receipt, seconds=time.perf_counter()-tick,
            depth_is_camera_z=True, physical_core_returns_never_used=True,
            actual_effect_region_ground_returns_retained=True,
            motion_compensation='source exports; TOP per-pixel compensation, other lasers frame-pose approximation'))


def sample_depth_on_grid(depth, neighbors, inside, physical_unknown):
    """Same legal geometry sampling for training, real inference and adapters.

    Confidence is support-weighted after resampling, not a new per-pixel sensor
    measurement. Unknown physical pixels may interpolate inferred sources only.
    """
    from .input_builder import _sample, _four
    z = np.asarray(depth['depth_m'], np.float32)
    confidence = np.asarray(depth['depth_confidence'], np.float32)
    valid = np.isfinite(z) & (z > 0) & np.isfinite(confidence) & (confidence > 0)
    normalized = np.where(valid, 20/(20+np.maximum(z, 0)), 0).astype(np.float32)
    weight = np.where(valid, confidence, 0)
    denom = _sample(weight, neighbors)
    value = _sample(normalized*weight, neighbors)/np.maximum(denom, 1e-12)
    inferred = np.asarray(depth['depth_inferred'], bool) & valid
    infer_weight = np.where(inferred, confidence, 0)
    infer_denom = _sample(infer_weight, neighbors)
    infer_value = _sample(normalized*infer_weight, neighbors)/np.maximum(infer_denom, 1e-12)
    value[physical_unknown] = infer_value[physical_unknown]
    denom[physical_unknown] = np.minimum(infer_denom[physical_unknown], .25)
    good = inside & (denom > 0)
    value[~good] = 0; denom[~good] = 0
    observed = _four(np.asarray(depth['depth_observed'], bool), neighbors).all(-1) & good & ~physical_unknown
    inferred_view = good & ~observed
    # Expected retained-body reveal is meaningful precisely where road depth is
    # unavailable, so this label must not be gated by `good` road-depth support.
    reveal = _four(np.asarray(depth['depth_retained_reveal'], bool), neighbors).any(-1) & inside
    assert not observed[physical_unknown].any()
    assert np.isfinite(value).all() and np.isfinite(denom).all()
    return dict(normalized_depth=value.astype(np.float32), depth_source_confidence=denom.astype(np.float32),
                depth_observed=observed, depth_inferred=inferred_view, depth_retained_reveal=reveal)


def reconcile_road_prior(lane8, physical_unknown, retained_reveal):
    """Do not assert road/paint behind a nearer retained object's 3D support.

    This removes an inferred claim; it does not supervise or generate that
    object's hidden appearance. Actual visible paint channels remain unchanged.
    """
    lanes = np.asarray(lane8)
    unknown = np.asarray(physical_unknown, bool)
    reveal = np.asarray(retained_reveal, bool)
    if lanes.shape != (*unknown.shape, 8) or reveal.shape != unknown.shape:
        raise ValueError('lane8 and reveal roles must share the same camera view')
    result = lanes.copy()
    blocked = unknown & reveal
    result[blocked, 2:8] = 0
    return result
