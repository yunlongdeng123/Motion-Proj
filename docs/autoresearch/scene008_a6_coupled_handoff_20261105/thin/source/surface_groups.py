"""BUILD-only measured rear-panel grouping for one coupled handoff control.

``build_group(asset, selection_rank=0)`` is a pure CPU operation.  It consumes the output of
``rescue.build_anchors()`` augmented with independently counted BUILD RGB
support (``rgb_frame_count``).  It never reads the old proxy ledger, opacity,
renders, held views, or a model.  A NO_GROUP result is an expected, auditable
outcome; thresholds are fixed and are not relaxed to obtain a component.

For a PASS, group coordinates are ``(xyz - center) @ basis``.  The columns of
``basis`` are two tangents followed by the rear-oriented fitted normal.  Proxy
centers and colors are exact copies of selected input anchors: plane fitting
sets the common orientation and group frame but never projects a center.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.spatial import cKDTree


CONFIG = {
    "neighbor_distance_m": .08,
    "normal_angle_deg": 20.,
    "min_component_points": 8,
    "min_component_source_frames": 2,
    "min_anchor_rgb_frames": 2,
    "min_abs_plane_normal_x": .65,
    "max_plane_q95_residual_m": .06,
    "tangent_grid_m": .08,
    "sigma_spacing_fraction": .66,
    "sigma_min_m": .003,
    "sigma_max_m": .035,
    "normal_bound_margin_m": .03,
    "max_center_projection_m": .03,
}


def panel(asset):
    """The prior positive.panel() formula, with no GPU-dependent imports."""
    xyz = np.asarray(asset["xyz"], dtype=np.float64)
    width = float(np.asarray(asset["box"])[1])
    core = xyz[(np.abs(xyz[:, 1]) < width / 4)
               & np.isfinite(xyz).all(axis=1)]
    if not len(core):
        raise ValueError("No finite points in the rear-panel central-width core")
    x = np.quantile(core[:, 0], [.12, .68]) + [-.045, .045]
    z = np.quantile(core[:, 2], [.12, .75]) + [-.035, .035]
    return np.array([[x[0], -width / 4, z[0]],
                     [x[1], width / 4, z[1]]], dtype=np.float32)


def _empty(box, audit, reason):
    audit["reason"] = reason
    return {
        "status": "NO_GROUP",
        "qualified_component_count": int(audit.get("qualified_component_count", 0)),
        "ledger": {
            "xyz": np.empty((0, 3), np.float32),
            "normal": np.empty((0, 3), np.float32),
            "sigma": np.empty(0, np.float32),
            "rgb": np.empty((0, 3), np.float32),
            "box": np.asarray(box).copy(),
            "source": np.empty(0, np.int64),
        },
        # These placeholders are not a usable shell unless status == PASS.
        "center": np.zeros(3, np.float64),
        "basis": np.eye(3, dtype=np.float64),
        "bounds": np.zeros((2, 3), np.float64),
        "audit": audit,
    }


def _fingerprint(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("ascii"))
        digest.update(str((value.dtype.str, value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _components(xyz, normal):
    """Connected components of the distance-and-unoriented-normal graph."""
    pairs = cKDTree(xyz).query_pairs(CONFIG["neighbor_distance_m"], output_type="ndarray")
    cosine = np.cos(np.deg2rad(CONFIG["normal_angle_deg"]))
    if len(pairs):
        dot = np.abs(np.einsum("ij,ij->i", normal[pairs[:, 0]], normal[pairs[:, 1]]))
        pairs = pairs[dot >= cosine]
    parent = np.arange(len(xyz))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for first, second in pairs:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([find(k) for k in range(len(xyz))])
    return [np.flatnonzero(roots == k) for k in np.unique(roots)], int(len(pairs))


def _plane(xyz):
    center = xyz.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh((xyz - center).T @ (xyz - center) / len(xyz))
    normal = vectors[:, 0]
    # Rear is negative object-local x.  Fix the sign deterministically.
    if normal[0] > 0:
        normal = -normal
    residual = np.abs((xyz - center) @ normal)
    return center, normal, eigenvalues, residual


def _basis(normal):
    # Keep the vertical tangent near +z, even when the two tangent PCA
    # eigenvalues are similar.  cross(u,v)=normal gives a rigid right-hand frame.
    vertical = np.array([0., 0., 1.])
    vertical -= normal * (vertical @ normal)
    vertical /= np.linalg.norm(vertical)
    horizontal = np.cross(vertical, normal)
    return np.column_stack((horizontal, vertical, normal))


def _spacing_sigma(tangent):
    """Use nearest non-self tangent spacing after measured-row deduplication."""
    if len(tangent) < 2:
        return np.full(len(tangent), CONFIG["sigma_min_m"])
    spacing = cKDTree(tangent).query(tangent, k=2, workers=1)[0][:, 1]
    return np.clip(spacing * CONFIG["sigma_spacing_fraction"],
                   CONFIG["sigma_min_m"], CONFIG["sigma_max_m"])


def build_group(asset, selection_rank=0):
    """Return one measured component and its retirement OBB, or NO_GROUP.

    Required row fields: xyz/normal/rgb [N,3], sigma/source [N], and
    rgb_frame_count [N].  box is [3].  ``source`` is an actual representative
    raw-frame ID, not a projected visibility witness.  ``rgb_frame_count`` must
    count distinct BUILD frames with actual RGB evidence before fusion.  If
    rgb_frame_min/max are supplied they are retained without substitution.

    ``selection_rank`` chooses the nth qualified component, ordered by
    descending source-point count and then earliest input-row index.  The
    default selects the same largest component as the original helper.  A
    rank outside the qualified list returns NO_GROUP and the available count.

    Input arrays are not mutated.  Every rejection is recorded in audit, and
    ledger.source_anchor_index indexes the exact caller-provided asset rows.
    """
    audit = {
        "schema": 1,
        "config": dict(CONFIG),
        "selection_inputs": "BUILD measured xyz, normals, source frames, actual RGB support only",
        "forbidden_selection_inputs": "old399 membership, fitted opacity, rendered damage, HELD views",
        "rgb_support_provenance": "caller supplies distinct actual BUILD RGB frame counts; no projected-witness fallback",
        "frame_convention": "local=(xyz-center)@basis; basis columns=tangent_u,tangent_v,normal",
        "components": [],
    }
    missing = [k for k in ("xyz", "normal", "sigma", "rgb", "box", "source", "rgb_frame_count")
               if k not in asset]
    box = np.asarray(asset.get("box", np.zeros(3)))
    if (isinstance(selection_rank, (bool, np.bool_))
            or not isinstance(selection_rank, (int, np.integer)) or selection_rank < 0):
        audit["requested_selection_rank"] = str(selection_rank)
        return _empty(box, audit, "invalid_selection_rank_requires_nonnegative_integer")
    selection_rank = int(selection_rank)
    audit["selection_rank"] = selection_rank
    if missing:
        audit["missing_fields"] = missing
        return _empty(box, audit, "missing_required_input_fields")
    arrays = {key: np.asarray(asset[key]) for key in
              ("xyz", "normal", "sigma", "rgb", "source", "rgb_frame_count")}
    xyz = arrays["xyz"]
    if xyz.ndim != 2 or xyz.shape[1:] != (3,):
        return _empty(box, audit, "invalid_xyz_shape")
    count = len(xyz)
    audit["input_anchor_count"] = count
    for key in ("normal", "rgb"):
        if arrays[key].shape != (count, 3):
            audit["invalid_field"] = key
            return _empty(box, audit, "invalid_row_field_shape")
    for key in ("sigma", "source", "rgb_frame_count"):
        if arrays[key].shape not in ((count,), (count, 1)):
            audit["invalid_field"] = key
            return _empty(box, audit, "invalid_row_field_shape")
        arrays[key] = arrays[key].reshape(-1)
    for key in ("rgb_frame_min", "rgb_frame_max"):
        if key in asset:
            value = np.asarray(asset[key])
            if value.shape not in ((count,), (count, 1)):
                audit["invalid_field"] = key
                return _empty(box, audit, "invalid_row_field_shape")
            arrays[key] = value.reshape(-1)
    if box.shape != (3,) or not np.isfinite(box).all() or np.any(box <= 0):
        return _empty(box, audit, "invalid_box_lwh")
    if not count:
        return _empty(box, audit, "empty_input_asset")
    audit["input_sha256"] = _fingerprint(dict(arrays, box=box))
    if any(not np.issubdtype(value.dtype, np.number) for value in arrays.values()):
        return _empty(box, audit, "nonnumeric_input_field")

    xyz64 = np.asarray(xyz, dtype=np.float64)
    normal = np.asarray(arrays["normal"], dtype=np.float64)
    length = np.linalg.norm(normal, axis=1)
    rgb = arrays["rgb"]
    frame_count = arrays["rgb_frame_count"]
    source = arrays["source"]
    valid_geometry = np.isfinite(xyz64).all(axis=1)
    valid_normal = np.isfinite(normal).all(axis=1) & (length > 1e-12)
    valid_rgb = np.isfinite(rgb).all(axis=1) & ((rgb >= 0) & (rgb <= 1)).all(axis=1)
    valid_source = np.isfinite(source) & (source == np.floor(source))
    rgb_supported = (np.isfinite(frame_count) & (frame_count == np.floor(frame_count))
                     & (frame_count >= CONFIG["min_anchor_rgb_frames"]))
    try:
        bounds = panel(asset)
    except ValueError as error:
        audit["panel_error"] = str(error)
        return _empty(box, audit, "no_panel_core")
    in_panel = ((xyz64 >= bounds[0]) & (xyz64 <= bounds[1])).all(axis=1)
    masks = {
        "invalid_xyz": ~valid_geometry,
        "invalid_normal": ~valid_normal,
        "invalid_rgb": ~valid_rgb,
        "invalid_source_frame": ~valid_source,
        "insufficient_actual_rgb_frames": ~rgb_supported,
        "outside_prior_positive_panel": ~in_panel,
    }
    audit["panel_bounds_object_local_m"] = bounds.tolist()
    audit["eligibility_rejections"] = {key: np.flatnonzero(value).tolist()
                                        for key, value in masks.items()}
    audit["panel_anchor_count"] = int(in_panel.sum())
    audit["panel_rgb_supported_anchor_count"] = int((in_panel & rgb_supported).sum())
    eligible = ~(np.stack(list(masks.values())).any(axis=0))
    indices = np.flatnonzero(eligible)
    audit["eligible_anchor_count"] = int(len(indices))
    audit["eligible_source_frames"] = np.unique(source[indices]).astype(int).tolist()
    if not len(indices):
        return _empty(box, audit, "no_eligible_panel_anchors")
    normalized = normal[indices] / length[indices, None]
    components, edges = _components(xyz64[indices], normalized)
    audit["graph_edge_count"] = edges
    candidates = []
    for component_id, members in enumerate(components):
        rows = indices[members]
        frames = np.unique(source[rows]).astype(int)
        item = {"id": component_id, "size": int(len(rows)),
                "source_frames": frames.tolist(), "source_frame_count": int(len(frames)),
                "source_anchor_index": rows.tolist(), "rejection_reasons": []}
        reasons = item["rejection_reasons"]
        if len(rows) < CONFIG["min_component_points"]:
            reasons.append("component_below_8_points")
        if len(frames) < CONFIG["min_component_source_frames"]:
            reasons.append("component_below_2_actual_source_frames")
        if len(rows) >= 3:
            center, fit_normal, eigenvalues, residual = _plane(xyz64[rows])
            q95 = float(np.quantile(residual, .95))
            item.update(plane_center_m=center.tolist(), plane_normal=fit_normal.tolist(),
                        plane_eigenvalues_m2=eigenvalues.tolist(), plane_q95_residual_m=q95,
                        plane_max_residual_m=float(residual.max()))
            # Collinear/identical points do not establish a surface orientation.
            if eigenvalues[1] <= 1e-12:
                reasons.append("degenerate_plane_support")
            if abs(fit_normal[0]) < CONFIG["min_abs_plane_normal_x"]:
                reasons.append("plane_not_rear_facing_abs_normal_x_below_0.65")
            if q95 > CONFIG["max_plane_q95_residual_m"]:
                reasons.append("plane_q95_residual_above_0.06m")
            if not reasons:
                candidates.append((len(rows), component_id, rows, center, fit_normal))
        audit["components"].append(item)
    audit["component_count"] = len(components)
    audit["qualified_component_count"] = len(candidates)
    ranked = sorted(candidates, key=lambda x: (-x[0], x[2][0]))
    audit["qualified_component_order"] = [candidate[1] for candidate in ranked]
    for rank, candidate in enumerate(ranked):
        audit["components"][candidate[1]]["qualified_rank"] = rank
    if not candidates:
        return _empty(box, audit, "no_component_meets_fixed_surface_contract")
    if selection_rank >= len(ranked):
        return _empty(box, audit, "selection_rank_out_of_range")
    # The predeclared rank selects by geometry alone, before rendering/fitting.
    _, selected_id, rows, center, fit_normal = ranked[selection_rank]
    for item in audit["components"]:
        if not item["rejection_reasons"] and item["id"] != selected_id:
            item["rejection_reasons"].append(
                "smaller_qualified_component_not_selected" if selection_rank == 0
                else "qualified_component_not_selected_by_requested_rank")
        item["selected"] = item["id"] == selected_id
    basis = _basis(fit_normal)
    local = (xyz64[rows] - center) @ basis
    cells = np.floor(local[:, :2] / CONFIG["tangent_grid_m"]).astype(np.int64)
    distance = np.linalg.norm(local[:, :2] - (cells + .5) * CONFIG["tangent_grid_m"], axis=1)
    # The nearest measured point to each cell center represents that cell.  No
    # cell center, centroid or plane projection becomes a Gaussian center.
    order = np.lexsort((rows, distance, cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    first = np.r_[True, np.any(np.diff(sorted_cells, axis=0) != 0, axis=1)]
    chosen = order[first]
    chosen = chosen[np.argsort(rows[chosen])]
    representatives = rows[chosen]
    sigma = _spacing_sigma(local[chosen, :2]).astype(np.float32)
    obb_bounds = np.stack((local.min(axis=0), local.max(axis=0)))
    obb_bounds[0, 2] -= CONFIG["normal_bound_margin_m"]
    obb_bounds[1, 2] += CONFIG["normal_bound_margin_m"]
    ledger = {
        "xyz": xyz[representatives].copy(),
        "normal": np.repeat(fit_normal[None], len(representatives), axis=0).astype(np.float32),
        "sigma": sigma,
        "rgb": rgb[representatives].copy(),
        "box": box.copy(),
        "source": source[representatives].copy(),
        "source_anchor_index": representatives.copy(),
        "source_normal": arrays["normal"][representatives].copy(),
        "source_sigma": arrays["sigma"][representatives].copy(),
        "group_id": np.zeros(len(representatives), np.int32),
        "tangent_grid_cell": cells[chosen].copy(),
        "center_projection_m": np.zeros(len(representatives), np.float32),
        "rgb_frame_count": frame_count[representatives].copy(),
    }
    for key in ("rgb_frame_min", "rgb_frame_max"):
        if key in arrays:
            ledger[key] = arrays[key][representatives].copy()
    audit.update(
        reason=("accepted_largest_qualified_rear_panel_component" if selection_rank == 0
                else "accepted_ranked_qualified_rear_panel_component"),
        selected_component_id=selected_id,
        selected_component_anchor_count=int(len(rows)),
        proxy_anchor_count=int(len(representatives)),
        retained_source_anchor_index=representatives.tolist(),
        tangent_grid_duplicate_anchor_index=np.setdiff1d(rows, representatives).tolist(),
        center_projection_max_m=0.,
        measured_centers_unchanged=True,
        color_rule="exact BUILD asset RGB at actual measured representative; no color fitting",
        sigma_rule="0.66 * nearest other deduplicated tangent-center spacing, clipped to [0.003,0.035] m",
        obb_support="full selected source component before tangent-grid deduplication",
        obb_normal_margin_m=CONFIG["normal_bound_margin_m"],
        obb_extent_m=(obb_bounds[1] - obb_bounds[0]).tolist(),
    )
    # Keep exact measured xyz/rgb equality visible as a runtime invariant.
    assert np.array_equal(ledger["xyz"], xyz[representatives])
    assert np.array_equal(ledger["rgb"], rgb[representatives])
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-12, rtol=0)
    assert np.isclose(np.linalg.det(basis), 1., atol=1e-12, rtol=0)
    assert np.all(local >= obb_bounds[0] - 1e-12) and np.all(local <= obb_bounds[1] + 1e-12)
    return {"status": "PASS", "qualified_component_count": len(ranked),
            "ledger": ledger, "center": center,
            "basis": basis, "bounds": obb_bounds, "audit": audit}
