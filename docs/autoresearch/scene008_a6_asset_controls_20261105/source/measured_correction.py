"""M3: BUILD-only measured normal-center correction of the complete M2 asset.

CPU-only, one deterministic rule on the whole car. No native-Gaussian centers,
RGB values, HELD observations, renderer, trajectory or unknown-surface targets
are used. The original sample_rgb_valid has a 20cm visibility tolerance; it is
not treated as a precision surface label. Cross-frame robust planes supply the
actual constraints. Same-source projected-depth checks establish provenance only.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix, diags, vstack
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from surface_sampling import GEOMETRY_FIELDS, hash_geometry, hash_provenance

RUN = Path(__file__).resolve().parent.parent
OBS = RUN.parent/'scene008_instances_20261105_r1/observations'
HELD = (89, 95, 101, 105, 113, 121, 125, 130, 137)
BUILD = tuple(raw for raw in range(89, 138) if raw not in HELD)
NOISE = .02  # Declared engineering budget, NOT calibrated Waymo sensor noise.
MIN_POINTS, MIN_FRAMES, MIN_POINTS_PER_FRAME = 6, 2, 3
PRIOR, SMOOTH = .05, .25
COS30, COS15 = np.cos(np.pi/6), np.cos(np.pi/12)


def contract():
    return dict(arm='M3_BUILD_measured_normal_center_correction', version=1,
        source='Exactly the complete M2 geometry_dense.npz; one asset, identical N and SH0 capacity',
        build=BUILD, internal_held=HELD, held_read_or_selected=False,
        measurements='Raw sample_xyz_local_m, sample_rgb_valid, sample_raw_frame, sample_source_cloud_row; never fused all-frame anchors',
        trust='BUILD + finite + original rgb_valid + unique(raw,cloud_row); reproject into original visible/depth_valid and match object/scene camera-projected nearest LiDAR sample within 2*noise_budget',
        source_validation_limit='The depth bank and samples share raw LiDAR: this is provenance checking, not independent accuracy evidence. Scene depth is the camera-projected nearest sample among five LiDAR sensors and both range returns, not physical first-return camera-ray truth. Original rgb_valid allows 20cm.',
        noise_budget_m=NOISE, noise_budget_semantics='Fixed engineering tolerance; no sensor-accuracy claim or data-driven rate search',
        spacing='h_i = median distances to six nearest OTHER M2 centers, recomputed on original M2 centers',
        support_radius='r_i = clip(3*h_i, 4*noise_budget, .25m)',
        plane='Frame-equal weighted PCA with three Huber reweights at 2*noise_budget; final PCA; >=6 points and >=2 frames each contributing >=3 inliers',
        plane_validation='MAD sigma<=noise_budget; weighted |residual| q90<=3*noise_budget; lambda_min/lambda_middle<=.2; sqrt(lambda_middle)>=noise_budget; |dot(plane_normal,source_normal)|>=cos30deg',
        cross_validation='Alternating sorted distinct BUILD frames define two folds. Both plane fits pass same quality; normal agreement<=15deg; their normal-line intersection offsets differ<=2*noise_budget; held-fold |residual| median<=2*noise_budget and q90<=3*noise_budget',
        temporal_validation='q90-q10 of per-frame median signed plane residual <=2*noise_budget; no frame-count duplication by dense points',
        movement='x_new = x_M2 + delta_i*unchanged_source_normal_i; abs(delta_i)<=min(.12m,r_i/2); unsupported or invalid-normal delta exactly zero',
        graph='Undirected 8NN edges within 2*max(h_i,h_j), source normal dot>=cos30deg; exp(-d^2/(2*h_i*h_j))*dot^2, symmetric degree normalization',
        boundary='Unknown nodes fixed at zero. Eligible bound is multiplied by min(1, graph distance to an unknown node/(2*h_i)); disconnected fully supported components have taper one',
        objective='sum_i confidence_i*(delta_i-plane_offset_i)^2 + .05*sum_i delta_i^2 + .25*sum_edges normalized_weight*(delta_i-delta_j)^2',
        confidence='min(1,distinct_frames/4) * inlier_fraction * normal_alignment^2',
        solver='One bounded scipy.sparse lsq_linear solve; tolerance 1e-8, max_iter100; no variant/threshold/rate search or failed-solve fallback',
        unchanged=['N and row order','normal','sigma','thickness','opacity','all original provenance flags','all original appearance seeds','trajectory','scene/a37/static/road/renderer'],
        flag_semantics='Original out_of_box remains M2 ancestry; correction_current_out_of_box records the newly derived position flag',
        appearance_comparison='Root refits M2/M3 with identical SH0 capacity, 2000-step objective, M2 initial prior/flags and exact original27429 BUILD pixels; no new-alpha target filtering',
        evaluation='Unchanged all49 full fixed masks and keep/delete scene guards; no lower-only/reveal selection',
        limitations=['Partial FRONT coverage; unobserved geometry remains the shared PVD prior.',
                    'Distinct adjacent frames are correlated and share poses/calibration; cross-fitting does not establish independent truth.',
                    'No normals, footprints or opacity fit; bounded normal offsets cannot repair tangential errors or wrong topology.',
                    'Local plane quality may reject curved, narrow or mixed surfaces; no eligibility expansion after results.',
                    'Parent PVD conditioning may include HELD source points; shared M2/M3 prior remains identical. M3 adds only BUILD evidence.'],
        GPU_used=False, parameter_search=False)


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _q(values, weights, quantile):
    order = np.argsort(values, kind='stable'); values, weights = values[order], weights[order]
    return float(np.interp(quantile*weights.sum(), np.cumsum(weights), values))


def _quantiles(values):
    values = np.asarray(values)
    return np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist() if len(values) else []


class FrameBank:
    """Directory of original raw frames, or a BUILD-only packed CPU-test bank."""
    def __init__(self, path):
        self.path = Path(path)
        payload = self.path.read_bytes() if self.path.is_file() else None
        self.packed = np.load(BytesIO(payload), allow_pickle=False) if payload is not None else None
        self.packed_sha256 = hashlib.sha256(payload).hexdigest() if payload is not None else None
        self.frame_sha256 = {}
        self.read_raw = []

    def frame(self, raw):
        if raw not in BUILD:
            raise ValueError('M3 may read only BUILD frames')
        self.read_raw.append(int(raw))
        names = ('K', 'world_to_camera', 'world_from_local', 'visible', 'depth_m', 'depth_valid', 'scene_depth_m')
        if self.packed is not None:
            return {key: self.packed[f'r{raw}_{key}'] for key in names}
        source = self.path/f'{raw:03d}.npz'
        payload = source.read_bytes()
        self.frame_sha256[source.name] = hashlib.sha256(payload).hexdigest()
        with np.load(BytesIO(payload), allow_pickle=False) as bank:
            return {key: bank[key] for key in names}

    def source_provenance(self):
        if self.packed is not None:
            return dict(mode='packed_BUILD_CPU_bank', file=str(self.path), file_sha256=self.packed_sha256,
                        raw_frames_read=sorted(set(self.read_raw)), held_frames_read=[])
        return dict(mode='original_BUILD_frame_files', directory=str(self.path),
                    file_sha256=dict(sorted(self.frame_sha256.items())),
                    raw_frames_read=sorted(set(self.read_raw)), held_frames_read=[])

    def close(self):
        if self.packed is not None:
            self.packed.close()


def trusted_samples(observations, frames):
    """Return exact original sample indices and their unchanged measured points."""
    xyz = np.asarray(observations['sample_xyz_local_m'], np.float64)
    raw = np.asarray(observations['sample_raw_frame'])
    cloud = np.asarray(observations['sample_source_cloud_row'])
    selected = np.isin(raw, BUILD) & observations['sample_rgb_valid'] & np.isfinite(xyz).all(1)
    indices = np.flatnonzero(selected)
    _, first = np.unique(np.column_stack((raw[indices], cloud[indices])), axis=0, return_index=True)
    indices = indices[np.sort(first)]
    passed, per_frame, differences = [], {}, []
    for r in BUILD:
        rows = indices[raw[indices] == r]; frame = frames.frame(r)
        matrix = frame['world_to_camera']@frame['world_from_local'][0]
        camera = xyz[rows]@matrix[:3, :3].T+matrix[:3, 3]
        z = camera[:, 2]
        uv = camera@frame['K'].T
        uv = uv[:, :2]/np.maximum(z[:, None], .01)
        integer = np.floor(uv).astype(np.int64)
        height, width = frame['scene_depth_m'].shape
        x, y = np.clip(integer[:, 0], 0, width-1), np.clip(integer[:, 1], 0, height-1)
        inside = (z > .1) & (integer[:, 0] >= 0) & (integer[:, 0] < width) & (integer[:, 1] >= 0) & (integer[:, 1] < height)
        known = inside & frame['visible'][0, y, x] & frame['depth_valid'][0, y, x]
        difference = np.abs(z-frame['depth_m'][0, y, x])
        nearest_consistent = (difference <= 2*NOISE) & (frame['scene_depth_m'][y, x] > 0) & (z <= frame['scene_depth_m'][y, x]+2*NOISE)
        keep = known & nearest_consistent
        passed.extend(rows[keep]); differences.extend(difference[keep])
        per_frame[str(r)] = dict(candidate=int(len(rows)), projected_visible_depth=int(known.sum()), trusted=int(keep.sum()))
    index = np.asarray(passed, np.int64)
    audit = dict(all_samples=len(xyz), original_rgb_valid=int(observations['sample_rgb_valid'].sum()),
        build_rgb_valid_finite=int(selected.sum()), duplicate_raw_cloud_rows=int(selected.sum()-len(indices)),
        trusted_samples=len(index), trusted_distinct_frames=int(len(np.unique(raw[index]))),
        source_self_consistency_abs_depth_m_quantiles=_quantiles(differences), per_frame=per_frame,
        raw_frames_read=frames.read_raw, held_frames_read=[], fused_anchor_arrays_used=False,
        note='Same raw LiDAR source; these depth residuals are not independent accuracy estimates.')
    return xyz[index], raw[index], index, audit


def _plane(points, frames):
    if len(points) < 3:
        return None
    _, inverse, counts = np.unique(frames, return_inverse=True, return_counts=True)
    base = 1/counts[inverse].astype(np.float64); weight = base.copy()
    for iteration in range(4):  # Initial PCA plus exactly three reweightings.
        center = np.einsum('n,ni->i', weight, points)/weight.sum()
        centered = points-center
        covariance = (centered*weight[:, None]).T@centered/weight.sum()
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        normal = eigenvector[:, 0]; residual = centered@normal
        if iteration < 3:
            weight = base*np.minimum(1., 2*NOISE/np.maximum(np.abs(residual), 1e-12))
    median = _q(residual, base, .5)
    sigma = 1.4826*_q(np.abs(residual-median), base, .5)
    quality = bool(eigenvalue[1] >= NOISE**2 and eigenvalue[0] <= .2*eigenvalue[1]
                   and sigma <= NOISE and _q(np.abs(residual), base, .9) <= 3*NOISE)
    return dict(center=center, normal=normal, residual=residual, base=base,
                sigma=sigma, eigenvalue=eigenvalue, quality=quality)


def _anchor(center, source_normal, points, frames):
    if len(points) < MIN_POINTS or len(np.unique(frames)) < MIN_FRAMES:
        return None, 'insufficient_points_or_frames'
    initial = _plane(points, frames)
    if not initial['quality']:
        return None, 'initial_plane_quality'
    inlier = np.abs(initial['residual']) <= 2*NOISE
    labels, counts = np.unique(frames[inlier], return_counts=True)
    accepted_frames = labels[counts >= MIN_POINTS_PER_FRAME]
    use = inlier & np.isin(frames, accepted_frames)
    if use.sum() < MIN_POINTS or len(accepted_frames) < MIN_FRAMES:
        return None, 'insufficient_inlier_frame_support'
    p, f = points[use], frames[use]; fitted = _plane(p, f)
    if not fitted['quality']:
        return None, 'final_plane_quality'
    alignment = abs(float(fitted['normal']@source_normal))
    if alignment < COS30:
        return None, 'source_normal_disagreement'
    folds = np.isin(f, accepted_frames[::2])
    a, b = _plane(p[folds], f[folds]), _plane(p[~folds], f[~folds])
    if a is None or b is None or not a['quality'] or not b['quality']:
        return None, 'crossfit_plane_quality'
    offsets = []
    for plane, hold_points, hold_frames in ((a, p[~folds], f[~folds]), (b, p[folds], f[folds])):
        if abs(float(plane['normal']@fitted['normal'])) < COS15:
            return None, 'crossfit_normal_disagreement'
        denominator = float(plane['normal']@source_normal)
        if abs(denominator) < COS30:
            return None, 'crossfit_source_normal_disagreement'
        offsets.append(float((plane['center']-center)@plane['normal']/denominator))
        _, inv, count = np.unique(hold_frames, return_inverse=True, return_counts=True)
        error = np.abs((hold_points-plane['center'])@plane['normal']); weight = 1/count[inv]
        if _q(error, weight, .5) > 2*NOISE or _q(error, weight, .9) > 3*NOISE:
            return None, 'crossfit_held_frame_residual'
    if abs(offsets[0]-offsets[1]) > 2*NOISE:
        return None, 'crossfit_offset_disagreement'
    medians = np.array([np.median(fitted['residual'][f == r]) for r in accepted_frames])
    if np.quantile(medians, .9)-np.quantile(medians, .1) > 2*NOISE:
        return None, 'temporal_plane_disagreement'
    offset = float((fitted['center']-center)@fitted['normal']/(fitted['normal']@source_normal))
    confidence = min(1., len(accepted_frames)/4)*float(use.mean())*alignment**2
    return dict(target=offset, confidence=confidence, frames=len(accepted_frames), points=int(use.sum()),
                noise=fitted['sigma'], alignment=alignment, used=np.flatnonzero(use)), 'accepted'


def _graph(xyz, normals, spacing, normal_valid):
    distance, neighbor = cKDTree(xyz).query(xyz, k=min(9, len(xyz)))
    row = np.repeat(np.arange(len(xyz)), neighbor.shape[1]-1)
    column = neighbor[:, 1:].ravel(); distance = distance[:, 1:].ravel()
    edge = np.unique(np.sort(np.column_stack((row, column)), axis=1), axis=0)
    left, right = edge.T
    distance = np.linalg.norm(xyz[left]-xyz[right], axis=1)
    dot = np.einsum('ij,ij->i', normals[left], normals[right])
    keep = (distance > 0) & (distance <= 2*np.maximum(spacing[left], spacing[right])) & (dot >= COS30) & normal_valid[left] & normal_valid[right]
    left, right, distance, dot = left[keep], right[keep], distance[keep], dot[keep]
    kernel = np.exp(-distance**2/(2*spacing[left]*spacing[right]))*dot**2
    degree = np.bincount(np.r_[left, right], weights=np.r_[kernel, kernel], minlength=len(xyz))
    weight = kernel/np.sqrt(degree[left]*degree[right])
    length = coo_matrix((np.r_[distance, distance], (np.r_[left, right], np.r_[right, left])), shape=(len(xyz), len(xyz))).tocsr()
    return left, right, weight, length


def correct_geometry(arrays, observations, frame_bank, provenance=None):
    """Return corrected arrays and CPU audit; inputs are never mutated."""
    original_hash = hash_geometry(arrays)
    if str(np.asarray(arrays.get('geometry_hash', original_hash)).item()) != original_hash:
        raise ValueError('Source geometry hash mismatch')
    if 'correction_delta_m' in arrays:
        raise ValueError('M3 is one correction of M2, never iterative reapplication')
    xyz, normals = np.asarray(arrays['xyz'], np.float64), np.asarray(arrays['normal'], np.float64)
    count = len(xyz)
    if count < 7 or not np.isfinite(xyz).all() or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError('Expected at least seven finite centers with unchanged unit normals')
    normal_valid = np.asarray(arrays['normal_valid'], bool)
    point, raw, source_index, source_audit = trusted_samples(observations, frame_bank)
    distance, _ = cKDTree(xyz).query(xyz, k=7)
    spacing = np.median(distance[:, 1:], axis=1)
    if np.any(spacing <= 0):
        raise ValueError('Degenerate M2 sampling spacing; no fallback radius')
    radius = np.clip(3*spacing, 4*NOISE, .25)
    neighbors = cKDTree(point).query_ball_point(xyz, radius) if len(point) else [[] for _ in xyz]
    eligible = np.zeros(count, bool); target = np.zeros(count); confidence = np.zeros(count)
    frame_count = np.zeros(count, np.int16); point_count = np.zeros(count, np.int32)
    noise = np.zeros(count); alignment = np.zeros(count); support = []
    rejection = Counter(); offsets = [0]; flattened = []
    for i, ids in enumerate(neighbors):
        support.append(len(np.unique(raw[ids])))
        if normal_valid[i]:
            fit, reason = _anchor(xyz[i], normals[i], point[ids], raw[ids])
        else:
            fit, reason = None, 'source_normal_invalid'
        rejection[reason] += 1
        if fit is not None:
            eligible[i] = True; target[i] = fit['target']; confidence[i] = fit['confidence']
            frame_count[i], point_count[i] = fit['frames'], fit['points']
            noise[i], alignment[i] = fit['noise'], fit['alignment']
            flattened.extend(source_index[np.asarray(ids)[fit['used']]])
        offsets.append(len(flattened))
    left, right, weight, length_graph = _graph(xyz, normals, spacing, normal_valid)
    active = np.flatnonzero(eligible); limit = np.minimum(.12, radius/2)
    taper = np.zeros(count); delta = np.zeros(count)
    if len(active):
        inactive = np.flatnonzero(~eligible)
        boundary = dijkstra(length_graph, directed=False, indices=inactive, min_only=True) if len(inactive) else np.full(count, np.inf)
        taper[active] = np.minimum(1., boundary[active]/(2*spacing[active]))
        bound = limit[active]*taper[active]
        if np.any(bound <= 0):
            raise ArithmeticError('Nonpositive bound for supported center')
        index = np.full(count, -1, np.int64); index[active] = np.arange(len(active))
        edge_keep = eligible[left] | eligible[right]
        l, r, w = left[edge_keep], right[edge_keep], np.sqrt(SMOOTH*weight[edge_keep])
        edge_row = np.arange(len(l)); le, re = eligible[l], eligible[r]
        graph = coo_matrix((np.r_[w[le], -w[re]], (np.r_[edge_row[le], edge_row[re]], np.r_[index[l[le]], index[r[re]]])), shape=(len(l), len(active))).tocsr()
        matrix = vstack((diags(np.sqrt(confidence[active])), diags(np.full(len(active), np.sqrt(PRIOR))), graph), format='csr')
        rhs = np.r_[np.sqrt(confidence[active])*target[active], np.zeros(len(active)+len(l))]
        solved = lsq_linear(matrix, rhs, bounds=(-bound, bound), tol=1e-8, lsmr_tol='auto', max_iter=100)
        if not solved.success:
            raise ArithmeticError('Bounded correction did not converge: '+solved.message)
        delta[active] = solved.x
        solver = dict(status=int(solved.status), iterations=int(solved.nit), optimality=float(solved.optimality),
                      loss_zero=float(rhs@rhs), loss_final=float(np.sum((matrix@solved.x-rhs)**2)))
    else:
        solver = dict(status='NO_ELIGIBLE_ANCHORS_EXACT_NOOP', loss_zero=0., loss_final=0.)
    result = {key: np.array(value, copy=True) for key, value in arrays.items()}
    result['xyz'] = (xyz+delta[:, None]*normals).astype(np.float32)
    actual = result['xyz'].astype(np.float64)-xyz
    projected = np.einsum('ij,ij->i', actual, normals)/np.einsum('ij,ij->i', normals, normals)
    tangent_error = np.linalg.norm(actual-projected[:, None]*normals, axis=1)
    if not np.array_equal(result['xyz'][~eligible], arrays['xyz'][~eligible]) or np.max(tangent_error) > 5e-7 or np.any(np.abs(projected) > limit+5e-7):
        raise ArithmeticError('Normal-only, zero-unknown, or displacement-bound invariant failed')
    for key in arrays:
        if key not in ('xyz', 'geometry_hash', 'provenance_hash', 'metadata_json') and not np.array_equal(result[key], arrays[key]):
            raise ArithmeticError('Unexpected non-center change: '+key)
    result.update(correction_delta_m=delta.astype(np.float32), correction_eligible=eligible,
        correction_target_delta_m=target.astype(np.float32), correction_confidence=confidence.astype(np.float32),
        correction_radius_m=radius.astype(np.float32), correction_spacing_m=spacing.astype(np.float32),
        correction_limit_m=limit.astype(np.float32), correction_boundary_taper=taper.astype(np.float32),
        correction_anchor_frame_count=frame_count, correction_anchor_point_count=point_count,
        correction_plane_noise_m=noise.astype(np.float32), correction_normal_alignment=alignment.astype(np.float32),
        correction_anchor_offsets=np.asarray(offsets, np.int64), correction_anchor_sample_index=np.asarray(flattened, np.int64),
        correction_source_geometry_hash=np.asarray(original_hash),
        correction_current_out_of_box=(np.abs(result['xyz']) > np.asarray(arrays['box'])/2+1e-6).any(1))
    audit = dict(status='CPU_M3_MEASURED_GEOMETRY_BUILT', rows=count, input_geometry_hash=original_hash,
        source=source_audit, spacing_m_quantiles=_quantiles(spacing), radius_m_quantiles=_quantiles(radius),
        preliminary_support_2frames=int((np.asarray(support) >= 2).sum()),
        preliminary_support_6points_2frames=int(sum(len(v) >= MIN_POINTS and s >= MIN_FRAMES for v, s in zip(neighbors, support))),
        eligible_centers=int(eligible.sum()), zero_unknown_centers=int((~eligible).sum()), rejection_reasons=dict(rejection),
        eligible_original_mothers=int(len(np.unique(np.asarray(arrays.get('sampling_parent_index', np.arange(count)))[eligible]))),
        eligible_frame_count_quantiles=_quantiles(frame_count[eligible]), eligible_inlier_point_count_quantiles=_quantiles(point_count[eligible]),
        target_delta_m_quantiles=_quantiles(target[eligible]), actual_delta_m_quantiles=_quantiles(projected[eligible]),
        eligible_plane_noise_m_quantiles=_quantiles(noise[eligible]), boundary_taper_quantiles=_quantiles(taper[eligible]),
        moved_centers=int((np.linalg.norm(actual, axis=1) > 0).sum()), max_tangent_roundoff_m=float(tangent_error.max()),
        max_abs_displacement_m=float(np.max(np.abs(projected))),
        new_out_of_box_count=int(result['correction_current_out_of_box'].sum()), inherited_out_of_box_count=int(arrays['out_of_box'].sum()),
        graph_edges=len(left), graph_displacement_jump_m_quantiles=_quantiles(np.abs(delta[left]-delta[right])),
        unchanged_original_fields=[key for key in arrays if key not in ('xyz', 'geometry_hash', 'provenance_hash', 'metadata_json')],
        solver=solver, held_read_or_used=False, GPU_used=False, parameter_search=False,
        interpretation='Bounded normal-only geometric heuristic with real BUILD support. Not a true surface certificate or independent held-out discovery.')
    audit['geometry_hash'] = hash_geometry(result)
    audit.update(deepcopy(provenance or {}))
    audit['source_provenance'] = dict(deepcopy(provenance or {}), frames=frame_bank.source_provenance())
    result['geometry_hash'] = np.asarray(audit['geometry_hash'])
    result['provenance_hash'] = np.asarray(hash_provenance(result))
    audit['provenance_hash'] = str(result['provenance_hash'])
    metadata = json.loads(str(np.asarray(arrays.get('metadata_json', '{}')).item()))
    metadata = dict(deepcopy(metadata), measured_normal_correction=dict(contract=contract(), audit=audit),
                    geometry_hash=audit['geometry_hash'], provenance_hash=audit['provenance_hash'],
                    source_provenance=audit['source_provenance'],
                    out_of_box_flag_semantics='Inherited M2 source flag; use correction_current_out_of_box for current positions',
                    sampling_flag_semantics='All original conditioning/sampling flags and UV descriptors are M2 ancestry, not a claim that corrected centers still equal the original measured/source points; correction_delta_m is the normal displacement from M2')
    result['metadata_json'] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return result, audit


def build_corrected_geometry(input_path=None, observation_path=None, frames_path=None,
                             output_path=None, audit_path=None, contract_path=None):
    input_path = Path(input_path) if input_path else RUN/'geometry_dense.npz'
    observation_path = Path(observation_path) if observation_path else OBS/'a6.npz'
    frames_path = Path(frames_path) if frames_path else OBS/'frames'
    output_path = Path(output_path) if output_path else RUN/'geometry_measured.npz'
    audit_path = Path(audit_path) if audit_path else output_path.with_name('measured_correction_audit.json')
    contract_path = Path(contract_path) if contract_path else output_path.with_name('measured_correction_contract.json')
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Never overwrite the M2 source asset')
    # Persist the fixed contract BEFORE any observation/plane fit or correction.
    fixed = dict(contract(), input_file=str(input_path), input_sha256=_sha(input_path),
                 observations_file=str(observation_path), observations_sha256=_sha(observation_path), frames=str(frames_path))
    if contract_path.exists() and json.loads(contract_path.read_text(encoding='utf-8')) != json.loads(json.dumps(fixed)):
        raise ValueError('Existing M3 contract differs; do not overwrite preregistration')
    _json(contract_path, fixed)
    provenance = dict(input_file=str(input_path), input_sha256=fixed['input_sha256'],
                      observations_file=str(observation_path), observations_sha256=fixed['observations_sha256'],
                      frames_source=str(frames_path), contract_file=str(contract_path), contract_sha256=_sha(contract_path),
                      builder_file=str(Path(__file__)), builder_sha256=_sha(Path(__file__)),
                      metadata_errata='M3 top-level geometry/provenance hashes refreshed; source and contract SHA embedded before geometry construction. Projected nearest LiDAR depth is not physical first-return truth. No numerical method or threshold change.')
    with np.load(input_path, allow_pickle=False) as bank:
        arrays = {key: bank[key] for key in bank.files}
    with np.load(observation_path, allow_pickle=False) as bank:
        observations = {key: bank[key] for key in bank.files}
    frames = FrameBank(frames_path)
    try:
        corrected, audit = correct_geometry(arrays, observations, frames, provenance)
    finally:
        frames.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **corrected)
    temporary.replace(output_path)
    with np.load(output_path, allow_pickle=False) as saved:
        if hash_geometry(saved) != audit['geometry_hash'] or hash_provenance(saved) != audit['provenance_hash']:
            raise ArithmeticError('Saved corrected geometry/provenance roundtrip failed')
    audit.update(input_file=str(input_path), input_sha256=fixed['input_sha256'], observations_sha256=fixed['observations_sha256'],
                 output_file=str(output_path), output_sha256=_sha(output_path), contract_file=str(contract_path), contract_sha256=_sha(contract_path))
    _json(audit_path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path); parser.add_argument('--observations', type=Path)
    parser.add_argument('--frames', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--audit', type=Path); parser.add_argument('--contract', type=Path)
    parser.add_argument('--contract-only', action='store_true')
    args = parser.parse_args()
    if args.contract_only:
        _json(args.contract or RUN/'measured_correction_design.json', contract())
        return
    audit = build_corrected_geometry(args.input, args.observations, args.frames, args.output, args.audit, args.contract)
    print(json.dumps({key: audit[key] for key in ('status', 'rows', 'eligible_centers', 'moved_centers', 'geometry_hash', 'output_file')}))


if __name__ == '__main__':
    main()
