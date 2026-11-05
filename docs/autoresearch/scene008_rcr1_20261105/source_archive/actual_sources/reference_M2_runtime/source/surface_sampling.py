"""M2: fixed nine-child tangent discretization of one complete SH0 asset.

CPU only.  There is no native-a6 import, renderer change, image input, fitted
geometry, parameter sweep, or parent-plus-child rendering.  The normal-incidence
normalizations below do NOT establish per-pixel or multi-view equivalence.
"""
import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import spence

RUN = Path(__file__).resolve().parent.parent
PARENT = RUN.parent/'scene008_a6_complete_asset_20261105'
FLOAT_FIELDS = ('xyz', 'normal', 'sigma', 'opacity', 'thickness', 'box',
                'knn_spacing', 'normal_radius', 'normal_confidence')
BOOL_FIELDS = ('generated', 'conditioning', 'normal_valid', 'out_of_box', 'conditioning_held')
INT_FIELDS = ('source_index', 'conditioning_observation_index', 'conditioning_raw_frame', 'conditioning_cloud_row')
GEOMETRY_FIELDS = FLOAT_FIELDS+BOOL_FIELDS+INT_FIELDS
CHILDREN, SIGMA_RATIO, QUADRATURE_ORDER = 9, .5, 192
MIN_ALPHA, MAX_ALPHA, TERMINAL_T = 1/255, .99, .0001
GRID = np.array([(0, 0)]+[(u, v) for u in (-1, 0, 1) for v in (-1, 0, 1) if (u, v) != (0, 0)], np.float64)
APPEARANCE_FIELDS = {'rgb', 'colour_inferred', 'canonical_sh', 'coefficients', 'sh', 'sh_degree'}
RESERVED = set(GEOMETRY_FIELDS) | APPEARANCE_FIELDS | {'geometry_hash', 'provenance_hash', 'metadata_json'}


def _hash_arrays(arrays, keys, provenance=False):
    digest = hashlib.sha256()
    for key in sorted(keys):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode('utf-8'))
        header = (value.shape, value.dtype.str) if provenance else (value.dtype.str, value.shape)
        digest.update(str(header).encode('ascii')); digest.update(value.tobytes())
    return digest.hexdigest()


def hash_geometry(arrays):
    """Byte-compatible with full_pvd.hash_geometry for arbitrary row counts."""
    return _hash_arrays(arrays, GEOMETRY_FIELDS)


def hash_provenance(arrays):
    """Byte-compatible with SHAsset's hash of additional source/child arrays."""
    return _hash_arrays(arrays, set(arrays)-RESERVED, provenance=True)


def optical_mass(peak):
    """Integral tau du, u=r^2/2, tau=-log(1-native_alpha).

    The corresponding tangent-plane mass is 2*pi*sigma^2*this value.  The
    native .99 per-splat cap and 1/255 alpha cutoff are included analytically.
    """
    if peak <= MIN_ALPHA:
        return 0.
    flat = max(0., np.log(peak/MAX_ALPHA))
    return float(flat*(-np.log1p(-MAX_ALPHA))+spence(1-min(peak, MAX_ALPHA))-spence(1-MIN_ALPHA))


def alpha_area(peak):
    return float(2*np.pi*(MAX_ALPHA*max(0., np.log(peak/MAX_ALPHA))+min(peak, MAX_ALPHA)-MIN_ALPHA))


def family_alpha(x, y, parent_peak, outer_peak, step):
    """Native cap/cutoff/terminal rule for a normal-incidence isolated family."""
    transmittance = np.ones(np.broadcast_shapes(np.shape(x), np.shape(y)), np.float64)
    value, alive = np.zeros_like(transmittance), np.ones_like(transmittance, bool)
    for child, (u, v) in enumerate(GRID):
        peak = parent_peak if child == 0 else outer_peak
        alpha = np.minimum(MAX_ALPHA, peak*np.exp(-.5*((x-u*step)**2+(y-v*step)**2)/SIGMA_RATIO**2))
        alpha = np.where(alpha >= MIN_ALPHA, alpha, 0.)
        after = transmittance*(1-alpha)
        accept = alive & (after >= TERMINAL_T)
        value += np.where(accept, transmittance*alpha, 0.)
        transmittance = np.where(accept, after, transmittance)
        alive &= after >= TERMINAL_T
    return value


@lru_cache(maxsize=256)
def calibration(parent_peak):
    """Two deterministic scalar normalizations; no data/quality optimization.

    Half tangent sigma gives 2x linear resolution.  The central child retains
    the actual source point and peak.  Eight outer peaks preserve the sum of
    per-splat optical-depth mass; a unique grid step preserves isolated-family
    projected alpha area under one fixed quadrature.  All settings are global.
    """
    parent_peak = float(parent_peak)
    if not np.isfinite(parent_peak) or not MIN_ALPHA < parent_peak < 1:
        raise ValueError('This fixed control requires visible, finite peaks in (1/255,1); no rows are dropped')
    ratio2 = SIGMA_RATIO**2
    target = optical_mass(parent_peak)*(1-ratio2)/(8*ratio2)
    outer = brentq(lambda value: optical_mass(value)-target, MIN_ALPHA, parent_peak, xtol=1e-14, rtol=1e-14)
    parent_radius = float(np.sqrt(2*np.log(parent_peak/MIN_ALPHA)))
    outer_radius = float(SIGMA_RATIO*np.sqrt(2*np.log(outer/MIN_ALPHA)))
    # Every child cutoff disk stays inside its mother's cutoff disk in the
    # normal-incidence idealization; no affine screen-space expansion is used.
    max_step = (parent_radius-outer_radius)/np.sqrt(2)
    nodes, weights = leggauss(QUADRATURE_ORDER)
    x, y = np.meshgrid(nodes*parent_radius, nodes*parent_radius)
    area_weights = np.outer(weights, weights)*parent_radius**2
    target_area = alpha_area(parent_peak)
    residual = lambda step: float(np.sum(family_alpha(x, y, parent_peak, outer, step)*area_weights)-target_area)
    low, high = residual(0.), residual(max_step)
    if low > 0 or high < 0:
        raise ArithmeticError('Fixed nine-child normalization is not bracketed; no rule/parameter fallback')
    step = brentq(residual, 0., max_step, xtol=1e-11, rtol=1e-12)
    refined = family_alpha(x, y, parent_peak, outer, step)
    original = np.minimum(MAX_ALPHA, parent_peak*np.exp(-.5*(x*x+y*y)))
    original = np.where(original >= MIN_ALPHA, original, 0.)
    mass_ratio = ratio2*(optical_mass(parent_peak)+8*optical_mass(outer))/optical_mass(parent_peak)
    area_ratio = float(np.sum(refined*area_weights)/target_area)
    support_bound = max(SIGMA_RATIO*parent_radius, np.sqrt(2)*step+outer_radius)
    if abs(mass_ratio-1) > 2e-10 or abs(area_ratio-1) > 2e-5 or support_bound > parent_radius+1e-10:
        raise ArithmeticError('Fixed optical/area/support normalization failed; no clipping or retuning')
    return dict(parent_peak=parent_peak, central_peak=parent_peak, outer_peak=float(outer),
        grid_step_sigma=float(step), sigma_ratio=SIGMA_RATIO, optical_mass_ratio=float(mass_ratio),
        quadrature_alpha_area_ratio=area_ratio, parent_cutoff_radius_sigma=parent_radius,
        child_union_radius_bound_sigma=float(support_bound), quadrature_order=QUADRATURE_ORDER,
        profile_maxabs_alpha_change=float(np.max(np.abs(refined-original))),
        strong_support_hole_area=float(np.sum(((original >= .5) & (refined < .5))*area_weights)),
        strong_support_expansion_area=float(np.sum(((original < .5) & (refined >= .5))*area_weights)),
        alpha02_hole_area=float(np.sum(((original >= .2) & (refined < .2))*area_weights)),
        alpha02_expansion_area=float(np.sum(((original < .2) & (refined >= .2))*area_weights)))


def tangent_basis(normals):
    n = np.asarray(normals, np.float64)
    n = n/np.linalg.norm(n, axis=1, keepdims=True)
    reference = np.eye(3)[np.argmin(np.abs(n), axis=1)]
    u = reference-np.einsum('ij,ij->i', reference, n)[:, None]*n
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(n, u)
    return u, v


def refine_surface(arrays):
    """Return one free-N -> nine-N SH0 geometry table and its CPU audit.

    Input uses the full_pvd/SHAsset geometry schema.  All mother rows are
    replaced, including low-confidence/out-of-box sources.  A conditioning
    flag inherited by children means source ancestry, not new measurements.
    """
    missing = set(GEOMETRY_FIELDS)-set(arrays)
    if missing:
        raise ValueError('Missing complete geometry fields: '+','.join(sorted(missing)))
    if 'sampling_parent_index' in arrays:
        raise ValueError('This is one predeclared refinement, not repeated subdivision')
    count = len(arrays['xyz'])
    if count <= 0 or np.shape(arrays['xyz']) != (count, 3) or np.shape(arrays['normal']) != (count, 3):
        raise ValueError('Expected nonempty xyz/normal[N,3]')
    for key in GEOMETRY_FIELDS:
        value = np.asarray(arrays[key])
        if not np.isfinite(value).all() or (key not in ('xyz', 'normal', 'box') and value.shape != (count,)):
            raise ValueError('Invalid geometry array: '+key)
    if np.shape(arrays['box']) != (3,) or np.any(np.asarray(arrays['box']) <= 0):
        raise ValueError('Invalid global car box')
    if np.any(np.asarray(arrays['sigma']) <= 0) or np.any(np.asarray(arrays['thickness']) <= 0):
        raise ValueError('Nonpositive Gaussian scales')
    if not np.allclose(np.linalg.norm(arrays['normal'], axis=1), 1., atol=2e-6, rtol=0):
        raise ValueError('Unit normals required; no normal repair in this control')
    parent_hash = hash_geometry(arrays)
    if 'geometry_hash' in arrays and str(np.asarray(arrays['geometry_hash']).item()) != parent_hash:
        raise ValueError('Source geometry hash mismatch')
    if int(np.asarray(arrays.get('sh_degree', 0)).item()) > 0:
        raise ValueError('M2 is SH0; use the geometry/M0 source, not fitted SH3 material')
    for key in ('canonical_sh', 'coefficients', 'sh'):
        if key in arrays and np.asarray(arrays[key]).shape[1] > 1 and np.any(np.asarray(arrays[key])[:, 1:] != 0):
            raise ValueError('Nonzero higher SH would confound the SH0 sampling control')
    parent = np.repeat(np.arange(count, dtype=np.int64), CHILDREN)
    child = np.tile(np.arange(CHILDREN, dtype=np.int64), count)
    result = {}
    for key, value in arrays.items():
        if key in {'geometry_hash', 'provenance_hash', 'metadata_json'} | {'canonical_sh', 'coefficients', 'sh'}:
            continue
        value = np.asarray(value)
        if value.dtype.kind == 'O':
            raise ValueError('Object arrays are not allowed')
        result[key] = value[parent].copy() if key != 'box' and value.ndim and value.shape[0] == count else value.copy()
    u, v = tangent_basis(arrays['normal'])
    peak = np.asarray(arrays['opacity'], np.float64)
    calibrations = {float(p):deepcopy(calibration(float(p))) for p in np.unique(peak)}
    step = np.array([calibrations[float(p)]['grid_step_sigma'] for p in peak])
    outer = np.array([calibrations[float(p)]['outer_peak'] for p in peak])
    uv = GRID[child]*step[parent, None]*np.asarray(arrays['sigma'])[parent, None]
    offset = uv[:, :1]*u[parent]+uv[:, 1:]*v[parent]
    result['xyz'] = (np.asarray(arrays['xyz'])[parent]+offset).astype(np.float32)
    result['sigma'] = (np.asarray(arrays['sigma'])[parent]*SIGMA_RATIO).astype(np.float32)
    result['opacity'] = np.where(child == 0, peak[parent], outer[parent]).astype(np.float32)
    result['out_of_box'] = (np.abs(result['xyz']) > np.asarray(arrays['box'])/2+1e-6).any(1)
    for key in FLOAT_FIELDS: result[key] = np.asarray(result[key], np.float32)
    for key in BOOL_FIELDS: result[key] = np.asarray(result[key], bool)
    for key in INT_FIELDS: result[key] = np.asarray(result[key], np.int64)
    result.update(sampling_parent_index=parent, sampling_child_index=child,
        sampling_asset_row_id=np.arange(len(parent), dtype=np.int64), sampling_uv_m=uv.astype(np.float32),
        sampling_exact_source_point=child == 0, sampling_interpolated=child != 0,
        sampling_source_out_of_box=np.asarray(arrays['out_of_box'])[parent].astype(bool),
        sampling_parent_geometry_hash=np.asarray(parent_hash), sh_degree=np.int16(0))
    result['rgb'] = np.asarray(result.get('rgb', np.full((len(parent), 3), .5)), np.float32)
    if result['rgb'].shape != (len(parent), 3) or not np.isfinite(result['rgb']).all():
        raise ValueError('Bad SH0 RGB seed')
    result['colour_inferred'] = np.asarray(result.get('colour_inferred', np.ones(len(parent), bool)), bool) | (child != 0)
    geometry_hash = hash_geometry(result)
    audit = dict(status='CPU_DISCRETIZATION_BUILT', method='M2_nine_child_SH0_fixed_discretization',
        mother_rows=count, child_rows=len(parent), children_per_mother=CHILDREN, parent_rows_appended=0,
        parent_geometry_hash=parent_hash, geometry_hash=geometry_hash, sigma_ratio=SIGMA_RATIO,
        thickness_policy='Unchanged source normal thickness', normal_policy='Unchanged source normal; canonical deterministic tangent basis',
        calibration=list(calibrations.values()), source_normal_invalid=int((~np.asarray(arrays['normal_valid'])).sum()),
        source_out_of_box=int(np.asarray(arrays['out_of_box']).sum()), child_out_of_box=int(result['out_of_box'].sum()),
        true_source_positions_retained=int((child == 0).sum()), independent_measurements_added=0,
        runtime_requirement='Exactly one child asset, native_a6_rows_input=0; verify in actual scene runtime',
        mapping='parent_index is immediate mother row; source_index and conditioning flags are ancestry; only exact_source_point retains an original position',
        normalization='Preserve sum of individual native-capped/cutoff tangent optical-depth masses and normal-incidence isolated-family alpha area; no image fitting',
        limits=['This changes centers, scales, opacities, alpha contours and potentially depth ordering; not per-pixel geometric equivalence.',
                'Nonzero cutoff disks stay within each mother disk only in the ideal normal-incidence model; perspective, EWA filtering and oblique views require GPU checks.',
                'Equal integrated alpha does not imply equal thresholded silhouette: strong/weak expansion and holes are reported above.',
                'SH0 is unchanged as the material class; each child gets its own RGB only in a later BUILD-only fit.',
                'knn_spacing/normal_radius/confidence are inherited source-PCA descriptors, not re-estimated child spacing.',
                'No source row is dropped for clipping-quality flags or box position; no actual observations or hidden texture are invented.'],
        GPU_used=False, geometry_fit_updates=0, appearance_fit_updates=0, parameter_sweep=False)
    metadata = json.loads(str(np.asarray(arrays.get('metadata_json', '{}')).item()))
    metadata = dict(metadata, points=len(parent), geometry_hash=geometry_hash,
                    parent_geometry_hash=parent_hash, surface_sampling=audit,
                    source_count_semantics='Original conditioning/generated counts refer to mother-source provenance, not additional observations',
                    appearance='SH0 source RGB copied to children; offset-child colours initially inferred')
    result.update(geometry_hash=np.asarray(geometry_hash), metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)))
    result['provenance_hash'] = np.asarray(hash_provenance(result))
    audit['provenance_hash'] = str(result['provenance_hash'])
    metadata['provenance_hash'] = audit['provenance_hash']
    result['metadata_json'] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return result, audit


def build_dense_geometry(input_path=None, output_path=None, audit_path=None):
    input_path = Path(input_path) if input_path is not None else PARENT/'geometry.npz'
    output_path = Path(output_path) if output_path is not None else RUN/'geometry_dense.npz'
    audit_path = Path(audit_path) if audit_path is not None else RUN/'surface_sampling_audit.json'
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Never overwrite the source asset')
    with np.load(input_path, allow_pickle=False) as bank: source = {key:bank[key] for key in bank.files}
    arrays, audit = refine_surface(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name+'.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream, **arrays)
    temporary.replace(output_path)
    with np.load(output_path, allow_pickle=False) as bank:
        if hash_geometry(bank) != audit['geometry_hash'] or hash_provenance(bank) != audit['provenance_hash']:
            raise ArithmeticError('Saved dense geometry/provenance roundtrip failed')
    audit.update(source_file=str(input_path), source_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        output_file=str(output_path), output_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest())
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return audit


def self_check():
    normal = np.array([[0., 0., 1.], [1., 0., 0.], [.3, -.4, np.sqrt(.75)]], np.float32)
    n = len(normal)
    arrays = dict(xyz=np.array([[0., 0., .1], [3., 0., 0.], [.1, .2, .3]], np.float32), normal=normal,
        sigma=np.array([.06, .03, .011162], np.float32), opacity=np.full(n, .995, np.float32),
        thickness=np.full(n, .003, np.float32), box=np.array([4.33404, 2.07754, 1.44], np.float32),
        knn_spacing=np.full(n, .1, np.float32), normal_radius=np.full(n, .25, np.float32), normal_confidence=np.ones(n, np.float32),
        generated=np.array([False, True, True]), conditioning=np.array([True, False, False]),
        normal_valid=np.array([True, False, True]), out_of_box=np.array([False, True, False]),
        conditioning_held=np.array([True, False, False]), source_index=np.arange(n, dtype=np.int64),
        conditioning_observation_index=np.array([50, -1, -1], np.int64), conditioning_raw_frame=np.array([89, -1, -1], np.int64),
        conditioning_cloud_row=np.array([100, -1, -1], np.int64), rgb=np.full((n, 3), .5, np.float32))
    original = {key:value.copy() for key, value in arrays.items()}
    child, audit = refine_surface(arrays)
    assert len(child['xyz']) == 9*n and audit['parent_rows_appended'] == 0
    np.testing.assert_array_equal(child['xyz'][::9], arrays['xyz'])
    for key in arrays: np.testing.assert_array_equal(arrays[key], original[key])
    np.testing.assert_array_equal(child['sampling_parent_index'], np.repeat(np.arange(n), 9))
    offset = child['xyz'].astype(float)-arrays['xyz'][child['sampling_parent_index']]
    assert np.max(np.abs(np.einsum('ij,ij->i', offset, normal[child['sampling_parent_index']]))) < 2e-7
    assert np.allclose(child['sigma'], np.repeat(arrays['sigma']*.5, 9))
    assert (child['normal_valid'] == False).sum() == 9 and child['conditioning_held'].sum() == 9
    assert np.array_equal(child['normal'], np.repeat(normal, 9, axis=0))
    for entry in audit['calibration']:
        assert abs(entry['optical_mass_ratio']-1) < 2e-10
        assert abs(entry['quadrature_alpha_area_ratio']-1) < 2e-5
        assert entry['strong_support_hole_area'] == 0
        assert entry['child_union_radius_bound_sigma'] < entry['parent_cutoff_radius_sigma']
    for mother in range(n):
        group = child['sampling_parent_index'] == mother
        serialized_mass = sum((float(s)/float(arrays['sigma'][mother]))**2*optical_mass(float(o))
                              for s, o in zip(child['sigma'][group], child['opacity'][group]))
        assert abs(serialized_mass/optical_mass(float(arrays['opacity'][mother]))-1) < 1e-6
    digest = hash_geometry(child); child['rgb'][:] = .8; assert hash_geometry(child) == digest
    assert hash_provenance(child) == str(child['provenance_hash'])
    print(json.dumps(dict(status='PASS_CPU_MATH', free_N=n, child_N=len(child['xyz']),
                         calibration=audit['calibration'], GPU_used=False), allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description='CPU fixed nine-child SH0 sampling control')
    parser.add_argument('--input', type=Path); parser.add_argument('--output', type=Path); parser.add_argument('--audit', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check: self_check()
    else:
        audit = build_dense_geometry(args.input, args.output, args.audit)
        print(json.dumps({key:audit[key] for key in ('status', 'mother_rows', 'child_rows', 'geometry_hash', 'output_file')}))


if __name__ == '__main__': main()
