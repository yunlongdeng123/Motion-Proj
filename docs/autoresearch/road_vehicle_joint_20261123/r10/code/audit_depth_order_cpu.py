"""Read-only CPU depth ordering against an UNAPPROVED quadratic profile.

No renderer, torch, Gaussian updates, or geometry activation.  The primary
pixel measurement is the existing shared expected-depth buffer.  The separate
row measurement uses original native Gaussian centres and measured ROI VJPs;
it is deliberately not described as a per-pixel native depth decomposition.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

from local_surface_projection import raster_surface


REGIONS = ('red_diagnostic', 'blue_cyan_diagnostic',
           'ground_band_before_support', 'full_original_band')
MASS_KEYS = dict(red_diagnostic='allview_red_mass',
                 blue_cyan_diagnostic='allview_blue_mass',
                 ground_band_before_support='allview_ground_mass',
                 full_original_band=None)
TOLERANCES = (0., .02, .05, .10, .20)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def pin(path):
    p = Path(path)
    return dict(path=p.as_posix(), bytes=p.stat().st_size, sha256=sha(p))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def stats(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(count=0)
    return dict(count=int(a.size), min=float(a.min()), p05=float(np.quantile(a, .05)),
                median=float(np.median(a)), mean=float(a.mean()),
                p95=float(np.quantile(a, .95)), max=float(a.max()))


def weighted(a, weight):
    a = np.asarray(a, np.float64)
    w = np.asarray(weight, np.float64)
    ok = np.isfinite(a) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return dict(rows=0, alphaT_mass=0.)
    a, w = a[ok], w[ok]
    order = np.argsort(a)
    a, w = a[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    return dict(rows=int(a.size), alphaT_mass=float(w.sum()),
                mean=float(np.sum(a*w)/w.sum()),
                p05=float(np.interp(.05, cdf, a)),
                median=float(np.interp(.5, cdf, a)),
                p95=float(np.interp(.95, cdf, a)),
                min=float(a[0]), max=float(a[-1]))


def basis(xy, centre):
    d = np.asarray(xy, np.float64) - centre
    x, y = d[..., 0], d[..., 1]
    return np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], -1)


class Profile:
    def __init__(self, packet):
        self.packet = packet
        self.xyz = packet['profile_world_xyz']
        self.uv = packet['uv_global']
        self.mask = packet['global_hull_interpolation_mask']
        self.coeff = packet['fit_coefficients']
        self.boot = packet['bootstrap_coefficients']
        self.centre = packet['fit_XY_centre']
        assert self.boot.shape == (512, 6)
        assert not packet['renderable_mask'].any()
        assert not bool(packet['geometry_confidence_is_calibrated_probability'])
        design = np.column_stack([self.xyz[..., :2].reshape(-1, 2),
                                  np.ones(self.mask.size)])
        self.xy_to_uv = np.linalg.lstsq(design, self.uv.reshape(-1, 2), rcond=None)[0]
        self.affine_error = float(abs(design @ self.xy_to_uv - self.uv.reshape(-1, 2)).max())
        assert self.affine_error < 1e-7
        fitted_z = basis(self.xyz[..., :2], self.centre) @ self.coeff
        self.profile_formula_error = float(abs(fitted_z-self.xyz[..., 2]).max())
        assert self.profile_formula_error < 1e-8
        self.uv0 = self.uv[0, 0]
        assert np.max(abs(np.diff(self.uv[..., 0], axis=1)-1)) < 1e-7
        assert np.max(abs(np.diff(self.uv[..., 1], axis=0)-1)) < 1e-7

    def inside(self, xy):
        uv = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], -1) @ self.xy_to_uv
        uv = uv - self.uv0
        good = np.isfinite(uv).all(-1)
        safe = np.where(good[..., None], uv, 0.)
        ij = np.floor(safe).astype(np.int64)
        h, w = self.mask.shape
        good &= (ij[..., 0] >= 0) & (ij[..., 0] < w-1) & (ij[..., 1] >= 0) & (ij[..., 1] < h-1)
        u = np.clip(ij[..., 0], 0, w-2)
        v = np.clip(ij[..., 1], 0, h-2)
        frac = safe - ij
        # Same two triangle vertex sets as the parent's CPU projector.
        first = frac[..., 1] <= frac[..., 0]
        good &= self.mask[v, u] & self.mask[v+1, u+1]
        good &= np.where(first, self.mask[v, u+1], self.mask[v+1, u])
        return good

    def roots(self, coefficients, camera_origin, ray_world, nominal_hint=None):
        """Positive, hull-supported intersections; t equals camera-z in metres."""
        c = np.asarray(coefficients, np.float64)[:, None, :]
        rx, ry, rz = [ray_world[None, :, i] for i in range(3)]
        dx, dy = camera_origin[:2] - self.centre
        aa = c[..., 3]*rx*rx + c[..., 4]*rx*ry + c[..., 5]*ry*ry
        bb = (c[..., 1]*rx+c[..., 2]*ry+2*c[..., 3]*dx*rx+
              c[..., 4]*(dx*ry+dy*rx)+2*c[..., 5]*dy*ry-rz)
        cc = (c[..., 0]+c[..., 1]*dx+c[..., 2]*dy+c[..., 3]*dx*dx+
              c[..., 4]*dx*dy+c[..., 5]*dy*dy-camera_origin[2])
        disc = bb*bb-4*aa*cc
        root = np.sqrt(np.maximum(disc, 0.))
        # Stable quadratic evaluation; linear degenerate cases are explicit.
        q = -.5*(bb+np.copysign(root, bb))
        with np.errstate(divide='ignore', invalid='ignore'):
            t1, t2 = q/aa, cc/q
            linear = -cc/bb
        linear_case = abs(aa) < 1e-14
        t1 = np.where(linear_case, linear, t1)
        t2 = np.where(linear_case, np.nan, t2)
        outcomes = []
        for t in (t1, t2):
            valid = ((disc >= 0) | linear_case) & np.isfinite(t) & (t > 1e-8)
            xy = camera_origin[:2]+t[..., None]*ray_world[None, :, :2]
            valid &= self.inside(xy)
            outcomes.append(np.where(valid, t, np.nan))
        a, b = outcomes
        if nominal_hint is None:
            choose_b = np.isfinite(b) & (~np.isfinite(a) | (b < a))
        else:
            choose_b = np.isfinite(b) & (~np.isfinite(a) |
                       (abs(b-nominal_hint[None]) < abs(a-nominal_hint[None])))
        return np.where(choose_b, b, a)

    def ray_intervals(self, E, ray_camera, nominal_hint=None):
        origin = -E[:3, :3].T @ E[:3, 3]
        rays = ray_camera @ E[:3, :3]
        nominal = self.roots(self.coeff[None], origin, rays,
                             nominal_hint=nominal_hint)[0]
        n = len(ray_camera)
        lower = np.full(n, np.nan)
        upper = lower.copy()
        counts = np.zeros(n, np.int16)
        for start in range(0, n, 1024):
            stop = min(start+1024, n)
            values = self.roots(self.boot, origin, rays[start:stop], nominal[start:stop])
            counts[start:stop] = np.isfinite(values).sum(0)
            # Quantiles only enter classifications when ALL512 rays stay supported.
            full = np.isfinite(values).all(0)
            if full.any():
                lo, hi = np.quantile(values[:, full], [.025, .975], axis=0)
                lo = np.minimum(lo, nominal[start:stop][full])
                hi = np.maximum(hi, nominal[start:stop][full])
                lower[start:stop][full] = lo
                upper[start:stop][full] = hi
        return nominal, lower, upper, counts


def classes(depth, lo, hi, support, tol, extent=0.):
    """0=unsupported, 1=before, 2=overlapping uncertainty/near, 3=behind."""
    out = np.zeros(np.shape(depth), np.uint8)
    ok = support & np.isfinite(depth) & (depth > 0) & np.isfinite(lo) & np.isfinite(hi)
    out[ok] = 2
    out[ok & (depth+extent < lo-tol)] = 1
    out[ok & (depth-extent > hi+tol)] = 3
    return out


def class_counts(c, mask, w=None):
    names = ('unsupported', 'before_profile', 'near_or_uncertain', 'behind_profile')
    result = {name:int(np.count_nonzero(mask & (c == i))) for i, name in enumerate(names)}
    result['denominator'] = int(mask.sum())
    if w is not None:
        result['measured_alphaT_mass'] = {name:float(np.sum(w[mask & (c == i)])) for i, name in enumerate(names)}
        result['mass_denominator'] = float(w[mask].sum())
    return result


def camera_rays(x, y, K):
    samples = np.column_stack([x+.5, y+.5, np.ones_like(x)])
    ray = samples @ np.linalg.inv(K).T
    return ray/ray[:, 2:3]


def quat_matrix(q):
    q = np.asarray(q, np.float64)
    norm = np.linalg.norm(q, axis=1)
    assert np.isfinite(q).all() and (norm > 0).all()
    w, x, y, z = (q/norm[:, None]).T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1).reshape(-1, 3, 3)


def main(a):
    start = time.time()
    root, out = a.root.resolve(), a.output.resolve()
    assert not out.exists(), 'Use a new audit directory; never replace evidence.'
    out.mkdir(parents=True)
    pins = {}

    def read_npz(name, path):
        pins[name] = pin(path)
        return load(path)

    profile_path = root/'audit/local_surface_inference_v1_replay1/inference_candidate_DIAGNOSTIC_ONLY.npz'
    p = Profile(read_npz('profile', profile_path))
    pins['profile_C0_C1'] = pin(profile_path.parent/'parent_boundary_C0_C1.json')
    pins['projector'] = pin(Path(__file__).with_name('local_surface_projection.py'))
    policy = read_npz('candidate_policy', root/'body_overlap_policy_v1/diagnostic_native_candidates.npz')
    frames = list(map(int, policy['frames']))
    gids = policy['global_row_ids']
    assert len(gids) == len(np.unique(gids)) and np.all(np.diff(gids) > 0)
    for key in ['allview_K_mass', 'allview_body_mass', 'allview_ground_mass',
                'allview_red_mass', 'allview_blue_mass']:
        assert policy[key].shape == (len(gids), len(frames))
        assert np.isfinite(policy[key]).all() and (policy[key] >= 0).all()
    assert np.all(policy['allview_K_mass'] == 0)
    assert np.all((policy['owners_by_view'] <= 0) | (policy['owners_by_view'] == 10))
    result = dict(schema='R10.uncertified_profile_depth_order_CPU.v1', complete=False,
                  geometry_renderable=False, asset_initialization_authorized=False,
                  inputs_read_only=True, Gaussian_updates=0, GPU_forward=0, GPU_backward=0,
                  frames=frames, candidate_count=len(gids),
                  original_actor_sha256='ab4fa110440d579bcfd57e4bd0b8a7f61c3f7821189450b58bd027c2f28f67a7',
                  profile_status='NOT_C0_C1_MATCHED_NOT_RENDERABLE',
                  depth_domain='physical metric world and camera z, original calibrated pinhole1152x768',
                  uncertainty='nominal plus central95% of512 frame-cluster bootstrap ray intersections; conditional profile sensitivity, not calibrated truth coverage',
                  classification='before: depth < lower-tolerance; behind: depth > upper+tolerance; otherwise near/uncertain; only all512 hull-supported intersections are classified',
                  expected_depth_scope='existing shared alphaT-normalized depth mixture including actor, road and native context, not native-only geometry',
                  native_row_scope='actual5444 body-overlap diagnostic native centres; ROI masses are measured alphaT VJPs integrated over each whole ROI, not pixel-resolved depth weights',
                  Gaussian_extent_scope='3sigma camera-axis range is supplemental and not a certified full footprint occlusion bound',
                  tolerances_m=list(TOLERANCES), UV_affine_error_texels=p.affine_error,
                  profile_formula_error_m=p.profile_formula_error, views={})
    for frame_index, raw in enumerate(frames):
        print(f'CPU frame {raw:03d}: loading and projecting diagnostic profile', flush=True)
        rr = f'{raw:03d}'
        cache = read_npz(f'{rr}.camera', root.parent/'road_shared_appearance_20261122_r7/projection_cache_v1'/rr/'view_cache.npz')
        evaluation = read_npz(f'{rr}.fixed_eval', root/'fixed_eval_v1_replay1'/rr/'fixed_eval.npz')
        base = read_npz(f'{rr}.baseline', root/'measure_v1'/rr/'baseline_joint.npz')
        loc = read_npz(f'{rr}.native_locators', root/'body_overlap_diagnostic_v1'/rr/'retired_prepared_geometry.npz')
        E, K = cache['world_to_camera'], cache['camera_K']
        assert E.shape == (4, 4) and K.shape == (3, 3)
        assert np.max(abs(E[:3, :3]@E[:3, :3].T-np.eye(3))) < 1e-10
        masks = {key:evaluation[key].astype(bool) for key in REGIONS}
        union = np.logical_or.reduce(list(masks.values()))
        y, x = np.nonzero(union)
        H, W = union.shape
        projected = raster_surface(p.xyz, p.uv, p.mask, E, K, [W, H])
        hint = projected['camera_depth_m'][y, x]
        analytic, lo, hi, boot_count = p.ray_intervals(E, camera_rays(x, y, K), hint)
        support = projected['projection_valid'][y, x] & (boot_count == len(p.boot))
        projection_diff = np.abs(analytic-hint)
        check = projected['projection_valid'][y, x] & np.isfinite(analytic)
        if check.any():
            assert np.max(projection_diff[check]) < .005, 'Analytic quadratic and triangulated2cm profile disagree'
        depth = np.asarray(base['depth_expected_m'], np.float64)[y, x]
        alpha = base['alpha'][y, x]
        support &= np.isfinite(depth) & (depth > 0) & (alpha > 1e-6)
        view = dict(raw=raw, pixel_regions={}, native_candidate_regions={},
                    camera_K=K.tolist(), world_to_camera=E.tolist(),
                    projected_profile_pixels=int(projected['projection_valid'].sum()),
                    triangulated_vs_analytic_ray_depth_abs_m=stats(projection_diff[check]))
        pixel_packet = dict(pixel_y=y, pixel_x=x, expected_shared_depth_m=depth,
                            nominal_profile_depth_m=hint, analytic_profile_depth_m=analytic,
                            bootstrap_lower_depth_m=lo, bootstrap_upper_depth_m=hi,
                            bootstrap_supported_count=boot_count, comparison_supported=support,
                            scene_alpha=alpha)
        for name, full_mask in masks.items():
            selected = full_mask[y, x]
            one = dict(fixed_pixels=int(full_mask.sum()),
                       nominal_projection_supported=int(np.count_nonzero(selected & projected['projection_valid'][y, x])),
                       all512_bootstrap_supported=int(np.count_nonzero(selected & (boot_count == len(p.boot)))),
                       valid_shared_expected_depth=int(np.count_nonzero(selected & np.isfinite(depth) & (depth > 0))),
                       shared_depth_m=stats(depth[selected]),
                       profile_depth_m=stats(hint[selected & support]),
                       profile_conditional_depth_interval_width_m=stats((hi-lo)[selected & support]),
                       shared_minus_profile_depth_m=stats((depth-hint)[selected & support]),
                       classification_by_tolerance_m={})
            for tol in TOLERANCES:
                cls = classes(depth, lo, hi, support, tol)
                one['classification_by_tolerance_m'][str(tol)] = class_counts(cls, selected)
                if tol == .05:
                    pixel_packet[name] = selected
                    pixel_packet[name+'_class_5cm'] = np.where(selected, cls, 0)
            for key in ['road_alphaT', 'actor_alphaT', 'context_alphaT']:
                one[key] = stats(base[key][full_mask])
            view['pixel_regions'][name] = one
        # Original native centre positions and camera-z, never actor subtraction.
        assert np.array_equal(loc['global_row_ids'], gids)
        assert not bool(loc['clean_road_geometry_certificate'])
        world = loc['means_world_locator_only']
        mfw = loc['model_from_world']
        reconstructed = world@mfw[:3, :3].T+mfw[:3, 3]
        transform_error = float(abs(reconstructed-loc['means_model']).max())
        assert transform_error < 2e-5
        camera = world@E[:3, :3].T+E[:3, 3]
        z = camera[:, 2]
        q = camera@K.T
        with np.errstate(divide='ignore', invalid='ignore'):
            row_x, row_y = (q[:, :2]/z[:, None]-.5).T
            row_ray = camera/z[:, None]
        row_profile, row_lo, row_hi, row_boot_count = p.ray_intervals(E, row_ray)
        row_support = np.isfinite(row_profile) & (row_boot_count == len(p.boot)) & (z > 0)
        rotation = quat_matrix(loc['rotations_model'])
        camera_z_from_model = E[2, :3]@np.linalg.inv(mfw[:3, :3])
        local_axis = np.einsum('j,njk->nk', camera_z_from_model, rotation)
        sigma_z = np.sqrt(np.sum((local_axis*loc['scales_model'])**2, axis=1))
        row_packet = dict(global_row_ids=gids, owner_ids=loc['owner_ids'],
                          camera_depth_m=z, projected_center_xy=np.stack([row_x, row_y], -1),
                          profile_at_center_ray_depth_m=row_profile,
                          bootstrap_lower_depth_m=row_lo, bootstrap_upper_depth_m=row_hi,
                          bootstrap_supported_count=row_boot_count, sigma_camera_z_m=sigma_z)
        view['locator_model_roundtrip_max_abs_model_units'] = transform_error
        for name, mass_key in MASS_KEYS.items():
            if mass_key is None:
                continue
            w = np.asarray(policy[mass_key][:, frame_index], np.float64)
            active = w > 0
            row_packet[name+'_alphaT_mass'] = w
            one = dict(positive_mass_rows=int(active.sum()), measured_alphaT_mass=float(w.sum()),
                       native_center_camera_depth_m=weighted(z, w),
                       sigma_camera_z_m=weighted(sigma_z, w),
                       same_center_ray_profile_depth_m=weighted(row_profile, np.where(row_support, w, 0)),
                       native_center_minus_same_ray_profile_m=weighted(z-row_profile, np.where(row_support, w, 0)),
                       center_ray_order_by_tolerance_m={},
                       center_with_3sigma_camera_z_range_order_by_tolerance_m={})
            for tol in TOLERANCES:
                cls = classes(z, row_lo, row_hi, row_support, tol)
                extended = classes(z, row_lo, row_hi, row_support, tol, 3*sigma_z)
                one['center_ray_order_by_tolerance_m'][str(tol)] = class_counts(cls, active, w)
                one['center_with_3sigma_camera_z_range_order_by_tolerance_m'][str(tol)] = class_counts(extended, active, w)
                if tol == .05:
                    row_packet[name+'_center_class_5cm'] = cls
                    row_packet[name+'_3sigma_class_5cm'] = extended
            # Conservative ROI-wide depth-range comparison supplements centre rays.
            selected = masks[name][y, x]
            available = selected & support
            if available.any():
                rlo, rhi = float(lo[available].min()), float(hi[available].max())
                all_roi_supported = bool(np.all(support[selected]))
                cls = classes(z, np.full_like(z, rlo), np.full_like(z, rhi),
                              np.isfinite(z) & (z > 0), .05)
                one['whole_ROI_depth_range_order_5cm'] = dict(
                    road_depth_range_m=[rlo, rhi], full_ROI_profile_supported=all_roi_supported,
                    scope='global ROI range; VJP does not resolve contributions onto unsupported pixels',
                    **class_counts(cls, active, w))
            view['native_candidate_regions'][name] = one
        destination = out/rr
        destination.mkdir()
        np.savez_compressed(destination/'pixel_depth_diagnostic.npz', **pixel_packet)
        np.savez_compressed(destination/'native_center_diagnostic.npz', **row_packet)
        (destination/'summary.json').write_text(json.dumps(view, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        result['views'][rr] = view
        print(f'CPU frame {rr}: complete; red_pixels={view["pixel_regions"]["red_diagnostic"]["fixed_pixels"]}', flush=True)
    result['input_pins'] = pins
    result['input_hashes_unchanged'] = all(sha(v['path']) == v['sha256'] for v in pins.values())
    assert result['input_hashes_unchanged']
    result['torch_imported'] = 'torch' in sys.modules
    assert not result['torch_imported']
    result['complete'] = True
    result['seconds'] = time.time()-start
    (out/'depth_order_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(complete=True, output=out.as_posix(), seconds=result['seconds'], GPU_calls=0, updates=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
