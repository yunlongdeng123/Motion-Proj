"""Read-only independent audit of a range-expanded, unchanged plane hypothesis.

Projection analytically intersects every actual calibrated pixel ray with the
unchanged plane. It never invents material, native alpha, or a geometry approval.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
from diagnose_nonplanar_inference_cpu import load, read, write, pin, stats, design

RAWS = [86, 110, 134, 179, 198]


def png(path, rgb):
    Image.fromarray(np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(path)


def intersect(view, shape, b, centre):
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    E, K = view['world_to_camera'], view['camera_K']
    camera_rays = np.stack([xx+.5, yy+.5, np.ones(shape)], -1) @ np.linalg.inv(K).T
    C = -E[:3, :3].T @ E[:3, 3]
    direction = camera_rays @ E[:3, :3]
    n = np.array([-b[1], -b[2], 1.])
    plane_distance = b[0]-b[1]*centre[0]-b[2]*centre[1]
    denominator = direction @ n
    t = (plane_distance-C@n)/np.where(abs(denominator) > 1e-12, denominator, np.nan)
    world = C+t[..., None]*direction
    good = np.isfinite(world).all(-1) & (t > 0) & (abs(denominator) > 1e-12)
    camera = world @ E[:3, :3].T+E[:3, 3]
    return world, good & (camera[..., 2] > .1), camera[..., 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    C, out = a.campaign_root, a.output
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    R = C/'road_underbody_local_20261123_r10'
    G = R/'audit/local_surface_expanded_candidate_v1/expanded_candidate_geometry.npz'
    oldpath = R/'audit/local_surface_candidate_v2/candidate_geometry.npz'
    g, old = load(G), load(oldpath)
    producer_path = G.parent/'expanded_candidate_receipt.json'
    producer = read(producer_path)
    assert producer['geometry_npz']['sha256'] == pin(G)['sha256']
    for key in ['fit_coefficients', 'fit_XY_centre', 'bootstrap_coefficients']:
        assert g[key].dtype == old[key].dtype and g[key].shape == old[key].shape
        assert g[key].tobytes() == old[key].tobytes(), key
    b, centre = g['fit_coefficients'], g['fit_XY_centre']
    assert not b[3:].any()
    asset = g['cell_asset_mask']
    assert np.array_equal(asset, g['cell_reconstruction_mask'] | g['cell_blend_ring_mask'])
    assert not (g['cell_reconstruction_mask'] & g['cell_blend_ring_mask']).any()
    assert not (asset & ~g['cell_candidate_mask']).any()
    assert not (asset & (g['cell_geometry_source_kind'] != 2)).any()
    assert not g['renderable_mask'].any()
    assert np.array_equal(g['cell_uv_global'], g['uv_global'][:-1, :-1]+.5)
    for key in ['world_xyz', 'normal_world', 'cell_world_xyz', 'cell_normal_world',
                'height_bootstrap_SD_m', 'cell_height_bootstrap_SD_m']:
        assert np.isfinite(g[key]).all(), key
    grid = load(C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz')
    M, O = grid['world_from_uv_xy'][:2, :2], grid['world_from_uv_xy'][:2, 3]
    expected_xy = g['uv_global']@M.T+O
    expected_cellxy = g['cell_uv_global']@M.T+O
    assert np.array_equal(expected_xy, g['world_xyz'][..., :2])
    assert np.array_equal(expected_cellxy, g['cell_world_xyz'][..., :2])
    z_error = abs(design(g['world_xyz'][..., :2].reshape(-1, 2), centre)@b-g['world_xyz'][..., 2].ravel()).max()
    cell_z_error = abs(design(expected_cellxy.reshape(-1, 2), centre)@b-g['cell_world_xyz'][..., 2].ravel()).max()
    assert max(z_error, cell_z_error) < 1e-12
    normal_error = max(abs(np.linalg.norm(g[k], axis=-1)-1).max() for k in ['normal_world', 'cell_normal_world'])
    assert normal_error < 1e-12
    # Independent explicit 512-sample prediction check, not just the covariance shortcut.
    A = design(expected_cellxy[asset][::97], centre)
    direct_sd = np.std(A@g['bootstrap_coefficients'].T, axis=1, ddof=1)
    sd_error = abs(direct_sd-g['cell_height_bootstrap_SD_m'][asset][::97]).max()
    assert sd_error < 1e-10
    # The inherited old-rectangle world coordinates stay exactly the same.
    old_u, old_v = old['uv_global'][..., 0], old['uv_global'][..., 1]
    u0, v0, u1, v1 = g['bounds_uvxy'].astype(int)
    exact_overlap = np.array_equal(g['world_xyz'][old_v-v0, old_u-u0], old['world_xyz'])
    assert exact_overlap
    strict = g['strict_witness_mask_legacy']
    assert int(strict.sum()) == 837
    # Derive vertex footprint without altering the candidate NPZ.
    vertex_asset = np.zeros(g['candidate_mask'].shape, bool)
    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        vertex_asset[dy:dy+asset.shape[0], dx:dx+asset.shape[1]] |= asset
    np.savez_compressed(out/'derived_scope_and_extrapolation.npz',
        vertex_asset_support_mask=vertex_asset,
        original_fit_extrapolated_asset_vertices=vertex_asset & (g['original_fit_hull_margin_m'] < 0),
        candidate_geometry_sha256=np.asarray(pin(G)['sha256']))
    physical_path = R/'audit/ground_contact_prior_raw_GT_v2/GT_bottom_and_road_evidence.npz'
    physical = load(physical_path)
    P, raws = physical['official_training_road_worldXYZ'], physical['official_training_road_source_raw']
    Q = expected_cellxy[asset]
    bbox = np.stack([g['world_xyz'][..., :2].reshape(-1, 2).min(0)-3,
                     g['world_xyz'][..., :2].reshape(-1, 2).max(0)+3])
    keep = (P[:, :2] >= bbox[0]).all(-1) & (P[:, :2] <= bbox[1]).all(-1) & (abs(P[:, 2]+20.817) <= 1.5)
    P, raws = P[keep], raws[keep]
    d, ids = cKDTree(P[:, :2]).query(Q, k=16)
    delta = P[ids, :2]-Q[:, None, :]
    quad = (delta[..., 0] >= 0)*2+(delta[..., 1] >= 0)
    quadrants = sum(np.any(quad == k, axis=1) for k in range(4))
    frame_counts = np.zeros(len(Q), np.int16)
    for raw in np.unique(raws):
        frame_counts += cKDTree(P[raws == raw, :2]).query(Q)[0] <= 1.
    residual = design(P[:, :2], centre)@b-P[:, 2]
    physical_rows = [dict(raw=int(raw), points=int((raws == raw).sum()),
                          signed_plane_minus_official_road_m=stats(residual[raws == raw]),
                          absolute_plane_minus_official_road_m=stats(abs(residual[raws == raw])))
                     for raw in np.unique(raws)]
    # These broad-context physical residuals are diagnostics, not the old heldout score.
    prior_path = R/'audit/local_surface_candidate_v2/independent_limited_geometry_review.json'
    prior = read(prior_path)
    contact = read(R/'audit/ground_contact_prior_raw_GT_v2/ground_contact_prior_audit.json')
    rows, previews = [], []
    for raw in RAWS:
        folder = out/f'{raw:03d}'
        folder.mkdir()
        cp = C/f'road_shared_appearance_20261122_r7/projection_cache_v1/{raw:03d}/view_cache.npz'
        view = load(cp)
        mp = R/f'edit_footprint_masks_v2/{raw:03d}/scope_masks.npz'
        masks = load(mp)
        shape = masks['inside_allowed_edit'].shape
        world, good, depth = intersect(view, shape, b, centre)
        uv = (world[..., :2]-O)@np.linalg.inv(M).T
        iu = np.floor(np.nan_to_num(uv[..., 0], nan=-1e8)).astype(np.int64)-u0
        iv = np.floor(np.nan_to_num(uv[..., 1], nan=-1e8)).astype(np.int64)-v0
        incontext = good & (iu >= 0) & (iv >= 0) & (iu < asset.shape[1]) & (iv < asset.shape[0])
        cu, cv = np.clip(iu, 0, asset.shape[1]-1), np.clip(iv, 0, asset.shape[0]-1)
        supported = incontext & asset[cv, cu]
        original_fit_inside = incontext & (g['original_fit_hull_margin_m'][cv, cu] >= 0)
        masks_protected = masks['actor_protected'] | masks['K_protected']
        display = supported & ~masks_protected
        expected = load(G.parent/f'{raw:03d}_requested_coverage.npz')
        assert np.array_equal(supported & masks['inside_allowed_edit'], expected['expanded_geometry_supported'])
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        hatch = np.where((((xx//8)+(yy//8)) % 2)[..., None], .11, .17)*np.ones((*shape, 3))
        neutral = hatch.copy()
        neutral[supported & original_fit_inside] = [.48, .61, .65]
        neutral[supported & ~original_fit_inside] = [.67, .53, .30]
        gridline = (abs(uv[..., 0]-np.rint(uv[..., 0]/10)*10) < .55) | (abs(uv[..., 1]-np.rint(uv[..., 1]/10)*10) < .55)
        neutral[supported & gridline] = [.77, .84, .84]
        photo = np.clip(view['photo_pinhole_srgb'], 0, 1)
        overlay = photo.copy()
        overlay[display] = .68*photo[display]+.32*neutral[display]
        overlay[display & gridline] = [.10, .80, .72]
        edge = supported & ~binary_erosion(supported)
        overlay[edge & ~masks_protected] = [1., .6, .1]
        image = Image.fromarray(np.rint(overlay*255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        contacts = []
        for rr in contact['GT_bottom_rows']:
            if rr['raw'] != raw or not rr.get('present'):
                continue
            X = np.asarray(rr['bottom_world_corners_and_centre'])
            E, K = view['world_to_camera'], view['camera_K']
            xc = X@E[:3, :3].T+E[:3, 3]
            q = xc@K.T
            xy = q[:, :2]/xc[:, 2, None]-.5
            if (xc[:, 2] > .1).all():
                draw.line([tuple(xy[k]) for k in [0, 1, 3, 2, 0]],
                          fill=(60, 235, 70) if rr['alias'] == 'a20' else (235, 80, 215), width=2)
            contacts.append(dict(alias=rr['alias'],
                road_minus_GT_bottom_world_Z_m=(design(X[:, :2], centre)@b-X[:, 2]).tolist(),
                no_vehicle_geometry_fitted_or_moved=True))
        png(folder/'geometry_surface_only.png', neutral)
        image.save(folder/'geometry_photo_context.png')
        np.savez_compressed(folder/'surface_scope.npz',
            atlas_uv_global=np.where(incontext[..., None], uv, np.nan).astype(np.float32),
            camera_depth_m=np.where(incontext, depth, 0).astype(np.float32),
            geometry_candidate_context=incontext, asset_scope=supported,
            fit_hull_inside=original_fit_inside, display_visible=display,
            inside_allowed_edit=masks['inside_allowed_edit'], K_protected=masks['K_protected'],
            actor_protected=masks['actor_protected'])
        region_rows = {}
        for name in ['inside_allowed_edit', 'full_original_band', 'inside_replacement', 'K_protected', 'actor_protected']:
            mask = masks[name]
            part = dict(outside_candidate_context=int((mask & ~incontext).sum()),
                        candidate_not_in_asset=int((mask & incontext & ~supported).sum()),
                        asset_supported=int((mask & supported).sum()))
            assert sum(part.values()) == int(mask.sum())
            region_rows[name] = dict(fixed_pixels=int(mask.sum()), partition=part,
                asset_supported_in_original_fit_hull=int((mask & supported & original_fit_inside).sum()),
                asset_supported_model_extrapolation=int((mask & supported & ~original_fit_inside).sum()))
        row = dict(raw=raw, camera=pin(cp), scope_masks=pin(mp), regions=region_rows,
                   GT_bottom_contact=contacts, shader='neutral geometry, no material or native alpha',
                   context_photo_only=True, protects_only_display_not_physical_road=True)
        write(folder/'projection_receipt.json', row)
        rows.append(row)
        panel = Image.new('RGB', (1152, 410), (20, 23, 30))
        panel.paste(Image.fromarray(np.rint(photo*255).astype(np.uint8)).resize((576, 384)), (0, 26))
        panel.paste(image.resize((576, 384)), (576, 26))
        ImageDraw.Draw(panel).text((8, 7), f'{raw:03d}  original photo | expanded inferred plane + GT bottom', fill=(235, 235, 235))
        previews.append(panel)
    sheet = Image.new('RGB', (1152, 410*len(previews)), (20, 23, 30))
    for j, im in enumerate(previews):
        sheet.paste(im, (0, 410*j))
    sheet.save(out/'five_view_expanded_geometry_contactsheet.png')
    vertex_fit_extrap = vertex_asset & (g['original_fit_hull_margin_m'] < 0)
    result = dict(schema='R10.independent_expanded_geometry_candidate_audit.v1', complete=True,
        decision='CANDIDATE_INTEGRITY_PASS_APPROVAL_PENDING', candidate_geometry=pin(G), producer_receipt=pin(producer_path),
        source_kind=2, all_geometry_is_model_inference=True, geometry_accepted_for_zero_step=False,
        independent_checks=dict(model_and_512_bootstrap_byteexact=True, original_candidate_overlap_worldXYZ_byteexact=True,
            integer_vertex_and_half_cell_UV=True, analytic_XYZ_and_cell_Z_max_error_m=float(max(z_error, cell_z_error)),
            normal_unit_max_error=float(normal_error), explicit_512_bootstrap_SD_max_error_m=float(sd_error),
            strict837_preserved=True, mask_partition_and_inference_labels_exact=True, renderable_false_preserved=True),
        asset_cells=int(asset.sum()), asset_vertices=int(vertex_asset.sum()),
        asset_vertices_inside_original_fit_hull=int((vertex_asset & ~vertex_fit_extrap).sum()),
        asset_vertices_outside_original_fit_hull=int(vertex_fit_extrap.sum()),
        asset_vertices_inside_broader_training_official_road_validation_hull=int((vertex_asset & (g['physical_validation_hull_margin_m'] >= 0)).sum()),
        asset_vertex_original_fit_leverage_ratio=stats(g['original_fit_leverage_over_max_training'][vertex_asset]),
        conditional_parameter_bootstrap_SD_m=stats(g['cell_height_bootstrap_SD_m'][asset]),
        conditional_SD_does_not_include_model_misspecification=True,
        nearest_official_road_distance_at_asset_cells_m=stats(d[:, 0]),
        asset_nearest16_official_point_direction_quadrants=stats(quadrants),
        asset_distinct_training_frames_with_official_road_within1m=stats(frame_counts),
        expanded_training_official_road_profile_diagnostic=physical_rows,
        inherited_fixed_heldout_scores=prior['fixed_heldout_height_normal'],
        inherited_heldout_does_not_measure_entire_expanded_domain=True,
        original_LOO_and_heldout_selection_unchanged=True, GT_bottom_not_fitted=True,
        source_coverage_and_material_acceptance='separate source preparation; no RGB written',
        five_view_projection=rows, contactsheet=pin(out/'five_view_expanded_geometry_contactsheet.png'),
        root_approval='PENDING', visual_review='PENDING', whole_scene_quality='NOT_ASSESSED',
        GPU_forward=0, backward=0, updates=0, seconds=time.time()-start)
    write(out/'independent_geometry_audit.json', result)
    print(json.dumps(dict(complete=True, audit=pin(out/'independent_geometry_audit.json'),
                         asset_cells=int(asset.sum()), asset_vertices=int(vertex_asset.sum()),
                         fit_extrapolated_asset_vertices=int(vertex_fit_extrap.sum()))), flush=True)


if __name__ == '__main__':
    main()
