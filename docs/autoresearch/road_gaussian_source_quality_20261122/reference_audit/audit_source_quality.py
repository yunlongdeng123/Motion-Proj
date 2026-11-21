"""CPU audit of existing source Gaussian patches; no native/GPU rendering."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

C = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
BASE = C / 'road_gaussian_transfer_20261122_r2_isometric'
OUT = C / 'road_gaussian_source_quality_20261122_r1/reference_audit'


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(value):
    value = np.asarray(value)
    return dict(n=int(value.size), min=float(value.min()), q05=float(np.quantile(value, .05)),
        median=float(np.median(value)), q95=float(np.quantile(value, .95)), max=float(value.max()),
        mean=float(value.mean()), RMS=float(np.sqrt(np.mean(value**2))))


def cosine(a, b):
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denominator) if denominator else None


def project(world, T, K):
    camera = world @ T[:3, :3].T + T[:3, 3]
    homogeneous = camera @ K.T
    return homogeneous[:, :2]/camera[:, 2:]-.5, camera


def screen_covariance(packet, T, K):
    M = packet['model_from_world'][:3, :3]
    scale = np.linalg.norm(M, axis=0).mean()
    normal = packet['normal_world'] @ M.T / scale
    # Use the exact legacy model-normal shortest arc.
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    q = np.c_[1+normal[:, 2], -normal[:, 1], normal[:, 0], np.zeros(len(normal))]
    q[np.linalg.norm(q, axis=1) < 1e-8] = [0., 1., 0., 0.]
    q /= np.linalg.norm(q, axis=1)[:, None]
    model_rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    unit = M/scale
    world_rotation = np.einsum('ij,njk->nik', unit.T, model_rotation)
    sigma = packet['gaussian_sigma_m']
    covariance = (world_rotation*sigma[:, None, :]**2) @ world_rotation.transpose(0, 2, 1)
    _, camera = project(packet['world_xyz'], T, K)
    z = camera[:, 2]
    J = np.zeros((len(z), 2, 3))
    J[:, 0, 0] = K[0, 0]/z
    J[:, 0, 2] = -K[0, 0]*camera[:, 0]/z**2
    J[:, 1, 1] = K[1, 1]/z
    J[:, 1, 2] = -K[1, 1]*camera[:, 1]/z**2
    J = J @ T[:3, :3]
    screen = J @ covariance @ J.transpose(0, 2, 1)
    return screen, J, camera


def measure(photo, gs, mask, sigmas=None):
    result = dict(pixels=int(mask.sum()), RGB_MAE=float(np.abs(gs[mask]-photo[mask]).mean()),
        photo_RGB_std=float(photo[mask].std()), GS_RGB_std=float(gs[mask].std()), bands=[])
    for sigma in (sigmas if sigmas is not None else [.5, 1., 2., 4.]):
        hp = photo-gaussian_filter(photo, (sigma, sigma, 0))
        hg = gs-gaussian_filter(gs, (sigma, sigma, 0))
        p, g = hp[mask], hg[mask]
        ep, eg = float(np.mean(p*p)), float(np.mean(g*g))
        result['bands'].append(dict(highpass_gaussian_sigma_px=sigma, photo_RMS=float(np.sqrt(ep)),
            GS_RMS=float(np.sqrt(eg)), GS_to_photo_RMS=float(np.sqrt(eg/ep)) if ep else None,
            cosine=cosine(p, g), residual_RMS=float(np.sqrt(np.mean((p-g)**2)))))
    blur_curve = []
    for sigma in [0., .5, 1., 1.5, 2., 3.]:
        candidate = photo if sigma == 0 else gaussian_filter(photo, (sigma, sigma, 0))
        a, b = candidate[mask], gs[mask]
        ac, bc = a-a.mean(0), b-b.mean(0)
        gain = float(np.sum(ac*bc)/np.sum(ac*ac))
        bias = b.mean(0)-gain*a.mean(0)
        residual = b-(a*gain+bias)
        blur_curve.append(dict(photo_blur_sigma_px=sigma, fitted_scalar_gain=gain,
            fitted_RGB_bias=bias.tolist(), affine_residual_MSE=float(np.mean(residual**2))))
    result['photo_blur_affine_match_curve'] = blur_curve
    return result


def energy_bounds(photo, gs, mask, historical_n):
    hp = photo-gaussian_filter(photo, (1, 1, 0))
    hg = gs-gaussian_filter(gs, (1, 1, 0))
    p, g = np.mean(hp[mask]**2, axis=1), np.mean(hg[mask]**2, axis=1)
    assert len(p) >= historical_n
    bounds = lambda x: [float(np.sqrt(np.sort(x)[:historical_n].mean())),
        float(np.sqrt(np.sort(x)[-historical_n:].mean()))]
    bp, bg = bounds(p), bounds(g)
    return dict(core_geometric_n=len(p), historical_alpha95_n=historical_n,
        pixels_excluded_by_unavailable_isolated_alpha=len(p)-historical_n,
        photo_RMS_interval_any_subset_of_that_size=bp, GS_RMS_interval_any_subset_of_that_size=bg,
        GS_to_photo_RMS_conservative_interval=[bg[0]/bp[1], bg[1]/bp[0]],
        note='Bounds apply to the known historical alpha95 mask as one subset; no isolated-alpha mask is invented.')


def plane_diagnostic(packet):
    core = packet['core_mask']
    basis = packet['source_tangent_frame_world'][:3, :3]
    local = (packet['world_xyz']-packet['source_center_world']) @ basis
    X = np.c_[local[core, :2], np.ones(core.sum())]
    y = local[core, 2]
    keep = np.ones(len(y), bool)
    for _ in range(4):
        coefficients = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        residual = y-X@coefficients
        center = np.median(residual)
        mad = max(1.4826*np.median(np.abs(residual-center)), .0005)
        keep = np.abs(residual-center) <= 3*mad
    residual = y-X@coefficients
    spacing = cKDTree(packet['world_xyz'][core]).query(packet['world_xyz'][core], k=2)[0][:, 1]
    return dict(core_rows=int(core.sum()), halo_rows=int((~core).sum()),
        tangent_sigma_m=stats(packet['gaussian_sigma_m'][core, :2]),
        normal_sigma_m=stats(packet['gaussian_sigma_m'][core, 2]),
        core_nearest_neighbor_spacing_m=stats(spacing),
        source_frame_height_about_median_m=stats(np.abs(y-np.median(y))),
        robust_plane_signed_residual_m=stats(residual),
        robust_plane_absolute_residual_m=stats(np.abs(residual)),
        robust_plane_coefficients=coefficients.tolist(),
        robust_plane_inliers=int(keep.sum()), diagnostic_only_geometry_modified=False)


def visual(case, photo_u8, gs_u8, mask):
    width, height = photo_u8.shape[1], photo_u8.shape[0]
    pad = 16
    canvas = Image.new('RGB', (max(900, 12*width+3*pad), 6*height+height+116), (16, 20, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), f"{case['patch']} raw{case['raw']:03d}: native {width} x {height}; photo / existing isolated GS", fill='white')
    draw.text((pad, 28), 'Top: original native pixels, including halo. Bottom: same geometric core, 6x nearest; no enhancement.', fill='white')
    canvas.paste(Image.fromarray(photo_u8), (pad, 48))
    canvas.paste(Image.fromarray(gs_u8), (2*pad+width, 48))
    for col, source in enumerate([photo_u8, gs_u8]):
        pixels = np.where(mask[:, :, None], source, np.array([35, 40, 50], dtype=np.uint8))
        scaled = Image.fromarray(pixels).resize((6*width, 6*height), Image.Resampling.NEAREST)
        x = pad+col*(6*width+pad)
        draw.text((x, height+64), 'photo core' if col == 0 else 'existing GS core', fill='white')
        canvas.paste(scaled, (x, height+84))
    path = OUT / f"{case['patch']}_raw{case['raw']:03d}_native_and_6x.png"
    canvas.save(path)
    return str(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((BASE/'source_native/receipt.json').read_text())
    result = dict(schema='source_gaussian_quality_CPU_audit.v1', base=str(BASE), GPU_calls=0,
        optimizer_updates=0, new_diffusion_calls=0, original_assets_modified=False,
        source_asset=receipt['reference_asset'], cases=[], geometry={}, input_files=[],
        mask_contract='historical float native metrics use erode3 polygon & isolated alpha>=.95; new PNG metrics use explicitly labelled geometric core. Historical alpha raster was not saved.',
        source_library_scope='all-frame independent support/ranking is audited by the multiview agent; query records are not independent source pixels')
    packets = {name: load(BASE/f'source_selection/candidates/{name}.npz') for name in ['patch_00','patch_02']}
    result['geometry'] = {name: plane_diagnostic(packet) for name, packet in packets.items()}
    for historic in receipt['quality']:
        patch, raw = historic['patch'], historic['raw']
        packet = packets[patch]
        camera = load(C/f'road_common1152_20261121/maps/{raw:03d}.npz')
        T, K = camera['world_to_camera'], camera['camera_K']
        uv, _ = project(packet['source_world_roi_corners'], T, K)
        polygon = Image.new('L', (1152, 768))
        ImageDraw.Draw(polygon).polygon([tuple(point) for point in uv], fill=1)
        polygon = np.asarray(polygon) > 0
        box = historic['bbox']; x0, y0, x1, y1 = box
        crop = lambda a: a[y0:y1, x0:x1]
        masks = {f'geometric_erode{n}': crop(binary_erosion(polygon, iterations=n)) for n in [3, 6]}
        photo_path = BASE/f'source_native/{patch}/{raw:03d}_observation_crop.png'
        gs_path = BASE/f'source_native/{patch}/{raw:03d}_isolated_crop.png'
        photo_u8, gs_u8 = np.array(Image.open(photo_path)), np.array(Image.open(gs_path))
        photo, gs = photo_u8.astype(np.float64)/255, gs_u8.astype(np.float64)/255
        native_photo = np.array(Image.open(C/f'road_common1152_20261121/teacher/raw{raw:03d}/observation_1152.png'))
        assert np.array_equal(photo_u8, crop(native_photo))
        metrics = {name: measure(photo, gs, mask) for name, mask in masks.items() if mask.any()}
        footprint, jacobian, xyz = screen_covariance(packet, T, K)
        eig = np.linalg.eigvalsh(footprint)
        core = packet['core_mask']
        tangent_basis = packet['source_tangent_frame_world'][:3, :3]
        height = ((packet['world_xyz']-packet['source_center_world']) @ tangent_basis)[:, 2]
        projected_height = np.linalg.norm(jacobian @ tangent_basis[:, 2], axis=1)*np.abs(height-np.median(height[core]))
        case = dict(patch=patch, raw=raw, historical_float_native=historic,
            input_quantization='supplementary RGB data are saved rounded 8-bit effective native crops; historical float metrics remain authoritative',
            native_crop_wh=[x1-x0, y1-y0], projected_core_uv=uv.tolist(),
            projected_polygon_pixels=int(polygon.sum()), crop_pixels=(x1-x0)*(y1-y0),
            displayed_rect_outside_core_fraction=1-float(crop(polygon).mean()), metrics=metrics,
            historical_alpha95_energy_bounds=energy_bounds(photo, gs, masks['geometric_erode3'], historic['score_n']),
            screen_footprint=dict(camera_depth_m=stats(xyz[core, 2]),
                geometric_sigma_major_px=stats(np.sqrt(eig[core, 1])),
                geometric_sigma_minor_px=stats(np.sqrt(eig[core, 0])),
                sigma_major_with_reference_AA_03px2=stats(np.sqrt(eig[core, 1]+.3)),
                sigma_minor_with_reference_AA_03px2=stats(np.sqrt(eig[core, 0]+.3)),
                approximate_FWHM_major_px=stats(2.354820045*np.sqrt(eig[core, 1]+.3)),
                height_scatter_projected_displacement_px=stats(projected_height[core]),
                projection_note='first-order pinhole projection of actual world covariance; +0.3 px^2 shown explicitly as the native-style reference filter, not a new native render'))
        case['comparison_image'] = visual(case, photo_u8, gs_u8, masks['geometric_erode3'])
        np.savez_compressed(OUT/f'{patch}_raw{raw:03d}_audit_arrays.npz', photo_rgb=photo_u8,
            GS_rgb=gs_u8, projected_polygon=crop(polygon), **masks)
        result['cases'].append(case)
        result['input_files'] += [dict(path=str(p), sha256=sha(p)) for p in [photo_path, gs_path]]
    result['input_files'] += [dict(path=str(BASE/'source_native/receipt.json'), sha256=sha(BASE/'source_native/receipt.json'))]
    target = OUT/'source_quality.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(result=str(target), geometry=result['geometry'], cases=[dict(patch=c['patch'],raw=c['raw'],
        core=c['metrics']['geometric_erode3'], native_alpha95_bounds=c['historical_alpha95_energy_bounds'],
        footprint=c['screen_footprint']) for c in result['cases']]),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
