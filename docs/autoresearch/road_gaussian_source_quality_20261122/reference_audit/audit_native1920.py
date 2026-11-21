"""Use already-exported exact native 1920 core/alpha; no GPU or training."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, binary_erosion
from audit_source_quality import C, BASE, OUT, load, sha, stats, cosine, measure, screen_covariance, plane_diagnostic

BANK = Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')


def actual_camera(raw):
    frames = json.loads((BANK/'receipt.json').read_text())['frames']
    frame_id = next(row['id'] for row in frames if int(row['raw']) == raw)
    observation = json.loads((BANK/f'observations/{raw:03d}.json').read_text())
    calibration = {line.split(':')[0]: np.fromstring(line.split(':')[1], sep=' ')
        for line in (BANK/f'calib/{frame_id}.txt').read_text().splitlines()}
    K = calibration['P0'].reshape(3, 4)[:, :3]
    T = np.eye(4)
    T[:3] = calibration['Tr_velo_to_cam_0'].reshape(3, 4)
    T = T @ np.linalg.inv(np.asarray(observation['ego_world']).reshape(4, 4))
    return K, T


def difference_of_gaussians(photo, gs, mask):
    rows = []
    # Native 1920 pixels = 5/3 pixels per old 1152 pixel. Both ends are low-pass
    # filtered, so sensor/Nyquist frequencies are excluded from these bands.
    for base_scale in [.35, .6, .9]:
        sigma = base_scale*5/3
        a = gaussian_filter(photo, (sigma, sigma, 0))-gaussian_filter(photo, (2*sigma, 2*sigma, 0))
        b = gaussian_filter(gs, (sigma, sigma, 0))-gaussian_filter(gs, (2*sigma, 2*sigma, 0))
        a, b = a[mask], b[mask]
        ep, eg = np.mean(a*a), np.mean(b*b)
        rows.append(dict(scale_in_1152_px=base_scale, sigma_native1920_px=sigma,
            upper_gaussian_sigma_native1920_px=2*sigma, cosine=cosine(a, b),
            photo_RMS=float(np.sqrt(ep)), GS_RMS=float(np.sqrt(eg)),
            GS_to_photo_RMS=float(np.sqrt(eg/ep)), residual_RMS=float(np.sqrt(np.mean((a-b)**2)))))
    return rows


def contact(row, data):
    photo = np.rint(np.clip(data['observation'], 0, 1)*255).astype(np.uint8)
    gs = np.rint(np.clip(data['isolated_rgb'], 0, 1)*255).astype(np.uint8)
    full = np.rint(np.clip(data['full_road_rgb'], 0, 1)*255).astype(np.uint8)
    mask = data['score_mask']
    width, height = photo.shape[1], photo.shape[0]
    pad = 16
    canvas = Image.new('RGB', (max(1000, 12*width+3*pad), 7*height+150), (15, 19, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), f"{row['patch']} raw{row['raw']:03d}: exact native1920 crop {width}x{height}; score {mask.sum()} pixels", fill='white')
    for col, (label, array) in enumerate([('photo 1:1', photo), ('isolated GS 1:1', gs), ('full parent GS 1:1', full)]):
        x = pad+col*(width+pad)
        draw.text((x, 30), label, fill='white')
        canvas.paste(Image.fromarray(array), (x, 48))
    draw.text((pad, height+64), 'Same alpha>=.95 core mask; 6x nearest display only. No enhancement.', fill='white')
    for col, array in enumerate([photo, gs]):
        x = pad+col*(6*width+pad)
        array = np.where(mask[:, :, None], array, np.array([35, 41, 52], np.uint8))
        canvas.paste(Image.fromarray(array).resize((6*width, 6*height), Image.Resampling.NEAREST), (x, height+86))
    path = OUT/f"{row['patch']}_raw{row['raw']:03d}_exact_native1920.png"
    canvas.save(path)
    return str(path)


def main():
    receipt_path = OUT.parent/'native1920/receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['complete'] and receipt['optimizer_updates'] == 0
    result = dict(schema='source_quality_exact_native1920_CPU_audit.v1',
        native_export_receipt=str(receipt_path), GPU_calls_by_this_audit=0, optimizer_updates=0,
        new_diffusion_calls=0, original_assets_modified=False, cases=[], geometry={}, input_identity=[],
        primary_patches=['patch_00','patch_02'], secondary_control='patch_01',
        frequency_contract='1920 highpass sigma is also given in physical-equivalent 1152-pixel units; DoG bands low-pass both ends. Do not compare sigma1.2@1920 directly with sigma1@1152.',
        geometry_contract='Plane fitting is a diagnostic of retained 3D source geometry; no XYZ, normal, sigma, opacity or RGB is modified.')
    for name in ['patch_00','patch_01','patch_02']:
        packet = load(BASE/f'source_selection/candidates/{name}.npz')
        geometry = plane_diagnostic(packet)
        unit = packet['normal_world']/np.linalg.norm(packet['normal_world'],axis=1)[:,None]
        reference = packet['source_tangent_frame_world'][:3,2]
        geometry['normal_angle_to_source_chart_deg'] = stats(np.degrees(np.arccos(np.clip(unit[packet['core_mask']]@reference,-1,1))))
        result['geometry'][name] = geometry
    for row in receipt['rows']:
        path = Path(row['output'])/'core_float.npz'
        data = load(path)
        photo, gs = data['observation'].astype(np.float64), data['isolated_rgb'].astype(np.float64)
        mask = data['score_mask']
        assert mask.sum() == row['score_pixels']
        assert abs(float(np.abs(photo[mask]-gs[mask]).mean())-row['RGB_MAE']) < 2e-8
        metrics = measure(photo,gs,mask,sigmas=[.5*5/3,1.2,1.*5/3,1.5*5/3,2.*5/3])
        for band in metrics['bands']:
            band['equivalent_sigma_1152_px'] = band['highpass_gaussian_sigma_px']*.6
        inner = mask & binary_erosion(data['core'],iterations=8)
        packet = load(BASE/f"source_selection/candidates/{row['patch']}.npz")
        core = packet['core_mask'];K,T=actual_camera(row['raw'])
        covariance,J,xyz = screen_covariance(packet,T,K)
        eig = np.linalg.eigvalsh(covariance)
        local = (packet['world_xyz']-packet['source_center_world']) @ packet['source_tangent_frame_world'][:3,:3]
        coefficients=np.array(result['geometry'][row['patch']]['robust_plane_coefficients'])
        residual=local[:,2]-np.c_[local[:,:2],np.ones(len(local))]@coefficients
        height_to_pixels=np.linalg.norm(J@packet['source_tangent_frame_world'][:3,2],axis=1)
        yy,xx=np.where(mask)
        min_border_distance=int(min(yy.min(),xx.min(),mask.shape[0]-1-yy.max(),mask.shape[1]-1-xx.max()))
        case=dict(patch=row['patch'],raw=row['raw'],exact_score_pixels=int(mask.sum()),
            crop_wh=row['crop_wh'],core_polygon_pixels=int(data['core'].sum()),
            crop_outside_physical_core_fraction=1-float(data['core'].mean()),
            historical_native1920_receipt_row=row,metrics=metrics,
            DoG_physical_bands=difference_of_gaussians(photo,gs,mask),
            deeper_core_pixels=int(inner.sum()),
            deeper_core_metrics=measure(photo,gs,inner,sigmas=[1.2,5/3]) if inner.any() else None,
            isolation_full_MAE=float(np.abs(gs[mask]-data['full_road_rgb'][mask]).mean()),
            screen_footprint=dict(geometric_sigma_major_px=stats(np.sqrt(eig[core,1])),
                geometric_sigma_minor_px=stats(np.sqrt(eig[core,0])),
                reference_AA_FWHM_major_px=stats(2.354820045*np.sqrt(eig[core,1]+.3)),
                reference_AA_FWHM_minor_px=stats(2.354820045*np.sqrt(eig[core,0]+.3)),
                robust_plane_residual_apparent_displacement_px=stats((abs(residual)*height_to_pixels)[core]),
                camera_depth_m=stats(xyz[core,2])),
            crop_filter_boundary_check=dict(min_score_to_crop_border_px=min_border_distance,
                maximum_filter_radius_in_this_analysis_px=int(4*2.*5/3+.5),
                all_reported_bands_have_saved_support=min_border_distance>=int(4*2.*5/3+.5)))
        case['comparison_image']=contact(row,data)
        result['cases'].append(case)
        result['input_identity'].append(dict(path=str(path),sha256=sha(path)))
    result['input_identity'].append(dict(path=str(receipt_path),sha256=sha(receipt_path)))
    target=OUT/'source_quality_native1920.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(target),cases=[dict(patch=c['patch'],raw=c['raw'],
        exact_n=c['exact_score_pixels'],bands=c['metrics']['bands'],DoG=c['DoG_physical_bands'],
        footprint=c['screen_footprint'],isolation_full_MAE=c['isolation_full_MAE'],
        crop_boundary=c['crop_filter_boundary_check']) for c in result['cases']]),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
