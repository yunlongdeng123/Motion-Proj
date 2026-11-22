"""Independent CPU photo qualification and pixel-exact 00/02 inspection.

No GS rendering, optimization, resampling of diagnostic crops, or remote writes.
The inherited foreground/shadow policy is reproduced, using each patch's center.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, laplace, sobel

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
OLD = WORK / 'road_gaussian_source_quality_20261122_r1'
REMOTE = '/root/autodl-tmp/drivingeditor_repro/campaign/source_fidelity_20261122_r2/photo_audit'
SELECTED = {'patch_00': [86, 110, 137, 134], 'patch_02': [86, 110, 115, 117]}
EXTRA = {'patch_00': [], 'patch_02': [100, 116]}
LUMA = np.array([.2126, .7152, .0722], np.float32)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def linear(c):
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def packet(cid):
    with np.load(OUT / '_inputs' / f'{cid}.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def bbox(uv):
    lo = np.maximum([0, 0], np.floor(np.asarray(uv).min(0)).astype(int))
    hi = np.minimum([1920, 1280], np.ceil(np.asarray(uv).max(0)).astype(int) + 1)
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def polygon_mask(uv):
    p = Image.new('L', (1920, 1280))
    ImageDraw.Draw(p).polygon([tuple(x) for x in uv], fill=255)
    return np.asarray(p) > 0


def mask_save(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def foreground_guard(ann, fg, labels):
    guard = np.zeros((320, 480), bool)
    rects = np.array([x['xyxy'] for x in ann['front_camera_rectangles']]).reshape(-1, 4) * .25
    for x0, y0, x1, y1 in rects:
        x0, x1 = max(0, int(np.floor(x0))), min(480, int(np.ceil(x1)))
        y0, y1 = max(0, int(np.floor(y0))), min(320, int(np.ceil(y1)))
        if x1 > x0 and y1 > y0:
            guard[y0:y1, x0:x1] = True
    guard = binary_dilation(guard, iterations=2)
    guard |= binary_dilation(fg | labels, iterations=2)
    return guard


def prepare(row, cid, center):
    raw = row['raw']
    inp = OUT / '_inputs' / f'{raw:03d}'
    im = Image.open(inp / f'0065{raw:03d}.png').convert('RGB')
    assert im.size == (1920, 1280)
    sem = np.asarray(Image.open(inp / f'{raw:03d}.png'))
    fg = np.asarray(Image.open(inp / 'dynamic_union.png')) > 0
    labels = np.asarray(Image.open(inp / 'labels.png')) > 0
    ann = json.loads((inp / f'{raw:03d}.json').read_text())
    assert sem.shape == (1280, 1920), sem.shape
    assert fg.shape == labels.shape == (320, 480), (fg.shape, labels.shape)
    guard = foreground_guard(ann, fg, labels)
    small = np.asarray(im.resize((480, 320), Image.Resampling.LANCZOS), np.float32) / 255
    road = np.asarray(Image.fromarray((sem == 0).astype(np.uint8)).resize((480, 320), Image.Resampling.NEAREST)) > 0
    lum = linear(small) @ LUMA
    clear = road & ~guard
    den = gaussian_filter(clear.astype(np.float32), 12)
    local = gaussian_filter(lum * clear, 12) / np.maximum(den, 1e-6)
    shadow = (lum < .72 * local) & (local - lum > .02) & (den > .2) & road
    near = False
    for box in ann['boxes']:
        if box['type'] in (1, 2, 4):
            rel = (center - np.array(box['center'])) @ np.array(box['axes'])
            if (abs(rel[:2]) <= np.array(box['half'])[:2] + 2).all():
                shadow |= (lum < .85 * local) & (local - lum > .012)
                near = True
                break
    guard_full = np.asarray(Image.fromarray(guard.astype(np.uint8)).resize(im.size, Image.Resampling.NEAREST)) > 0
    shadow_full = np.asarray(Image.fromarray(shadow.astype(np.uint8)).resize(im.size, Image.Resampling.NEAREST)) > 0
    core = polygon_mask(row['core_native_uv'])
    inner = binary_erosion(core, iterations=2)
    valid = inner & (sem == 0) & ~guard_full & ~shadow_full
    return im, core, valid, dict(
        near_object_stronger_shadow_proxy=near,
        independent_core_polygon_pixels=int(core.sum()),
        core_eroded2_pixels=int(inner.sum()),
        core_photo_valid_pixels=int(valid.sum()),
        eroded_core_nonroad_pixels=int((inner & (sem != 0)).sum()),
        eroded_core_foreground_guard_pixels=int((inner & guard_full).sum()),
        eroded_core_shadow_proxy_pixels=int((inner & shadow_full).sum()),
        full_core_dynamic_clear_fraction=float((core & ~guard_full).sum() / core.sum()),
        full_core_shadow_proxy_clear_fraction=float((core & ~shadow_full).sum() / core.sum()),
        full_core_road_semantic_fraction=float((core & (sem == 0)).sum() / core.sum()),
    )


def metrics(im, mask):
    arr = np.asarray(im, np.float32) / 255
    gray = arr @ LUMA
    vals = gray[mask]
    gx, gy = sobel(gray, axis=1) / 8, sobel(gray, axis=0) / 8
    hp = gray - gaussian_filter(gray, 1)
    return dict(
        valid_native_pixels=int(mask.sum()), srgb_luma_mean=float(vals.mean()),
        srgb_luma_std=float(vals.std()), srgb_luma_p01_p50_p99=np.quantile(vals, [.01, .5, .99]).tolist(),
        linear_luma_mean=float((linear(arr) @ LUMA)[mask].mean()),
        native_laplacian_variance=float(np.var(laplace(gray)[mask])),
        native_tenengrad_mean=float(np.mean((gx * gx + gy * gy)[mask])),
        native_highpass_sigma1_std=float(np.std(hp[mask])),
        native_highpass_sigma2_std=float(np.std((gray - gaussian_filter(gray, 2))[mask])),
        any_channel_low_clip_fraction=float((arr[mask].min(1) <= 1 / 255).mean()),
        any_channel_high_clip_fraction=float((arr[mask].max(1) >= 254 / 255).mean()),
        median_native_rgb=np.median(arr[mask], axis=0).tolist(),
    )


def crop_exact(im, uv, dest):
    box = bbox(uv)
    crop = im.crop(box)
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest)
    x0, y0, x1, y1 = box
    assert np.array_equal(np.asarray(Image.open(dest)), np.asarray(im)[y0:y1, x0:x1])
    return dict(path=str(dest), bbox_xyxy=box, native_crop_wh=list(crop.size), pixels_unchanged=True,
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest())


def square_uv(p, row, size=.6):
    r = p['source_tangent_frame_world'][:3, :3]
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size / 2
    world = p['source_center_world'] + np.c_[local, np.zeros(4)] @ r.T
    t, k = np.array(row['world_to_camera']), np.array(row['camera_K_native'])
    q = world @ t[:3, :3].T + t[:3, 3]
    proj = q @ k.T
    return proj[:, :2] / proj[:, 2:] - .5


def sheet(items, filename, columns=2, cell_w=450, cell_h=110):
    # Every source image is pasted 1:1; only the surrounding canvas is padded.
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * cell_w, rows * cell_h), '#101317')
    draw = ImageDraw.Draw(canvas)
    for i, (label, p) in enumerate(items):
        x, y = (i % columns) * cell_w, (i // columns) * cell_h
        draw.text((x + 10, y + 5), label, fill='#e9edf2')
        image = Image.open(p).convert('RGB')
        assert image.width <= cell_w - 20 and image.height <= cell_h - 28, (label, image.size)
        canvas.paste(image, (x + 10, y + 25))
    path = OUT / filename
    canvas.save(path)
    return str(path)


def main():
    records, audit, native_items, equal_items, pair_items = [], [], [], [], []
    reports = {cid: json.loads((OLD / f'takeback/source_view_ranking/{cid}_views.json').read_text()) for cid in SELECTED}
    packets = {cid: packet(cid) for cid in SELECTED}
    for cid, selected in SELECTED.items():
        lookup = {r['raw']: r for r in reports[cid]['all_frame_records']}
        for raw in selected + EXTRA[cid]:
            original = lookup[raw]
            assert original['photo_legal'], (cid, raw, original['rejection_reasons'])
            row = dict(original)
            im, core, valid, maskaudit = prepare(row, cid, packets[cid]['source_center_world'])
            dest = OUT / 'masks' / cid / f'raw{raw:03d}_core_photo_valid_full1920.png'
            cp = dest.with_name(f'raw{raw:03d}_core_polygon_full1920.png')
            mask_save(valid, dest)
            mask_save(core, cp)
            cropped = crop_exact(im, row['core_native_uv'], OUT / 'native_crops' / cid / f'raw{raw:03d}_core_native.png')
            uv06 = square_uv(packets[cid], row)
            crop06 = crop_exact(im, uv06, OUT / 'native_crops' / cid / f'raw{raw:03d}_physical0p6m_native.png')
            mask06 = polygon_mask(uv06) & valid
            a = dict(candidate_id=cid, raw=raw, photo_legal=True,
                     selection_kind='probe_selected' if raw in selected else 'supplementary_photo_reference',
                     original_RGB=row['original_RGB'], local_original_RGB=str(OUT / '_inputs' / f'{raw:03d}' / f'0065{raw:03d}.png'),
                     core_size_m=reports[cid]['core_size_m'], original_core_short_axis_native_px=row['core_min_caliper_native_px'],
                     inherited_box_clear_sample_fraction=row['box_clear_sample_fraction'],
                     inherited_old_fresh_certificate_supported=row['old_fresh_certificate_supported'],
                     inherited_accepted_distinct_world_4cm_cells=row['accepted_distinct_world_4cm_cells'],
                     mask_audit=maskaudit, core_metrics=metrics(im, valid),
                     same_physical_0p6m_metrics=metrics(im, mask06), core_crop=cropped, physical0p6m_crop=crop06)
            audit.append(a)
            if raw in selected:
                row.update(
                    role_in_this_probe='query' if raw in [86, 110] else 'candidate_reference',
                    original_RGB_local=a['local_original_RGB'],
                    core_photo_valid_mask=str(dest), core_photo_valid_mask_full1920=str(dest), core_polygon_mask=str(cp),
                    core_photo_valid_mask_full1920_remote_stage=f'{REMOTE}/masks/{cid}/{dest.name}',
                    core_polygon_mask_full1920_remote_stage=f'{REMOTE}/masks/{cid}/{cp.name}',
                    bbox_xyxy=cropped['bbox_xyxy'], core_photo_valid_native_pixels=int(valid.sum()),
                    original_crops=dict(core=cropped), qualification_scope='core only; no halo supervision',
                    local_mask_audit=maskaudit)
                records.append(row)
            size = reports[cid]['core_size_m'][0]
            native_items.append((f'{cid} raw{raw:03d} {size:.1f}m core | {cropped["native_crop_wh"][0]}x{cropped["native_crop_wh"][1]}px', cropped['path']))
            if raw in [86, 110]:
                equal_items.append((f'{cid} raw{raw:03d} same 0.6m square | 1:1 native', crop06['path']))
                oldr = next(x for x in json.loads((OLD / 'native1920/receipt.json').read_text())['rows'] if x['patch'] == cid and x['raw'] == raw)
                olddir = OLD / f'native1920/{cid}/{raw:03d}'
                photo = np.asarray(Image.open(olddir / 'observation.png'))
                x0, y0, x1, y1 = oldr['bbox_xyxy']
                assert np.array_equal(photo, np.asarray(im)[y0:y1, x0:x1]), (cid, raw)
                pair_items.extend([(f'{cid} raw{raw:03d} PHOTO native {photo.shape[1]}x{photo.shape[0]}', str(olddir / 'observation.png')),
                                   (f'{cid} raw{raw:03d} ORIGINAL GS same native crop', str(olddir / 'isolated.png'))])
            im.close()
    plan = dict(complete=True, source_candidate_default=None,
                source_candidates=list(SELECTED), selected_from_completed_199frame_photo_ranking=True,
                selected_raws_by_candidate=SELECTED, records=records,
                source_asset='/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/appearance_fit/asset',
                source_packet_by_candidate={cid: f'/root/autodl-tmp/drivingeditor_repro/campaign/road_gaussian_transfer_20261122_r1/source_selection/candidates/{cid}.npz' for cid in SELECTED},
                native_wh=[1920, 1280], native_render_required=True, no_photo_resampling=True,
                mask_policy='projected candidate-specific core eroded2px AND native road semantic0 AND NOT upsampled foreground guard AND NOT candidate-center-specific photometric shadow proxy',
                mask_near_shadow_candidate_center_corrected=True,
                photo_qualification_independent_of_fresh_depth_support=True,
                final_fit_mask='full1920 photo-valid mask INTERSECT separately computed isolated GS contribution qualification',
                source_geometry_scale_opacity_fixed=True,
                shadow_mask_kind='inherited local photometric proxy',
                camera_convention='Inherited FRONT pinhole K @ inverse ego_world, projection minus0.5px; Brown retained but not applied.',
                GPU_calls=0, optimizer_updates=0, new_diffusion_calls=0)
    write(OUT / 'probe_plan.json', plan)
    for cid in SELECTED:
        single = dict(plan)
        single.update(source_candidate=cid, records=[r for r in records if r['candidate_id'] == cid],
                      source_packet=plan['source_packet_by_candidate'][cid])
        write(OUT / f'{cid}_probe_plan.json', single)
    sheets = [sheet(native_items, 'native_reference_candidates_1to1.png', columns=2),
              sheet(equal_items, 'same_physical_0p6m_native_1to1.png', columns=2),
              sheet(pair_items, 'photo_vs_original_GS_native_1to1.png', columns=2)]
    write(OUT / 'photo_audit.json', dict(complete=True, audit_kind='independent native CPU photo audit',
          rows=audit, no_default_source=True, sheets=sheets,
          display_policy='Every image crop is pixel-exact and pasted without resizing; equal physical area does not imply equal display width.',
          legal_scope='Projected source core only; bounding rectangle corners are context only and excluded from metrics.',
          GPU_calls=0, optimizer_updates=0, new_diffusion_calls=0))
    print(json.dumps(dict(complete=True, plan=str(OUT / 'probe_plan.json'), rows=len(audit), sheets=sheets)))


if __name__ == '__main__':
    main()
