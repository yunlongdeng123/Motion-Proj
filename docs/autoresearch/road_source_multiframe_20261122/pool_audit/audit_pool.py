"""固定02多帧实拍池：原生像素、照片核心资格、相机中心基线。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
OLD = WORK / 'road_gaussian_source_quality_20261122_r1'
REMOTE = '/root/autodl-tmp/drivingeditor_repro/campaign/road_source_multiframe_20261122_r3/pool_audit'
TRAIN = [88, 90, 92, 94, 96, 99, 104, 115]
CONTROL = [88, 115]
QUERY = [86, 100, 110, 117]


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def contact_sheet(records, filename):
    columns, width, height = 2, 430, 95
    image = Image.new('RGB', (columns * width, ((len(records)+1)//2) * height), '#101317')
    draw = ImageDraw.Draw(image)
    for i, r in enumerate(records):
        x, y = i % columns * width, i // columns * height
        crop = Image.open(r['core_original_crop']['path']).convert('RGB')
        draw.text((x+10,y+5), f"{r['role']} raw{r['raw']:03d} | {crop.width}x{crop.height}px | 1:1 native", fill='#edf2f7')
        image.paste(crop, (x+10,y+25))
    image.save(OUT / filename)


def main():
    spec = importlib.util.spec_from_file_location('r2_photo_policy', OUT/'code_dependencies/audit_native_photos.py')
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    a.OUT = OUT
    report = json.loads((OLD/'takeback/source_view_ranking/patch_02_views.json').read_text())
    lookup = {r['raw']:r for r in report['all_frame_records']}
    legal = [r for r in lookup.values() if r.get('photo_legal')]
    packet = a.packet('patch_02')
    records, crops, validation = [], [], []
    for raw in TRAIN + QUERY:
        r = dict(lookup[raw])
        assert r['photo_legal']
        im, core, valid, maskaudit = a.prepare(r, 'patch_02', packet['source_center_world'])
        maskpath = OUT/'masks'/f'raw{raw:03d}_core_photo_valid_full1920.png'
        corepath = maskpath.with_name(f'raw{raw:03d}_core_polygon_full1920.png')
        a.mask_save(valid,maskpath)
        a.mask_save(core,corepath)
        crop = a.crop_exact(im,r['core_native_uv'],OUT/'native_crops'/f'raw{raw:03d}_core_native.png')
        rgb = np.asarray(im)
        low_clip = bool((rgb[valid].min(1)<=1).any())
        high_clip = bool((rgb[valid].max(1)>=254).any())
        strict = binary_erosion(valid,structure=np.ones((9,9),bool))
        role = 'TRAIN' if raw in TRAIN else ('QUERY_LOW_RES_STRESS' if raw==86 else 'QUERY_MAIN')
        r.update(role=role, historical_role='BUILD; QUERY only excludes this 400-step paired experiment optimizer inputs',
            original_RGB_local=str(OUT/'_inputs'/f'{raw:03d}'/f'0065{raw:03d}.png'),
            core_photo_valid_mask_full1920_local=str(maskpath),
            core_photo_valid_mask_full1920_remote_stage=f'{REMOTE}/masks/{maskpath.name}',
            core_polygon_mask_full1920_remote_stage=f'{REMOTE}/masks/{corepath.name}',
            core_photo_valid_mask=f'{REMOTE}/masks/{maskpath.name}',
            core_original_crop=crop,bbox_xyxy=crop['bbox_xyxy'],
            core_photo_valid_native_pixels=int(valid.sum()),mask_audit=maskaudit,
            core_photo_metrics=a.metrics(im,valid),
            strict_sigma1_filter_support_native_pixels=int(strict.sum()),
            strict_sigma1_photo_metrics=a.metrics(im,strict) if strict.sum()>=16 else None,
            original_RGB_sha256=hashlib.sha256(Path(OUT/'_inputs'/f'{raw:03d}'/f'0065{raw:03d}.png').read_bytes()).hexdigest(),
            any_exposure_clip=low_clip or high_clip)
        records.append(r)
        validation.append(dict(raw=raw,role=role,pixels_unchanged=crop['pixels_unchanged'],
            mask_subset_of_core=bool(not (valid & ~core).any()), mask_wh=[1920,1280],
            photo_legal=r['photo_legal'],any_exposure_clip=r['any_exposure_clip'],mask_audit=maskaudit))
        im.close()
    centers = {r['raw']:np.array(r['camera_origin_world']) for r in records}
    pairs = [dict(a=x,b=y,baseline_m=float(np.linalg.norm(centers[x]-centers[y])))
             for i,x in enumerate(TRAIN) for y in TRAIN[i+1:]]
    query_distance = [dict(raw=q,nearest_train=min(TRAIN,key=lambda t:np.linalg.norm(centers[q]-centers[t])),
        nearest_train_baseline_m=min(float(np.linalg.norm(centers[q]-centers[t])) for t in TRAIN)) for q in QUERY]
    R=packet['source_tangent_frame_world'][:3,:3]
    xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.array(packet['core_size_m'])/2
    corners=packet['source_center_world']+np.c_[xy,np.zeros(4)]@R.T
    stats=dict(photo_legal_views=len(legal),photo_legal_raws=[r['raw'] for r in legal],
        native_short_axis_min_training_px=min(lookup[r]['core_min_caliper_native_px'] for r in TRAIN),
        train_count=len(TRAIN),max_train_baseline_m=max(p['baseline_m'] for p in pairs),
        min_train_pair_baseline_m=min(p['baseline_m'] for p in pairs),
        two_frame_baseline_m=float(np.linalg.norm(centers[CONTROL[0]]-centers[CONTROL[1]])),
        train_pairs=pairs,query_nearest_train=query_distance)
    plan=dict(schema='source_multiframe_photo_pool.v1', complete=True, frozen_after_native_visual_review=False,
        run_id='road_source_multiframe_20261122_r3',source_candidate='patch_02',
        pool_8_raws=TRAIN,two_frame_control_raws=CONTROL,train_raws=TRAIN,query_raws=QUERY,
        main_query_raws=[100,110,117],low_resolution_stress_query_raws=[86],
        selected_from_completed_199frame_photo_ranking=True,records=records,baseline_statistics=stats,
        source_asset='/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/appearance_fit/asset',
        source_packet='/root/autodl-tmp/drivingeditor_repro/campaign/road_gaussian_transfer_20261122_r2_isometric/source_selection/candidates/patch_02.npz',
        core_size_m=packet['core_size_m'].tolist(),source_center_world=packet['source_center_world'].tolist(),
        source_tangent_frame_world=packet['source_tangent_frame_world'].tolist(),world_core_corners=corners.tolist(),
        fixed_geometry_scale_opacity=True,original_source_RGB_initialization=True,source_parameter_updates_by_this_audit=0,
        camera_policy='Reuse r2 Brown-image to native-pinhole deterministic mapping; preserve original PNGs and native K/T/minus0.5 contract.',
        photo_qualification_policy='Core eroded2 pixels AND road semantic0 AND NOT full foreground guard AND NOT candidate-specific local photometric shadow proxy; distinct from old fresh-depth support.',
        sampling_policy_requested='Random sample reference frames from each assigned pool during the same 400 optimizer steps; root owns training contract.',
        exposure_policy='Shared between paired arms; calibration handled separately from legal road outside the source core.',
        display_policy='Pixel-exact core crops, pasted 1:1 without resampling; equal physical size does not imply equal display width.',
        source_selection_kind='Manual/hard-coded qualified photo pool, not a street-wide automatic source search.',
        QUERY_is_historical_BUILD=True,QUERY_is_not_independent_scene_test=True,
        native_render_or_optimizer_calls_by_audit=0,GPU_calls=0,new_diffusion_calls=0)
    write('pool_plan.json',plan)
    write('native_quality_audit.json',dict(complete=True,rows=validation,statistics=stats,GPU_calls=0,optimizer_updates=0))
    write('candidate_baselines.json',dict(complete=True,statistics=stats,
        all_legal_candidates=[{k:r[k] for k in ['raw','camera_origin_world','core_min_caliper_native_px','highpass_sigma1_std','photo_luma_mean','photo_legal','road_semantic_fraction','dynamic_clear_fraction','shadow_proxy_clear_fraction']} for r in legal]))
    contact_sheet([r for r in records if r['raw'] in TRAIN],'train8_native_1to1.png')
    contact_sheet([r for r in records if r['raw'] in QUERY],'query4_native_1to1.png')
    contact_sheet([r for r in records if r['raw'] in CONTROL],'control2_endpoints_native_1to1.png')
    print(json.dumps(dict(complete=True,train_raws=TRAIN,query_raws=QUERY,train_min_baseline_m=stats['min_train_pair_baseline_m'],
                         two_frame_baseline_m=stats['two_frame_baseline_m'],plan=str(OUT/'pool_plan.json'))))


if __name__=='__main__':
    main()
