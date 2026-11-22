"""Document observed native visual findings, validate delivered probes and masks."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    audit = json.loads((OUT / 'photo_audit.json').read_text())
    plan = json.loads((OUT / 'probe_plan.json').read_text())
    by = {(r['candidate_id'], r['raw']): r for r in audit['rows']}
    checks = []
    for r in plan['records']:
        mask_path = Path(r['core_photo_valid_mask_full1920'])
        core_path = Path(r['core_polygon_mask'])
        mask, core = np.asarray(Image.open(mask_path)), np.asarray(Image.open(core_path))
        assert mask.shape == core.shape == (1280, 1920)
        assert set(np.unique(mask)) <= {0, 255}
        assert not ((mask > 0) & ~(core > 0)).any()
        assert int((mask > 0).sum()) == r['core_photo_valid_native_pixels']
        for field, expected in [('camera_K_native', (3, 3)), ('world_to_camera', (4, 4))]:
            assert np.asarray(r[field]).shape == expected
        assert r['photo_legal']
        checks.append(dict(candidate_id=r['candidate_id'], raw=r['raw'], mask_pixels=int((mask > 0).sum()),
                           mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                           original_RGB=r['original_RGB'], bbox_xyxy=r['bbox_xyxy']))
    references = dict(
        complete=True, recommended_material_candidate='patch_02',
        preferred_photo_reference_raw=115, secondary_photo_reference_raw=117,
        supplementary_photo_reference_raw=116,
        reference_reason='原生像素中02可见的颗粒与粗纹理较00更明显；115/116/117曝光相近且核心均通过照片资格。115与116的清晰度差很小，117提供约9cm额外相机基线。',
        comparison_candidate='patch_00', comparison_reference_raws=[137, 134],
        comparison_reason='00提供较多原生像素，但核心物理面积为0.8m方形，不能以显示宽度或总像素数与02的0.6m方形直接比较；137/134仅约1.05cm相机基线。',
        visual_observations=[
            '100%原生裁剪：00主要是平滑底色、缓慢亮度变化和少量暗点；02的颗粒起伏较明显。',
            '两者都属于低对比沥青原照，细纹理的可辨识度有限。',
            '相同0.6m物理方形按原生像素不缩放展示，02@110的局部高频强度仍高于00@110。',
            '86图像较亮，110到115/117较稳定；未观察到核心过曝、黑位剪切或明显动态遮挡。',
            '00原始GS出现条状/结块纹理，02原始GS更趋平滑；这些是GS资产表现，不能用来否定原照片。',
        ],
        quantitative_same_physical_0p6m_at_raw110=dict(
            patch00=by[('patch_00', 110)]['same_physical_0p6m_metrics'],
            patch02=by[('patch_02', 110)]['same_physical_0p6m_metrics']),
        qualification=dict(
            photo_legal_is_not_fresh_depth_legal=True,
            raw116_photo_legal=True, raw116_old_fresh_certificate_supported=False,
            all_admitted_masks_core_only=True, halo_used_for_supervision=False,
            per_candidate_center_used_for_shadow_proxy=True,
            same_physical_area_is_not_same_display_width=True,
            source_geometry_appearance_fitting_performed=False),
        next_use='在相同110视角先比较固定几何颜色表达能力；之后02@115/117与00@137/134分别做候选参考视角控制。照片资格掩码仍须与独立native GS贡献掩码相交。',
        GPU_calls=0, optimizer_updates=0)
    write('reference_decision.json', references)
    write('validation.json', dict(complete=True, checks=checks, asserted_native_full_masks=True,
          asserted_mask_subset_of_core=True, asserted_photo_legal=True,
          asserted_pixel_exact_original_crops=True, selected_probe_records=len(checks), GPU_calls=0, optimizer_updates=0))
    # Context is also pasted/cropped 1:1. Core outlines identify what was audited;
    # neither context nor annotation overlays are admitted to measurements.
    frames = [110, 115, 117, 137]
    canvas = Image.new('RGB', (850, 250 * len(frames)), '#101317')
    draw = ImageDraw.Draw(canvas)
    for i, raw in enumerate(frames):
        box = [1100, 1040, 1920, 1260]
        im = Image.open(OUT / '_inputs' / f'{raw:03d}' / f'0065{raw:03d}.png').convert('RGB')
        context = im.crop(box)
        d = ImageDraw.Draw(context)
        for cid, color in [('patch_00', '#49d6a1'), ('patch_02', '#ffb86b')]:
            candidates = [r for r in plan['records'] if r['candidate_id'] == cid and r['raw'] == raw]
            if not candidates:
                continue
            uv = np.array(candidates[0]['core_native_uv']) - np.array(box[:2])
            d.line([tuple(x) for x in np.vstack([uv, uv[0]])], fill=color, width=1)
            d.text(tuple(uv.mean(0)), cid, fill=color)
        draw.text((10, i * 250 + 4), f'raw{raw:03d} context 820x220 native pixels | green00 orange02 core', fill='#e9edf2')
        canvas.paste(context, (10, i * 250 + 25))
    canvas.save(OUT / 'qualified_core_context_native_1to1.png')
    print(json.dumps(dict(complete=True, selected_probe_records=len(checks), recommended='patch_02 raw115',
                         secondary='raw117', decision=str(OUT / 'reference_decision.json'))))


if __name__ == '__main__':
    main()
