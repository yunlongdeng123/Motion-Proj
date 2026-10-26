"""CPU evidence for cross-request backend variation; no counterfactual accuracy GT."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from analyze_motion_suite import video


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args(); root = args.suite / 'continuous'; dest = root / 'fixed_backend_audit'
    original = json.loads((dest / 'summary.json').read_text())
    names = ['delete_original', 'insert_reused', 'move_inserted', 'turn_inserted', 'revisit_original_camera']
    paths = sorted((root / names[0] / 'native_rgb').glob('*.png')); assert len(paths) == 31
    records = []; pairwise_records = []; counts = []; native_errors = []
    for path in paths:
        common = np.ones((320, 480), bool); arrays = {}; images = {}
        for name in names:
            arrays[name] = np.load(root / name / 'float' / (path.stem + '.npz'))
            weights = arrays[name]['dynamic_weight'][0]
            common &= np.array(Image.open(root / name / 'mask' / path.name)) > 0
            common &= ~binary_dilation(weights > .001, iterations=3)
            images[name] = np.array(Image.open(root / name / 'propainter/model_png' / path.name).convert('RGB')) / 255.
        counts.append(int(common.sum()))
        for name in names[1:]:
            reference = arrays[names[0]]['rgb']; current = arrays[name]['rgb']
            delta = np.abs(current - reference).transpose(1, 2, 0)
            metric = float(delta[common].mean()) if common.any() else None
            native_errors.append(metric)
            video_delta = np.abs(images[name] - images[names[0]])
            records.append(dict(raw=int(path.stem), revision=name, common_pixels=int(common.sum()),
                                native_float_MAE=metric, generated_png_MAE=float(video_delta[common].mean()) if common.any() else None))
            pairwise = np.ones((320, 480), bool)
            for member in [names[0], name]:
                pairwise &= np.array(Image.open(root / member / 'mask' / path.name)) > 0
                pairwise &= ~binary_dilation(arrays[member]['dynamic_weight'][0] > .001, iterations=3)
            pairwise_records.append(dict(raw=int(path.stem), revision=name, common_pixels=int(pairwise.sum()),
                                         native_float_MAE=float(delta[pairwise].mean()) if pairwise.any() else None,
                                         generated_png_MAE=float(video_delta[pairwise].mean()) if pairwise.any() else None))
        if path == paths[3 * len(paths) // 4]:
            sheet = Image.new('RGB', (2400, 1050), '#142332'); draw = ImageDraw.Draw(sheet)
            edge = binary_dilation(common) & ~common
            for column, name in enumerate(names):
                native = np.array(Image.open(root / name / 'native_rgb' / path.name).convert('RGB'))
                generated = np.clip(images[name] * 255, 0, 255).astype('uint8')
                difference = np.abs(images[name] - images[names[0]]).mean(2)
                heat = np.zeros((320, 480, 3), dtype='uint8')
                heat[common, 0] = np.minimum(difference[common] * 5 * 255, 255).astype('uint8')
                heat[edge] = [255, 255, 0]
                for row, (label, pixels) in enumerate([('native3D', native), ('same ProPainter', generated), ('difference x5; yellow=common domain', heat)]):
                    x = column * 480; y = row * 350
                    sheet.paste(Image.fromarray(pixels), (x, y + 30))
                    draw.text((x + 5, y + 8), name + ' / ' + label + ' / raw' + path.stem, fill='white')
            sheet.save(dest / 'comparison.jpg', quality=95)
        for array in arrays.values(): array.close()
    for name in names: video(root / name / 'propainter/model_png', 31)
    per_revision = {}
    for name in names[1:]:
        rows = [r for r in records if r['revision'] == name and r['common_pixels']]
        denominator = sum(r['common_pixels'] for r in rows)
        per_revision[name] = dict(frame_pixel_observations=denominator,
                                  native_float_MAE_macro=float(np.mean([r['native_float_MAE'] for r in rows])),
                                  generated_png_MAE_macro=float(np.mean([r['generated_png_MAE'] for r in rows])),
                                  generated_png_MAE_pixel_weighted=sum(r['generated_png_MAE'] * r['common_pixels'] for r in rows) / denominator)
    result = dict(frames=31, revisions=5, fresh_backend_calls=3, ordinary_cache_reuses=2,
                  common_pixels_per_frame=counts, common_frame_pixel_observations=sum(counts),
                  common_pixels_min=min(counts), common_pixels_max=max(counts),
                  common_pixels_mean=float(np.mean(counts)), common_pixels_median=float(np.median(counts)),
                  common_domain_is_small_not_full_background=True,
                  native_float_change_MAE_macro=float(np.mean([v for v in native_errors if v is not None])),
                  generated_png_change_MAE_macro=original['generated_background_change_MAE_macro'], per_revision=per_revision, rows=records,
                  domain='intersection of five disocclusion masks, excluding dynamic support above .001 plus 3px dilation',
                  native_float_vs_generated_png_different_precision=True, same_input_rerun_variance_tested=False,
                  camera_pose_unchanged=True, cross_request_consistency_not_accuracy=True, counterfactual_RGB_GT=False,
                  repeated_cache_is_not_new_inference=True, quality_accepted=False, human_verdict=None)
    result['pairwise_domain_control'] = dict(
        domain='each revision intersected only with delete_original, same dynamic exclusion; denominators differ across revisions',
        rows=pairwise_records, per_revision={})
    for name in names[1:]:
        rows=[r for r in pairwise_records if r['revision']==name and r['common_pixels']]
        n=sum(r['common_pixels'] for r in rows)
        result['pairwise_domain_control']['per_revision'][name]=dict(
            frame_pixel_observations=n,
            native_float_MAE_pixel_weighted=sum(r['native_float_MAE']*r['common_pixels'] for r in rows)/n,
            generated_png_MAE_pixel_weighted=sum(r['generated_png_MAE']*r['common_pixels'] for r in rows)/n)
    (dest / 'cpu_delivery_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows', 'common_pixels_per_frame']}))


if __name__ == '__main__': main()
