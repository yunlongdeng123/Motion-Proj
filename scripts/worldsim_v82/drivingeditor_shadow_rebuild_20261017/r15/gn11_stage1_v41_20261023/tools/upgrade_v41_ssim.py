"""复用已验收的固定像素证据，只在新目录升级 SSIM 统计域，不扩大 RGB 域。"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.ndimage import binary_erosion
from PIL import Image
from v41_dense_geometry import MASK_VERSION, MASK_POLICY, finalize_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('只写新目录')
    meta = json.loads((args.source/'manifest.json').read_text())
    if meta['version'] != 'V41_FIXED_DENSE_EVIDENCE_R1': raise ValueError('需要原始 R1 证据')
    bundle = json.loads(Path(meta['source_bundle']).read_text())
    shutil.copytree(args.source, args.output)
    for row in meta['frames']:
        fid = row['frame']
        path = args.output/'frames'/f'{fid}.npz'
        with np.load(path) as z: data = {k:z[k].copy() for k in z.files}
        visible, windows = finalize_masks(data['visible'], data['source_count_capped_two'], data['protect'])
        if not np.array_equal(visible, data['visible']): raise AssertionError('不得改变 RGB 域')
        row['fully_valid_ssim_windows'] = int(binary_erosion(visible, structure=np.ones((11, 11)), border_value=0).sum())
        row['ssim_windows'] = int(windows.sum()); data['ssim_valid'] = windows
        np.savez_compressed(path, **data)
        image = np.asarray(Image.open(bundle['frames'][fid]['original_rgb']).convert('RGB').resize((480, 320), Image.Resampling.LANCZOS)).astype(float)
        for mask, color, mix in ((data['protect'], [220, 50, 50], .25),
            (data['candidate'] & ~visible, [255, 180, 0], .5), (visible, [0, 255, 70], .6),
            (windows, [0, 220, 255], .7)):
            image[mask] = image[mask]*(1-mix)+np.asarray(color)*mix
        Image.fromarray(np.uint8(image.clip(0, 255))).save(args.output/'overlays'/f'{fid}.png')
    meta.update(version=MASK_VERSION, policy=MASK_POLICY, derived_from=str(args.source.resolve()),
        SSIM='Gaussian-weighted valid-pixel moments; >=80% valid pixels, no hidden values in moments')
    meta['totals']['ssim_windows'] = sum(r['ssim_windows'] for r in meta['frames'])
    meta['totals']['fully_valid_ssim_windows'] = sum(r['fully_valid_ssim_windows'] for r in meta['frames'])
    (args.output/'manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(meta['totals'], indent=2))


if __name__ == '__main__': main()
