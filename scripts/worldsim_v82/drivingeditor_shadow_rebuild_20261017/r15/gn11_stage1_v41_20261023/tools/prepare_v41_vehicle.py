"""CPU 派生车身保护域；仅缩小道路标签域，不增加几何/颜色先验。"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
from v41_dense_geometry import finalize_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    args = p.parse_args(); sys.path.insert(0, str(args.engine))
    from utils.v41_vehicle import load_vehicle_domains, VEHICLE_POLICY
    if args.output.exists(): raise FileExistsError('保护原证据，只写新目录')
    meta = json.loads((args.source/'manifest.json').read_text())
    if meta['version'] != 'V41_FIXED_DENSE_EVIDENCE_R2': raise ValueError('需要 R2 固定证据')
    bundle = json.loads(Path(meta['source_bundle']).read_text())
    if bundle['actors'] != VEHICLE_POLICY['actors']: raise ValueError('身份顺序错误')
    shutil.copytree(args.source, args.output)
    for row in meta['frames']:
        fid = row['frame']; path = args.output/'frames'/f'{fid}.npz'
        with np.load(path, allow_pickle=False) as z: data = {k:z[k].copy() for k in z.files}
        labels, bodies, edges = load_vehicle_domains(bundle['frames'][fid])
        vehicle = bodies.any(0) | edges.any(0)
        data['protect'] |= vehicle
        visible, windows = finalize_masks(data['visible'], data['source_count_capped_two'], data['protect'])
        data.update(visible=visible, ssim_valid=windows, vehicle_body=bodies, vehicle_edge=edges, actor_labels=labels)
        row.update(visible_pixels=int(visible.sum()), ssim_windows=int(windows.sum()),
            protect_pixels=int(data['protect'].sum()), vehicle_body_pixels=[int(m.sum()) for m in bodies],
            vehicle_edge_pixels=[int(m.sum()) for m in edges], vehicle_road_target_overlap=int((vehicle & visible).sum()))
        row.pop('fully_valid_ssim_windows', None)
        np.savez_compressed(path, **data)
        image = np.asarray(Image.open(bundle['frames'][fid]['original_rgb']).convert('RGB').resize((480, 320), Image.Resampling.LANCZOS)).astype(float)
        for mask, color, mix in ((data['protect'], [220, 50, 50], .25),
            (data['candidate'] & ~visible & ~data['protect'], [255, 180, 0], .5),
            (visible, [0, 255, 70], .6), (windows, [0, 220, 255], .7),
            (bodies.any(0), [255, 70, 180], .5), (edges.any(0), [255, 80, 80], .7)):
            image[mask] = image[mask]*(1-mix)+np.asarray(color)*mix
        Image.fromarray(np.uint8(image.clip(0, 255))).save(args.output/'overlays'/f'{fid}.png')
    meta.update(version='V41_FIXED_DENSE_EVIDENCE_R3', derived_from=str(args.source.resolve()))
    meta['policy']['vehicle'] = VEHICLE_POLICY
    meta['totals'].pop('fully_valid_ssim_windows', None)
    for key in ('visible_pixels', 'ssim_windows', 'protect_pixels', 'vehicle_road_target_overlap'):
        meta['totals'][key] = sum(r[key] for r in meta['frames'])
    meta['dense_domain_ready'] = all(r['visible_pixels'] > r['old_reliable_centers'] and r['ssim_windows'] > 0 and r['vehicle_road_target_overlap'] == 0 for r in meta['frames'])
    (args.output/'manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(meta['totals'], indent=2))


if __name__ == '__main__': main()
