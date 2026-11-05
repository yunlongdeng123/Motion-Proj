"""Frozen-M2 BUILD-only input rasters and normalized all-object RGB operators.

No optimization, native-a6 appearance render, geometry update, or scene edit.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.sparse import load_npz, save_npz
import torch

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
CONTROLS = CAMPAIGN/'scene008_a6_asset_controls_20261105'
RUN = CAMPAIGN/'scene008_a6_appearance_formal_20261105'
sys.path.insert(0, str(CONTROLS/'source'))
from sh_asset import SHAsset
from scene_runtime import Runtime, raster, tagged
from linear_weights import design

RAWS = (106, 114, 129)
HELD = (89, 95, 101, 105, 113, 121, 125, 130, 137)
SIZE = 512
THRESHOLD = .1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def rgb8(value):
    return np.round(np.clip(value, 0, 1)*255).astype(np.uint8)


def png(path, value):
    Image.fromarray(value).save(path)


def square_transform(support):
    ys, xs = np.where(support)
    assert len(xs), 'Empty full-object alpha support'
    bbox = np.array([xs.min(), ys.min(), xs.max()+1, ys.max()+1], np.int64)
    bw, bh = bbox[2:]-bbox[:2]
    margin = max(8, int(np.ceil(max(bw, bh)*.16)))
    side = int(max(bw, bh)+2*margin)
    # Integral crop, potentially extending beyond the source image; pad is black.
    left = int(np.floor((bbox[0]+bbox[2]-side)/2))
    top = int(np.floor((bbox[1]+bbox[3]-side)/2))
    box = np.array([left, top, left+side, top+side], np.int64)
    scale = SIZE/side
    # Pixel-center affine: x512=(xraw-left+.5)*scale-.5.
    forward = np.array([[scale, 0, (.5-left)*scale-.5],
                        [0, scale, (.5-top)*scale-.5], [0, 0, 1]], np.float64)
    return bbox, box, forward, np.linalg.inv(forward)


def transform(value, box, nearest=False):
    im = Image.fromarray(value)
    im = im.crop(tuple(int(v) for v in box))
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.array(im.resize((SIZE, SIZE), resample=resample), copy=True)


def save_transform(dest, raw, original_shape, bbox, box, forward, inverse):
    h, w = original_shape
    ys, xs = np.mgrid[:h, :w]
    homogeneous = np.stack((xs, ys, np.ones_like(xs)), -1)
    original_to_generated = homogeneous@forward.T
    gy, gx = np.mgrid[:SIZE, :SIZE]
    generated_to_original = np.stack((gx, gy, np.ones_like(gx)), -1)@inverse.T
    crop_h = crop_w = int(box[2]-box[0])
    np.savez_compressed(dest/'transform.npz',
        raw=np.int64(raw), original_hw=np.array([h, w]), generated_hw=np.array([SIZE, SIZE]),
        object_bbox_xyxy=bbox, crop_xyxy=box, crop_hw=np.array([crop_h, crop_w]),
        pad_ltrb=np.array([max(0,-box[0]), max(0,-box[1]), max(0,box[2]-w), max(0,box[3]-h)]),
        original_to_generated_pixel_center=forward,
        generated_to_original_pixel_center=inverse,
        original_pixel_to_generated_xy=original_to_generated[...,:2],
        generated_pixel_to_original_xy=generated_to_original[...,:2],
        rgb_interpolation=np.asarray('Pillow bilinear'),
        mask_interpolation=np.asarray('Pillow nearest; masks derived before resize'),
        coordinate_definition=np.asarray('zero-based integer pixel centers; homogeneous 3x3 affines'))
    points = np.array([[0,0,1],[w-1,h-1,1],[w/2,h/2,1]], np.float64)
    assert np.max(np.abs(points@forward.T@inverse.T-points)) < 1e-10


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raws', type=int, nargs='+', default=list(RAWS))
    args = parser.parse_args()
    assert all(raw in RAWS and raw not in HELD for raw in args.raws)
    start = time.time()
    dest = RUN/'inputs'
    dest.mkdir(parents=True, exist_ok=True)
    source_asset = CONTROLS/'m2/complete_a6.npz'
    source_design = CONTROLS/'m2/design.npz'
    old_A = load_npz(source_design)
    mass = np.asarray(old_A.sum(0)).ravel()
    unknown_columns = mass == 0
    assert old_A.shape[1] == 18432 and bool((old_A.data >= 0).all())
    asset = SHAsset.from_geometry(source_asset).cuda()
    original_asset_hash = sha(source_asset)
    assert len(asset.xyz) == 18432
    np.savez_compressed(dest/'column_support.npz',
        contribution=mass, unknown_columns=unknown_columns,
        known_columns=~unknown_columns,
        initial_m2_rgb=asset.rgb.cpu().numpy(),
        geometry_hash=np.asarray(asset.geometry_hash),
        source_asset=np.asarray(str(source_asset)), source_asset_sha256=np.asarray(original_asset_hash),
        source_design=np.asarray(str(source_design)), source_design_sha256=np.asarray(sha(source_design)),
        rule=np.asarray('unknown iff controls/m2/design.npz column total contribution == 0 exactly'))
    rt = Runtime()
    records = []
    for raw in args.raws:
        per = dest/f'raw{raw:03d}'
        per.mkdir(exist_ok=True)
        d = rt.raw(raw)
        rows = asset.rows(d['pose'], d['cam'], rt.pipe, 1, precomputed=True)
        black = raster(d['cam'], rt.pipe, 0, tagged(rows, 'asset'), rt.n.bg.new_zeros(3))
        alpha = black['alpha'][0].cpu().numpy()
        m2_premul = black['rgb'].permute(1,2,0).cpu().numpy()
        m2_straight = m2_premul/np.maximum(alpha[...,None], 1e-6)
        rgb = d['rgb'].permute(1,2,0).cpu().numpy()
        h, w = alpha.shape
        known = d['a6_visible'].cpu().numpy().astype(bool)
        support = alpha > THRESHOLD
        unknown = support & ~known
        assert known.shape == support.shape == (320,480)
        target_candidate = rgb8(m2_straight)
        target_candidate[~support] = 0
        grey_prior = np.zeros_like(target_candidate)
        grey_prior[support] = 128
        real_rgb_source = np.asarray(d['obs']['rgb'])
        source_normalized = real_rgb_source.astype(np.float32)
        if float(source_normalized.max()) > 1:
            source_normalized = source_normalized/255.
        assert np.array_equal(source_normalized, rgb), 'Observation RGB normalization changed'
        real_rgb8 = rgb8(source_normalized)
        grey_prior[known] = real_rgb8[known]
        known_rgb = np.zeros_like(real_rgb8)
        known_rgb[known] = real_rgb8[known]
        candidate_with_known = target_candidate.copy()
        candidate_with_known[known] = real_rgb8[known]
        bbox, box, forward, inverse = square_transform(support | known)
        save_transform(per, raw, (h,w), bbox, box, forward, inverse)
        png(per/'real_rgb_original.png', real_rgb8)
        png(per/'known_rgb_original.png', known_rgb)
        png(per/'known_mask_original.png', known.astype(np.uint8)*255)
        png(per/'object_mask_original.png', support.astype(np.uint8)*255)
        png(per/'inpaint_mask_original.png', unknown.astype(np.uint8)*255)
        png(per/'gray_prior_original.png', grey_prior)
        png(per/'m2_target_original.png', target_candidate)
        png(per/'m2_candidate_with_known_original.png', candidate_with_known)
        png(per/'m2_premultiplied_original.png', rgb8(m2_premul))
        # Resize RGB, then enforce the independently nearest-resized exact masks.
        known512 = transform(known.astype(np.uint8)*255, box, True) > 0
        object512 = transform(support.astype(np.uint8)*255, box, True) > 0
        unknown512 = object512 & ~known512
        real512 = transform(real_rgb8, box)
        gray512 = np.zeros((SIZE,SIZE,3), np.uint8)
        gray512[object512] = 128
        gray512[known512] = real512[known512]
        m2512 = transform(target_candidate, box)
        m2512[~object512] = 0
        candidate512 = m2512.copy()
        candidate512[known512] = real512[known512]
        known_rgb512 = np.zeros_like(gray512)
        known_rgb512[known512] = real512[known512]
        png(per/'input_512.png', gray512)
        png(per/'known_rgb_512.png', known_rgb512)
        png(per/'known_mask_512.png', known512.astype(np.uint8)*255)
        png(per/'object_mask_512.png', object512.astype(np.uint8)*255)
        png(per/'inpaint_mask_512.png', unknown512.astype(np.uint8)*255)
        png(per/'m2_target_512.png', m2512)
        png(per/'m2_candidate_with_known_512.png', candidate512)
        png(per/'real_rgb_crop_512.png', real512)
        np.savez_compressed(per/'raster_original.npz',
            raw=np.int64(raw), alpha=alpha, m2_rgb_unpremultiplied=m2_straight,
            m2_rgb_premultiplied=m2_premul, real_rgb_uint8=real_rgb8,
            real_rgb_source_exact=real_rgb_source, real_rgb_normalized_exact=rgb,
            known_mask=known, object_mask=support, inpaint_mask=unknown,
            model_from_local=d['pose'].cpu().numpy(),
            world_view_transform=d['cam'].world_view_transform.cpu().numpy(),
            full_proj_transform=d['cam'].full_proj_transform.cpu().numpy())
        print(json.dumps(dict(stage='images_ready', raw=raw, path=str(per),
                             known_pixels=int(known.sum()), unknown_pixels=int(unknown.sum()))), flush=True)
        lin = design(d['cam'], rt.pipe, rows, torch.as_tensor(support, device='cuda'))
        W = lin['matrix']
        A = W.multiply(1/np.maximum(lin['alpha'][:,None],1e-6)).tocsr()
        A.eliminate_zeros()
        assert A.shape == (int(support.sum()),18432)
        assert np.array_equal(lin['pixels'], np.flatnonzero(support))
        conservation = float(np.max(np.abs(np.asarray(A.sum(1)).ravel()-1)))
        colour_error = float(np.max(np.abs(A@asset.rgb.cpu().numpy()-m2_straight.reshape(-1,3)[lin['pixels']])))
        assert conservation < 5e-5 and colour_error < 5e-5
        save_npz(per/'design_A_normalized.npz', A)
        save_npz(per/'design_W_premultiplied.npz', W)
        pixel = lin['pixels']
        np.savez_compressed(per/'operator_samples.npz',
            pixels=pixel, pixel_xy=np.column_stack((pixel%w,pixel//w)),
            alpha=lin['alpha'], original_hw=np.array([h,w]),
            known=known.ravel()[pixel], inpaint=unknown.ravel()[pixel],
            real_rgb=rgb.reshape(-1,3)[pixel],
            initial_m2_rgb=m2_straight.reshape(-1,3)[pixel],
            unknown_column_fraction=np.asarray(A[:,unknown_columns].sum(1)).ravel(),
            column_unknown=unknown_columns, alpha_threshold=np.float64(THRESHOLD))
        record = dict(raw=raw, split='BUILD', path=str(per), original_hw=[h,w], generated_hw=[SIZE,SIZE],
            object_pixels=int(support.sum()), known_pixels=int(known.sum()),
            inpaint_pixels=int(unknown.sum()), known512_pixels=int(known512.sum()),
            inpaint512_pixels=int(unknown512.sum()), crop_xyxy=box.tolist(),
            shape=list(A.shape), nnz=int(A.nnz), row_sum_maxabs=conservation,
            color_reconstruction_maxabs=colour_error,
            native_alpha_maxabs=lin['alpha_maxabs'], native_rgb_maxabs=lin['colour_maxabs'],
            design_sha256=sha(per/'design_A_normalized.npz'), transform_sha256=sha(per/'transform.npz'),
            raw_observation_sha256=sha(CAMPAIGN/'scene008_instances_20261105_r1/observations/frames'/f'{raw:03d}.npz'))
        write(per/'receipt.json', record)
        records.append(record)
        print(json.dumps(dict(stage='frame_complete', **record)), flush=True)
        del d, rows, black, lin
        rt._data.clear()
        torch.cuda.empty_cache()
    assert asset.assert_geometry_unchanged() and rt.modelunchanged()
    assert sha(source_asset) == original_asset_hash
    write(dest/'collection.json', dict(status='PASS', requested_raws=args.raws, frames=records,
        source_raws_used=args.raws, HELD_used=False, native_a6_appearance_rendered=False,
        geometry_updates=0, appearance_updates=0, scene_updates=0,
        geometry_hash=asset.geometry_hash, columns=18432, unknown_columns=int(unknown_columns.sum()),
        known_columns=int((~unknown_columns).sum()),
        source_asset=str(source_asset), source_asset_sha256=original_asset_hash,
        unknown_rgb='Gray128 placeholder in input_512.png; generated RGB required; no hidden truth implied',
        inpaint_rule='full frozen M2 alpha>0.1 AND NOT BUILD observation a6_visible',
        known_rule='Exact original BUILD observation a6_visible; never native-a6 predicted RGB',
        black_background='Outside union(object,known) is black and not an inpaint target',
        fit_operator='A=W/alpha, native alpha*T rows, original camera pixels',
        transform='Exact integral crop plus black padding and512resize; pixel-center affine in transform.npz',
        seconds=time.time()-start, peak_vram_bytes=int(torch.cuda.max_memory_allocated())))
    print(json.dumps(dict(stage='complete', status='PASS', frames=len(records), seconds=time.time()-start)), flush=True)


if __name__ == '__main__':
    main()
