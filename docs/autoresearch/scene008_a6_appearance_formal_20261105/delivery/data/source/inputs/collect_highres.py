"""BUILD-only true-HR frozen-M2 operators mapped to unchanged512 proposals.

No new proposal/canvas/crop, and no overwrite of the original480 operators.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.sparse import load_npz,save_npz
import torch

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD=CAM/'scene008_a6_asset_controls_20261105'
RUN=CAM/'scene008_a6_appearance_formal_20261105'
sys.path.insert(0,str(RUN/'source'))
from highres_fit import camera_hr,RAW as RAW_HR
from scene_runtime import Runtime,raster,tagged
from sh_asset import SHAsset
from linear_weights import design

RAWS=(106,114,129)
HELD=(89,95,101,105,113,121,125,130,137)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def rgb8(value):
    return np.round(np.clip(value,0,1)*255).astype(np.uint8)


def png(path,value):
    Image.fromarray(value).save(path)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raws',nargs='+',type=int,default=list(RAWS))
    args=parser.parse_args()
    assert all(raw in RAWS and raw not in HELD for raw in args.raws)
    start=time.time();root=RUN/'inputs_hr';root.mkdir(exist_ok=True)
    asset_path=OLD/'m2/complete_a6.npz'
    identity=sha(asset_path)
    asset=SHAsset.from_geometry(asset_path).cuda();assert len(asset.xyz)==18432
    rt=Runtime();records=[]
    with np.load(RUN/'inputs/column_support.npz') as saved:
        unknown_columns=saved['unknown_columns'].copy()
    immutable={str(p):sha(p) for raw in args.raws for p in (RUN/'inputs'/f'raw{raw:03d}').glob('*512.png')}
    for raw in args.raws:
        per=root/f'raw{raw:03d}';per.mkdir(exist_ok=False)
        lowroot=RUN/'inputs'/f'raw{raw:03d}'
        d=rt.raw(raw);c=camera_hr(d['cam'])
        assert (c.image_height,c.image_width)==(1280,1920)
        source_rgb=RAW_HR/'image/image_0'/f'0008{raw:03d}.png'
        real=np.array(Image.open(source_rgb).convert('RGB'))
        assert real.shape==(1280,1920,3)
        lowvisible=d['a6_visible'].cpu().numpy().astype(bool)
        visible=np.repeat(np.repeat(lowvisible,4,axis=0),4,axis=1)
        conservative=np.repeat(np.repeat(binary_erosion(lowvisible,iterations=1),4,axis=0),4,axis=1)
        rows=asset.rows(d['pose'],c,rt.pipe,1,precomputed=True)
        black=raster(c,rt.pipe,0,tagged(rows,'asset'),rt.n.bg.new_zeros(3))
        alpha=black['alpha'][0].cpu().numpy()
        premul=black['rgb'].permute(1,2,0).cpu().numpy()
        straight=premul/np.maximum(alpha[...,None],1e-6)
        support=alpha>.1
        unknown=support&~visible
        conservative_known=support&conservative
        candidate=rgb8(straight);candidate[~support]=0
        known_rgb=np.zeros_like(real);known_rgb[conservative_known]=real[conservative_known]
        png(per/'m2_target_hr.png',candidate)
        png(per/'known_rgb_hr.png',known_rgb)
        png(per/'known_mask_conservative_hr.png',conservative_known.astype(np.uint8)*255)
        png(per/'visible_mask_lifted_hr.png',visible.astype(np.uint8)*255)
        png(per/'object_mask_hr.png',support.astype(np.uint8)*255)
        png(per/'inpaint_mask_hr.png',unknown.astype(np.uint8)*255)
        shutil.copyfile(source_rgb,per/'real_rgb_hr.png')
        assert sha(per/'real_rgb_hr.png')==sha(source_rgb)
        # Compose EXACT requested pixel-center affine with the existing512 map.
        with np.load(lowroot/'transform.npz') as old:
            low_to_512=old['original_to_generated_pixel_center'].copy()
            crop=old['crop_xyxy'].copy()
        hr_to_low=np.array([[.25,0,-.375],[0,.25,-.375],[0,0,1]],np.float64)
        hr_to_512=low_to_512@hr_to_low
        out_to_hr=np.linalg.inv(hr_to_512)
        np.savez_compressed(per/'transform.npz',raw=np.int64(raw),
            highres_hw=np.array([1280,1920]),lowres_hw=np.array([320,480]),generated_hw=np.array([512,512]),
            original_low_crop_xyxy=crop,hr_to_low_pixel_center=hr_to_low,
            low_to_generated_pixel_center=low_to_512,
            highres_to_generated_pixel_center=hr_to_512,
            generated_to_highres_pixel_center=out_to_hr,
            original_transform_path=np.asarray(str(lowroot/'transform.npz')),
            original_transform_sha256=np.asarray(sha(lowroot/'transform.npz')),
            convention=np.asarray('zero-based pixel centers; low=(hr+.5)/4-.5; generated=oldaffine(low)'))
        lin=design(c,rt.pipe,rows,torch.as_tensor(support,device='cuda'))
        W=lin['matrix'];A=W.multiply(1/lin['alpha'][:,None]).tocsr();A.eliminate_zeros()
        pixels=lin['pixels'];xy=np.column_stack((pixels%1920,pixels//1920))
        xy512=(np.column_stack((xy,np.ones(len(xy))))@hr_to_512.T)[:,:2]
        assert np.array_equal(pixels,np.flatnonzero(support))
        assert A.shape==(int(support.sum()),18432)
        fit_alpha=lin['alpha']>.9
        known_samples=conservative_known.ravel()[pixels]
        visible_samples=visible.ravel()[pixels]
        # Record the existing512 mask sampled at nearest pixel centers too;
        # a fitter may use this for exact preservation of proposal known areas.
        ii=np.floor(xy512+.5).astype(np.int64)
        in_canvas=(ii[:,0]>=0)&(ii[:,0]<512)&(ii[:,1]>=0)&(ii[:,1]<512)
        assert in_canvas.all()
        old_known=np.asarray(Image.open(lowroot/'known_mask_512.png'))>0
        old_inpaint=np.asarray(Image.open(lowroot/'inpaint_mask_512.png'))>0
        generated_known=old_known[ii[:,1],ii[:,0]]
        generated_inpaint=old_inpaint[ii[:,1],ii[:,0]]
        row_error=float(np.max(np.abs(np.asarray(A.sum(1)).ravel()-1)))
        color_error=float(np.max(np.abs(A@asset.rgb.cpu().numpy()-straight.reshape(-1,3)[pixels])))
        assert row_error<5e-5 and color_error<5e-5
        save_npz(per/'design_A_normalized.npz',A)
        save_npz(per/'design_W_premultiplied.npz',W)
        np.savez_compressed(per/'operator_samples.npz',
            pixels=pixels,pixel_xy=xy,generated_xy=xy512,
            lowres_xy=(xy+.5)/4-.5,
            alpha=lin['alpha'],fit_alpha_gt_09=fit_alpha,
            known=known_samples,known_conservative=known_samples,
            visible_lifted=visible_samples,inpaint=~visible_samples,
            generated_known_nearest=generated_known,generated_inpaint_nearest=generated_inpaint,
            fit_generated=fit_alpha&generated_inpaint,
            fit_observed=fit_alpha&known_samples,
            real_rgb=real.reshape(-1,3)[pixels].astype(np.float32)/255,
            real_rgb_uint8=real.reshape(-1,3)[pixels],
            initial_m2_rgb=straight.reshape(-1,3)[pixels],
            unknown_column_fraction=np.asarray(A[:,unknown_columns].sum(1)).ravel(),
            column_unknown=unknown_columns,original_hw=np.array([1280,1920]),
            alpha_threshold=np.float64(.1),fit_alpha_threshold=np.float64(.9))
        np.savez_compressed(per/'raster_hr.npz',alpha=alpha,
            known_mask_conservative=conservative_known,visible_mask_lifted=visible,
            object_mask=support,inpaint_mask=unknown,
            model_from_local=d['pose'].cpu().numpy(),
            world_view_transform=c.world_view_transform.cpu().numpy(),
            full_proj_transform=c.full_proj_transform.cpu().numpy())
        roundtrip=float(np.max(np.abs(hr_to_512@out_to_hr-np.eye(3))))
        record=dict(raw=raw,split='BUILD',path=str(per),shape=list(A.shape),nnz=A.nnz,
            original_hw=[1280,1920],proposal_hw=[512,512],
            conservative_known_pixels=int(conservative_known.sum()),visible_lifted_pixels=int(visible.sum()),
            object_pixels=int(support.sum()),inpaint_pixels=int(unknown.sum()),
            fit_alpha_gt09_pixels=int(fit_alpha.sum()),fit_observed_pixels=int((fit_alpha&known_samples).sum()),
            fit_generated_pixels=int((fit_alpha&generated_inpaint).sum()),
            row_sum_maxabs=row_error,color_reconstruction_maxabs=color_error,
            alpha_reconstruction_maxabs=lin['alpha_maxabs'],native_rgb_maxabs=lin['colour_maxabs'],
            transform_roundtrip_maxabs=roundtrip,
            source_rgb=str(source_rgb),source_rgb_sha256=sha(source_rgb),
            same_original_proposal_input_sha256=sha(lowroot/'input_512.png'),
            same_original_proposal_mask_sha256=sha(lowroot/'inpaint_mask_512.png'),
            geometry_hash=asset.geometry_hash,design_sha256=sha(per/'design_A_normalized.npz'),
            samples_sha256=sha(per/'operator_samples.npz'),transform_sha256=sha(per/'transform.npz'))
        assert roundtrip<1e-10
        write(per/'receipt.json',record);records.append(record)
        print(json.dumps(dict(stage='HR_frame_complete',**record)),flush=True)
        del d,rows,black,lin,A,W
        rt._data.clear();torch.cuda.empty_cache()
    assert asset.assert_geometry_unchanged() and rt.modelunchanged()
    assert sha(asset_path)==identity
    assert all(sha(path)==old for path,old in immutable.items())
    write(root/'collection.json',dict(status='PASS',frames=records,source_raws=args.raws,
        HELD_used=False,geometry_updates=0,appearance_updates=0,neural_training_updates=0,
        new_proposals_created=0,original512_inputs_unchanged=True,original480_operators_preserved=True,
        geometry_hash=asset.geometry_hash,columns=18432,
        camera='Imported exact source/highres_fit.py camera_hr',
        full_operator_domain='Frozen complete M2 alpha>0.1, true1920x1280 raster',
        fitting_filter='fit_alpha_gt_09; fit_observed uses original visible erode1+nearest4 and actual HR RGB; fit_generated uses unchanged512 proposal inpaint mask',
        seconds=time.time()-start,peak_vram_bytes=int(torch.cuda.max_memory_allocated())))
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.npz','.png','.json') and p.name!='file_manifest.json']
    write(root/'file_manifest.json',dict(files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]))
    print(json.dumps(dict(stage='HR_complete',status='PASS',seconds=time.time()-start,frames=len(records))),flush=True)


if __name__=='__main__':
    main()
