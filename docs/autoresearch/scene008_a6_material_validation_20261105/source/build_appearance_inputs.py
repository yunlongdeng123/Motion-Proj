"""Model-agnostic appearance inputs: true RGB and explicit unknown masks.

No hidden/HELD RGB or fitted M2 material is included in these conditioning
packages. Geometry/depth/silhouette are labelled fixed prior predictions.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign');OLD=CAM/'scene008_a6_asset_controls_20261105';RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(OLD/'source'))
from sh_asset import SHAsset
from scene_runtime import Runtime,raster,tagged
HELD=(89,95,101,105,113,121,125,130,137)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

@torch.no_grad()
def main():
    start=time.time();dest=RUN/'appearance_inputs';dest.mkdir(exist_ok=False)
    rt=Runtime();asset=SHAsset.from_geometry(OLD/'geometry_dense.npz').cuda()
    assert bool((asset.rgb==.5).all())
    with np.load(RUN/'data/sealed_targets.npz') as bank:
        # Only row identities and BUILD metadata; no sealed target RGB is read.
        raw_ids=bank['raw'];pixels=bank['pixels'];BUILD=bank['BUILD']
    rowmask={'full':np.ones(len(raw_ids),bool)}
    for case in ('left','right','top'):
        with np.load(RUN/'cases'/case/'split.npz') as split:retained=split['train_rows'].copy()
        mask=np.zeros(len(raw_ids),bool);mask[np.isin(raw_ids,BUILD)]=retained;rowmask[case]=mask
    records={case:[] for case in rowmask};total_known={case:0 for case in rowmask}
    for raw in BUILD:
        raw=int(raw);assert raw not in HELD
        d=rt.raw(raw);rows=tagged(asset.rows(d['pose'],d['cam'],rt.pipe,16),'asset')
        pkg=raster(d['cam'],rt.pipe,3,rows,rt.n.bg.new_zeros(3))
        alpha=pkg['alpha'][0].cpu().numpy();depth=(pkg['depth_sum'][0]/pkg['alpha'][0].clamp_min(1e-5)/rt.scale).cpu().numpy()
        yy,xx=np.where((alpha>.01)|d['a6_visible'].cpu().numpy().astype(bool))
        x0,x1=max(0,int(xx.min())-9),min(480,int(xx.max())+10);y0,y1=max(0,int(yy.min())-9),min(320,int(yy.max())+10)
        full_rgb=d['rgb'].permute(1,2,0).cpu().numpy()
        for case,allow in rowmask.items():
            folder=dest/case;folder.mkdir(exist_ok=True)
            mask=np.zeros((320,480),bool);select=(raw_ids==raw)&allow
            mask.ravel()[pixels[select]]=True
            rgb=np.zeros_like(full_rgb);rgb[mask]=full_rgb[mask]
            assert np.array_equal(rgb[mask],full_rgb[mask]) and not np.any(rgb[~mask])
            crop=np.s_[y0:y1,x0:x1];rgb_crop=rgb[crop];known=mask[crop]
            np.savez_compressed(folder/f'{raw:03d}.npz',known_RGB_srgb=rgb_crop,RGB_known_mask=known,
                prior_alpha=alpha[crop],prior_expected_depth_m=depth[crop],
                camera_K=d['obs']['K'] if 'K' in d['obs'] else np.asarray([[d['cam'].fx,0,d['cam'].cx],[0,d['cam'].fy,d['cam'].cy],[0,0,1]]),
                model_from_local=d['pose'].cpu().numpy(),model_to_camera=d['cam'].world_view_transform.T.cpu().numpy(),
                crop_xyxy=np.asarray([x0,y0,x1,y1]),raw=np.asarray(raw),source_pixel_ids=pixels[select])
            rgba=np.dstack((np.uint8(np.clip(rgb_crop,0,1)*255+.5),known.astype(np.uint8)*255))
            Image.fromarray(rgba,'RGBA').save(folder/f'{raw:03d}_preview.png')
            records[case].append(dict(raw=raw,known_pixels=int(known.sum()),npz=f'{case}/{raw:03d}.npz',sha256=sha(folder/f'{raw:03d}.npz')))
            total_known[case]+=int(known.sum())
        rt._data.clear();torch.cuda.empty_cache()
    geo=dict(np.load(OLD/'geometry_dense.npz'))
    # Explicit geometry allowlist: never export fitted RGB/canonical SH/prior colour.
    np.savez_compressed(dest/'geometry_only.npz',**{k:geo[k] for k in ('xyz','normal','sigma','opacity','thickness','box','normal_valid','sampling_parent_index','sampling_child_index')})
    expected={'full':27429}
    prep=json.loads((RUN/'cases/preparation.json').read_text())
    expected.update({r['case']:r['train_rows'] for r in prep['cases']});assert total_known==expected
    assert rt.modelunchanged() and asset.assert_geometry_unchanged()
    write(dest/'contract.json',dict(status='PASS_INPUT_INTERFACE_ONLY_NOT_A_COMPLETION_MODEL',views_per_case=40,raw_ids=BUILD.tolist(),
        source='Original reviewed a6 BUILD RGB only. H-contributing rows removed across all views for pseudo cases.',
        known_pixels=total_known,cases=records,geometry_sha256=sha(dest/'geometry_only.npz'),seconds=time.time()-start,
        units='Object-local metres; explicit model_from_local and model_to_camera; crop offsets refer to480x320 full image',
        RGB='NPZ float32 sRGB authoritative. RGBA PNG is a quantized preview; alpha0 means UNKNOWN, never black/background truth.',
        shape='prior_alpha and prior_expected_depth_m come from fixed PVD/M2 geometry. They are not observations or amodal ground truth.',
        camera='K stored in FULL480x320 coordinates; subtract crop x0,y0 from principal point when using crop pixels.',
        forbidden_inputs=['Any HELD RGB','Masked-region RGB','Full-fit M2 material','Native predictions outside real visible mask','A37/road colours'],
        hidden_colour_labels_exported=False,fitted_material_exported=False,model_training_steps=0))

if __name__=='__main__':main()
