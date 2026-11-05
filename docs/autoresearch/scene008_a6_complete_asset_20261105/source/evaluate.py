"""49 paired frames; one fixed asset; independent black and street evaluation."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from full_pvd import FixedAsset,RUN
from scene_runtime import Runtime,tagged,raster
from metrics import evaluate_frame,evaluate_pair,fixed_masks
from fit_asset import HELD,BUILD

def cpu(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
def image(x):
    a=cpu(x)
    if a.ndim==3 and a.shape[0] in (1,3):a=a.transpose(1,2,0)
    if a.ndim==3 and a.shape[2]==1:a=np.repeat(a,3,2)
    return Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))
def write(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def solo(rt,raw,asset):
    d=rt.raw(raw);rows=tagged(asset.rows(d['pose'],d['cam'],rt.pipe,1,precomputed=True),'asset')
    out=raster(d['cam'],rt.pipe,0,rows,rt.n.bg.new_zeros(3));out['component_alpha_T']=out['extra'][:4]
    return out
def strip(images,labels,crop=None,height=270):
    tiles=[]
    for im,label in zip(images,labels):
        im=image(im)
        if crop is not None:im=im.crop(crop)
        width=round(im.width*height/im.height);im=im.resize((width,height),Image.Resampling.NEAREST)
        tile=Image.new('RGB',(width,height+32),(242,246,249));tile.paste(im,(0,32))
        ImageDraw.Draw(tile).text((8,8),label,fill=(24,42,56));tiles.append(tile)
    result=Image.new('RGB',(sum(t.width for t in tiles),height+32))
    left=0
    for tile in tiles:result.paste(tile,(left,0));left+=tile.width
    return result

@torch.no_grad()
def main():
    start=time.time();dest=RUN/'evaluation';dest.mkdir(exist_ok=False)
    frames=dest/'frames';frames.mkdir();rt=Runtime()
    asset=FixedAsset.from_geometry(RUN/'appearance/complete_a6_fitted.npz').cuda()
    prior=FixedAsset.from_geometry(RUN/'appearance/complete_a6_prior.npz').cuda()
    assert prior.geometry_hash==asset.geometry_hash
    records=[];flags=[];conservation=0
    for raw in range(89,138):
        d=rt.raw(raw);native=rt.native_reference(raw);fit=solo(rt,raw,asset);before=solo(rt,raw,prior)
        bk=rt.render_scene(raw,baseline=True);bd=rt.render_scene(raw,baseline=True,delete=True)
        ak=rt.render_scene(raw,asset);ad=rt.render_scene(raw,asset,delete=True)
        record=dict(raw=raw,split='internal_HELD' if raw in HELD else 'BUILD',
            solo_native=evaluate_frame(d,native,native,rt.scale,state='solo'),
            solo_prior=evaluate_frame(d,before,native,rt.scale,state='solo'),
            solo_fitted=evaluate_frame(d,fit,native,rt.scale,state='solo'),
            scene=evaluate_pair(d,ak,ad,bk,bd,rt.scale),
            input_native_a6_rows=ak['meta']['native_a6_rows_input'],geometry_hash=asset.geometry_hash)
        records.append(record)
        for pkg in (native,fit,before,bk,bd,ak,ad):
            err=float((pkg['component_alpha_T'].sum(0)-pkg['alpha'][0]).abs().max());conservation=max(conservation,err)
        support=(fit['alpha'][0]>.01)|(native['alpha'][0]>.01)|d['a6_visible'].bool()
        yy,xx=torch.where(support);crop=(max(0,int(xx.min())-10),max(0,int(yy.min())-10),
              min(480,int(xx.max())+11),min(320,int(yy.max())+11))
        truth=d['rgb']*d['a6_visible'][None]
        strip([truth,native['rgb'],before['rgb'],fit['rgb'],fit['alpha'].expand(3,-1,-1)],
              ['Observed a6 RGB','Native a6 / black','Full PVD / prior RGB','Full PVD / fitted RGB','Fixed full-PVD alpha'],crop).save(frames/f'{raw:03d}_objects.jpg',quality=93)
        strip([bk['rgb'],ak['rgb'],bd['rgb'],ad['rgb']],
              ['Native / a37 present','Complete asset / present','Native / delete a37','Complete asset / delete a37'],None,height=320).save(frames/f'{raw:03d}_street.jpg',quality=92)
        strip([ak['component_alpha_T'][0:1].expand(3,-1,-1),ak['component_alpha_T'][1:2].expand(3,-1,-1),
               ad['component_alpha_T'][0:1].expand(3,-1,-1),ad['component_alpha_T'][3:4].expand(3,-1,-1)],
              ['Asset / a37 present','a37 contribution','Asset / delete a37','Legacy static / delete a37'],crop).save(frames/f'{raw:03d}_groups.jpg',quality=93)
        if raw in (105,113,130):
            image(fit['rgb']).crop(crop).resize((480,round((crop[3]-crop[1])*480/(crop[2]-crop[0]))),Image.Resampling.NEAREST).save(frames/f'{raw:03d}_asset_black.png')
        np.savez_compressed(frames/f'{raw:03d}_replay.npz',
            native_solo=cpu(native['rgb']).astype(np.float16),asset_solo=cpu(fit['rgb']).astype(np.float16),
            alpha=cpu(fit['alpha']).astype(np.float16),keep=cpu(ak['rgb']).astype(np.float16),delete=cpu(ad['rgb']).astype(np.float16),
            keep_groups=cpu(ak['component_alpha_T']).astype(np.float16),delete_groups=cpu(ad['component_alpha_T']).astype(np.float16),
            crop=np.array(crop),visible=cpu(d['a6_visible']),geometry_hash=np.asarray(asset.geometry_hash))
        write(dest/'progress.json',dict(completed=len(records),last_raw=raw))
        print(json.dumps(dict(stage='evaluate',raw=raw,completed=len(records))),flush=True)
        del d,native,fit,before,bk,bd,ak,ad;rt._data.clear();torch.cuda.empty_cache()
    assert rt.modelunchanged() and asset.assert_geometry_unchanged() and conservation<2e-5
    write(dest/'metrics.json',dict(rows=records,runtime=rt.meta,geometry_hash=asset.geometry_hash,
         geometry_unchanged=True,frozen_scene_unchanged=True,native_a6_rows_in_replacement=0,
         component_alpha_conservation_maxabs=conservation,seconds=time.time()-start,
         independent_generalization=False,all49_complete=True))
    # Return to a fresh asset loaded from disk after evaluation, not an in-memory fallback.
    fresh=FixedAsset.from_geometry(RUN/'appearance/complete_a6_fitted.npz')
    assert fresh.geometry_hash==asset.geometry_hash and np.array_equal(fresh.rgb.numpy(),asset.rgb.cpu().numpy())
    write(dest/'receipt.json',dict(status='PASS_ENGINEERING',frames=49,seconds=time.time()-start,
      geometry_unchanged=True,same_asset_keep_delete=True,independent_asset_reload=True,
      scientific_acceptance='See quality metrics and visual review; engineering PASS is not quality acceptance'))

if __name__=='__main__':main()
