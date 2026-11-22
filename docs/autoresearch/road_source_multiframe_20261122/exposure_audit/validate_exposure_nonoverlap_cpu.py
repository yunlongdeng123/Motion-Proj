"""共享gain保持不变，额外验证不接触任何fit像素的外部对应。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json
import numpy as np
from PIL import Image


def safe(uv,fitmask):
    floor=np.floor(uv).astype(int);fraction=uv-floor;valid=np.ones(len(uv),bool)
    for dy in (0,1):
        for dx in (0,1):
            weight=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1])
            x=floor[:,0]+dx;y=floor[:,1]+dy
            inside=(x>=0)&(x<1920)&(y>=0)&(y<1280)
            hit=np.zeros(len(uv),bool);hit[inside]=fitmask[y[inside],x[inside]]
            valid &= (weight==0)|(inside&~hit)
    return valid


def main():
    out=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_source_multiframe_20261122_r3/exposure_audit')
    exposure=json.loads((out/'shared_exposure.json').read_text())
    with np.load(out/'external_correspondence_samples.npz',allow_pickle=False) as z:t={k:z[k] for k in z.files}
    source_path=Path(exposure['source_packet'])
    with np.load(source_path,allow_pickle=False) as s:F=s['source_tangent_frame_world'][:3,:3];O=s['source_center_world']
    anchor_fit=np.asarray(Image.open(out/'masks/115/fit_Brown_bilinear_support.png'))>0
    anchor_ids=t['raw_115_query_id'];anchor_uv=t['raw_115_Brown_uv'];anchor_rgb=t['raw_115_RGB']
    luma=np.array([.2126,.7152,.0722]);rows=[]
    for record in exposure['records']:
        raw=record['raw'];prefix=f'raw_{raw:03d}_';common=t[prefix+'shared_query_id'];ids=t[prefix+'query_id']
        fp=np.searchsorted(ids,common);ap=np.searchsorted(anchor_ids,common)
        worldpos=np.searchsorted(t['world_query_ids'],common);local=(t['world_xyz'][worldpos]-O)@F
        keys=np.floor(local[:,:2]/.08).astype(np.int64);unique,inverse=np.unique(keys,axis=0,return_inverse=True)
        validation=t[prefix+'validation_cells'];take=validation[inverse]
        fitmask=np.asarray(Image.open(out/f'masks/{raw:03d}/fit_Brown_bilinear_support.png'))>0
        take &= safe(t[prefix+'Brown_uv'][fp],fitmask)&safe(anchor_uv[ap],anchor_fit)
        f=[];a=[];cellkeys=[]
        for i,key in enumerate(unique):
            select=take&(inverse==i)
            if select.any():
                f.append(np.median(t[prefix+'RGB'][fp[select]],0));a.append(np.median(anchor_rgb[ap[select]],0));cellkeys.append(key.tolist())
        f=np.asarray(f);a=np.asarray(a);gain=record['gain']
        before=float(abs(f@luma-a@luma).mean());after=float(abs(gain*f@luma-a@luma).mean())
        rgb=float(abs(gain*f-a).mean())
        rows.append(dict(raw=raw,gain_unchanged=gain,strict_validation_world_cells=len(f),strict_matching_world_queries=int(take.sum()),
            validation_cell_keys=cellkeys,strict_validation_raw_luma_MAE=before,
            strict_validation_canonical_luma_MAE=after,strict_validation_canonical_RGB_MAE=rgb,
            minimum8cells_pass=len(f)>=8,residual_thresholds_pass=after<=.006 and rgb<=.012,
            no_bilinear_support_pixel_used_in_any_fit=True,source_and_anchor_both_checked=True))
    result=dict(schema='frozen_exposure_strict_pixel_disjoint_validation.v1',complete=True,all_gains_unchanged=True,
        source_core_samples_used=0,records=rows,all_strict_external_validation_passed=all(r['minimum8cells_pass'] and r['residual_thresholds_pass'] for r in rows),
        gain_fit_repeated=False,GPU_calls=0,optimizer_updates=0)
    (out/'strict_nonoverlap_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
