"""颜色固定几何写回的逐像素松弛下界；不是可实现的三维方法。"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import json
import numpy as np

BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD=BASE/'scene008_road_seam_contact_20261107_r1'
RUN=BASE/'scene008_road_fusion_bake_20261107_r1'


def stat(x,m):
    x=np.asarray(x,dtype=np.float64)[m]
    return dict(count=int(x.size),mean=float(x.mean()) if x.size else None,
                sum=float(x.sum()),max=float(x.max()) if x.size else None)


def main():
    records=[]
    for raw in (105,113,128,130,137):
        with np.load(OLD/f'blue/fivekey_world_patch_masks_v1/{raw:03d}.npz') as z:
            masks={k:z[k].copy() for k in z.files if z[k].dtype==bool}
        if raw==128:
            with np.load(OLD/'blue/fixed_masks_raw128.npz') as z:
                masks['primary_original_road']=z['original_road'].copy()
        with np.load(OLD/f'diagnostic/layers_v1/{raw:03d}/observations.npz') as z:
            target=z['rgb'].transpose(2,0,1).astype(np.float64)
        for state in ('keep','delete'):
            with np.load(OLD/f'results/n2_mip_full49/{raw:03d}/candidate_{state}.npz') as z:
                rgb=z['rgb'].astype(np.float64)
                w=z['newroad_alpha_T'].astype(np.float64)
                fixed=rgb-z['newroad_rgb_T'].astype(np.float64)
            # All writable road colours in [0,1], fixed geometry and all other layers.
            # Allowing independent colours per ray is a relaxation of shared 3D.
            lo=fixed;hi=fixed+w[None]
            lower=np.maximum(lo-target,0)+np.maximum(target-hi,0)
            residual=abs(rgb-target).mean(0);floor=lower.mean(0)
            item=dict(raw=raw,state=state,regions={})
            for key,mask in masks.items():
                item['regions'][key]=dict(original_MAE=stat(residual,mask),
                    pixel_relaxed_min_MAE=stat(floor,mask),
                    pixels_outside_any_channel_interval=int((mask&(lower.max(0)>1e-6)).sum()),
                    pixels_newroad_mass_lt001=int((mask&(w<.01)).sum()),
                    pixels_newroad_mass_lt01=int((mask&(w<.1)).sum()),
                    newroad_mass=stat(w,mask),
                    scope='Same fixed masks; original visible-road RGB only. Not hidden truth, not a shared-asset reachable solution.')
            records.append(item)
    out=dict(schema='FIXED_GEOMETRY_NEWROAD_COLOUR_PIXEL_RELAXATION_V1',
        formula='F=baselineRGB-newroadRGBT; feasible pixel RGB interval=[F,F+Wnew]; min MAE is distance of observed RGB to interval.',
        use='Early capacity diagnostic. Zero bound does not establish shared 3D feasibility or seam success.',
        geometry_or_alpha_changed=False,forwards=0,backwards=0,records=records)
    folder=RUN/'root';folder.mkdir(exist_ok=True)
    (folder/'pixel_capacity_realRGB.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(raw=r['raw'],state=r['state'],world=r['regions']['world_patch_clean_road']) for r in records],ensure_ascii=False))


if __name__=='__main__':main()
