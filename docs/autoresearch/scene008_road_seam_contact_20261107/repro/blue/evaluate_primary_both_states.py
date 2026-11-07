"""Read-only CPU addition of explicit KEEP/DELETE labels on the original sealed ROI."""
from pathlib import Path
import json,hashlib
import numpy as np
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_seam_contact_20261107_r1')
B=RUN/'blue';m=np.load(B/'fixed_masks_raw128.npz');obs=np.load(RUN/'diagnostic/layers_v1/128/observations.npz');target=obs['rgb'].transpose(2,0,1)
roots={'R0':RUN/'diagnostic/layers_v1','n2':RUN/'results/n2_full49','AA':RUN/'results/n2_mip_full49','n2_material':RUN/'results/n2_material_full49','AA_frozen_material':RUN/'results/n2_mip_material_pilot'}
regions={k:m[k] for k in m.files if m[k].dtype==bool};regions['global_protected']=obs['protect'].astype(bool)&~obs['visible'].any(0)
results=[]
for state in ('keep','delete'):
 for method,root in roots.items():
  path=root/'128'/(f'r0_{state}.npz' if method=='R0' else f'candidate_{state}.npz');d=np.load(path);resid=d['rgb']-target;r={}
  for key,mask in regions.items():
   a=abs(resid[:,mask]).mean(0);r[key]=dict(count_pixels=int(mask.sum()),abs_RGB_error_sum=float(a.sum()),abs_RGB_error_mean=float(a.mean()) if len(a) else None)
  edges={}
  for edge,ta in [('near_x',1),('far_x',1),('lateral_y',0)]:
   mm=m[f'band_{edge}_minus'];mp=m[f'band_{edge}_plus'];bins=np.floor(m['local_xyz'][...,ta]/.2).astype(np.int32);pairs=[]
   for bi in np.unique(bins[mm|mp]):
    mn=mm&(bins==bi);pl=mp&(bins==bi)
    if mn.sum()>=3 and pl.sum()>=3:
     jump=resid[:,pl].mean(1)-resid[:,mn].mean(1)
     pairs.append(dict(tangent_bin=int(bi),minus_count=int(mn.sum()),plus_count=int(pl.sum()),abs_channel_jump=float(abs(jump).mean())))
   edges[edge]=dict(band_minus_count=int(mm.sum()),band_plus_count=int(mp.sum()),paired_bins=len(pairs),matched_minus_count=sum(x['minus_count'] for x in pairs),matched_plus_count=sum(x['plus_count'] for x in pairs),equal_bin_abs_residual_jump_mean=float(np.mean([x['abs_channel_jump'] for x in pairs])) if pairs else None,bins=pairs)
  results.append(dict(raw=128,state=state,domain='PRIMARY_FIXED_APPROXIMATE_USER_BBOX',method=method,regions=r,edges=edges,source_path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
out=dict(schema='BLUE_RAW128_PRIMARY_FIXED_ROI_BOTH_STATES_V1',raw=128,bbox_xyxy_halfopen=[165,188,208,234],mask_sha256=hashlib.sha256((B/'fixed_masks_raw128.npz').read_bytes()).hexdigest(),selection='Exactly original pre-n2 frozen masks, no reselection or metric change.',rows=results)
(B/'raw128_fixed_roi_both_states.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps([{'method':r['method'],'state':r['state'],'road_MAE':r['regions']['original_road']['abs_RGB_error_mean'],'near_x':r['edges']['near_x']} for r in results]))
