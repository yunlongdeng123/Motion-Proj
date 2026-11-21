"""由块归属和外边界自动定义窄接缝带，无场景/帧号分支。"""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

def seam_rows(root):
    root=Path(root);out=root/'qualification/seam_rows.npz'
    if out.exists():
        with np.load(out) as z:return {k:z[k] for k in z.files}
    with np.load(root/'proposal/ownership.npz') as z:q={k:z[k] for k in z.files}
    with np.load(root/'proposal/transferred_rows.npz') as z:a={k:z[k] for k in z.files}
    keys=q['cell_keys'];lo=keys.min(0);hi=keys.max(0)+1
    labels=np.full(tuple((hi-lo)[::-1]),-1,np.int64);ij=keys-lo
    labels[ij[:,1],ij[:,0]]=q['cell_patch_id']
    boundary=np.zeros(labels.shape,bool)
    different=(labels[:,1:]!=labels[:,:-1])&(labels[:,1:]>=0)&(labels[:,:-1]>=0)
    boundary[:,1:]|=different;boundary[:,:-1]|=different
    different=(labels[1:]!=labels[:-1])&(labels[1:]>=0)&(labels[:-1]>=0)
    boundary[1:]|=different;boundary[:-1]|=different
    size=np.asarray(q['cell_size']);distance=ndi.distance_transform_edt(~boundary,sampling=size[::-1])
    cell=np.floor((a['target_uv']-q['grid_origin'])/size).astype(np.int64)-lo
    x=np.clip(cell[:,0],0,labels.shape[1]-1);y=np.clip(cell[:,1],0,labels.shape[0]-1)
    width=max(.04,3*float(np.median(a['gaussian_sigma_m'][:,:2])))
    outer=a['ownership_weight']<1-1e-6;mask=outer|(distance[y,x]<=width)
    result=dict(new_seam_mask=mask,outer_transition=outer,patch_seam_distance_m=distance[y,x].astype(np.float32),
        patch_seam_width_m=np.asarray(width))
    out.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(out,**result);return result
