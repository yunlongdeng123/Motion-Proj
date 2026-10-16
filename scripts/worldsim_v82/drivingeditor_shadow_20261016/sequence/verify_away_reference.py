"""Independent motion-based audit of the selected bright stable road reference."""
import os,json,warnings
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1');NEW=ROOT/'shadow_sequence_20261016_v2';OLD=ROOT/'shadow_BC_20261016_v1'
with np.load(OLD/'training_data_v2/training_data.npz') as f:poses=f['poses'];sizes=f['sizes_model'];valid=f['pose_valid'];scale=float(f['world_scale'])
with np.load(NEW/'surface_guarded_data/surface.npz') as f:d={k:f[k] for k in f.files}
dist=[]
for i in range(49):
    q=d['xyz'];local=np.einsum('kni,kij->knj',q[None]-poses[i,:,:3,3][:,None],poses[i,:,:3,:3])/scale
    distance=np.linalg.norm(np.maximum(abs(local[:,:,:2])-sizes[i,:,None,:2]/scale/2,0),axis=-1);distance[~valid[i]]=np.inf
    dist.append(distance.min(0))
away=d['free']&(np.array(dist)>4.)
with warnings.catch_warnings():
    warnings.simplefilter('ignore');ref=np.nanmedian(np.where(away[:,:,None],d['observed'],np.nan),axis=0)
count=away.sum(0);first=away.argmax(0);last=48-away[::-1].argmax(0)
checked=(count>=3)&(last-first>=6)&(d['confidence']>.9)
diff=abs(ref-d['reference']).mean(1)
consistent=checked&(diff<.02)
result={'scope':'motion-based cross check, not independent shadow GT','far_criterion_m_from_each_selected_body_box':4.,
    'current_interior_texels':int((d['confidence']>.9).sum()),'far_reference_confirmed_texels':int(checked.sum()),
    'far_reference_confirmed_fraction':float(checked.sum()/(d['confidence']>.9).sum()),
    'mean_linear_MAE_current_vs_far_median':float(diff[checked].mean()),
    'fraction_confirmed_with_MAE_under_002':float(consistent.sum()/checked.sum()),
    'near_actors_all_observed_times_not_certified_texels':int(((d['confidence']>.9)&~checked).sum()),
    'reference_differences_over_005_texels':int((checked&(diff>.05)).sum()),
    'global_constant_shadow_never_observed_clean_is_unidentifiable':True}
with np.load(OLD/'training_data_v2/training_data.npz') as f:oldxyz=f['xyz_model'];pixels=f['pixels'].astype(int);accepted=f['accepted_reference']
fp=np.asarray(Image.open(ROOT/'black_shadow_audit_20261016_v1/focus_cpu/reviewed_artifact_footprint.png'))>0
footprint=fp[pixels[:,1],pixels[:,0]]&accepted
xy=((oldxyz-d['origin'])@d['basis']/scale-d['lo'])/float(d['step']);xy=np.rint(xy).astype(int)
xy[:,0]=xy[:,0].clip(0,int(d['shape'][0])-1);xy[:,1]=xy[:,1].clip(0,int(d['shape'][1])-1)
index=xy[:,1]*int(d['shape'][0])+xy[:,0]
result.update(original_black_footprint_points=int(footprint.sum()),black_footprint_nearest_atlas_far_checked_fraction=float(checked[index[footprint]].mean()),
    black_footprint_nearest_atlas_far_consistent_fraction=float(consistent[index[footprint]].mean()))
visual=np.zeros((len(checked),3),dtype=np.uint8);visual[d['confidence']>.9]=[202,151,56];visual[consistent]=[50,166,99]
Image.fromarray(visual.reshape(int(d['shape'][1]),int(d['shape'][0]),3)).save(NEW/'away_reference_coverage.png')
np.savez_compressed(NEW/'away_reference_audit.npz',far_reference=np.nan_to_num(ref,nan=0).astype('float32'),far_checked=checked,far_consistent=consistent,far_count=count)
(NEW/'away_reference_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
