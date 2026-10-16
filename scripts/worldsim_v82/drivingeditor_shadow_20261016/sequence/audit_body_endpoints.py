import os,json,sys,itertools
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1');NEW=ROOT/'shadow_sequence_20261016_v2';OLD=ROOT/'shadow_BC_20261016_v1'
sys.path.insert(0,str(OLD/'implementation_v1'))
from prepare_e0 import project
ev=ROOT/'rigid_post30k/scene0008_089_137/evaluation';frames=json.loads((ev/'scene_evaluation_manifest.json').read_text())['streams']['observed']['frames']
with np.load(OLD/'training_data_v2/training_data.npz') as f:poses=f['poses'];sizes=f['sizes_model'];valid=f['pose_valid']
rows=[];signs=np.array(list(itertools.product([-.5,.5],repeat=3)))
for i in [0,12,24,36,48]:
    row=frames[i]
    with np.load(ev/row['npz']) as f:gt=f['gt'];masks=f['observed_masks'].astype(bool)
    rgb=np.moveaxis(gt,0,-1).clip(0,1);overlay=rgb.copy()
    for k in range(3):overlay[masks[k]]=.4*overlay[masks[k]]+.6*np.eye(3)[k]
    image=Image.fromarray(np.rint(overlay*255).astype('uint8'));draw=ImageDraw.Draw(image);actors=[]
    for k in range(3):
        corners=(signs*sizes[i,k])@poses[i,k,:3,:3].T+poses[i,k,:3,3]
        uv,front=project(corners,row['camera']);center,fr=project(poses[i,k,:3,3][None],row['camera'])
        bounds=[float(uv[:,0].min()),float(uv[:,1].min()),float(uv[:,0].max()),float(uv[:,1].max())]
        actors.append({'actor':k,'valid':bool(valid[i,k]),'body_mask_pixels':int(masks[k].sum()),'center_in_front_of_camera':bool(fr[0]),'projected_center':center.tolist(),'box':bounds})
        if front.all():draw.rectangle(bounds,outline=['red','green','blue'][k],width=2)
    image.save(NEW/('body_audit_'+row['frame_id']+'.png'));rows.append({'frame':row['frame_id'],'actors':actors})
(NEW/'body_endpoint_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]))
