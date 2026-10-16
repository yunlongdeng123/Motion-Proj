import sys,json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
NEW=ROOT/'shadow_sequence_20261016_v2';OLD=ROOT/'shadow_BC_20261016_v1'
sys.path.insert(0,str(OLD/'implementation_v1'))
from prepare_e0 import sample
with np.load(OLD/'training_data_v2/training_data.npz') as f:
    uv=f['uv'];accepted=f['accepted_reference'];free=f['body_free'];fid=f['frame_ids'];ref=f['reference_linear']
with np.load(NEW/'surface_guarded_data/surface.npz') as f:guards=f['protection']
rows=[]
for i in range(49):
    ins=(uv[i,:,0]>=0)&(uv[i,:,0]<=478)&(uv[i,:,1]>=0)&(uv[i,:,1]<=318)
    guard=sample(guards[i,:,:,None].astype(float),uv[i])[0][:,0]>.01
    visible=accepted&ins&free[i]&~guard
    rgb=np.asarray(Image.open(NEW/'diagnostic49/retained_static'/(fid[i]+'.png'))).astype(float)/255
    pred=sample(rgb,uv[i])[0].clip(0,1);pred=np.where(pred<=.04045,pred/12.92,((pred+.055)/1.055)**2.4)
    rows.append({'frame':fid[i],'reference_points':int(accepted.sum()),'inside_image':int((accepted&ins).sum()),'inview_bodyfree_unselected_guardfree':int(visible.sum()),'retained_visible_MAE_8bit_audit_only':float(abs(pred[visible]-ref[visible]).mean())})
out={'rows':rows,'historical_all_reference_MAE_not_visibility_filtered':True,'corrected_metrics_added_in_full49_evaluation':True}
(NEW/'old_metric_visibility_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(rows[-1]))
