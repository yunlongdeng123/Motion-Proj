import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,statistics
from PIL import Image
import numpy as np
H=pathlib.Path('/root/autodl-tmp/codex_tmp/v43_case15_isolation_20261101_r1');P=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027/motion_scene_inputs_v1');d=json.loads((H/'mask_proposals_r3_inherited/summary.json').read_text());ids={'waymo-train-212_a35':79,'waymo-train-212_a56':145,'waymo-train-564_a22':82};contexts={'waymo-train-212':'14705303724557273004_3105_000_3125_000','waymo-train-564':'473735159277431842_630_095_650_095'};summary=[]
for r in d['rows']:
 if r['case'] not in ids:continue
 ident=ids[r['case']];scene=r['case'].rsplit('_',1)[0];root=P/contexts[scene];out=H/'prepared_source_masks'/r['case'];(out/'body_masks').mkdir(parents=True,exist_ok=True);records=[]
 assert r['top1_instance_ID_frame_counts']=={str(ident):49}
 for f in r['frames']:
  src=root/'mask/mask_0/Annotations'/(f['frame_id']+'.png');labels=np.asarray(Image.open(src));mask=labels==ident;assert mask.any();dest=out/'body_masks'/(f['frame_id']+'.png');Image.fromarray((mask*255).astype(np.uint8)).save(dest);records.append({'raw':f['raw'],'frame_id':f['frame_id'],'source_prediction':str(src),'source_instance_id':ident,'output':str(dest),'pixels':int(mask.sum())})
 m={'case':r['case'],'track_id':r['track'],'source_context':contexts[scene],'schema':'provisional_associated_source_masks_v1','source_instance_id':ident,'frames':records,'frame_count':len(records),'nonzero_frames':sum(f['pixels']>0 for f in records),'median_pixels':statistics.median(f['pixels'] for f in records),'independent_pixel_GT':False,'used_to_update_G_or_train':False,'source_identity_association':'fixed pre-existing DEVA instance ID across49 source frames; reviewed contact by assistant; still provisional input','formal_admission':'not admitted; native contributor/occlusion/quality controls remain','human_verdict':None,'original_scene_files_modified':False}
 (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));summary.append({k:v for k,v in m.items() if k!='frames'});print(r['case'],ident,len(records),m['median_pixels'],flush=True)
(H/'prepared_source_masks/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
