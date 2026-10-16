from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=R/'ownership_controls_r12_sam3';sam=root/'shared_sam3_masks_r2';out=root/'input_visual_review';out.mkdir(exist_ok=True)
s=json.loads((sam/'summary.json').read_text());m=json.loads((R/'data_r2/manifest.json').read_text());targets=[5,6,12,18,25,26,37];counts={a:{} for a in targets};meta={r['frame_id']:r for r in s['frames']}
for rec in s['frames']:
 for x in rec['actors']:counts[x['actor']][x['used']]=counts[x['actor']].get(x['used'],0)+1
chosen={5:'0008107',6:'0008105',12:'0008137',18:'0008137',25:'0008137',26:'0008133',37:'0008121'}
tiles=[]
for actor,fid in list(chosen.items())+[(37,'0008089')]:
 row=next(r for r in m['frames'] if r['frame_id']==fid);image=np.asarray(Image.open(R/'data_r2'/row['rgb']).convert('RGB'));d=np.load(root/'raw_measurements'/(fid+'.npz'));p=np.load(sam/(fid+'.npz'));old=np.load(R/'data_r2'/row['supervision']);j=targets.index(actor);record=next(x for x in meta[fid]['actors'] if x['actor']==actor)
 ij=d['pixel'].astype(np.int64);ids=np.flatnonzero(d['BUILD']&d['in_image']);bins=ij[ids,1]*480+ij[ids,0];depth=np.full(320*480,np.inf);np.minimum.at(depth,bins,d['camera_z'][ids]);front=ids[d['camera_z'][ids]<=depth[bins]+.1];front=front[d['candidate_owner'][front]==actor]
 mask=p['masks'][...,actor];raw=p['SAM_prediction'][...,j];orig=old['masks'][...,actor]
 union=mask|raw|orig
 if not union.any():raise RuntimeError(('no crop evidence',actor,fid))
 yy,xx=np.where(union);x0=max(0,int(xx.min())-15);x1=min(480,int(xx.max())+16);y0=max(0,int(yy.min())-15);y1=min(320,int(yy.max())+16)
 panels=[]
 for mode,label in [('RGB','RGB'),('old','original DEVA'),('raw','native SAM3'),('used','shared input + LiDAR')]:
  rgb=image.copy()
  sel={'old':orig,'raw':raw,'used':mask}.get(mode)
  if sel is not None:
   rgb[sel]=(.6*rgb[sel]+.4*np.array([255,60,20])).astype(np.uint8);edge=sel&~binary_erosion(sel);rgb[edge]=[255,80,30]
  if mode=='used':
   for pt in ij[front]:rgb[pt[1],pt[0]]=[0,255,200]
  crop=Image.fromarray(rgb[y0:y1,x0:x1]);crop.thumbnail((235,160));panel=Image.new('RGB',(245,190),(20,30,45));panel.paste(crop,((245-crop.width)//2,24));ImageDraw.Draw(panel).text((5,5),label,fill=(255,255,255));panels.append(panel)
 tile=Image.new('RGB',(980,218),(20,30,45));draw=ImageDraw.Draw(tile);draw.text((8,4),f'a{actor} raw{int(fid)-8000} | {record["used"]} | same-frame BUILD returns={len(front)}',fill=(255,255,255))
 for k,panel in enumerate(panels):tile.paste(panel,(245*k,26))
 tile.save(out/f'a{actor:02d}_{fid}.png');tiles.append(tile)
canvas=Image.new('RGB',(980,len(tiles)*218),(20,30,45))
for i,tile in enumerate(tiles):canvas.paste(tile,(0,i*218))
canvas.save(out/'contact.png');summary=dict(model='SAM3',source_counts=counts,review_frames=chosen,extra_low_stability_old_case={'actor':37,'frame':'0008089'},LiDAR='BUILD-only actual frontmost candidate returns, candidate boxes not body truth',contour='real actual native/shared masks, no fabricated labels',status='REQUIRES_ROOT_VISUAL_REVIEW',CPU_finalize_seconds=s['seconds'],native_predictions_reused_without_GPU=True);(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
