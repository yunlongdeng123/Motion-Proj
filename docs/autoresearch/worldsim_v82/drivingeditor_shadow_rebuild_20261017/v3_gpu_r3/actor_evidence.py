"""真实逐Actor相机/LiDAR证据准入；不使用渲染结果决定资格。"""
from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation
import sys
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(root/'implementation_r2/shadow_rebuild_v3'))
from shadow_rebuild.prepare import transform,project
data=root/'data_r2';m=json.loads((data/'manifest.json').read_text());scene=Path(m['source_scene'])
out=root/'actor_evidence_r1';out.mkdir(exist_ok=False)
origin=np.array(m['original_world_origin_m']);actors=m['actor_ids'];measure=[[] for _ in actors]
with np.load(data/'seed.npz') as f:counts=np.bincount(f['owner'][f['owner']>=0],minlength=len(actors))
best={};threshold=dict(min_body_pixels=256,min_max_body_pixels=400,min_visible_lidar_hits=8,min_unique_lidar_pixels=4,min_joint_frames=3,min_seed_gaussians=100)
for row in m['frames']:
 fid=row['frame_id'];sup=dict(np.load(data/row['supervision']));area=sup['masks'].sum((0,1))
 candidates=np.flatnonzero((area>=threshold['min_body_pixels'])&sup['mask_known'])
 annotation=json.loads((scene/'annotations'/(fid+'.json')).read_text());labels={l['id']:l for l in annotation['laser_labels']}
 if len(candidates):
  xyz=np.fromfile(scene/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[:,:3]
  ego=np.loadtxt(scene/'pose'/(fid+'.txt'));ego[:3,3]-=origin
  world=transform(xyz,ego);k=np.array(row['intrinsic']);cam=np.array(row['world_from_camera'])
  uv,z,inview=project(world,k,cam,sup['depth'].shape)
 for a,track in enumerate(actors):
  hits=0;unique=0;projected=np.empty((0,2),int)
  if a in candidates and track in labels:
   b=labels[track]['box'];center=np.array([b['center_x'],b['center_y'],b['center_z']]);c,s=np.cos(b['heading']),np.sin(b['heading'])
   rot=np.array([[c,-s,0],[s,c,0],[0,0,1]]);size=np.array([b['length'],b['width'],b['height']])
   inside=(np.abs((xyz-center)@rot)<=size/2+.03).all(-1)
   ids=np.flatnonzero(inside&inview)
   yy,xx=uv[ids,1],uv[ids,0]
   okay=(sup['depth'][yy,xx]>0)&(np.abs(sup['depth'][yy,xx]-z[ids])<.2)&sup['masks'][yy,xx,a]
   projected=uv[ids[okay]];hits=len(projected);unique=len(np.unique(projected,axis=0))
  qualified=bool(area[a]>=256 and hits>=8 and unique>=4 and sup['mask_known'][a])
  entry=dict(frame_id=fid,raw_index=row['source_index'],body_pixels=int(area[a]),mask_known=bool(sup['mask_known'][a]),
             visible_lidar_hits=hits,unique_lidar_pixels=unique,joint_evidence=qualified)
  measure[a].append(entry)
  if qualified and (a not in best or area[a]>best[a][0]):best[a]=(int(area[a]),row,projected,sup['masks'][...,a].copy())
rows=[]
for a,track in enumerate(actors):
 qualifying=[r for r in measure[a] if r['joint_evidence']]
 eligible=bool(len(qualifying)>=3 and max(r['body_pixels'] for r in measure[a])>=400 and counts[a]>=100)
 reasons=[]
 if len(qualifying)<3:reasons.append('fewer than 3 clearly visible camera+LiDAR frames')
 if max(r['body_pixels'] for r in measure[a])<400:reasons.append('camera footprint too small')
 if counts[a]<100:reasons.append('insufficient reconstructed seed geometry')
 preview=None
 if eligible:
  _,frame,points,mask=best[a];rgb=np.array(Image.open(data/frame['rgb']).convert('RGB'))
  edge=binary_dilation(mask)^mask;overlay=rgb.copy();overlay[edge]=[255,210,0]
  for x,y in points:overlay[max(0,y-1):y+2,max(0,x-1):x+2]=[0,240,255]
  y,x=np.nonzero(mask);lo=np.maximum([x.min()-12,y.min()-12],[0,0]);hi=np.minimum([x.max()+13,y.max()+13],[rgb.shape[1],rgb.shape[0]])
  left=Image.fromarray(rgb[lo[1]:hi[1],lo[0]:hi[0]]).resize((400,260))
  right=Image.fromarray(overlay[lo[1]:hi[1],lo[0]:hi[0]]).resize((400,260))
  panel=Image.new('RGB',(800,310),(18,27,40));panel.paste(left,(0,45));panel.paste(right,(400,45))
  draw=ImageDraw.Draw(panel);draw.text((8,7),f'a{a:02d} track={track} frame={frame["frame_id"]}',fill='white')
  draw.text((8,25),'real RGB crop                      body contour + measured visible LiDAR projection',fill='white')
  preview=f'a{a:02d}_{frame["frame_id"]}.png';panel.save(out/preview)
 rows.append(dict(actor_index=a,raw_track_id=track,eligible_delete=eligible,seed_gaussians=int(counts[a]),
       joint_evidence_frames=len(qualifying),max_body_pixels=max(r['body_pixels'] for r in measure[a]),
       max_visible_lidar_hits=max(r['visible_lidar_hits'] for r in measure[a]),preview=preview,excluded_reasons=reasons,frames=measure[a]))
report=dict(schema='camera_lidar_actor_evidence_v1',thresholds=threshold,actors=rows,
 eligible_actor_indices=[r['actor_index'] for r in rows if r['eligible_delete']],
 eligible_track_ids=[r['raw_track_id'] for r in rows if r['eligible_delete']],
 selection_uses_all49_camera_and_lidar_observations=True,not_independent_test_cohort=True,human_verdict=None)
(out/'actor_evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(eligible=[{k:r[k] for k in ['actor_index','raw_track_id','joint_evidence_frames','max_body_pixels','max_visible_lidar_hits','preview']} for r in rows if r['eligible_delete']],excluded=len(rows)-len(report['eligible_actor_indices']))))
