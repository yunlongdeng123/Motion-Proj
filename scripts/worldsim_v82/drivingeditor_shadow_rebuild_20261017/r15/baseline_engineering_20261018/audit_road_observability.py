"""Read-only CPU audit of actual motion, road exposure opportunities and LiDAR support.

Ground queries are a measured-plane hypothesis, not hidden-road ground truth.
All frames are projected in world coordinates; no model edits or training.
"""
from pathlib import Path
import os,json,itertools,datetime
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
from PIL import Image,ImageDraw,ImageFont
E=Path(__file__).resolve().parent;ROOT=E.parent;SID='scene0008_089_137';SRC=ROOT/'inputs'/SID
OUT=E/'road_observability_20261019';A=E/'full5_pipeline/inputs'/SID;B=E/'all_vehicle_pipeline/inputs'/SID
IDS=[5,26,18,6,37];RAW=list(range(89,138));W,H=480,320
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def transform(x,m):return x@m[:3,:3].T+m[:3,3]
def box_data(label):
 b=label['box'];h=b['heading'];r=np.array([[np.cos(h),-np.sin(h),0],[np.sin(h),np.cos(h),0],[0,0,1.]])
 return np.array([b['center_x'],b['center_y'],b['center_z']]),np.array([b['length'],b['width'],b['height']]),r
def project(x,frame):
 v=transform(x,np.linalg.inv(frame['pose']));c=transform(v,frame['tr']);p=c@frame['K'].T
 uv=p[:,:2]/np.maximum(p[:,2:3],1e-9);pix=np.floor(uv+.5).astype(np.int64)
 inside=(c[:,2]>.1)&(pix[:,0]>=0)&(pix[:,0]<W)&(pix[:,1]>=0)&(pix[:,1]<H)
 return v,c,uv,pix,inside
def blocked_by_boxes(v,frame):
 # Segment from the actual camera centre to the hypothesized road query.
 origin=frame['camera_origin_vehicle'];directions=v-origin;blocked=np.zeros(len(v),bool)
 for label in frame['labels']:
  center,size,r=box_data(label);o=(origin-center)@r;d=directions@r
  safe=np.where(np.abs(d)<1e-12,1e-12,d);t0=(-size/2-o)/safe;t1=(size/2-o)/safe
  lo=np.minimum(t0,t1).max(1);hi=np.maximum(t0,t1).min(1)
  blocked|=(hi>=np.maximum(lo,0))&(lo<.995)&(hi>0)
 return blocked
def summarize(counts):
 return dict(points=len(counts),never_exposed_fraction=float(np.mean(counts==0)),at_least_3_frames_fraction=float(np.mean(counts>=3)),median_exposed_frames=float(np.median(counts)),mean_exposed_frames=float(np.mean(counts)),max_exposed_frames=int(counts.max()))
def main():
 assert not OUT.exists(),'audit output exists; preserve and inspect instead of overwriting'
 OUT.mkdir();(OUT/'fullframes').mkdir()
 old=json.loads((ROOT/'shadow_rebuild_20261017_v3/backend_restore_r15/data/manifest.json').read_text());tracks={a:old['actor_ids'][a] for a in IDS}
 readiness=json.loads((E/'full5_pipeline/metadata'/SID/'CPU_readiness.json').read_text());plane=np.array(readiness['SGCM_geometry']['ground_fit']['plane_anchor'])
 frames=[];ground=[];ground_frame=[];anchor=np.loadtxt(SRC/'pose/0008089.txt');anchor_inv=np.linalg.inv(anchor)
 for index,raw in enumerate(RAW):
  fid=f'0008{raw:03d}';ann=json.loads((SRC/'annotations'/f'{fid}.json').read_text());pose=np.loadtxt(SRC/'pose'/f'{fid}.txt')
  cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (SRC/'calib'/f'{fid}.txt').read_text().splitlines() if ':' in s}
  tr=np.eye(4);tr[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);K=cal['P0'].reshape(3,4)[:,:3]*.25;K[2,2]=1
  maskA=np.asarray(Image.open(A/'matched_training_labels'/f'{fid}.png'))>0
  maskB=np.asarray(Image.open(B/'dynamic_masks/dynamic_mask_0'/f'{fid}.png').resize((W,H),Image.Resampling.NEAREST))>0
  proposal=np.load(E/'all_vehicle_native49/native_predictions'/f'{fid}.npz',allow_pickle=False)['masks'].any(0)
  proposal=np.asarray(Image.fromarray(proposal).resize((W,H),Image.Resampling.NEAREST))>0
  labels=[l for l in ann['laser_labels'] if l['type']=='TYPE_VEHICLE']
  frame=dict(raw=raw,fid=fid,pose=pose,K=K,tr=tr,camera_origin_vehicle=np.linalg.inv(tr)[:3,3],timestamp=ann['timestamp_micros'],labels=labels,selected={a:next((l for l in labels if l['id']==tracks[a]),None) for a in IDS},maskA=maskA,maskAll=maskA|maskB|proposal)
  frame['camera_origin_world']=transform(frame['camera_origin_vehicle'][None],pose)[0];frames.append(frame)
  # Actual original LiDAR, near a plane fitted to observed static lower envelope.
  xyz=np.fromfile(SRC/'velodyne'/f'{fid}.bin',np.float32).reshape(-1,6)[:,:3].astype(np.float64)
  world=transform(xyz,pose);local=transform(world,anchor_inv);distance=local@plane[:3]+plane[3]
  keep=(np.abs(distance)<=.15)&(local[:,0]>=4)&(local[:,0]<=90)&(local[:,1]>=-3)&(local[:,1]<=6)
  xyz=xyz[keep];local=local[keep]
  in_vehicle=np.zeros(len(xyz),bool)
  for l in labels:
   c,s,r=box_data(l);in_vehicle|=(np.abs((xyz-c)@r)<=s/2).all(1)
  ground.append(local[~in_vehicle]);ground_frame.append(np.full((~in_vehicle).sum(),index,np.int16))
  write(OUT/'progress.json',dict(stage='READ_MEASURED_INPUTS_CPU',frames=index+1,GPU_used=False))
 ground=np.concatenate(ground);ground_frame=np.concatenate(ground_frame)
 # Fixed 0.2m lattice inside each real box footprint, placed on the independently fitted road plane.
 query=[];groups=[]
 for f in frames:
  for a in IDS:
   label=f['selected'][a]
   if label is None:continue
   c,size,r=box_data(label);xs=np.arange(-size[0]/2+.1,size[0]/2,.2);ys=np.arange(-size[1]/2+.1,size[1]/2,.2)
   xy=np.array(list(itertools.product(xs,ys)));v=np.c_[xy,np.zeros(len(xy))]@r.T+c;world=transform(v,f['pose']);local=transform(world,anchor_inv)
   local[:,2]=-(local[:,:2]@plane[:2]+plane[3])/plane[2];world=transform(local,anchor)
   start=sum(len(q) for q in query);query.append(world);groups.append(dict(actor=a,anchor_raw=f['raw'],start=start,end=start+len(world),size_lwh=size.tolist(),center_vehicle=c.tolist()))
 world=np.concatenate(query);local=transform(world,anchor_inv);n=len(world)
 counts={key:np.zeros(n,np.int16) for key in ['inside','A_static_supervision_opportunity','non_vehicle_exposure','non_vehicle_eroded_2px']}
 exposure=np.zeros((49,n),bool);aOpportunity=np.zeros((49,n),bool)
 for i,f in enumerate(frames):
  v,cam,uv,pix,inside=project(world,f);ids=np.flatnonzero(inside);counts['inside']+=inside
  am=np.zeros(n,bool);vm=np.zeros(n,bool);dil=np.zeros(n,bool)
  am[ids]=~f['maskA'][pix[ids,1],pix[ids,0]];vm[ids]=~f['maskAll'][pix[ids,1],pix[ids,0]]
  dilation=binary_dilation(f['maskAll'],iterations=2);dil[ids]=~dilation[pix[ids,1],pix[ids,0]]
  occluded=blocked_by_boxes(v,f);vm&=~occluded;dil&=~occluded
  counts['A_static_supervision_opportunity']+=am;counts['non_vehicle_exposure']+=vm;counts['non_vehicle_eroded_2px']+=dil;exposure[i]=vm;aOpportunity[i]=am
  write(OUT/'progress.json',dict(stage='WORLD_COORDINATE_ALL49_PROJECTION_CPU',frames=i+1,road_queries=n,GPU_used=False))
 # Independent LiDAR support, irrespective of camera visibility; not an RGB completion target.
 support=np.zeros(n,np.int16)
 for source_frame in range(49):
  measured=ground[ground_frame==source_frame]
  if len(measured):
   distance,_=cKDTree(measured[:,:2]).query(local[:,:2],distance_upper_bound=.25,workers=1);support+=np.isfinite(distance)
 summaries=[]
 for g in groups:
  sl=slice(g['start'],g['end']);row=dict(g)
  for key,values in counts.items():row[key]=summarize(values[sl])
  row['actual_LiDAR_support_0p25m']=summarize(support[sl])
  ai=g['anchor_raw']-89;row['exposure_frame_fractions']=exposure[:,sl].mean(1).tolist();row['A_supervision_frame_fractions']=aOpportunity[:,sl].mean(1).tolist()
  row['anchor_image_inside_fraction']=float(np.mean(project(world[sl],frames[ai])[-1]));summaries.append(row)
 motions=[]
 duration=(frames[-1]['timestamp']-frames[0]['timestamp'])/1e6
 for a in IDS:
  rows=[(f,f['selected'][a]) for f in frames if f['selected'][a] is not None];centers=np.array([transform(box_data(l)[0][None],f['pose'])[0] for f,l in rows]);rels=np.array([box_data(l)[0] for f,l in rows]);speed=[np.linalg.norm([l.get('metadata',{}).get('speed_x',0),l.get('metadata',{}).get('speed_y',0)]) for f,l in rows]
  slot=[5,26,18,6,37].index(a)+1;pixels=np.array([int(np.sum(np.asarray(Image.open(A/'matched_training_labels'/(f['fid']+'.png')))==slot)) for f in frames])
  motions.append(dict(actor=a,net_displacement_world_m=float(np.linalg.norm(centers[-1]-centers[0])),path_length_world_m=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).sum()),first_relative_x_y_m=rels[0,:2].tolist(),last_relative_x_y_m=rels[-1,:2].tolist(),median_labeled_speed_m_s=float(np.median(speed)),visible_mask_frames=int(np.sum(pixels>0)),mask_pixels_by_frame=pixels.tolist()))
 ego=np.array([f['pose'][:3,3] for f in frames]);stationary_disp=np.linalg.norm(ego-ego[0],axis=1);cam_origins=np.array([f['camera_origin_world'] for f in frames])
 report=dict(status='COMPLETE_READONLY_WORLD_ROAD_OBSERVABILITY_AUDIT',frames=49,raw_range=[89,137],duration_seconds=duration,camera='FRONT',render_WH=[W,H],actors=IDS,GPU_used=False,training_updates=0,inputs_modified=False,
  ego=dict(net_translation_m=float(np.linalg.norm(ego[-1]-ego[0])),path_length_m=float(np.linalg.norm(np.diff(ego,axis=0),axis=1).sum()),frames_within_0p1m_of_start=int(np.sum(stationary_disp<.1)),camera_baseline_m=float(np.linalg.norm(cam_origins[-1]-cam_origins[0])),displacement_by_frame_m=stationary_disp.tolist()),
  motion=motions,ground_plane_anchor=plane.tolist(),ground_fit=readiness['SGCM_geometry']['ground_fit'],measured_ground_points=len(ground),ground_queries=n,
  geometry_semantics='Real per-frame Waymo vehicle-box XY footprint sampled on an independently measured SGCM road plane; same fixed world queries tested in all49 frames. Boxes define diagnostic patches only, never training masks or initialization.',
  exposure_semantics='In FRONT FOV, outside reviewed A/B/native-SAM vehicle mask union and not occluded by any actual vehicle OBB along camera segment. This is a ground-plane exposure opportunity, not proof of photometric/semantic visibility or clean-road ground truth.',
  lidar_semantics='Actual original LiDAR within0.15m measured plane, outside actual vehicle boxes; ground-query XY support radius0.25m, unique source-frame counts. Sparse physical support is not RGB truth.',
  limitations=['No matched vehicle-free ground-truth image.','Planar-road approximation; masks/3D boxes and temporal alignment have finite error.','Non-vehicle occluders are not explicitly ray-tested; road query range restricted to the actual traffic lanes.','At least one unmasked projection is not a quality pass; shadows and limited view diversity remain.','Data missing after the49-frame window is not used or silently added.'],
  footprint_summaries=summaries,human_verdict=None)
 write(OUT/'receipt.json',report)
 np.savez_compressed(OUT/'projection_evidence.npz',world=world,anchor_xyz=local,exposure=exposure,A_opportunity=aOpportunity,LiDAR_source_frame_count=support)
 # Full-frame evidence. No crops, only point overlays at actual projected positions.
 for raw in [89,107,128,137]:
  f=frames[raw-89];im=Image.open(SRC/'image/image_0'/(f['fid']+'.png')).convert('RGB').resize((960,640),Image.Resampling.LANCZOS);d=ImageDraw.Draw(im)
  for g in groups:
   if g['anchor_raw']!=raw:continue
   sl=slice(g['start'],g['end']);v,cam,uv,pix,inside=project(world[sl],f);ex=counts['non_vehicle_exposure'][sl]
   for p,count,ok in zip(uv,ex,inside):
    if ok:
     x,y=p*2;color='#42df8b' if count>=3 else '#ffce52' if count else '#ff5555';d.ellipse((x-1.6,y-1.6,x+1.6,y+1.6),fill=color)
   label=f['selected'][g['actor']];c,size,r=box_data(label);_,_,cuv,_,visible=project(transform(c[None],f['pose']),f)
   if visible[0]:d.text(tuple(cuv[0]*2),f"a{g['actor']}",fill='yellow',stroke_width=2,stroke_fill='black')
  d.text((12,12),f'raw{raw} | fixed world ground queries: red=0, amber=1-2, green>=3 non-vehicle exposure frames',fill='white',stroke_width=2,stroke_fill='black');im.save(OUT/'fullframes'/f'{f["fid"]}_exposure.png')
 write(OUT/'progress.json',dict(stage='COMPLETE_READONLY_CPU',frames=49,GPU_used=False))
 print(json.dumps({k:report[k] for k in ['status','duration_seconds','ego','motion','measured_ground_points','ground_queries']},ensure_ascii=False))
 print('ANCHOR128',json.dumps([dict(actor=r['actor'],exposure=r['non_vehicle_exposure'],LiDAR=r['actual_LiDAR_support_0p25m'],inside=r['anchor_image_inside_fraction']) for r in summaries if r['anchor_raw']==128]))
if __name__=='__main__':main()
