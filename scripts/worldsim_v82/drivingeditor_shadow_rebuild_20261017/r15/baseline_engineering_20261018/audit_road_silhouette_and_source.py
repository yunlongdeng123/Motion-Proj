"""CPU-only extension: same-world road behind body pixels, plus original199-frame context.
The wider source is diagnostic only and never enters either training experiment.
"""
from pathlib import Path
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
E=Path(__file__).resolve().parent;sys.path.insert(0,str(E));import audit_road_observability as au
ROOT=E.parent;sys.path.insert(0,str(ROOT));import waymo_numpy_cpu as rawreader
OUT=E/'road_observability_20261019';DST=OUT/'silhouette_and_source';A=au.A;SRC=au.SRC
def main():
 assert not DST.exists(),'preserve earlier audit'
 DST.mkdir();(DST/'source_fullframes').mkdir()
 base=json.loads((OUT/'receipt.json').read_text());plane=np.array(base['ground_plane_anchor']);anchor=np.loadtxt(SRC/'pose/0008089.txt');raw=128;fid='0008128'
 pose=np.loadtxt(SRC/'pose'/f'{fid}.txt');cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (SRC/'calib'/f'{fid}.txt').read_text().splitlines() if ':' in s}
 tr=np.eye(4);tr[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);K=cal['P0'].reshape(3,4)[:,:3]*.25;K[2,2]=1
 c2anchor=np.linalg.inv(anchor)@pose@np.linalg.inv(tr);labels=np.asarray(Image.open(A/'matched_training_labels'/f'{fid}.png'));queries=[];groups=[]
 for actor in au.IDS:
  slot=[5,26,18,6,37].index(actor)+1;ys,xs=np.where(labels==slot)
  if not len(xs):continue
  camdirs=np.c_[xs,ys,np.ones(len(xs))]@np.linalg.inv(K).T;directions=camdirs@c2anchor[:3,:3].T;origin=c2anchor[:3,3]
  distance=-(origin@plane[:3]+plane[3])/(directions@plane[:3]);local=origin+distance[:,None]*directions
  valid=(distance>0)&(local[:,0]>=4)&(local[:,0]<=90)&(local[:,1]>=-3)&(local[:,1]<=6)
  start=sum(len(q) for q in queries);queries.append(au.transform(local[valid],anchor));groups.append(dict(actor=actor,start=start,end=start+int(valid.sum()),body_pixels=len(xs),valid_ground_queries=int(valid.sum()),valid_body_ground_fraction=float(valid.mean()),pixels_xy=np.c_[xs[valid],ys[valid]].tolist()))
 world=np.concatenate(queries);n=len(world);exposure=[];conservative=[];colors=[]
 for raw in au.RAW:
  fid=f'0008{raw:03d}';ann=json.loads((SRC/'annotations'/f'{fid}.json').read_text());pose=np.loadtxt(SRC/'pose'/f'{fid}.txt');cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (SRC/'calib'/f'{fid}.txt').read_text().splitlines() if ':' in s}
  tr=np.eye(4);tr[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);K=cal['P0'].reshape(3,4)[:,:3]*.25;K[2,2]=1
  mask=np.asarray(Image.open(au.B/'dynamic_masks/dynamic_mask_0'/f'{fid}.png').resize((480,320),Image.Resampling.NEAREST))>0
  mask|=np.asarray(Image.open(A/'matched_training_labels'/f'{fid}.png'))>0
  native=np.load(E/'all_vehicle_native49/native_predictions'/f'{fid}.npz')['masks'].any(0);mask|=np.asarray(Image.fromarray(native).resize((480,320),Image.Resampling.NEAREST))>0
  frame=dict(pose=pose,tr=tr,K=K,camera_origin_vehicle=np.linalg.inv(tr)[:3,3],labels=[l for l in ann['laser_labels'] if l['type']=='TYPE_VEHICLE'])
  v,c,uv,pix,inside=au.project(world,frame);ids=np.flatnonzero(inside);seen=np.zeros(n,bool);strict=np.zeros(n,bool)
  seen[ids]=~mask[pix[ids,1],pix[ids,0]];dil=binary_dilation(mask,iterations=2);strict[ids]=~dil[pix[ids,1],pix[ids,0]]
  blocked=au.blocked_by_boxes(v,frame);seen&=~blocked;strict&=~blocked;exposure.append(seen);conservative.append(strict)
  rgb=np.asarray(Image.open(SRC/'image/image_0'/f'{fid}.png').resize((480,320),Image.Resampling.LANCZOS));sample=np.full((n,3),np.nan);sample[seen]=rgb[pix[seen,1],pix[seen,0]];colors.append(sample)
 counts=np.sum(exposure,axis=0);strict=np.sum(conservative,axis=0);colors=np.array(colors)
 static=np.asarray(Image.open(E/'GN11_stage1_full49/0008128/static.png'));gt=np.asarray(Image.open(E/'GN11_stage1_full49/0008128/gt.png'))
 assert static.shape==gt.shape==(320,480,3)
 rows=[]
 for g in groups:
  sl=slice(g['start'],g['end']);r=dict(g);r['exposure_49']=au.summarize(counts[sl]);r['exposure_49_mask_dilated_2px']=au.summarize(strict[sl]);available=counts[sl]>=3
  pixels=np.array(g['pixels_xy']);median=np.full((len(pixels),3),np.nan)
  if available.any():median[available]=np.nanmedian(colors[:,sl][:,available],axis=0)
  delta=np.abs(static[pixels[:,1],pixels[:,0]].astype(float)-median).mean(1)
  r['median_reference_absolute_RGB_error_255_where_3plus_exposures']=float(np.median(delta[available])) if available.any() else None
  r['reference_is_GT']=False;rows.append(r)
 # Native original source199 frames. Box-only visibility opportunity is an optimistic upper bound.
 prep=json.loads((SRC/'preparation_manifest.json').read_text());pb=rawreader.protobuf();full=[]
 anchorbytes={};rawpath=prep['raw_path']
 for index,payload,crc in rawreader.records(rawpath,198):
  f=pb.Frame.FromString(payload);pose=np.array(f.pose.transform).reshape(4,4);camera=next(c for c in f.context.camera_calibrations if c.name==1)
  transform=np.eye(4);transform[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]];tr=transform@np.linalg.inv(np.array(camera.extrinsic.transform).reshape(4,4));k=camera.intrinsic;K=np.array([[k[0]/4,0,k[2]/4],[0,k[1]/4,k[3]/4],[0,0,1.]])
  labels=[dict(box=dict(center_x=l.box.center_x,center_y=l.box.center_y,center_z=l.box.center_z,length=l.box.length,width=l.box.width,height=l.box.height,heading=l.box.heading)) for l in f.laser_labels if l.type==1]
  frame=dict(pose=pose,tr=tr,K=K,camera_origin_vehicle=np.linalg.inv(tr)[:3,3],labels=labels)
  v,c,uv,pix,inside=au.project(world,frame);seen=inside&~au.blocked_by_boxes(v,frame)
  full.append(dict(raw=index,in_training_window=89<=index<=137,camera_world=au.transform(frame['camera_origin_vehicle'][None],pose)[0].tolist(),box_only_unoccluded_fractions={g['actor']:float(np.mean(seen[g['start']:g['end']])) for g in groups},query_opportunities=seen))
  if index in [0,48,88,138,160,180,198]:
   image=next(im for im in f.images if im.name==1);import io
   with Image.open(io.BytesIO(image.image)) as im:im.resize((960,640),Image.Resampling.LANCZOS).save(DST/'source_fullframes'/f'raw{index:03d}.jpg',quality=90)
  if index%25==0:au.write(DST/'progress.json',dict(stage='READ_NATIVE_SOURCE_ONLY_NO_EXTRA_TRAINING',raw=index))
 fullseen=np.array([x.pop('query_opportunities') for x in full]);inside=fullseen[89:138].sum(0);outside=np.r_[fullseen[:89],fullseen[138:]].sum(0)
 for g,r in zip(groups,rows):
  sl=slice(g['start'],g['end']);r['box_only_full199_opportunities']=au.summarize(fullseen.sum(0)[sl]);r['box_only_training49_opportunities']=au.summarize(inside[sl]);r['box_only_outside49_opportunities']=au.summarize(outside[sl])
  scores=[(x['raw'],x['box_only_unoccluded_fractions'][g['actor']]) for x in full if not x['in_training_window']];r['best_outside49_box_only_frame']=max(scores,key=lambda x:x[1])
 result=dict(status='COMPLETE_READONLY_SILHOUETTE_AND_RAW_SOURCE_CONTEXT',anchor_raw=128,training_frames=49,diagnostic_source_frames=199,additional_training_views=0,training_inputs_modified=False,GPU_used=False,training_updates=0,road_queries=n,rows=rows,source_frames=full,
  semantics='Actual body-mask pixels in raw128 are back-projected onto the measured road plane, then fixed world positions are tested in all training frames. Wider199 uses actual native calibration/ego/3D boxes only: box-only candidate exposure is an optimistic upper bound, not confirmed clean road.',
  pseudo_reference='Median actual RGB of projected queries with >=3 non-vehicle exposure opportunities. Plane/visibility/shadow errors remain; not vehicle-free ground truth.',human_verdict=None)
 au.write(DST/'receipt.json',result);np.savez_compressed(DST/'evidence.npz',world=world,exposure=np.array(exposure),full_source_box_opportunity=fullseen)
 print(json.dumps(dict(status=result['status'],road_queries=n,rows=[{k:v for k,v in r.items() if k!='pixels_xy'} for r in rows]),ensure_ascii=False))
if __name__=='__main__':main()
