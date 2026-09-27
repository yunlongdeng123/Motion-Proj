"""r32：共同可见路面LiDAR估计高度平移，再以真实RGB验证；不把GT平面定位当6DoF真值。"""
import sys,time,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_road_plane_control as p
import hybrid_road_patch_control as rp
from repair_common import dump,read
from geometry import transform
from scipy.spatial import cKDTree,Delaunay
import numpy as np,cv2
from PIL import Image,ImageDraw
cv2.setNumThreads(4);ROOT=p.BASE/'r32';CACHE={};LOCAL={};HEIGHT={};DATA=p.DATA;HW=p.HW

def local_ground(f):
 if f in LOCAL:return LOCAL[f]
 fr,c,k,im,exclude=p.get(f);points=transform(np.fromfile(DATA/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(DATA/'lidar_pose'/f'{f:03}.txt'));uv,z=p.project(points,c,k);ij=np.rint(uv).astype(int);good=(z>3)&(z<50)&(ij[:,0]>=0)&(ij[:,0]<1024)&(ij[:,1]>=0)&(ij[:,1]<576);roi,_=p.road_roi();ids=np.flatnonzero(good);ids=ids[roi[ij[ids,1],ij[ids,0]]&~exclude[ij[ids,1],ij[ids,0]]];q=points[ids];q=q[(q[:,2]-c[2,3]>-3)&(q[:,2]-c[2,3]<-.3)];assert len(q)>75
 rng=np.random.default_rng(42);best=np.zeros(len(q),bool)
 for _ in range(300):
  a=q[rng.choice(len(q),3,False)];n=np.cross(a[1]-a[0],a[2]-a[0]);size=np.linalg.norm(n)
  if size<1e-6:continue
  n/=size
  if abs(n[2])<np.cos(np.deg2rad(15)):continue
  keep=np.abs(q@n-n@a[0])<.05
  if keep.sum()>best.sum():best=keep
 assert best.sum()>=50;g=q[best];mid=g.mean(0);_,_,v=np.linalg.svd(g-mid,full_matrices=False);n=v[-1];n*=np.sign(n[2]);d=-n@mid
 LOCAL[f]=(g,n,d);return LOCAL[f]

def fit_heights():
 anchors=[0,5,10,15,20,25,30,35,40,45,49];correction={0:0.};rows=[]
 for a,b in zip(anchors,anchors[1:]):
  source,n,_=local_ground(b);target,_,_=local_ground(a);dist,nearest=cKDTree(target[:,:2]).query(source[:,:2],workers=4);sel=np.flatnonzero(dist<=.20);assert len(sel)>=50,(a,b,len(sel));s=source[sel];t=target[nearest[sel]]
  # 比较同一水平位置附近的地面高度，并去掉最近邻XY偏差的局部坡度项。
  slope=-n[:2]/n[2];target_at_source=t[:,2]+(s[:,:2]-t[:,:2])@slope;delta=target_at_source-s[:,2];hold=np.arange(len(sel))%3==0;shift=float(np.median(delta[~hold]));error=delta[hold]-shift
  assert np.median(np.abs(error))<.05 and np.percentile(np.abs(error),90)<.12,(a,b,error)
  correction[b]=correction[a]+shift;rows.append(dict(source_frame=b,target_frame=a,pairs=len(sel),fit_pairs=int((~hold).sum()),heldout_pairs=int(hold.sum()),height_increment_m=shift,height_correction_m=correction[b],heldout_abs_median_before=float(np.median(np.abs(delta[hold]))),heldout_abs_median_after=float(np.median(np.abs(error))),heldout_abs_q90_after=float(np.percentile(np.abs(error),90))))
 for f in range(50):HEIGHT[f]=float(np.interp(f,anchors,[correction[a] for a in anchors]))
 dump(ROOT/'height_alignment.json',dict(anchor_frames=anchors,correction_z_by_frame=HEIGHT,pairs=rows,scope='Only z translation, fixed xy/rotation. Heldout correspondences test fit consistency, not independent scene truth.',human_verdict=None));return rows

def source(f):
 if f in CACHE:return CACHE[f]
 fr,c,k,im,exclude=p.get(f);c=c.copy();c[2,3]+=HEIGHT[f]
 points=transform(np.fromfile(DATA/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(DATA/'lidar_pose'/f'{f:03}.txt'));points[:,2]+=HEIGHT[f];uv,z=p.project(points,c,k);ij=np.rint(uv).astype(int);valid=(z>3)&(z<50)&(ij[:,0]>=0)&(ij[:,0]<1024)&(ij[:,1]>=0)&(ij[:,1]<576);ids=np.flatnonzero(valid);ids=ids[~exclude[ij[ids,1],ij[ids,0]]];ids=ids[np.abs(points[ids]@rp.NORMAL+rp.OFFSET)<=.05];world=points[ids];uv=uv[ids];support=np.zeros(HW,np.uint8);triangles=0
 if len(ids)>=3:
  tri=Delaunay(uv).simplices;w=world[tri];u=uv[tri];edges=np.stack([np.linalg.norm(w[:,a]-w[:,b],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1);pixels=np.stack([np.linalg.norm(u[:,a]-u[:,b],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1);keep=(edges.max(1)<=3)&(pixels.max(1)<=64);triangles=int(keep.sum())
  for t in np.rint(u[keep]).astype('int32'):cv2.fillConvexPoly(support,t,1)
 support=cv2.erode(support,np.ones((3,3),np.uint8))>0;support&=~exclude;rec=dict(frame=f,height_correction_m=HEIGHT[f],ground_lidar_points=len(ids),triangles=triangles,ground_rgb_pixels=int(support.sum()));CACHE[f]=(fr,c,k,im,exclude,support,rec);return CACHE[f]

def main():
 assert not ROOT.exists();ROOT.mkdir();beg=time.time()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r32',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',actor='12',camera=0,
  question='Does limited z-only registration of shared static road points improve actual RGB reprojection and recover valid DELETE evidence before assuming backgrounds unobserved?',
  source_note='nuScenes official ego_pose schema says planar xy localization and translation.z=0. This is motivation, not proof that all processed-pose errors arise from omitted height.',primary_source='https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#ego_pose',
  intervention='Estimate scalar z shifts from nearest common ground XY, interpolate between anchor times. Apply same z shift to camera and LiDAR at each frame; GT image exclusion envelopes remain the original same-frame projections. Everything else uses r27 frozen source support/rendering rules.',
  height_fit=dict(anchors=[0,5,10,15,20,25,30,35,40,45,49],local_ground='same visible image ROI with allGT actors excluded; RANSAC300 seed42,5cm inlier,normalwithin15deg',max_xy_pair_distance_m=.2,fit='median target minus source z after local-slope XY correction',heldout='everythird correspondence excluded from median; requiremedian<5cm/q90<12cm',gauge='f0 correction0',xy_and_rotation='unchanged'),
  limits='LiDAR-assisted POC with local road-domain assumption. Scalar z cannot solve arbitrary pose/depth errors or prove hidden pixel truth. RGB query is evaluation only; no generated images in geometry.',
  fixed=dict(source_offsets=[5,10,20],source_plane='r25 f0',source_ground_distance_m=.05,triangle_max_edge_m=3,triangle_max_edge_px=64,source_erode_px=1,rgb_consensus_max_error=20,cpu_threads=4,gpu_forwards=0),
  stop_rule='Reject if shared-road heldout or actual visible RGB alignment worsens. Target-hole support is diagnostic until visually verified; no automatic writing into candidate background or Omega.',failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 alignment=fit_heights();rp.source=source;positives=[]
 for f in [0,10,20]:
  domain=np.array(Image.open(p.ROOT/f'f{f:03}/evaluation_mask.png'))>0;im,near,hit,con,chit,first,pair,counts=rp.render(f,domain);rp.save(ROOT/'visible'/f'f{f:03}',im,domain,near,hit,con,chit,first,pair);row=dict(frame=f,denominator=int(domain.sum()),sources=counts,arms={})
  oldrgb=np.array(Image.open(p.ROOT/f'f{f:03}/nearest_rgb.png'));oldmask=np.array(Image.open(p.ROOT/f'f{f:03}/nearest_mask.png'))>0
  for name,res,m in [('nearest',near,hit),('consensus',con,chit)]:
   err=np.abs(res.astype('float32')-im).mean(-1);old_error=np.abs(oldrgb.astype('float32')-im).mean(-1);shared=m&oldmask
   row['arms'][name]=dict(covered=int(m.sum()),coverage=float(m.sum()/max(1,domain.sum())),mean_mae=float(err[m].mean()) if m.any() else None,q90_mae=float(np.percentile(err[m],90)) if m.any() else None,shared_with_r25_nearest_pixels=int(shared.sum()),r25_mae_on_shared=float(old_error[shared].mean()) if shared.any() else None,corrected_mae_on_shared=float(err[shared].mean()) if shared.any() else None)
  positives.append(row)
 dump(ROOT/'visible_validation.json',positives)
 rows=[]
 for f in range(30):
  domain=np.array(Image.open(rp.OLD/'mask'/f'{f:05}.png'))>0;im,near,hit,con,chit,first,pair,counts=rp.render(f,domain);rp.save(ROOT/f'f{f:03}',im,domain,near,hit,con,chit,first,pair);assert np.all(first[hit]!=f) and np.all(pair[chit]!=f);rows.append(dict(frame=f,mask_pixels=int(domain.sum()),nearest_pixels=int(hit.sum()),consensus_pixels=int(chit.sum()),sources=counts))
 summary=dict(state='complete_pending_visual_review',seconds=time.time()-beg,frames=30,total_mask_pixels=sum(r['mask_pixels'] for r in rows),nearest=sum(r['nearest_pixels'] for r in rows),consensus=sum(r['consensus_pixels'] for r in rows),height_correction_f49_m=HEIGHT[49],visible=positives,rows=rows,human_verdict=None,background_input_dir=None);dump(ROOT/'state.json',summary)
 dump(ROOT/'source_patch_metrics.json',[v[-1] for k,v in sorted(CACHE.items())])
 for mode,frames in [('visible',[0,10,20]),('hole',[0,5,10,15,20,29])]:
  sheet=Image.new('RGB',(1200,len(frames)*250),(12,20,30))
  for j,f in enumerate(frames):
   out=ROOT/'visible'/f'f{f:03}' if mode=='visible' else ROOT/f'f{f:03}';mask=np.array(Image.open(out/'mask.png'))>0;y,x=np.where(mask)
   if mode=='hole':left=int(np.clip((x.min()+x.max())/2-192,0,640));top=int(np.clip((y.min()+y.max())/2-108,0,360));crop=(left,top,left+384,top+216)
   else:crop=(0,0,1024,576)
   for col,key in enumerate(['original','nearest_diagnostic','consensus_diagnostic']):
    tile=Image.new('RGB',(400,250),(12,20,30));tile.paste(Image.open(out/f'{key}.png').crop(crop).resize((400,225)),(0,25));ImageDraw.Draw(tile).text((5,5),f'f{f} {key}',fill='white');sheet.paste(tile,(col*400,j*250))
  sheet.save(ROOT/f'{mode}_contact.jpg',quality=97)
 print('R32_COMPLETE',{k:v for k,v in summary.items() if k not in ['visible','rows']},positives,flush=True)
if __name__=='__main__':main()
