"""r19：仅用原视频和GT相机检验围栏平面对齐，冻结所有生成图。"""
import sys,datetime,time
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import read,dump,transform
from video_review import scene_frame
from repair_common import camera
import cv2,numpy as np
from PIL import Image,ImageDraw
cv2.setNumThreads(4)
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'r19/geometry_attempt01';DATA=Path('/root/autodl-tmp/data/v76_vadgs/scene_0255')
SPEC=next(s for s in read(BASE/'r2/registration.json')['scenes'] if s['name']=='scene_0255');INST=read(DATA/'instances/instances_info.json');HW=(900,1600)

def cam(f):return camera(scene_frame(SPEC['spec'],f,INST),3,HW)
def project(world,c,k):
 q=transform(world,np.linalg.inv(c))@k.T;return q[:,:2]/q[:,2:]
def main():
 assert not ROOT.exists();ROOT.mkdir();start=time.time();frames=list(range(65,111));geom=read(BASE/'r8/registration.json')['geometry'];scale=np.array([1600/960,900/536]);shift=scale*.5-.5
 g={k:[np.array(v,float)*scale+shift for v in geom[k]] for k in ['lines','bars']};g['panel']=np.array(geom['panel'],float)*scale+shift
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r19',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',scope='Original foreground-fence alignment only, no new inpainting inference',
  hypothesis='Different-time original views expose clean background behind the thin fence; a camera-constrained foreground plane can align its true material samples without carrying actor25 pixels.',
  input_roles='GT camera poses/intrinsics and original RGB f65..110 CAM3; assistant-traced r8 fence geometry seeds support. No generated RGB in fitting.',
  controls='Panel features fit the plane; non-panel rail features evaluate it. Query RGB is visible fence evidence, not hidden background truth.',
  method='OpenCV corner detection and forward/backward LK; triangulation with GT cameras; robust panel plane then reprojection audit',
  source='https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html',fixed=dict(seed=42,cpu_threads=4,fb_limit_px=.7,min_tracked_timestamps=20,plane_distance_m=.08),
  stop_rule='First audit only; do not reuse source RGB for compositing if projected rails/panel visibly misalign.',failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 ims={f:np.array(Image.open(DATA/'images'/f'{f:03}_3.jpg').convert('RGB')) for f in frames};grays={f:cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for f,im in ims.items()}
 support=np.zeros(HW,np.uint8);panel=np.zeros_like(support)
 cv2.fillPoly(panel,[np.rint(g['panel']).astype('int32')],255)
 for l in g['lines']+g['bars']:cv2.polylines(support,[np.rint(l).astype('int32')],False,255,5)
 support|=panel
 pts=cv2.goodFeaturesToTrack(grays[65],300,.002,4,mask=support,blockSize=3);assert pts is not None and len(pts)>=15
 xy=np.rint(pts[:,0]).astype(int);ispanel=panel[xy[:,1],xy[:,0]]>0
 track=np.full((len(frames),len(pts),2),np.nan,np.float32);track[0]=pts[:,0];valid=np.ones(len(pts),bool);old=pts
 for i,f in enumerate(frames[1:],1):
  nxt,s,_=cv2.calcOpticalFlowPyrLK(grays[f-1],grays[f],old,None,winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
  back,s2,_=cv2.calcOpticalFlowPyrLK(grays[f],grays[f-1],nxt,None,winSize=(21,21),maxLevel=4)
  valid&=s[:,0].astype(bool)&s2[:,0].astype(bool)&(np.linalg.norm(back[:,0]-old[:,0],axis=-1)<.7)
  track[i,valid]=nxt[valid,0];old=nxt
 cameras={f:cam(f) for f in frames};c0,k0=cameras[65];P0=k0@np.linalg.inv(c0)[:3]
 track_summary=[dict(frame=f,valid=int(np.isfinite(track[i,:,0]).sum()),panel=int((np.isfinite(track[i,:,0])&ispanel).sum())) for i,f in enumerate(frames)]
 np.savez_compressed(ROOT/'raw_tracking.npz',points0=pts[:,0],tracks=track,is_panel=ispanel)
 sheet=Image.new('RGB',(1000,300*5),(20,25,35))
 for i,f in enumerate([65,75,85,95,110]):
  im=Image.fromarray(ims[f]);d=ImageDraw.Draw(im);q=track[f-65];good=np.isfinite(q[:,0])
  for j in np.flatnonzero(good):
   x,y=q[j];d.ellipse((x-2,y-2,x+2,y+2),outline='yellow' if ispanel[j] else 'cyan',width=1)
  if good.any():
   lo=q[good].min(0)-30;hi=q[good].max(0)+30;im=im.crop((*lo,*hi)).resize((1000,300))
  d=ImageDraw.Draw(im);d.text((8,8),f'f{f} panel yellow / rail-support cyan',fill='white');sheet.paste(im,(0,i*300))
 sheet.save(ROOT/'tracking_review.jpg',quality=96)
 points=[];kept=[];reproj=[];rejected=[]
 for j in range(len(pts)):
  ids=np.flatnonzero(np.isfinite(track[:,j,0]));far=[i for i in ids if frames[i]>=85]
  if len(ids)<20 or len(far)<5:rejected.append(dict(point=j,reason='short_track',valid_timestamps=len(ids)));continue
  candidates=[]
  for i in far[::3]:
   c,k=cameras[frames[i]];P=k@np.linalg.inv(c)[:3];x=cv2.triangulatePoints(P0,P,track[0,j].reshape(2,1),track[i,j].reshape(2,1));x=(x[:3]/x[3:]).T[0];candidates.append(x)
  world=np.median(candidates,axis=0);errs=np.array([np.linalg.norm(project(world[None],*cameras[frames[i]])[0]-track[i,j]) for i in ids]);z=transform(world[None],np.linalg.inv(c0))[0,2]
  if not np.isfinite(world).all() or z<2 or z>40 or np.median(errs)>1.5:rejected.append(dict(point=j,reason='triangulation_or_reprojection',z=float(z),median_error=float(np.median(errs))));continue
  points.append(world);kept.append(j);reproj.append(float(np.median(errs)))
 points=np.array(points);kept=np.array(kept,dtype=int);fit=ispanel[kept]
 dump(ROOT/'tracking_attrition.json',dict(frames=track_summary,rejected=rejected,retained=len(points),retained_panel=int(fit.sum())))
 if fit.sum()<5:
  dump(ROOT/'state.json',dict(state='rejected_insufficient_triangulated_panel_tracks',retained=len(points),retained_panel=int(fit.sum()),human_verdict=None));print('R19_GEOMETRY_REJECTED',len(points),int(fit.sum()),flush=True);return
 rng=np.random.default_rng(42);fp=points[fit];best=None
 for _ in range(300):
  tri=fp[rng.choice(len(fp),3,replace=False)];n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);norm=np.linalg.norm(n)
  if norm<1e-5:continue
  n/=norm;d=-n@tri[0];inside=np.abs(fp@n+d)<.08
  if best is None or inside.sum()>best[0].sum():best=(inside,n,d)
 assert best is not None and best[0].sum()>=5
 centroid=fp[best[0]].mean(0);_,_,vt=np.linalg.svd(fp[best[0]]-centroid);normal=vt[-1];offset=-normal@centroid
 rays=np.c_[pts[:,0],np.ones(len(pts))]@np.linalg.inv(k0).T@c0[:3,:3].T;depth=-(normal@c0[:3,3]+offset)/(rays@normal);planepts=c0[:3,3]+rays*depth[:,None]
 rows=[];homographies=[];atlas=Image.new('RGB',(800*3,400*3),(20,25,35));crop=(570,490,980,665)
 for row,f in enumerate([65,80,90,100,110]):
  c,k=cameras[f];pix=project(planepts,c,k);i=f-65;ok=np.isfinite(track[i,:,0])&(depth>0);errors=np.linalg.norm(pix[ok]-track[i,ok],axis=-1);pc=ispanel[ok]
  stat=dict(frame=f,panel_tracks=int(pc.sum()),panel_error_median=float(np.median(errors[pc])) if pc.any() else None,rail_tracks=int((~pc).sum()),rail_error_median=float(np.median(errors[~pc])) if (~pc).any() else None,rail_error_q90=float(np.percentile(errors[~pc],90)) if (~pc).any() else None);rows.append(stat)
  # 用已知相机与同一拟合平面约束整张H，避免无约束退化拟合。
  relative=np.linalg.inv(c)@c0;n0=normal@c0[:3,:3];d0=float(normal@c0[:3,3]+offset);H=k@(relative[:3,:3]-relative[:3,3,None]*n0[None]/d0)@np.linalg.inv(k0);H/=H[2,2];homographies.append(dict(frame=f,H_65_to_frame=H.tolist()))
  aligned=cv2.warpPerspective(ims[f],np.linalg.inv(H),(1600,900),flags=cv2.INTER_LINEAR);Image.fromarray(aligned).save(ROOT/f'canonical_{f}.png')
  over=Image.fromarray(aligned);draw=ImageDraw.Draw(over)
  for l in g['lines']+g['bars']:draw.line([tuple(p) for p in l],fill='cyan',width=1)
  if row<3:
   for col,(title,im) in enumerate([('source original',Image.fromarray(ims[f])),('plane aligned to f65',Image.fromarray(aligned)),('fixed fence geometry',over)]):
    if col==0:
     corners=cv2.perspectiveTransform(np.array([[crop[0],crop[1]],[crop[2],crop[3]]],np.float32).reshape(-1,1,2),H)[:,0];box=(*np.floor(corners.min(0)).astype(int),*np.ceil(corners.max(0)).astype(int))
    else:box=crop
    im=im.crop(box).resize((800,400));ImageDraw.Draw(im).text((6,6),f'f{f} {title}',fill='yellow');atlas.paste(im,(col*800,row*400))
 np.savez_compressed(ROOT/'tracking.npz',points0=pts[:,0],tracks=track,is_panel=ispanel,triangulated=points,triangulated_point_indices=kept,plane_world_normal=normal,plane_world_offset=offset)
 dump(ROOT/'homographies.json',homographies);atlas.save(ROOT/'alignment_review.jpg',quality=96)
 dump(ROOT/'geometry_audit.json',dict(frames=len(frames),initial_points=len(pts),initial_panel_points=int(ispanel.sum()),triangulated_points=len(points),triangulated_panel_points=int(fit.sum()),plane_fit_inliers=int(best[0].sum()),plane_normal=normal.tolist(),plane_offset=float(offset),absolute_normal_z=float(abs(normal[2])),sampled_reprojection=rows,elapsed_s=time.time()-start,human_verdict=None))
 dump(ROOT/'state.json',dict(state='geometry_complete_pending_visual_review',human_verdict=None));print('R19_GEOMETRY_COMPLETE',rows,flush=True)

if __name__=='__main__':main()
