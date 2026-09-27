"""r25：第三scene可见路面的真实RGB重投影，先验证几何再处理删除洞。"""
import os,sys,time,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,camera,hull_mask
from geometry import transform
from video_review import scene_frame
import cv2,numpy as np
from PIL import Image,ImageDraw
cv2.setNumThreads(4)
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'r25';OLD=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-REPAIR-20260927/r1')
SPEC=next(s for s in read(OLD/'registration.json')['scenes'] if s['name']=='official_000');DATA=Path(SPEC['data']);INST=read(DATA/'instances/instances_info.json');HW=(576,1024)
def get(f):
 fr=scene_frame(SPEC['spec'],f,INST);c,k=camera(fr,0,HW);im=np.array(Image.open(DATA/'images'/f'{f:03}_0.jpg').convert('RGB').resize((1024,576),Image.Resampling.BILINEAR));exclude=np.zeros(HW,bool)
 for b in fr['all_boxes']:exclude|=hull_mask(b,c,k,HW,pad=3,ground_extend=.5)
 return fr,c,k,im,exclude
def project(pts,c,k):
 q=transform(pts,np.linalg.inv(c));uv=q@k.T;return uv[:,:2]/np.maximum(uv[:,2:],1e-9),q[:,2]
def road_roi():
 m=np.zeros(HW,np.uint8)
 # 已查看f0原图后固定的一块可见道路；仅用于POC验证，不冒称自动语义分割。
 poly=np.array([[.28,.94],[.90,.94],[.70,.66],[.55,.55],[.46,.63]])*np.array([1024,576]);cv2.fillPoly(m,[np.rint(poly).astype('int32')],1);return m>0,poly
def fit_plane(fr,c,k,exclude):
 pts=transform(np.fromfile(DATA/'lidar/000.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(DATA/'lidar_pose/000.txt'));uv,z=project(pts,c,k);ij=np.rint(uv).astype(int);good=(z>3)&(z<50)&(ij[:,0]>=0)&(ij[:,0]<1024)&(ij[:,1]>=0)&(ij[:,1]<576)
 roi,_=road_roi();idx=np.flatnonzero(good);idx=idx[roi[ij[idx,1],ij[idx,0]]&~exclude[ij[idx,1],ij[idx,0]]];p=pts[idx];valid_height=(p[:,2]-c[2,3]>-3)&(p[:,2]-c[2,3]<-.3);idx=idx[valid_height];p=pts[idx];assert len(p)>=75,('not enough ground candidates',len(p))
 test=np.arange(len(p))%3==0;train=p[~test];rng=np.random.default_rng(42);best=None
 for _ in range(300):
  a=train[rng.choice(len(train),3,replace=False)];n=np.cross(a[1]-a[0],a[2]-a[0]);length=np.linalg.norm(n)
  if length<1e-6:continue
  n/=length
  if abs(n[2])<np.cos(np.deg2rad(15)):continue
  d=-n@a[0];ins=np.abs(train@n+d)<.05
  if best is None or ins.sum()>best[0].sum():best=(ins,n,d)
 assert best is not None and best[0].sum()>=50,('insufficient plane',None if best is None else int(best[0].sum()))
 keep=train[best[0]];center=keep.mean(0);_,_,v=np.linalg.svd(keep-center,full_matrices=False);n=v[-1];n*=np.sign(n[2]);d=-n@center;err=np.abs(p@n+d);hull=cv2.convexHull(keep[:,:2].astype('float32')).reshape(-1,2)
 record=dict(candidates=len(p),train_points=len(train),fit_inliers=int(best[0].sum()),normal=n.tolist(),offset=float(d),slope_deg=float(np.rad2deg(np.arccos(n[2]))),heldout_count=int(test.sum()),heldout_median_m=float(np.median(err[test])),heldout_q90_m=float(np.percentile(err[test],90)),heldout_within_5cm=float(np.mean(err[test]<.05)),training_convex_hull_xy=hull.tolist())
 return n,d,hull,record,uv[idx],test,err
def main():
 assert not ROOT.exists();ROOT.mkdir();beg=time.time()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r25',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',target='12',camera=0,
  question='Do original camera poses and a measured local ground plane reliably transport visible road RGB before deletion-hole reuse?',
  roles='f0 LiDAR/pose and GT camera/boxes are geometry input; f0 visible road polygon is assistant POC domain. Query RGB is evaluation only; never a donor at its own timestamp. No generated RGB or new model.',
  plane='RANSAC 300 proposals seed42, normal within15deg of world-up,5cm inlier threshold,at least50 train inliers. Every third candidate point held out. Plane-only approximation restricted to inlier xy hull.',
  query_frames=[0,10,20],source_offsets=[5,10,20],source_policy='Original CAM0 other timestamps, all GT actor projection envelopes excluded with3px dilation and0.5m ground extension; geometric support does not certify absence of unknown objects/shadows.',
  controls='Nearest available source vs at least two other timestamps agreeing within20 RGB levels. Same frozen plane and camera. Compare on actual visible road outside all query actor boxes.',
  fixed=dict(seed=42,cpu_threads=4,size=[1024,576]),stop_rule='No actual hole fill unless both held-out geometry and visible-road image alignment are credible. Preserve gaps and reject wrong-source/structural alignment; no threshold sweep.',
  failure_ledger_refs=['V77-F02'],resources=dict(gpu_forwards=0),human_verdict=None,background_input_dir=None))
 fr,c,k,im,exclude=get(0);n,d,hull,fit,uv,test,err=fit_plane(fr,c,k,exclude);dump(ROOT/'plane_fit.json',fit)
 overlay=Image.fromarray(im);dr=ImageDraw.Draw(overlay);_,poly=road_roi();dr.line([tuple(x) for x in np.vstack([poly,poly[0]])],fill='yellow',width=2)
 for q,is_test,e in zip(uv,test,err):
  x,y=q;dr.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill=('red' if e>.05 else 'cyan' if is_test else 'lime'))
 overlay.save(ROOT/'plane_points.jpg',quality=96)
 if fit['heldout_median_m']>.05 or fit['heldout_q90_m']>.12:
  dump(ROOT/'state.json',dict(state='rejected_ground_geometry',fit=fit,human_verdict=None,background_input_dir=None));print('R25_GROUND_REJECTED',fit,flush=True);return
 yy,xx=np.mgrid[:576,:1024];pix=np.stack([xx,yy,np.ones_like(xx)],-1).reshape(-1,3);reports=[];roi,_=road_roi()
 # Hull path uses measured plane extent, not extrapolation across the whole image.
 for qf in [0,10,20]:
  out=ROOT/f'f{qf:03}';out.mkdir();qfr,qc,qk,query,qexclude=get(qf);rays=pix@np.linalg.inv(qk).T@qc[:3,:3].T;den=rays@n;depth=-(qc[:3,3]@n+d)/np.where(np.abs(den)>1e-8,den,np.nan);world=qc[:3,3]+rays*depth[:,None]
  xy=world[:,:2].astype('float32');inside=np.zeros(len(world),bool);geometric=np.isfinite(depth)&(depth>3)&(depth<50)
  # OpenCV contour test on finite points only; roughly 100k road candidates.
  candidates=np.flatnonzero(geometric&roi.ravel()&~qexclude.ravel());inside[candidates]=[cv2.pointPolygonTest(hull,(float(xy[j,0]),float(xy[j,1])),False)>=0 for j in candidates];evaluation=inside.reshape(HW)
  projections=[];colors=[];valids=[]
  for sf in [qf+5,qf+10,qf+20]:
   assert sf!=qf;_,sc,sk,source,sexclude=get(sf);uv,sz=project(world,sc,sk);mx=uv[:,0].reshape(HW).astype('float32');my=uv[:,1].reshape(HW).astype('float32');valid=evaluation&(sz.reshape(HW)>.2)&np.isfinite(mx)&np.isfinite(my)&(mx>=0)&(mx<1023)&(my>=0)&(my<575)
   ex=cv2.remap(sexclude.astype('uint8'),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=1)>0;valid&=~ex;warped=cv2.remap(source,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);valids.append(valid);colors.append(warped);projections.append(dict(source_frame=sf,valid_pixels=int(valid.sum())))
  valid=np.stack(valids);color=np.stack(colors);nearest=np.zeros_like(query);hit=np.zeros(HW,bool)
  for j in range(3):take=valid[j]&~hit;nearest[take]=color[j][take];hit|=valid[j]
  consensus=np.zeros_like(query);chit=np.zeros(HW,bool)
  for a,b in [(0,1),(0,2),(1,2)]:
   agree=np.max(np.abs(color[a].astype('float32')-color[b]),-1)<=20;take=valid[a]&valid[b]&agree&~chit;consensus[take]=np.rint((color[a][take].astype('float32')+color[b][take])/2).astype('uint8');chit|=take
  q=Image.fromarray(query);dr=ImageDraw.Draw(q);dr.line([tuple(x) for x in np.vstack([poly,poly[0]])],fill='yellow',width=2);q.save(out/'query.jpg',quality=96);rows={};panels=[('query original',query)]
  for name,result,m in [('nearest',nearest,hit),('consensus',consensus,chit)]:
   error=np.abs(result.astype('float32')-query).mean(-1);admitted=error[m];Image.fromarray(result).save(out/f'{name}_rgb.png');Image.fromarray(np.uint8(m)*255).save(out/f'{name}_mask.png');view=query.copy();view[evaluation]=[90,20,110];view[m]=result[m];Image.fromarray(view).save(out/f'{name}_overlay.png');heat=query.copy();heat[m]=np.where((error[m]<=20)[:,None],[0,180,90],[235,40,40]);Image.fromarray(heat).save(out/f'{name}_error.png')
   rows[name]=dict(covered=int(m.sum()),denominator=int(evaluation.sum()),coverage=float(m.sum()/max(1,evaluation.sum())),mean_mae=float(admitted.mean()) if len(admitted) else None,median_mae=float(np.median(admitted)) if len(admitted) else None,q90_mae=float(np.percentile(admitted,90)) if len(admitted) else None,fraction_mae_le20=float(np.mean(admitted<=20)) if len(admitted) else None)
   panels.append((name,view))
  Image.fromarray(np.uint8(evaluation)*255).save(out/'evaluation_mask.png');sheet=Image.new('RGB',(1024,600*3),(10,20,30))
  for j,(name,panel) in enumerate(panels):sheet.paste(Image.fromarray(panel),(0,600*j+24));ImageDraw.Draw(sheet).text((8,600*j+5),f'f{qf} {name}; purple=unsupported',fill='white')
  sheet.save(out/'comparison.jpg',quality=96);report=dict(query_frame=qf,sources=projections,arms=rows);dump(out/'metrics.json',report);reports.append(report)
 dump(ROOT/'state.json',dict(state='visible_road_control_complete_pending_visual_review',seconds=time.time()-beg,plane=fit,queries=reports,human_verdict=None,background_input_dir=None));print('R25_VISIBLE_ROAD_COMPLETE',fit,reports,flush=True)
if __name__=='__main__':main()
