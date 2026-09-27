"""r27：以各来源实际可见的LiDAR地面小三角形代替仅f0中道支撑域。"""
import sys,time,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hybrid_road_plane_control as p
from repair_common import read,dump
from scipy.spatial import Delaunay
from geometry import transform
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=p.BASE/'r27';OLD=p.OLD/'official_000';CACHE={}
FIT=read(p.ROOT/'plane_fit.json');NORMAL=np.array(FIT['normal']);OFFSET=FIT['offset']

def source(f):
 if f in CACHE:return CACHE[f]
 fr,c,k,im,exclude=p.get(f)
 points=transform(np.fromfile(p.DATA/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(p.DATA/'lidar_pose'/f'{f:03}.txt'))
 uv,z=p.project(points,c,k);ij=np.rint(uv).astype(int);valid=(z>3)&(z<50)&(ij[:,0]>=0)&(ij[:,0]<1024)&(ij[:,1]>=0)&(ij[:,1]<576);idx=np.flatnonzero(valid);idx=idx[~exclude[ij[idx,1],ij[idx,0]]]
 residual=np.abs(points[idx]@NORMAL+OFFSET);idx=idx[residual<=.05];world=points[idx];uv=uv[idx];support=np.zeros(p.HW,np.uint8);accepted=[]
 if len(idx)>=3:
  tri=Delaunay(uv).simplices;w=world[tri];u=uv[tri];lengths=np.stack([np.linalg.norm(w[:,a]-w[:,b],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1);pixel_span=np.stack([np.linalg.norm(u[:,a]-u[:,b],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1)
  keep=(lengths.max(1)<=3)&(pixel_span.max(1)<=64)
  for t in np.rint(u[keep]).astype('int32'):cv2.fillConvexPoly(support,t,1)
  accepted=tri[keep]
 support=cv2.erode(support,np.ones((3,3),np.uint8))>0;support&=~exclude
 record=dict(frame=f,ground_lidar_points=len(idx),triangles=len(accepted),ground_rgb_pixels=int(support.sum()),median_plane_residual_m=float(np.median(np.abs(world@NORMAL+OFFSET))) if len(idx) else None)
 result=(fr,c,k,im,exclude,support,record);CACHE[f]=result
 return result

def render(qf,domain):
 _,c,k,im,exclude,_,_=source(qf);yy,xx=np.mgrid[:576,:1024];pixels=np.stack([xx,yy,np.ones_like(xx)],-1).reshape(-1,3);rays=pixels@np.linalg.inv(k).T@c[:3,:3].T;den=rays@NORMAL;depth=-(c[:3,3]@NORMAL+OFFSET)/np.where(np.abs(den)>1e-8,den,np.nan);world=c[:3,3]+rays*depth[:,None]
 valid_domain=domain&np.isfinite(depth.reshape(p.HW))&(depth.reshape(p.HW)>3)&(depth.reshape(p.HW)<50);colors=[];valids=[];counts=[]
 for f in [qf+5,qf+10,qf+20]:
  _,sc,sk,sim,sex,ground,rec=source(f);uv,z=p.project(world,sc,sk);mx=uv[:,0].reshape(p.HW).astype('float32');my=uv[:,1].reshape(p.HW).astype('float32');valid=valid_domain&(z.reshape(p.HW)>.2)&np.isfinite(mx)&np.isfinite(my)&(mx>=0)&(mx<1023)&(my>=0)&(my<575)
  valid&=cv2.remap(ground.astype('uint8'),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0;warped=cv2.remap(sim,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);valids.append(valid);colors.append(warped);counts.append(dict(frame=f,valid=int(valid.sum())))
 nearest=np.zeros_like(im);hit=np.zeros(p.HW,bool);first=np.full(p.HW,-1,np.int16)
 for j,f in enumerate([qf+5,qf+10,qf+20]):take=valids[j]&~hit;nearest[take]=colors[j][take];first[take]=f;hit|=valids[j]
 consensus=np.zeros_like(im);chit=np.zeros(p.HW,bool);pair=np.full((*p.HW,2),-1,np.int16);frames=[qf+5,qf+10,qf+20]
 for a,b in [(0,1),(0,2),(1,2)]:
  agree=np.max(np.abs(colors[a].astype('float32')-colors[b]),-1)<=20;take=valids[a]&valids[b]&agree&~chit;consensus[take]=np.rint((colors[a][take].astype('float32')+colors[b][take])/2).astype('uint8');pair[take]=[frames[a],frames[b]];chit|=take
 return im,nearest,hit,consensus,chit,first,pair,counts

def save(out,im,domain,near,hit,con,chit,first,pair):
 out.mkdir(parents=True)
 for name,arr in [('original',im),('mask',domain),('nearest_rgb',near),('nearest_mask',hit),('consensus_rgb',con),('consensus_mask',chit)]:Image.fromarray(np.uint8(arr)*255 if arr.dtype==bool else arr).save(out/f'{name}.png')
 np.savez_compressed(out/'provenance.npz',nearest_source_frame=first,consensus_source_frames=pair)
 for name,color,m in [('nearest',near,hit),('consensus',con,chit)]:
  view=im.copy();view[domain]=[110,20,140];view[m]=color[m];Image.fromarray(view).save(out/f'{name}_diagnostic.png')

def main():
 assert not ROOT.exists();ROOT.mkdir();beg=time.time()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r27',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',actor='12',frames=list(range(30)),camera=0,
  question='Can real measured source road patches cover the right-lane DELETE hole, beyond r25 center-lane support?',
  intervention='Keep r25 plane, camera, source offsets and RGB rules. Replace f0 training convex hull with locally interpolated LiDAR ground patches measured at each source timestamp.',
  fixed=dict(plane_source='r25 f0 held-out validated',source_offsets=[5,10,20],depth_range_m=[3,50],ground_distance_m=.05,triangle_max_world_edge_m=3,triangle_max_image_edge_px=64,source_mask_erosion_px=1,rgb_consensus_max_channel_error=20,size=[1024,576],cpu_threads=4),
  source_roles='Only original RGB and actual source LiDAR/GT camera/boxes. Every source differs from query timestamp. Query visible RGB is evaluation only. No generated RGB or mask-erased image as factual evidence.',
  source_support='2D Delaunay over projected source ground LiDAR points, keep triangles below fixed spatial/image edge limits, erode1px, exclude all GT box ground-extended envelopes. This is POC interpolation, not exact semantic/visibility GT.',
  limits='Local ground patches may still miss static occlusion, thin objects, target shadows or road non-planarity. No extrapolation to trees/buildings/other cars. RGB consensus is not proof of query visibility.',
  positive_control='Queries0,10,20 same r25 visible evaluation masks, run before actual hole and compare observed query RGB. Target-hole RGB has no GT.',
  stop_rule='One source-support control; preserve residual hole and fail visual structure errors. No threshold/seed sweep or downstream Omega until backgrounds reviewed.',resources=dict(gpu_forwards=0),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 positives=[]
 for f in [0,10,20]:
  domain=np.array(Image.open(p.ROOT/f'f{f:03}/evaluation_mask.png'))>0;im,near,hit,con,chit,first,pair,counts=render(f,domain);save(ROOT/'visible'/f'f{f:03}',im,domain,near,hit,con,chit,first,pair);row=dict(frame=f,denominator=int(domain.sum()),sources=counts,arms={})
  for name,res,m in [('nearest',near,hit),('consensus',con,chit)]:
   err=np.abs(res.astype('float32')-im).mean(-1)[m];row['arms'][name]=dict(covered=int(m.sum()),coverage=float(m.sum()/max(1,domain.sum())),mean_mae=float(err.mean()) if len(err) else None,q90_mae=float(np.percentile(err,90)) if len(err) else None)
  positives.append(row)
 dump(ROOT/'visible_validation.json',positives)
 rows=[]
 for f in range(30):
  domain=np.array(Image.open(OLD/'mask'/f'{f:05}.png').convert('L'))>0;assert domain.shape==p.HW;im,near,hit,con,chit,first,pair,counts=render(f,domain);save(ROOT/f'f{f:03}',im,domain,near,hit,con,chit,first,pair)
  assert np.all(first[hit]!=f) and np.all(pair[chit]!=f) and not np.any(hit&~domain) and not np.any(chit&~domain)
  row=dict(frame=f,mask_pixels=int(domain.sum()),nearest_pixels=int(hit.sum()),consensus_pixels=int(chit.sum()),sources=counts);dump(ROOT/f'f{f:03}/metrics.json',row);rows.append(row)
 source_records=[v[-1] for k,v in sorted(CACHE.items())];dump(ROOT/'source_patch_metrics.json',source_records)
 for f in [5,15,25]:
  _,c,k,im,ex,ground,_=source(f);v=im.copy();v[ground]=(.45*v[ground]+.55*np.array([30,210,170])).astype('uint8');Image.fromarray(v).save(ROOT/f'source_{f:03}_ground.jpg',quality=95)
 state=dict(state='measured_road_patches_complete_pending_visual_review',seconds=time.time()-beg,frames=30,total_mask_pixels=sum(r['mask_pixels'] for r in rows),nearest=sum(r['nearest_pixels'] for r in rows),consensus=sum(r['consensus_pixels'] for r in rows),visible_control=positives,rows=rows,human_verdict=None,background_input_dir=None);dump(ROOT/'state.json',state);print({k:v for k,v in state.items() if k not in ['rows','visible_control']},flush=True)
 sheet=Image.new('RGB',(1152,260*6),(10,20,30))
 for j,f in enumerate([0,5,10,15,20,29]):
  domain=np.array(Image.open(ROOT/f'f{f:03}/mask.png'))>0;y,x=np.where(domain);cx=(x.min()+x.max())/2;cy=(y.min()+y.max())/2;left=int(np.clip(cx-192,0,640));top=int(np.clip(cy-108,0,360))
  for col,key in enumerate(['original','nearest_diagnostic','consensus_diagnostic']):
   a=Image.open(ROOT/f'f{f:03}/{key}.png').crop((left,top,left+384,top+216));sheet.paste(a,(col*384,j*260+24));ImageDraw.Draw(sheet).text((col*384+5,j*260+5),f'f{f} {key}',fill='white')
 sheet.save(ROOT/'contact.jpg',quality=96)
if __name__=='__main__':main()
