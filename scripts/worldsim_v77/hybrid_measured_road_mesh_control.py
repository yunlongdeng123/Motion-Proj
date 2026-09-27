"""r37：来源局部实测地面小三角形的透视纹理投影，不用查询首帧平面反求深度。"""
import sys,time,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_road_plane_control as p
from hybrid_road_height_prefix_control import local_ground
from repair_common import dump,read,hull_mask
from geometry import transform
from scipy.spatial import Delaunay
import numpy as np,cv2
from PIL import Image,ImageDraw
cv2.setNumThreads(4);ROOT=p.BASE/'r37';CACHE={};OLD=p.OLD/'official_000'

def source(f):
 if f in CACHE:return CACHE[f]
 fr,c,k,im,exclude=p.get(f);sam=np.array(Image.open(OLD/'sam'/f'core_{f:05}.png').convert('L'))>0;assert sam.shape==p.HW;exclude=cv2.dilate(sam.astype('uint8'),np.ones((7,7),np.uint8))>0
 for b in fr['all_boxes']:
  if str(b['actor_id'])!='12':exclude|=hull_mask(b,c,k,p.HW,pad=3,ground_extend=.5)
 points=transform(np.fromfile(p.DATA/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(p.DATA/'lidar_pose'/f'{f:03}.txt'));uv,z=p.project(points,c,k);ij=np.rint(uv).astype(int);valid=(z>3)&(z<50)&(ij[:,0]>=0)&(ij[:,0]<1024)&(ij[:,1]>=0)&(ij[:,1]<576);ids=np.flatnonzero(valid);ids=ids[~exclude[ij[ids,1],ij[ids,0]]];g,n,d=local_ground(f);ids=ids[np.abs(points[ids]@n+d)<=.05];world=points[ids];uv=uv[ids];tri=Delaunay(uv).simplices;w=world[tri];u=uv[tri];edge=np.stack([np.linalg.norm(w[:,a]-w[:,b],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1);span=np.stack([np.linalg.norm(u[:,a]-u[:,b],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1);tri=tri[(edge.max(1)<=3)&(span.max(1)<=64)];support=np.zeros(p.HW,np.uint8)
 for t in np.rint(uv[tri]).astype('int32'):cv2.fillConvexPoly(support,t,1)
 support=cv2.erode(support,np.ones((3,3),np.uint8))>0;support&=~exclude
 info=dict(frame=f,source_points=len(ids),triangles=len(tri),supported_pixels=int(support.sum()),plane_normal=n.tolist(),plane_offset=float(d),plane_fit_points=len(g));CACHE[f]=(world,tri,c,k,im,support,info);return CACHE[f]

def project_pair(q,s,domain):
 _,qc,qk,query,_=p.get(q);world,tri,sc,sk,im,support,info=source(s);cp=transform(world,np.linalg.inv(qc));sh=transform(world,np.linalg.inv(sc))@sk.T;qh=cp@qk.T;uv=qh[:,:2]/np.maximum(qh[:,2:],1e-8);mx=np.full(p.HW,-1,np.float32);my=mx.copy();depth=np.full(p.HW,np.inf,np.float32);which=np.full(p.HW,-1,np.int32)
 for ti,ids in enumerate(tri):
  z=cp[ids,2]
  if z.min()<=.2:continue
  v=uv[ids];left=max(0,int(np.ceil(v[:,0].min())));right=min(1023,int(np.floor(v[:,0].max())));top=max(0,int(np.ceil(v[:,1].min())));bottom=min(575,int(np.floor(v[:,1].max())))
  if left>right or top>bottom:continue
  local_domain=domain[top:bottom+1,left:right+1]
  if not local_domain.any():continue
  mat=np.vstack([v.T,np.ones(3)])
  if abs(np.linalg.det(mat))<1e-8:continue
  yy,xx=np.mgrid[top:bottom+1,left:right+1];pixels=np.stack([xx.ravel(),yy.ravel(),np.ones(xx.size)]);lam=np.linalg.solve(mat,pixels);inside=(lam>=-1e-6).all(0)&local_domain.ravel();den=(lam/z[:,None]).sum(0);zz=1/np.maximum(den,1e-9);inside&=(zz>3)&(zz<50);coords=sh[ids].T@np.linalg.solve(qh[ids].T,pixels);uu=coords[0]/np.maximum(coords[2],1e-9);vv=coords[1]/np.maximum(coords[2],1e-9);inside&=(uu>=0)&(uu<1023)&(vv>=0)&(vv<575)
  ui=np.clip(np.rint(uu).astype(int),0,1023);vi=np.clip(np.rint(vv).astype(int),0,575);inside&=support[vi,ui];ys=yy.ravel()[inside];xs=xx.ravel()[inside];zs=zz[inside];take=zs<depth[ys,xs];ys,xs,zs=ys[take],xs[take],zs[take];mx[ys,xs]=uu[inside][take];my[ys,xs]=vv[inside][take];depth[ys,xs]=zs;which[ys,xs]=ti
 hit=which>=0;rgb=cv2.remap(im,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);return rgb,hit,depth,mx,my,which,info

def run_query(q,mode,domain):
 _,_,_,original,_=p.get(q);frames=[q+5,q+10,q+15] if q else [5,10,20];outs=[];nearest=np.zeros_like(original);hit=np.zeros(p.HW,bool);first=np.full(p.HW,-1,np.int16);details=[];dest=ROOT/mode/f'f{q:03}';dest.mkdir(parents=True)
 for s in frames:
  rgb,m,z,mx,my,tri,info=project_pair(q,s,domain);outs.append((rgb,m));take=m&~hit;nearest[take]=rgb[take];first[take]=s;hit|=m;details.append(dict(source=s,covered=int(m.sum())));np.savez_compressed(dest/f'source{s:03}_map.npz',mx=mx,my=my,depth=z,triangle=tri)
 con=np.zeros_like(original);cm=np.zeros(p.HW,bool);pair=np.full((*p.HW,2),-1,np.int16)
 for a,b in [(0,1),(0,2),(1,2)]:
  agree=np.max(np.abs(outs[a][0].astype('float32')-outs[b][0]),-1)<=20;take=outs[a][1]&outs[b][1]&agree&~cm;con[take]=np.rint((outs[a][0][take].astype('float32')+outs[b][0][take])/2).astype('uint8');cm|=take;pair[take]=[frames[a],frames[b]]
 np.savez_compressed(dest/'provenance.npz',nearest_source=first,consensus_sources=pair)
 metrics=dict(query=q,mode=mode,denominator=int(domain.sum()),sources=details,arms={})
 for name,rgb,m in [('nearest',nearest,hit),('consensus',con,cm)]:
  Image.fromarray(rgb).save(dest/f'{name}_rgb.png');Image.fromarray(np.uint8(m)*255).save(dest/f'{name}_mask.png');view=original.copy();view[domain]=[110,20,140];view[m]=rgb[m];Image.fromarray(view).save(dest/f'{name}_diagnostic.png');rec=dict(covered=int(m.sum()),coverage=float(m.sum()/max(1,domain.sum())))
  if mode=='visible':
   err=np.abs(rgb.astype('float32')-original).mean(-1);oldrgb=np.array(Image.open(p.ROOT/f'f{q:03}/nearest_rgb.png'));oldmask=np.array(Image.open(p.ROOT/f'f{q:03}/nearest_mask.png'))>0;shared=m&oldmask;old_error=np.abs(oldrgb.astype('float32')-original).mean(-1);rec.update(mean_mae=float(err[m].mean()) if m.any() else None,shared_pixels=int(shared.sum()),r25_mae_shared=float(old_error[shared].mean()) if shared.any() else None,measured_mesh_mae_shared=float(err[shared].mean()) if shared.any() else None)
  metrics['arms'][name]=rec
 Image.fromarray(original).save(dest/'original.png');Image.fromarray(np.uint8(domain)*255).save(dest/'mask.png');dump(dest/'metrics.json',metrics);return metrics

def main():
 assert not ROOT.exists();ROOT.mkdir();start=time.time()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r37',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',target='12',queries=[0,10],sources={'0':[5,10,20],'10':[15,20,25]},question='Does source-local measured road geometry restore usable RGB support when a common first-frame plane rejects actual ground?',intervention='Change geometric representation: fit each source local plane only for classifying measured ground LiDAR; project actual measured triangles into query with perspective-correct original RGB and z-buffer. No ray intersection with query/f0 plane and no camera z correction.',fixed='Local RANSAC from r36; source3–50m,5cm inlier,triangle3m/64px,1px source erosion,existing SAM2+3px target exclusion,otherGT envelopes; no thresholds relaxed. Other query actors excluded from road fill.',roles='All RGB, points and camera/GT from original processed data. Query visible RGB evaluation only. Triangle texture is interpolated original RGB, not exact hidden surface GT. GT/LiDAR-assisted POC.',primary_sources=['https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html','https://docs.opencv.org/3.4.8/d9/dab/tutorial_homography.html'],migration='Use known camera projection and local planar perspective texture; restrict interpolation to measured small triangles, not full-scene homography.',stop_rule='Evaluate visible controls before interpreting actual hole. Reject structural mismatch or adverse shared-pixel errors; retain positive/negative artifacts without automatic compositor admission.',resources=dict(cpu_threads=4,gpu_forwards=0),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 rows=[]
 for mode in ['visible','hole']:
  for q in [0,10]:
   if mode=='visible':domain=np.array(Image.open(p.ROOT/f'f{q:03}/evaluation_mask.png'))>0
   else:
    domain=np.array(Image.open(OLD/'mask'/f'{q:05}.png'))>0;fr,c,k,_,_=p.get(q)
    for b in fr['all_boxes']:
     if str(b['actor_id'])!='12':domain&=~hull_mask(b,c,k,p.HW,pad=3,ground_extend=.5)
   rows.append(run_query(q,mode,domain))
 sheet=Image.new('RGB',(1536,314*4),(12,20,30));dr=ImageDraw.Draw(sheet)
 for j,row in enumerate(rows):
  d=ROOT/row['mode']/f'f{row["query"]:03}';mask=np.array(Image.open(d/'mask.png'))>0
  if row['mode']=='hole':y,x=np.where(mask);left=int(np.clip((x.min()+x.max())/2-192,0,640));top=int(np.clip((y.min()+y.max())/2-108,0,360));box=(left,top,left+384,top+216)
  else:box=(0,0,1024,576)
  for col,key in enumerate(['original','nearest_diagnostic','consensus_diagnostic']):sheet.paste(Image.open(d/f'{key}.png').crop(box).resize((512,288)),(col*512,j*314+26));dr.text((col*512+5,j*314+5),f'{row["mode"]} f{row["query"]} {key}',fill='white')
 sheet.save(ROOT/'contact.jpg',quality=97);dump(ROOT/'state.json',dict(state='complete_pending_visual_review',seconds=time.time()-start,queries=rows,sources=[v[-1] for v in CACHE.values()],human_verdict=None,background_input_dir=None));print(rows,flush=True)
if __name__=='__main__':main()
