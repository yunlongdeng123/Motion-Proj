"""r38：用更晚后视原图的实测路面，检验短前视来源之外的真实背景覆盖。"""
import sys,time,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/work')
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_measured_road_mesh_control as mesh
import hybrid_road_plane_control as p
from repair_common import dump,camera,hull_mask
from video_review import scene_frame
from geometry import transform
from scipy.spatial import Delaunay
import numpy as np,cv2
from PIL import Image,ImageDraw
cv2.setNumThreads(4);ROOT=p.BASE/'r38';CACHE={};POLY=np.array([[.22,.96],[.96,.96],[.80,.58],[.52,.45],[.40,.58]])*np.array([1024,576])

def source(f):
 if f in CACHE:return CACHE[f]
 fr=scene_frame(p.SPEC['spec'],f,p.INST);c,k=camera(fr,5,p.HW);im=np.array(Image.open(p.DATA/'images'/f'{f:03}_5.jpg').convert('RGB').resize((1024,576),Image.Resampling.BILINEAR));exclude=np.zeros(p.HW,bool)
 for b in fr['all_boxes']:exclude|=hull_mask(b,c,k,p.HW,pad=3,ground_extend=.5)
 points=transform(np.fromfile(p.DATA/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(p.DATA/'lidar_pose'/f'{f:03}.txt'));uv,z=p.project(points,c,k);ij=np.rint(uv).astype(int);valid=(z>3)&(z<50)&(ij[:,0]>=0)&(ij[:,0]<1024)&(ij[:,1]>=0)&(ij[:,1]<576);ids=np.flatnonzero(valid);ids=ids[~exclude[ij[ids,1],ij[ids,0]]];roi=np.zeros(p.HW,np.uint8);cv2.fillPoly(roi,[np.rint(POLY).astype('int32')],1);trainids=ids[roi[ij[ids,1],ij[ids,0]]>0];trainids=trainids[(points[trainids,2]-c[2,3]>-3)&(points[trainids,2]-c[2,3]<-.3)];q=points[trainids];assert len(q)>75;hold=np.arange(len(q))%3==0;fit=q[~hold];rng=np.random.default_rng(42);best=np.zeros(len(fit),bool)
 for _ in range(300):
  a=fit[rng.choice(len(fit),3,False)];n=np.cross(a[1]-a[0],a[2]-a[0]);norm=np.linalg.norm(n)
  if norm<1e-8:continue
  n/=norm
  if abs(n[2])<np.cos(np.deg2rad(15)):continue
  m=np.abs(fit@n-n@a[0])<.05
  if m.sum()>best.sum():best=m
 assert best.sum()>=50;g=fit[best];mid=g.mean(0);_,_,v=np.linalg.svd(g-mid,full_matrices=False);n=v[-1];n*=np.sign(n[2]);d=-n@mid;error=np.abs(q[hold]@n+d);record=dict(frame=f,camera=5,ground_roi=POLY.tolist(),plane_normal=n.tolist(),plane_offset=float(d),fit_points=len(g),heldout_points=int(hold.sum()),heldout_median_m=float(np.median(error)),heldout_q90_m=float(np.percentile(error,90)));dump(ROOT/f'source{f:03}/plane_fit.json',record)
 assert np.median(error)<.05 and np.percentile(error,90)<.12,record
 ids=ids[np.abs(points[ids]@n+d)<=.05];world=points[ids];uv=uv[ids];tri=Delaunay(uv).simplices;w=world[tri];u=uv[tri];edge=np.stack([np.linalg.norm(w[:,a]-w[:,b],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1);span=np.stack([np.linalg.norm(u[:,a]-u[:,b],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1);tri=tri[(edge.max(1)<=3)&(span.max(1)<=64)];support=np.zeros(p.HW,np.uint8)
 for t in np.rint(uv[tri]).astype('int32'):cv2.fillConvexPoly(support,t,1)
 support=cv2.erode(support,np.ones((3,3),np.uint8))>0;support&=~exclude;record.update(source_points=len(ids),triangles=len(tri),supported_pixels=int(support.sum()));view=im.copy();view[support]=(.5*view[support]+.5*np.array([0,225,150])).astype('uint8');v=Image.fromarray(view);ImageDraw.Draw(v).line([tuple(x) for x in np.vstack([POLY,POLY[0]])],fill='yellow',width=2);v.save(ROOT/f'source{f:03}/support.jpg',quality=96);Image.fromarray(im).save(ROOT/f'source{f:03}/original.jpg',quality=96);CACHE[f]=(world,tri,c,k,im,support,record);return CACHE[f]

def main():
 assert not ROOT.exists();ROOT.mkdir();start=time.time();mesh.source=source
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r38',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',target='12',queries=[0,10],sources=[{'frame':50,'camera':5},{'frame':75,'camera':5}],question='Do later real rear-camera measurements cover the deleted actor area that nearby forward views cannot?',intervention='Change original source views to rear50/75 selected from r36 original-image inventory. Same measured triangle projection and original camera poses; source visible ground plane fitted in predeclared rear-view ROI.',fixed='RANSAC300 seed42,5cm inlier,normal15deg,heldout everythird median5cm/q90=12cm; geometry3–50m,triangles3m64px,source1pxerosion,allsourceGT envelopes,querynon-targetGT excluded; RGB pair maxchannel20.',roles='Future original RGB/LiDAR/poses/GT allowed as offline reconstruction input; no query RGB used as source, no generated evidence. Rear ROI assistant-selected before computation, not automatic road semantics. Single source support does not certify hidden appearance.',stop_rule='If visible source geometry or RGB alignment fails, reject direct writeback. Report partial coverage and uncompensated lighting honestly; no generation/geometry threshold grid.',resources=dict(cpu_threads=4,gpu_forwards=0),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 rows=[]
 for mode in ['visible','hole']:
  for q in [0,10]:
   fr,c,k,original,_=p.get(q)
   if mode=='visible':domain=np.array(Image.open(p.ROOT/f'f{q:03}/evaluation_mask.png'))>0
   else:
    domain=np.array(Image.open(mesh.OLD/'mask'/f'{q:05}.png'))>0
    for b in fr['all_boxes']:
     if str(b['actor_id'])!='12':domain&=~hull_mask(b,c,k,p.HW,pad=3,ground_extend=.5)
   out=ROOT/mode/f'f{q:03}';out.mkdir(parents=True);outs=[];first=np.full(p.HW,-1,np.int16);nearest=np.zeros_like(original);hit=np.zeros(p.HW,bool);detail=[]
   for s in [50,75]:
    rgb,m,z,mx,my,tri,info=mesh.project_pair(q,s,domain);outs.append((rgb,m));take=m&~hit;nearest[take]=rgb[take];first[take]=s;hit|=m;detail.append(dict(frame=s,camera=5,covered=int(m.sum())));np.savez_compressed(out/f'source{s:03}_map.npz',mx=mx,my=my,depth=z,triangle=tri);Image.fromarray(rgb).save(out/f'source{s:03}_rgb.png');Image.fromarray(np.uint8(m)*255).save(out/f'source{s:03}_mask.png')
   agree=np.max(np.abs(outs[0][0].astype('float32')-outs[1][0]),-1)<=20;cm=outs[0][1]&outs[1][1]&agree;con=np.zeros_like(original);con[cm]=np.rint((outs[0][0][cm].astype('float32')+outs[1][0][cm])/2).astype('uint8');row=dict(query=q,mode=mode,denominator=int(domain.sum()),sources=detail,arms={})
   np.savez_compressed(out/'provenance.npz',nearest_source_frame=first,nearest_source_camera=np.where(hit,5,-1).astype(np.int8),consensus_source_frames=np.where(cm[...,None],np.array([50,75]),-1).astype(np.int16))
   for name,rgb,m in [('nearest',nearest,hit),('consensus',con,cm)]:
    Image.fromarray(rgb).save(out/f'{name}_rgb.png');Image.fromarray(np.uint8(m)*255).save(out/f'{name}_mask.png');view=original.copy();view[domain]=[110,20,140];view[m]=rgb[m];Image.fromarray(view).save(out/f'{name}_diagnostic.png');rec=dict(covered=int(m.sum()),coverage=float(m.sum()/max(1,domain.sum())))
    if mode=='visible':
     err=np.abs(rgb.astype('float32')-original).mean(-1);oldrgb=np.array(Image.open(p.ROOT/f'f{q:03}/nearest_rgb.png'));oldmask=np.array(Image.open(p.ROOT/f'f{q:03}/nearest_mask.png'))>0;shared=m&oldmask;olderr=np.abs(oldrgb.astype('float32')-original).mean(-1);rec.update(mean_mae=float(err[m].mean()) if m.any() else None,shared_pixels=int(shared.sum()),r25_mae_shared=float(olderr[shared].mean()) if shared.any() else None,rear_mae_shared=float(err[shared].mean()) if shared.any() else None)
    row['arms'][name]=rec
   Image.fromarray(original).save(out/'original.png');Image.fromarray(np.uint8(domain)*255).save(out/'mask.png');dump(out/'metrics.json',row);rows.append(row)
 sheet=Image.new('RGB',(1536,1256),(12,20,30));dr=ImageDraw.Draw(sheet)
 for j,row in enumerate(rows):
  d=ROOT/row['mode']/f'f{row["query"]:03}';m=np.array(Image.open(d/'mask.png'))>0
  if row['mode']=='hole':y,x=np.where(m);left=int(np.clip((x.min()+x.max())/2-192,0,640));top=int(np.clip((y.min()+y.max())/2-108,0,360));box=(left,top,left+384,top+216)
  else:box=(0,0,1024,576)
  for col,key in enumerate(['original','nearest_diagnostic','consensus_diagnostic']):sheet.paste(Image.open(d/f'{key}.png').crop(box).resize((512,288)),(col*512,j*314+26));dr.text((col*512+5,j*314+5),f'{row["mode"]} f{row["query"]} {key}',fill='white')
 sheet.save(ROOT/'contact.jpg',quality=97);dump(ROOT/'state.json',dict(state='complete_pending_visual_review',seconds=time.time()-start,queries=rows,sources=[v[-1] for v in CACHE.values()],human_verdict=None,background_input_dir=None));print(rows,flush=True)
if __name__=='__main__':main()
