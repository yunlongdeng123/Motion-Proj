"""r35：只把来源目标车的GT地面包络替换成已有SAM2轮廓；保留其余证据限制。"""
import sys,time,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_road_plane_control as p
import hybrid_road_patch_control as rp
from repair_common import dump,hull_mask
from geometry import transform
import numpy as np,cv2
from scipy.spatial import Delaunay
from PIL import Image,ImageDraw
cv2.setNumThreads(4);ROOT=p.BASE/'r35'

def source(f):
 fr,c,k,im,oldexclude,oldground,_=rp.source(f)
 sam=np.array(Image.open(rp.OLD/'sam'/f'core_{f:05}.png').convert('L').resize((1024,576),Image.Resampling.NEAREST))>0
 exclude=cv2.dilate(sam.astype('uint8'),np.ones((7,7),np.uint8))>0
 for b in fr['all_boxes']:
  if str(b['actor_id'])!='12':exclude|=hull_mask(b,c,k,p.HW,pad=3,ground_extend=.5)
 points=transform(np.fromfile(p.DATA/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(p.DATA/'lidar_pose'/f'{f:03}.txt'));uv,z=p.project(points,c,k);ij=np.rint(uv).astype(int);valid=(z>3)&(z<50)&(ij[:,0]>=0)&(ij[:,0]<1024)&(ij[:,1]>=0)&(ij[:,1]<576);ids=np.flatnonzero(valid);ids=ids[~exclude[ij[ids,1],ij[ids,0]]];ids=ids[np.abs(points[ids]@rp.NORMAL+rp.OFFSET)<=.05];ground=np.zeros(p.HW,np.uint8);triangles=0
 if len(ids)>=3:
  world=points[ids];uv=uv[ids];tri=Delaunay(uv).simplices;w=world[tri];u=uv[tri];edges=np.stack([np.linalg.norm(w[:,a]-w[:,b],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1);pixels=np.stack([np.linalg.norm(u[:,a]-u[:,b],axis=1) for a,b in [(0,1),(1,2),(2,0)]],1);keep=(edges.max(1)<=3)&(pixels.max(1)<=64);triangles=int(keep.sum())
  for t in np.rint(u[keep]).astype('int32'):cv2.fillConvexPoly(ground,t,1)
 ground=cv2.erode(ground,np.ones((3,3),np.uint8))>0;ground&=~exclude
 out=ROOT/f'source{f:03}';out.mkdir(exist_ok=True)
 for name,m in [('sam_core',sam),('old_exclude',oldexclude),('new_exclude',exclude),('ground',ground)]:Image.fromarray(np.uint8(m)*255).save(out/f'{name}.png')
 return c,k,im,oldexclude,exclude,ground,dict(frame=f,sam_core_pixels=int(sam.sum()),ground_lidar_points=len(ids),triangles=triangles,ground_pixels=int(ground.sum()))

def main():
 assert not ROOT.exists();ROOT.mkdir();start=time.time()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r35',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',target='12',query_sources={'0':[5,10,20],'15':[20,25]},question='Does replacing the target source exclusion envelope with existing SAM2 recover usable measured road evidence?',intervention='Only target12 source exclusion: existing SAM2 core resized nearest +3px dilation, versus GT envelope pad3/groundextend0.5m. All other actors retain old envelopes; fixed r27 pose/plane/LiDAR5cm/triangle3m64px/erosion unchanged.',selection='r34 pairs with existing saved SAM2 source masks: f35 unavailable and omitted before running; no new propagation. Diagnostic source-pair comparison, not a full-window fill.',roles='Original RGB and processed geometry. Existing SAM2 is prediction, not GT. Exposed pixels may include unmodeled shadow; no automatic compositor admission.',stop_rule='If no actual ground support, stop this mask-only control. Do not relax road geometry or overwrite candidate backgrounds.',resources=dict(cpu_threads=4,gpu_forwards=0),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 rows=[];cache={}
 for q,fs in [(0,[5,10,20]),(15,[20,25])]:
  _,c,k,im,_,_,_=rp.source(q);mask=np.array(Image.open(rp.OLD/'mask'/f'{q:05}.png'))>0;y,x=np.where(mask);rays=np.stack([x,y,np.ones_like(x)],1)@np.linalg.inv(k).T@c[:3,:3].T;den=rays@rp.NORMAL;depth=-(c[:3,3]@rp.NORMAL+rp.OFFSET)/np.where(np.abs(den)>1e-8,den,np.nan);world=c[:3,3]+rays*depth[:,None];eligible=np.isfinite(depth)&(depth>3)&(depth<50)
  sheet=Image.new('RGB',(1200,300*len(fs)),(12,20,30));dr=ImageDraw.Draw(sheet)
  for j,f in enumerate(fs):
   if f not in cache:cache[f]=source(f)
   sc,sk,rgb,oldex,newex,ground,info=cache[f];uv,z=p.project(world,sc,sk);inview=eligible&(z>.2)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<1023)&(uv[:,1]>=0)&(uv[:,1]<575);ids=np.flatnonzero(inview);ij=np.rint(uv[ids]).astype(int);old=oldex[ij[:,1],ij[:,0]];new=newex[ij[:,1],ij[:,0]];hit=ground[ij[:,1],ij[:,0]];released=old&~new
   rows.append(dict(query=q,source=f,query_domain=int(mask.sum()),eligible=int(eligible.sum()),inview=len(ids),old_excluded=int(old.sum()),new_excluded=int(new.sum()),released=int(released.sum()),accepted=int(hit.sum())))
   view=rgb.copy();view[ij[released,1],ij[released,0]]=[0,220,220];oldview=rgb.copy();oldview[oldex]=(.5*oldview[oldex]+.5*np.array([255,65,65])).astype('uint8');newview=rgb.copy();newview[newex]=(.5*newview[newex]+.5*np.array([255,65,65])).astype('uint8');newview[ground]=(.4*newview[ground]+.6*np.array([30,240,90])).astype('uint8')
   for col,(name,img) in enumerate([('old GT exclusion',oldview),('SAM target + measured ground',newview),('cyan = newly released query rays',view)]):sheet.paste(Image.fromarray(img).resize((400,225)),(col*400,j*300+28));dr.text((col*400+4,j*300+5),f'q{q} s{f}: {name}',fill='white')
  sheet.save(ROOT/f'q{q:03}_contact.jpg',quality=97)
 dump(ROOT/'state.json',dict(state='complete_pending_visual_review',seconds=time.time()-start,rows=rows,sources=[v[-1] for v in cache.values()],human_verdict=None,background_input_dir=None));print(rows,flush=True)
if __name__=='__main__':main()
