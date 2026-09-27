"""r36：查看被释放路面附近的真实测量，区分局部地面范围与来源视角限制。"""
import sys,time,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_road_plane_control as p
import hybrid_road_patch_control as rp
from repair_common import dump,camera,hull_mask
from geometry import transform,project_bbox
from video_review import scene_frame
from hybrid_road_height_prefix_control import local_ground
import cv2,numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
cv2.setNumThreads(4);ROOT=p.BASE/'r36'

def main():
 assert not ROOT.exists();ROOT.mkdir();start=time.time()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r36',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_goal_turn='progress: r30–r35 checkpoint fbb8889f, actual28frames and source-mask counterexample',question='Do released source road pixels have real local LiDAR ground samples rejected only by the first-frame plane, and do later cameras show the area?',scene='official_000',target='12',query_sources={'0':[5,10,20],'15':[20,25]},view_inventory={'frames':[0,25,50,75,100],'cameras':list(range(6))},roles='Original RGB, processed LiDAR/poses/GT. Per-source local plane is fitted only to observed road with all objects excluded. Image-nearest LiDAR is a diagnostic association, not true per-pixel depth.',intervention='Read-only measurement/camera inventory; no correction, no generated input and no RGB compositing.',resources=dict(cpu_threads=4,gpu_forwards=0),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 rows=[];cache={}
 for q,frames in [(0,[5,10,20]),(15,[20,25])]:
  _,c,k,query,_,_,_=rp.source(q);mask=np.array(Image.open(rp.OLD/'mask'/f'{q:05}.png'))>0;y,x=np.where(mask);rays=np.stack([x,y,np.ones_like(x)],1)@np.linalg.inv(k).T@c[:3,:3].T;den=rays@rp.NORMAL;depth=-(c[:3,3]@rp.NORMAL+rp.OFFSET)/np.where(np.abs(den)>1e-8,den,np.nan);world=c[:3,3]+rays*depth[:,None];eligible=np.isfinite(depth)&(depth>3)&(depth<50)
  for s in frames:
   fr,sc,sk,rgb,oldex=p.get(s);newex=np.array(Image.open(p.BASE/'r35'/f'source{s:03}/new_exclude.png'))>0;uv,z=p.project(world,sc,sk);valid=eligible&(z>.2)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<1023)&(uv[:,1]>=0)&(uv[:,1]<575);ids=np.flatnonzero(valid);ij=np.rint(uv[ids]).astype(int);released=ids[oldex[ij[:,1],ij[:,0]]&~newex[ij[:,1],ij[:,0]]];ruv=uv[released];assert len(released)>0
   points=transform(np.fromfile(p.DATA/'lidar'/f'{s:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(p.DATA/'lidar_pose'/f'{s:03}.txt'));puv,pz=p.project(points,sc,sk);pi=np.rint(puv).astype(int);good=(pz>3)&(pz<50)&(pi[:,0]>=0)&(pi[:,0]<1024)&(pi[:,1]>=0)&(pi[:,1]<576);piids=np.flatnonzero(good);piids=piids[~newex[pi[piids,1],pi[piids,0]]];g,n,d=local_ground(s);local_res=np.abs(points@n+d);fixed_res=np.abs(points@rp.NORMAL+rp.OFFSET)
   distance,nearest=cKDTree(puv[piids]).query(ruv,workers=4);close=distance<=3;hit=piids[nearest[close]]
   rect=[max(0,int(ruv[:,0].min())-12),max(0,int(ruv[:,1].min())-12),min(1024,int(ruv[:,0].max())+13),min(576,int(ruv[:,1].max())+13)];near=(puv[piids,0]>=rect[0])&(puv[piids,0]<rect[2])&(puv[piids,1]>=rect[1])&(puv[piids,1]<rect[3]);nearids=piids[near]
   row=dict(query=q,source=s,released_pixels=len(released),image_nearest_lidar_within3px=int(close.sum()),unique_nearest_points=len(np.unique(hit)),nearest_fixed_plane_residual_median=float(np.median(fixed_res[hit])) if len(hit) else None,nearest_local_plane_residual_median=float(np.median(local_res[hit])) if len(hit) else None,nearby_points=len(nearids),nearby_fixed_plane_ground=int((fixed_res[nearids]<=.05).sum()),nearby_local_plane_ground=int((local_res[nearids]<=.05).sum()),local_plane_normal=n.tolist(),local_plane_offset=float(d),local_plane_fit_points=len(g),limits='Nearest projected measurements may hit another surface; do not interpret as hidden-road GT. Local plane still extrapolates from measured ROI.');rows.append(row)
   center=ruv.mean(0);left=int(np.clip(center[0]-192,0,640));top=int(np.clip(center[1]-108,0,360));crop=(left,top,left+384,top+216)
   overlay=Image.fromarray(rgb);dr=ImageDraw.Draw(overlay)
   for i in nearids:
    xx,yy=puv[i];color='lime' if local_res[i]<=.05 else 'red';dr.ellipse((xx-1.4,yy-1.4,xx+1.4,yy+1.4),fill=color)
   release=rgb.copy();ri=np.rint(ruv).astype(int);release[ri[:,1],ri[:,0]]=[0,225,240]
   sheet=Image.new('RGB',(1536,314),(12,20,30));dr=ImageDraw.Draw(sheet)
   for col,(name,im) in enumerate([('source original',Image.fromarray(rgb)),('cyan released projection',Image.fromarray(release)),('LiDAR green local ground / red other',overlay)]):sheet.paste(im.crop(crop).resize((512,288)),(col*512,26));dr.text((col*512+5,5),f'q{q} source{s}: {name}',fill='white')
   sheet.save(ROOT/f'q{q:03}_s{s:03}_measurement.jpg',quality=97)
 for f in [0,25,50,75,100]:
  fr=scene_frame(p.SPEC['spec'],f,p.INST);sheet=Image.new('RGB',(1536,628),(12,20,30));dr=ImageDraw.Draw(sheet)
  for cam in range(6):
   c,k=camera(fr,cam,p.HW);im=Image.open(p.DATA/'images'/f'{f:03}_{cam}.jpg').resize((1024,576));d=ImageDraw.Draw(im)
   for b in fr['all_boxes']:
    if b['actor_id'] not in ['12','34']:continue
    rect=project_bbox(b['pose'],b['size_lwh'],c,k,p.HW)
    if rect:d.rectangle(rect,outline='yellow' if b['actor_id']=='12' else 'cyan',width=2);d.text((max(0,rect[0]),max(0,rect[1]-14)),b['actor_id'],fill='yellow')
   col=cam%3;row=cam//3;sheet.paste(im.resize((512,288)),(col*512,row*314+26));dr.text((col*512+5,row*314+5),f'f{f} CAM{cam} original, target12 yellow / keep34 cyan',fill='white')
  sheet.save(ROOT/f'views_f{f:03}.jpg',quality=96)
 dump(ROOT/'state.json',dict(state='complete_pending_visual_review',seconds=time.time()-start,rows=rows,human_verdict=None,background_input_dir=None));print(rows,flush=True)
if __name__=='__main__':main()
