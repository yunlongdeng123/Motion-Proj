"""r12内部细拆：点在哪里丢失，以及原Ω深度和actor52的LiDAR是否一致。"""
import sys
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import VIDEO,camera,hull_mask
from geometry import unproject,box_mask
from video_review import scene_frame
from r3_visibility import segment_box_occlusion
from scipy.spatial import cKDTree
from PIL import ImageDraw
BASE=ROOT.parent;OUT=BASE/'r12/scene_0255';spec=next(s for s in read(ROOT/'registration.json')['scenes'] if s['name']=='scene_0255');data=Path(spec['data']);inst=read(data/'instances/instances_info.json')
cv2.setNumThreads(4)
rows=[];sheet=Image.new('RGB',(960*2,536*4),(15,20,25));picks=[(100,1),(115,1),(125,1),(135,3)]
for n,(f,cam) in enumerate(picks):
 fr=scene_frame(spec['spec'],f,inst);b=next(b for b in fr['all_boxes'] if b['actor_id']=='52');fd=VIDEO/'scene_0255/frames'/f'{f:03}';met=read(fd/'metrics.json')
 with np.load(met['prediction_path']) as z:d=z['depth'][0,cam,...,0]
 d=d*read(fd/'alignment.json')['calibrated_global_depth_scale'];hw=d.shape;c,k=camera(fr,cam,hw);p=unproject(d,k,c)
 hull=hull_mask(b,c,k,hw);grad=np.maximum(np.abs(np.gradient(d,axis=0)),np.abs(np.gradient(d,axis=1)));inbox=box_mask(p.reshape(-1,3),np.array(b['pose']),np.array(b['size_lwh'])+.1).reshape(hw)
 selected=inbox&(d>1)&(d<60)&np.isfinite(d)&(grad<.3);flat=np.flatnonzero(selected);points=p.reshape(-1,3)[flat];visible=np.ones(len(points),bool)
 for other in fr['all_boxes']:
  if other['actor_id']=='52':continue
  ix=np.flatnonzero(visible)
  if len(ix):visible[ix]&=~segment_box_occlusion(c[:3,3],points[ix],np.array(other['pose']),np.array(other['size_lwh'])+.1)
 lidar=transform(np.fromfile(data/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(data/'lidar_pose'/f'{f:03}.txt'))
 la=lidar[box_mask(lidar,np.array(b['pose']),np.array(b['size_lwh'])+.1)];dist,_=cKDTree(la).query(points[visible],workers=4)
 lc=transform(la,np.linalg.inv(c));q=lc@k.T;uv=np.rint(q[:,:2]/np.maximum(q[:,2:],1e-6)).astype(int);inside=(lc[:,2]>.5)&(uv[:,0]>=0)&(uv[:,0]<hw[1])&(uv[:,1]>=0)&(uv[:,1]<hw[0]);ok=inside.copy()
 for other in fr['all_boxes']:
  if other['actor_id']=='52':continue
  ix=np.flatnonzero(ok)
  if len(ix):ok[ix]&=~segment_box_occlusion(c[:3,3],la[ix],np.array(other['pose']),np.array(other['size_lwh'])+.1)
 use=np.flatnonzero(ok);err=d[uv[use,1],uv[use,0]]-lc[use,2];rat=d[uv[use,1],uv[use,0]]/lc[use,2]
 rec=dict(frame=f,camera=cam,depth_hw=list(hw),gt_hull_pixels=int(hull.sum()),omega_inside_actor_box=int(inbox.sum()),after_depth_gradient=int(selected.sum()),after_other_actor_visibility=int(visible.sum()),after_lidar20cm=int((dist<=.2).sum()),actor_lidar_points=len(la),visible_lidar_samples=len(use),
          omega_minus_lidar_z_quantiles_m=np.quantile(err,[.1,.5,.9]).tolist() if len(err) else None,omega_over_lidar_z_quantiles=np.quantile(rat,[.1,.5,.9]).tolist() if len(err) else None)
 rows.append(rec)
 im=rgb(data/'images'/f'{f:03}_{cam}.jpg');a=Image.fromarray(im);draw=ImageDraw.Draw(a)
 for j in use:
  x=uv[j,0]*W/hw[1];y=uv[j,1]*H/hw[0];draw.ellipse((x-1,y-1,x+1,y+1),fill='cyan')
 colors=im.copy();m=cv2.resize(selected.astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST)>0;colors[m]=[255,160,30];bim=Image.fromarray(colors)
 for j,t in enumerate([a,bim]):ImageDraw.Draw(t).text((8,8),f'f{f} CAM{cam} '+['actor52 raw LiDAR / cyan','Omega points inside box / orange'][j],fill='yellow');sheet.paste(t,(j*W,n*H))
 print(rec,flush=True)
sheet.resize((1280,1429)).save(OUT/'depth_attrition.jpg',quality=96);dump(OUT/'depth_attrition.json',dict(rows=rows,scope='Sparse LiDAR plus GT-box visibility diagnostic; static occlusion and exact surface visibility not certified; no threshold changes',human_verdict=None))
