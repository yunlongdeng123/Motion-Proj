"""r14：同组来源、同几何，只拆逐像素LiDAR邻近门控；查询RGB仅评价。"""
import sys,time
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera,hull_mask,VIDEO
from geometry import unproject,box_mask
from video_review import scene_frame
from r3_visibility import segment_box_occlusion
from hybrid_neighbor_evidence import project
from scipy.spatial import cKDTree
from PIL import ImageDraw
cv2.setNumThreads(4)
BASE=ROOT.parent;R2=ROOT;ROOT=BASE/'r14';CFG=read(ROOT/'registration.json');spec=next(s for s in read(R2/'registration.json')['scenes'] if s['name']=='scene_0255');data=Path(spec['data']);inst=read(data/'instances/instances_info.json')

def sources():
 records=[];pools=[]
 for r in CFG['sources']:
  f,cam=r['frame'],r['camera'];fr=scene_frame(spec['spec'],f,inst);b=next(b for b in fr['all_boxes'] if b['actor_id']=='52');fd=VIDEO/'scene_0255/frames'/f'{f:03}';met=read(fd/'metrics.json')
  with np.load(met['prediction_path']) as z:d=z['depth'][0,cam,...,0]
  d=d*read(fd/'alignment.json')['calibrated_global_depth_scale']*r['scale'];c,k=camera(fr,cam,d.shape);world=unproject(d,k,c);grad=np.maximum(np.abs(np.gradient(d,axis=0)),np.abs(np.gradient(d,axis=1)))
  sm=cv2.resize(mask(ROOT/f'{f:03}_{cam}/sam.png').astype('uint8'),(d.shape[1],d.shape[0]),interpolation=cv2.INTER_NEAREST)>0
  inside=box_mask(world.reshape(-1,3),np.array(b['pose']),np.array(b['size_lwh'])+.1).reshape(d.shape)
  valid=sm&inside&(d>1)&(d<60)&np.isfinite(d)&(grad<.3);yy,xx=np.where(valid);pts=world[valid];visible=np.ones(len(pts),bool)
  for other in fr['all_boxes']:
   if other['actor_id']=='52':continue
   ix=np.flatnonzero(visible)
   if len(ix):visible[ix]&=~segment_box_occlusion(c[:3,3],pts[ix],np.array(other['pose']),np.array(other['size_lwh'])+.1)
  yy,xx,pts=yy[visible],xx[visible],pts[visible]
  lidar=transform(np.fromfile(data/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(data/'lidar_pose'/f'{f:03}.txt'));la=lidar[box_mask(lidar,np.array(b['pose']),np.array(b['size_lwh'])+.1)];dist,_=cKDTree(la).query(pts,workers=4)
  im=np.array(Image.open(data/'images'/f'{f:03}_{cam}.jpg').convert('RGB').resize((d.shape[1],d.shape[0]),Image.Resampling.BILINEAR));local=transform(pts,np.linalg.inv(np.array(b['pose']))).astype('float32');col=im[yy,xx]
  pools.append(dict(local=local,rgb=col,uv=np.stack([xx,yy],-1).astype('int16'),strict=dist<=.2,distance=dist.astype('float32')))
  records.append(dict(**r,dense_points=len(pts),strict_points=int((dist<=.2).sum()),prediction_path=met['prediction_path'],depth_hw=list(d.shape)))
 dump(ROOT/'source_index.json',records);np.savez_compressed(ROOT/'sources.npz',**{f'{i}_{k}':v for i,p in enumerate(pools) for k,v in p.items()})
 return pools,records

def reconstruct(pools,records,fr,cam,exclude_frame,arm,delete_target=False):
 c,k=camera(fr,cam,HW);b=next(b for b in fr['all_boxes'] if b['actor_id']=='52');pose=np.array(b['pose']);roi=hull_mask(b,c,k,HW,pad=4)
 zbest=np.full(H*W,np.inf);color=np.zeros((H*W,3),np.uint8);sid=np.full(H*W,-1,np.int16);pid=np.full(H*W,-1,np.int32);projected=[]
 for j,source in enumerate(pools):
  if records[j]['frame']==exclude_frame:projected.append((np.array([],int),np.array([],float),np.array([],int),np.empty((0,3),np.uint8)));continue
  ids=np.flatnonzero(source['strict']) if arm=='strict' else np.arange(len(source['local']));s={key:source[key][ids] for key in ['local','rgb']};ix,z,pp=project(s,pose,c,k,roi);pp=ids[pp]
  world=transform(source['local'][pp],pose);visible=np.ones(len(ix),bool)
  for other in fr['all_boxes']:
   if other['actor_id']=='52' or (delete_target and other['actor_id']=='25'):continue
   ii=np.flatnonzero(visible)
   if len(ii):visible[ii]&=~segment_box_occlusion(c[:3,3],world[ii],np.array(other['pose']),np.array(other['size_lwh']))
  ix,z,pp=ix[visible],z[visible],pp[visible];col=source['rgb'][pp];projected.append((ix,z,pp,col));better=z<zbest[ix];use=ix[better];zbest[use]=z[better];color[use]=col[better];sid[use]=j;pid[use]=pp[better]
 second=np.full(H*W,-1,np.int16);sourceframes=np.array([r['frame'] for r in records]+[-10000]);sourcef=sourceframes[sid]
 for j,(ix,z,pp,col) in enumerate(projected):
  good=(np.abs(z-zbest[ix])<=.25)&(np.max(np.abs(col.astype('int16')-color[ix].astype('int16')),axis=-1)<=25)&(np.abs(records[j]['frame']-sourcef[ix])>=5)
  second[ix[good]]=j
 single=(sid>=0).reshape(HW);agree=(second>=0).reshape(HW);accepted=cv2.erode(agree.astype('uint8'),np.ones((3,3),np.uint8))>0
 return color.reshape(H,W,3),accepted,dict(source_id=sid.reshape(HW),source_point=pid.reshape(HW),second_source=second.reshape(HW),z=zbest.reshape(HW),single=single,agreement=agree)

def main():
 assert read(ROOT/'mask_state.json')['state']=='complete';assert not (ROOT/'visible_summary.json').exists();started=time.time();pools,index=sources();rows=[];sheet=Image.new('RGB',(640*4,360*len(index)),(12,22,30))
 for row,r in enumerate(index):
  f,cam=r['frame'],r['camera'];out=ROOT/f'{f:03}_{cam}';fr=scene_frame(spec['spec'],f,inst)
  # 此处mask与RGB只定义评价域和计算误差，不参与reconstruct的投影或一致性。
  truth=rgb(out/'rgb.png');sm=mask(out/'sam.png');evalmask=cv2.erode(sm.astype('uint8'),np.ones((3,3),np.uint8))>0;rec=dict(frame=f,camera=cam,eval_pixels=int(evalmask.sum()),excluded_source_frame=f,arms={})
  b=next(b for b in fr['all_boxes'] if b['actor_id']=='52');c,k=camera(fr,cam,HW);box=project_bbox(b['pose'],b['size_lwh'],c,k,HW);box=[max(0,int(box[0])-20),max(0,int(box[1])-20),min(W,int(box[2])+20),min(H,int(box[3])+25)];images=[('query truth / evaluation only',truth)]
  for arm in ['strict','dense']:
   col,accepted,p=reconstruct(pools,index,fr,cam,f,arm);hit=accepted&evalmask;err=np.abs(col.astype('int16')-truth.astype('int16')).mean(-1)
   rec['arms'][arm]=dict(single_support_pixels=int((p['single']&evalmask).sum()),two_source_pixels=int((p['agreement']&evalmask).sum()),accepted_pixels=int(hit.sum()),coverage=float(hit.sum()/max(1,evalmask.sum())),median_mae=float(np.median(err[hit])) if hit.any() else None,q95_mae=float(np.quantile(err[hit],.95)) if hit.any() else None,mae_le_20_pixels=int((hit&(err<=20)).sum()))
   draw=truth.copy();draw[sm]=[70,45,80];draw[sm&accepted]=col[sm&accepted];Image.fromarray(draw).save(out/f'{arm}_visible.png');write_mask(out/f'{arm}_accepted.png',accepted);np.savez_compressed(out/f'{arm}_provenance.npz',**p,rgb=col)
   images.append((f'{arm} warp / purple unknown',draw))
   if arm=='dense':
    errmap=np.zeros_like(truth);errmap[hit&(err<=20)]=[25,230,100];errmap[hit&(err>20)]=[245,65,60];errmap[sm&~accepted]=[70,45,80];images.append(('dense errors: green<=20 red>20',errmap))
  for j,(title,im) in enumerate(images):
   a=Image.fromarray(im).crop(box).resize((640,360));ImageDraw.Draw(a).text((8,8),f'f{f} CAM{cam} | {title}',fill='yellow');sheet.paste(a,(j*640,row*360))
  rows.append(rec);print(rec,flush=True)
 sheet.save(ROOT/'visible_control.jpg',quality=96)
 dump(ROOT/'visible_summary.json',dict(queries=rows,elapsed_s=time.time()-started,source_views=len(index),source_dense_points=sum(r['dense_points'] for r in index),source_strict_points=sum(r['strict_points'] for r in index),scope='Known-visible original RGB only. Same-time donors excluded; query masks/RGB used only in evaluation. No hidden surface admission.',human_verdict=None))

if __name__=='__main__':main()
