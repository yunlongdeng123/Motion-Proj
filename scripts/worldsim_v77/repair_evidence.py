"""从原始RGB与冻结Ω深度前向warp。LiDAR与双来源检查仅作为保守证据准入。"""
import time
from PIL import Image
from scipy.spatial import cKDTree
from repair_common import *
from r3_visibility import segment_box_occlusion
cv2.setNumThreads(4)
HW=(576,1024);H,W=HW

def grounded(b):
 p=np.array(b['pose']);s=np.array(b['size_lwh']);p[2,3]-=.3;s[2]+=.6;s[:2]+=.2
 return p,s

def source_points(s,frames,out):
 sources=[];stats=[]
 for f in s['donor_frames']:
  fr=frames[str(f)];fd=VIDEO/s['name']/'frames'/f'{f:03}';metrics=read(fd/'metrics.json');cached=pathlib.Path(metrics['prediction_path'])
  with np.load(cached) as p:depth=p['depth'][0,...,0]
  scale=read(fd/'alignment.json')['calibrated_global_depth_scale'];depth=depth*scale
  data=pathlib.Path(s['data']);lid=transform(np.fromfile(data/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(data/'lidar_pose'/f'{f:03}.txt'))
  boxes=[grounded(b) for b in fr['all_boxes']];bg=np.ones(len(lid),bool)
  for p,z in boxes:bg &= ~box_mask(lid,p,z)
  tree=cKDTree(lid[bg]);target=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor']);center=np.array(target['pose'])[:3,3]
  for c in range(6):
   cam,k=camera(fr,c,depth.shape[1:]);d=depth[c];points=unproject(d,k,cam);near=np.linalg.norm(points[...,:2]-center[:2],axis=-1)<28
   grad=np.maximum(np.abs(np.gradient(d,axis=0)),np.abs(np.gradient(d,axis=1)))
   valid=near&(d>1)&(d<60)&np.isfinite(d)&(grad<.3)
   yy,xx=np.where(valid);pt=points[valid]
   if not len(pt):continue
   # 光线不能穿过目标/其他对象贴地包络；深度点不能落在对象内。
   keep=np.ones(len(pt),bool)
   for p,z in boxes:
    ids=np.flatnonzero(keep)
    if len(ids):keep[ids]&=~(segment_box_occlusion(cam[:3,3],pt[ids],p,z)|box_mask(pt[ids],p,z))
   yy,xx,pt=yy[keep],xx[keep],pt[keep]
   if not len(pt):continue
   distance,_=tree.query(pt,workers=4);keep=distance<=.20;yy,xx,pt,dist=yy[keep],xx[keep],pt[keep],distance[keep]
   if not len(pt):continue
   rgb=np.array(Image.open(data/'images'/f'{f:03}_{c}.jpg').convert('RGB').resize((d.shape[1],d.shape[0]),Image.Resampling.BILINEAR))
   sources.append({'frame':f,'camera':c,'points':pt.astype('float32'),'rgb':rgb[yy,xx],'uv':np.stack([xx,yy],1).astype('int16'),'lidar_distance':dist.astype('float32')})
   stats.append({'frame':f,'camera':c,'points':len(pt),'raw_omega_path':str(cached),'depth_scale':scale})
  print(s['name'],'donor',f,'views',len(sources),flush=True)
 dump(out/'donor_index.json',stats)
 np.savez_compressed(out/'donors.npz',**{f'{i}_{k}':v for i,r in enumerate(sources) for k,v in r.items() if isinstance(v,np.ndarray)})
 return sources

def project_source(src,cam,k,mask):
 cp=transform(src['points'],np.linalg.inv(cam));q=cp@k.T;uv=np.rint(q[:,:2]/np.maximum(q[:,2:],1e-9)).astype(int)
 valid=(cp[:,2]>.5)&(uv[:,0]>=1)&(uv[:,0]<W-1)&(uv[:,1]>=1)&(uv[:,1]<H-1)
 pids=np.flatnonzero(valid);indices=[];zs=[];ids=[]
 for dx,dy in [(x,y) for x in [-1,0,1] for y in [-1,0,1]]:
  x=uv[pids,0]+dx;y=uv[pids,1]+dy;take=mask[y,x];indices.append(y[take]*W+x[take]);zs.append(cp[pids[take],2]);ids.append(pids[take])
 index=np.concatenate(indices);z=np.concatenate(zs);ids=np.concatenate(ids)
 if not len(index):return index,z,ids
 order=np.argsort(z,kind='stable');_,first=np.unique(index[order],return_index=True);sel=order[first]
 return index[sel],z[sel],ids[sel]

def main():
 reg=read(ROOT/'registration.json');allrows=[];started=time.monotonic()
 for s in reg['scenes']:
  out=ROOT/s['name'];frames=read(out/'frames.json');sources=source_points(s,frames,out);rows=[]
  for folder in ['evidence','residual','evidence_mask','provenance']:(out/folder).mkdir(exist_ok=False)
  for i,f in enumerate(s['source_frames']):
   fr=frames[str(f)];cam,k=camera(fr,s['camera']);mask=cv2.imread(str(out/'mask'/f'{i:05}.png'),0)>0
   # 每来源取最近点；选全来源最近候选后，再验证另一独立时刻颜色和深度。
   zbest=np.full(H*W,np.inf);color=np.zeros((H*W,3),np.uint8);sourceid=np.full(H*W,-1,np.int16);pointid=np.full(H*W,-1,np.int32)
   projected=[]
   for j,src in enumerate(sources):
    idx,z,pid=project_source(src,cam,k,mask);projected.append((idx,z,pid));better=z<zbest[idx];ix=idx[better];pp=pid[better]
    zbest[ix]=z[better];color[ix]=src['rgb'][pp];sourceid[ix]=j;pointid[ix]=pp
   other=np.full(H*W,-1,np.int16);opoint=np.full(H*W,-1,np.int32)
   fs=np.array([src['frame'] for src in sources]+[-1000]);basef=fs[sourceid]
   for j,(src,(idx,z,pid)) in enumerate(zip(sources,projected)):
    agreement=(np.abs(z-zbest[idx])<=.25)&(np.max(np.abs(src['rgb'][pid].astype('int16')-color[idx].astype('int16')),axis=1)<=25)&(np.abs(src['frame']-basef[idx])>=5)
    ix=idx[agreement];other[ix]=j;opoint[ix]=pid[agreement]
   accepted=(other>=0).reshape(HW)&mask
   # 点状/一像素偶合不当连续背景；只留下可3x3腐蚀的区域，保留真实RGB不平滑。
   accepted=cv2.erode(accepted.astype('uint8'),np.ones((3,3),np.uint8))>0
   rgb=np.array(Image.open(out/'rgb'/f'{i:05}.png'));filled=np.where(accepted[...,None],color.reshape(H,W,3),rgb);residual=mask&~accepted
   Image.fromarray(filled).save(out/'evidence'/f'{i:05}.png');cv2.imwrite(str(out/'residual'/f'{i:05}.png'),residual.astype('uint8')*255);cv2.imwrite(str(out/'evidence_mask'/f'{i:05}.png'),accepted.astype('uint8')*255)
   flat=np.flatnonzero(accepted);np.savez_compressed(out/'provenance'/f'{i:05}.npz',target_index=flat,source_id=sourceid[flat],source_point=pointid[flat],second_source=other[flat],second_point=opoint[flat],target_z=zbest[flat])
   core=cv2.imread(str(out/'sam'/f'core_{i:05}.png'),0)>0
   row={'frame':f,'mask_pixels':int(mask.sum()),'evidence_pixels':int(accepted.sum()),'core_pixels':int(core.sum()),'core_evidence_pixels':int((accepted&core).sum()),'residual_pixels':int(residual.sum()),'outside_mask_change_max':int(np.abs(filled.astype('int16')-rgb.astype('int16'))[~mask].max())};rows.append(row)
  dump(out/'evidence_stats.json',rows);record={'scene':s['name'],'source_views':len(sources),'mask_pixels':sum(r['mask_pixels'] for r in rows),'evidence_pixels':sum(r['evidence_pixels'] for r in rows),'core_pixels':sum(r['core_pixels'] for r in rows),'core_evidence_pixels':sum(r['core_evidence_pixels'] for r in rows)};allrows.append(record);print(record,flush=True)
 dump(ROOT/'evidence_summary.json',{'scenes':allrows,'elapsed_s':time.monotonic()-started,'policy':'Ω原RGB深度+GT相机+背景LiDAR20cm支持；贴地包络下延60cm；3x3投影splat；两来源相隔>=0.5s，深度差<=25cm、RGB最大差<=25/255，再腐蚀1px；推断的可观测支持，不是隐藏背景GT。','human_verdict':None})
if __name__=='__main__':main()
