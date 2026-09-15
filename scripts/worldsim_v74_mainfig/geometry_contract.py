import json,pathlib,numpy as np
P=pathlib.Path('/root/autodl-tmp');R=P/'runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1'
def affine(v,hw):
 h,w=hw;ow,oh=v['original_wh'];return np.array([[w/ow,0,(w/ow-1)/2],[0,h/oh,(h/oh-1)/2],[0,0,1]])
def outputs(path):
 rec=json.loads((path/'result.json').read_text());m=json.loads(pathlib.Path(rec['input']).read_text());views=m['views'][:rec['n_images']]
 a=np.load(path/'native_outputs.npz');h,w=rec['network_hw'];cam=[]
 kg=np.array([affine(v,(h,w))@np.array(v['intrinsics_original']) for v in views]);method=rec['method']
 yy,xx=np.mgrid[:h,:w];pix=np.stack([xx,yy,np.ones_like(xx)],-1)
 base=1.;scale_cv=None
 if method=='dvgt1':
  pts=a['points'][0].reshape(-1,h,w,3);conf=a['points_conf'][0].reshape(-1,h,w)
  rdf=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
  for p,v in zip(pts,views):
   T=np.linalg.inv(v['world_from_camera'])@np.array(views[0]['world_from_ego_camera'])
   cam.append((p/.1)@rdf.T@T[:3,:3].T+T[:3,3])
 elif method in ['vggt','omega512']:
  E=a['extrinsics'][0];K=a['intrinsics'][0];dep=a['depth'][0,...,0];conf=a['depth_conf'][0]
  centers=np.array([-e[:,:3].T@e[:,3] for e in E]);true=np.array([v['world_from_camera'] for v in views])[:,:3,3];rat=[]
  for i in range(len(views)):
   for j in range(i):
    d=np.linalg.norm(centers[i]-centers[j]);g=np.linalg.norm(true[i]-true[j])
    if d>.01 and g>.1:rat.append(g/d)
  base=float(np.median(rat)) if rat else 1.;scale_cv=float(np.std(rat)/np.mean(rat)) if rat else None
  for d,k in zip(dep,K):cam.append((pix@np.linalg.inv(k).T)*d[...,None]*base)
 else:
  cam=list(a['local_points'][0]);conf=a['conf'][0]
  if conf.ndim==4:conf=conf[...,0]
 return rec,views,np.asarray(cam),np.asarray(conf),kg,base,scale_cv
def build_control(path,scene):
 rec,views,cam,conf,K,base,cv=outputs(path);h,w=rec['network_hw'];ratios=[];checks=[]
 for i,v in enumerate(views[:6]):
  s=scene['views'][v['source_view_index']];uv=np.asarray(s['uv']);z=np.asarray(s['z_m']);mask=np.asarray(s['diagnostic_mask']).astype(bool)&(np.asarray(s['owners'])=='')
  sh,sw=s['image'].shape[-2:];ow,oh=v['original_wh'];xy=np.rint(uv*[w/sw,h/sh]+[(w/ow-1)/2,(h/oh-1)/2]).astype(int);valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)&mask&(z>1)&(z<80)
  xy=xy[valid];z=z[valid];pred=cam[i,xy[:,1],xy[:,0],2];ok=np.isfinite(pred)&(pred>.2)
  ratios.extend((z[ok]/pred[ok]).tolist())
  # 模型点图本身的像素投影偏差；用于暴露坐标/内参差异，不用于删例。
  yy,xx=np.mgrid[0:h:16,0:w:16]
  cp=cam[i,yy,xx];pr=cp@K[i].T;uvp=pr[...,:2]/pr[...,2:3]
  err=np.linalg.norm(uvp-np.stack([xx,yy],-1),axis=-1);good=np.isfinite(err)&(cp[...,2]>.2)
  checks.append({'camera':v['camera'],'build_background_n':int(ok.sum()),'median_z_ratio':float(np.median(z[ok]/pred[ok])) if ok.any() else None,'positive_depth_fraction':float((cam[i,...,2]>.2).mean()),'median_projected_pixel_error':float(np.median(err[good])) if good.any() else None})
 scale=float(np.median(ratios)) if ratios else None
 return {'method':rec['method'],'scene':rec['scene'],'variant':rec['variant'],'base_scale':base,'camera_baseline_scale_cv':cv,'build_global_scalar':scale,'anchor_n':len(ratios),'anchor_mask':'BUILD sample3, diagnostic_mask AND owner empty, depth 1..80m','views':checks}
if __name__=='__main__':
 import torch
 torch.set_num_threads(1)
 raw=P/'runs/worldsim_v73/WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01/20260907T161500Z__native-dpt-surround25-dev6-s7301-r3/build_observations.pt'
 scenes=torch.load(raw,weights_only=False,map_location='cpu',mmap=True);rows=[]
 for f in sorted((R/'predictions').glob('*/*/*/result.json')):
  rec=json.loads(f.read_text());s=next(x for x in scenes if x['scene_id']==rec['scene']);row=build_control(f.parent,s);rows.append(row)
  print(json.dumps(row),flush=True)
 (R/'build_contract.json').write_text(json.dumps(rows,indent=2))
