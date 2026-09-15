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
 if method in ['dvgt1','dvgt2']:
  pts=a['points'][0].reshape(-1,h,w,3);conf=a['points_conf'][0].reshape(-1,h,w)
  rdf=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
  for i,(p,v) in enumerate(zip(pts,views)):
   ego=views[0] if method=='dvgt1' else views[(i//6)*6]
   T=np.linalg.inv(v['world_from_camera'])@np.array(ego['world_from_ego_camera'])
   cam.append((p/.1)@rdf.T@T[:3,:3].T+T[:3,3])
 elif method in ['vggt','omega512','dggt']:
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
