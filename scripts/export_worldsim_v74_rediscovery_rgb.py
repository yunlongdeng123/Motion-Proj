import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,numpy as np,json,pathlib,tarfile
from PIL import Image
torch.set_num_threads(1)
R=pathlib.Path('/root/autodl-tmp/runs/worldsim_v74_h2/WS-V74-H2-REDISCOVERY-01/20260915-cpu-r1');out=R/'context';out.mkdir(exist_ok=True)
D=pathlib.Path('/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2')
raw=pathlib.Path('/root/autodl-tmp/runs/worldsim_v73/WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01/20260907T161500Z__native-dpt-surround25-dev6-s7301-r3/build_observations.pt')
scenes=torch.load(raw,map_location='cpu',weights_only=False,mmap=True)
records=[]
for entry in json.loads((R/'paired/manifest.json').read_text()):
 c=torch.load(D/entry['actor']['file'],map_location='cpu',weights_only=False);pts=c['points_actor_m'].numpy()
 scene=next(s for s in scenes if s['scene_id']==entry['actor']['scene']);best=None
 for j,idx in enumerate(c['view_indices']):
  idx=int(idx);v=scene['views'][idx];image=v['image'];height,width=image.shape[-2:]
  mat=c['camera_from_actor'][j].numpy();K=c['intrinsics'][j].numpy();cam=pts@mat[:3,:3].T+mat[:3,3];pix=cam@K.T;uv=pix[:,:2]/np.maximum(pix[:,2:3],1e-12)
  valid=(cam[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
  if best is None or int(valid.sum())>best[0]:best=(int(valid.sum()),idx,v,uv[valid],width,height)
 n,idx,v,uv,width,height=best;label=entry['source_selection_method']
 a=v['image'].numpy().transpose(1,2,0);path=out/(label+'.png');Image.fromarray(np.clip(a*255,0,255).astype('uint8')).save(path)
 bbox=[float(uv[:,0].min()),float(uv[:,1].min()),float(uv[:,0].max()),float(uv[:,1].max())] if len(uv) else None
 rec={'case':label,'source':str(raw),'selection':'max projected BUILD point count among original input images; no prediction or QUERY used','image':path.name,'view_index':idx,'projected_build_points':n,'bbox_from_build_points':bbox,'image_hw':[height,width],'view_metadata':{k:v for k,v in v.items() if isinstance(v,(str,int,float,bool,type(None)))}}
 records.append(rec);print(json.dumps(rec),flush=True)
(out/'manifest.json').write_text(json.dumps(records,indent=2))
with tarfile.open(R/'context.tar','w') as t:t.add(out,arcname='context')
