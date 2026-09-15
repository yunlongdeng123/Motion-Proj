import pathlib,json,torch,numpy as np,shutil,tarfile
from geometry_contract import P,R
torch.set_num_threads(1)
D=P/'runs/worldsim_v73/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2'
raw=P/'runs/worldsim_v73/WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01/20260907T161500Z__native-dpt-surround25-dev6-s7301-r3/build_observations.pt'
scenes=torch.load(raw,weights_only=False,map_location='cpu',mmap=True);cohort=json.loads((R/'cohort.json').read_text());out=R/'official_case_assets';out.mkdir(exist_ok=True)
chosen=[]
for log in sorted(set(x['log_id'] for x in cohort)):
 es=[x for x in cohort if x['log_id']==log and x.get('translation_speed_mps') is not None and x['translation_speed_mps']<=.5]
 entry=max(es,key=lambda x:x['build_points']);chosen.append(entry)
white=next(x for x in cohort if x['owner']=='204704542f8642dc8ab046ffbd70e0c5');chosen.append(white)
manifest=[]
for e in chosen:
 c=torch.load(D/e['file'],weights_only=False,map_location='cpu');s=next(x for x in scenes if x['scene_id']==e['scene']);vlist=json.loads((R/'inputs'/f'{e["scene"]}.json').read_text())['views'];b=np.asarray(c['points_actor_m']);best=None
 for v in vlist[:6]:
  idx=v['source_view_index']
  if idx not in list(map(int,c['view_indices'])):continue
  j=list(map(int,c['view_indices'])).index(idx);T=np.asarray(c['camera_from_actor'][j]);K=np.array(v['intrinsics_original']);cam=b@T[:3,:3].T+T[:3,3];q=cam@K.T;uv=q[:,:2]/q[:,2:3];w,h=v['original_wh'];valid=(cam[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
  if best is None or valid.sum()>best[0]:best=(int(valid.sum()),v,T,K,uv[valid])
 if best is None:
  manifest.append({'actor':e,'status':'NO_CASE_CAMERA_POSE_AT_SAMPLE3','selection':'largest metadata BUILD count; retained missing-input outcome'});print('NO_INPUT',e['scene'],e['owner'],flush=True);continue
 n,v,T,K,uv=best;dest=out/e['owner'];dest.mkdir(exist_ok=True);shutil.copy2(v['image'],dest/'rgb.jpg');np.savez_compressed(dest/'projection.npz',camera_from_actor=T,K=K,build=b,size=np.asarray(c['size_lwh_m']))
 bbox=[float(uv[:,0].min()),float(uv[:,1].min()),float(uv[:,0].max()),float(uv[:,1].max())] if len(uv) else None
 rec={'actor':e,'selection':'largest BUILD point count among metadata near-static actors per log; legacy white car additionally retained','camera':v,'visible_build_points':n,'bbox':bbox,'role':'DISCOVERY illustration; not independent confirmation'};manifest.append(rec)
 for model in ['dvgt1','vggt','pi3x','omega512']:
  for variant in ['six','twelve']:
   src=R/'evaluation_bgscale'/model/e['scene']/variant/e['owner'];dd=dest/model/variant;dd.mkdir(parents=True,exist_ok=True)
   for name in ['primary_surface.npz','cal_build_1.0_rays.npz']:
    if (src/name).exists():shutil.copy2(src/name,dd/name)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(R/'official_case_assets.tar','w') as t:t.add(out,arcname='official_case_assets')
print(json.dumps({'cases':[(e['scene'],e['owner'],e['heldout_actor_returns']) for e in chosen],'output':str(R/'official_case_assets.tar')}))
