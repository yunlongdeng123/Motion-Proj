"""AV2 单轮模型前表面筛查；固定日志、窗口和格子，最多两表面/日志供视觉审核。"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw,ImageFont
from motion_proj.worldsim_v81.geometry import transform,apply,project,remove_boxes,plane_fit,coverage,roi_mask,reference_filter
sys.path.insert(0,str(Path(__file__).parent))
from build_worldsim_v81_atlas import texture
O=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01');A=O/'av2_atlas';A.mkdir(exist_ok=True)
for d in ['reference_geometry','input_manifests','crops']:(A/d).mkdir(exist_ok=True)
def T(r):return transform({'rotation':[r[k] for k in ['qw','qx','qy','qz']],'translation':[r[k] for k in ['tx_m','ty_m','tz_m']]})
def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(a@b),0,1))))
allrows=[];chosen=[];windows=[]
for source in json.loads((O/'new_source_acquisition.json').read_text()):
 if source['status']!='READY':continue
 log=source['log'];root=Path(source['root']);pose=pd.read_feather(root/'city_SE3_egovehicle.feather').set_index('timestamp_ns');pt=np.array(pose.index);ann=pd.read_feather(root/'annotations.feather');intr=pd.read_feather(root/'calibration/intrinsics.feather').set_index('sensor_name');extr=pd.read_feather(root/'calibration/egovehicle_SE3_sensor.feather').set_index('sensor_name');t=source['target_lidar_timestamp_ns']
 def pose_at(ts):
  near=int(pt[np.argmin(abs(pt-ts))]);assert abs(near-ts)<2000000,(ts,near)
  return T(pose.loc[near])
 def boxes(ts):
  out=[];E=pose_at(ts)
  for _,r in ann[ann.timestamp_ns==ts].iterrows():
   B=E@T(r);out.append((B,np.array([r.length_m,r.width_m,r.height_m])/2+.35))
  return out
 def strip(world,bs):
  keep=np.ones(len(world),bool)
  for B,half in bs:keep &= ~np.all(abs(apply(world,np.linalg.inv(B)))<=half,axis=1)
  return world[keep]
 target_boxes=boxes(t);clouds={}
 for key in source['lidar_keys']:
  ts=int(Path(key).stem);df=pd.read_feather(root/'sensors/lidar'/Path(key).name);clouds[ts]=strip(strip(apply(df[['x','y','z']].values,pose_at(ts)),boxes(ts)),target_boxes)
 prompt=clouds[t];other=[ts for ts in sorted(clouds) if ts!=t];world=np.concatenate([clouds[ts] for ts in other]);src=np.concatenate([np.full(len(clouds[ts]),i) for i,ts in enumerate(other)])
 manifest={'window_id':'av2_'+log[:8],'log':log,'scene':log,'target_sample':str(t),'role':'DISCOVERY','dataset':'AV2','views':[],'context_views':[],'input_lidar_world':str(A/f'{log}_input_world.npy'),'heldout_input_disjoint':True};np.save(A/f'{log}_input_world.npy',prompt)
 for camera,keys in source['camera_keys'].items():
  r=intr.loc[camera];K=np.array([[r.fx_px,0,r.cx_px],[0,r.fy_px,r.cy_px],[0,0,1]])
  for i,key in enumerate(keys):
   ts=int(Path(key).stem);E=pose_at(ts);v={'camera':camera,'sample_token':str(t) if i==1 else str(t+(-500000000 if i==0 else 500000000)),'timestamp_us':ts//1000,'world_from_camera':(E@T(extr.loc[camera])).tolist(),'world_from_ego_camera':E.tolist(),'K':K.tolist(),'size':[int(r.width_px),int(r.height_px)],'image':str(root/'sensors/cameras'/camera/Path(key).name)}
   manifest['views' if i==1 else 'context_views'].append(v)
  v=manifest['views'][-1];C=np.array(v['world_from_camera']);W,H=v['size'];uv,z,xyz=project(world,C,K);valid=reference_filter(uv,z,src,W,H);pu,pz,px=project(prompt,C,K);ok=roi_mask(pu,pz,[0,0,W,H]);tree=cKDTree(pu[ok]);dist,nn=tree.query(uv[valid]);valid=valid[~((dist<10)&(z[valid]>pz[ok][nn]+.5+.01*z[valid]))];maskvalid=np.zeros(len(world),bool);maskvalid[valid]=True;rgb=np.array(Image.open(v['image']).convert('RGB'))
  for yi in range(3):
   for xi in range(5):
    box=np.array([xi*W//5,(yi+1)*H//5,(xi+1)*W//5,(yi+2)*H//5])+[32,32,-32,-32];inside=roi_mask(uv,z,box)&maskvalid;pts=xyz[inside];fit=plane_fit(pts);rr={'roi_id':f'av2_{log[:8]}_{camera}_{yi}{xi}','window_id':manifest['window_id'],'log':log,'scene':log,'camera':camera,'box':box.tolist(),'points':len(pts),'scans':len(np.unique(src[inside])),'coverage':coverage(uv[inside],box),'model_status':'NOT_RUN','human_verdict':None,'selection_role':'MODEL_BLIND_DISCOVERY'}
    if not fit:rr['reference_pass']=False;allrows.append(rr);continue
    rr.update(rms_m=fit['rms'],inlier_fraction=fit['inlier_fraction'],ground=bool(abs((C[:3,:3]@fit['normal'])[2])>.9));angles=[]
    for s in np.unique(src[inside]):
     f=plane_fit(pts[src[inside]!=s])
     if f:angles.append(angle(f['normal'],fit['normal']))
    spread=max(angles) if len(angles)==rr['scans'] and rr['scans']>=2 else None;rr.update(leave_scan_normal_spread_deg=spread,reference_pass=bool(len(pts)>=30 and rr['scans']>=2 and rr['coverage']>=.25 and fit['rms']<=.12 and fit['inlier_fraction']>=.85 and spread is not None and spread<=5))
    allrows.append(rr)
    if not rr['reference_pass'] or rr['ground']:continue
    crop=rgb[box[1]:box[3],box[0]:box[2]];rr.update(texture(crop));np.savez_compressed(A/'reference_geometry'/f'{rr["roi_id"]}.npz',uv=uv[inside],xyz=pts,depth_z=z[inside],source_scan=src[inside],K=K,world_from_camera=C,normal=fit['normal'],center=fit['center'],world=world[inside],input_sample=np.array(str(t)),reference_samples=np.array(other,dtype=str));Image.fromarray(crop).save(A/'crops'/f'{rr["roi_id"]}.jpg')
 (A/'input_manifests'/f'{manifest["window_id"]}.json').write_text(json.dumps(manifest,indent=2));windows.append(manifest['window_id']);pool=[r for r in allrows if r['log']==log and r['reference_pass'] and not r['ground'] and r.get('dark_fraction',1)<.2 and r.get('saturated_fraction',1)<.2];pool=sorted(pool,key=lambda r:(r['gradient_energy'],r['gray_entropy'],r['roi_id']))[:2];chosen.extend(pool)
 print(json.dumps({'log':log,'screened':105,'planar_non_ground':len(pool),'selected':[r['roi_id'] for r in pool]}),flush=True)
 (A/'registry.json').write_text(json.dumps(allrows,indent=2));(A/'selected_before_model.json').write_text(json.dumps(chosen,indent=2))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
for start in range(0,len(chosen),8):
 group=chosen[start:start+8];canvas=Image.new('RGB',(1200,((len(group)+1)//2)*250),'white');draw=ImageDraw.Draw(canvas)
 for j,r in enumerate(group):
  x=j%2*600;y=j//2*250;im=Image.open(A/'crops'/f'{r["roi_id"]}.jpg');im.thumbnail((580,200));canvas.paste(im,(x,y+45));draw.text((x+3,y+3),f'{start+j}: {r["roi_id"]}\nn={r["points"]} scans={r["scans"]} rms={r["rms_m"]:.3f}',font=font,fill='black')
 canvas.save(O/'figures'/f'av2_model_blind_{start//8}.png')
(A/'summary.json').write_text(json.dumps({'logs':len(windows),'roi_grid_cells':len(allrows),'reference_pass':sum(r['reference_pass'] for r in allrows),'frozen_visual_review_candidates':len(chosen),'model_inferences':0,'selection':'at most 2 lowest-gradient non-ground geometric passes per frozen log; no replacement after visual exclusion'},indent=2))
