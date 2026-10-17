from pathlib import Path
import sys,json,numpy as np
from dataclasses import asdict
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'backend_restore_r15';sys.path.insert(0,str(R/'implementation_r15_final/shadow_rebuild_v3'))
from shadow_rebuild.legacy.paper_sgcm import SGCMConfig,fit_ground_plane,transform_points,transform_plane
seed=dict(np.load(D/'data/seed.npz'));m=json.loads((D/'data/manifest.json').read_text());p=np.loadtxt(Path(m['source_scene'])/'pose'/(m['frames'][0]['frame_id']+'.txt'));p[:3,3]-=np.array(m['original_world_origin_m']);a=np.linalg.inv(p);bounds=[]
for actor in [5,6,12,18,25,26,37]:
    points=transform_points(seed['source_world'][seed['owner']==actor],a);bounds.append(np.stack((points.min(0),points.max(0))))
cfg=SGCMConfig();plane,fit=fit_ground_plane(transform_points(seed['xyz'][seed['owner']<0],a),bounds,cfg);regions=[]
for b in bounds:
    b=b.copy();b[0,:2]-=.25;b[1,:2]+=.25;xy=np.array([[x,y] for x in b[:,0] for y in b[:,1]]);z=-(xy@plane[:2]+plane[3])/plane[2];b[0,2]=min(b[0,2],z.min())-.25;b[1,2]=max(b[1,2],z.max())+.25;regions.append(b.tolist())
v=dict(config=asdict(cfg),plane_model=transform_plane(plane,p).tolist(),model_to_anchor=a.tolist(),regions_anchor=regions,fit=fit,input='same actual BUILD raw static and source_world Actor seeds; no QUERY or learned geometry',static_only=True,temporal_guard='v=0, lifetime=1e6; ordinal cameras inert')
(D/'sgcm_geometry.json').write_text(json.dumps(v,indent=2));c=json.loads((D/'config.json').read_text());c['sgcm_geometry_path']=str(D/'sgcm_geometry.json');(D/'config.json').write_text(json.dumps(c,indent=2));print(json.dumps(v,indent=2))
