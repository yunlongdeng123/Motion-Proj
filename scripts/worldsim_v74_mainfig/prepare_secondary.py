import json,pathlib,time,torch,numpy as np,importlib.util
from scipy.spatial import cKDTree
P=pathlib.Path('/root/autodl-tmp');R=P/'runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1';S=R/'secondary';S.mkdir(exist_ok=True)
D=P/'runs/worldsim_v73/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2'
cohort=json.loads((R/'cohort.json').read_text())
reg={'task':'WS-V74-MAINFIG-SECONDARY-01','run':'20260915-secondary-r1','registered_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'seed':7403,'trigger':'Four official geometry models show shared first-return crossings; extend bounded method-level references, not training.','cohort':'Same frozen 75 discovery objects / 6 scenes / 5 logs. No new selection by results.','models':{'nksr':'Official ks native field/dual mesh; BUILD LiDAR samples1,3,4,6 plus PCA16 normals; extra geometric information.','dvgt2':'Official checkpoint, causal geometry branch; same RGB six/twelve images; native points in each ego_n frame, GT scale factor0.1. Same declared grid adapter controls.','dggt':'Official nuScenes weights; same six/twelve RGB. Depth-grid diagnostic separated from Gaussian RGB+expected-depth rendering; expected depth is not first return.','noksr':'Official Carla Serial checkpoint if publicly retrievable; same BUILD-only points/normals, native meshing. Engineering blockers recorded, not counted as failure.'},'control':'No QUERY to solver, fixed primary scale protocol. Surface representations and information budgets reported separately.','failure_ledger_refs':['V74-H2-F12','V74-F06'],'failure_ledger_delta':'pending','human_verdict':None}
if not (S/'registration.json').exists():(S/'registration.json').write_text(json.dumps(reg,indent=2))
out=S/'build_inputs';out.mkdir(exist_ok=True);rows=[]
for e in cohort:
 f=out/(e['scene']+'__'+e['owner']+'.npz')
 if not f.exists():
  c=torch.load(D/e['file'],weights_only=False,map_location='cpu');p=np.asarray(c['points_actor_m'],dtype='float32')
  sensors=np.concatenate([np.asarray(x['origins_actor_m']) for x in c['rays'] if x['role']=='build'])
  sensors=np.unique(sensors,axis=0)
  if len(p)>=16 and len(sensors):
   idx=cKDTree(p).query(p,k=16)[1];nb=p[idx];d=nb-nb.mean(1,keepdims=True);_,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',d,d));norm=vec[:,:,0]
   nearest=cKDTree(sensors).query(p)[1];flip=np.sum(norm*(sensors[nearest]-p),axis=1)<0;norm[flip]*=-1
  else:norm=np.zeros_like(p)
  np.savez_compressed(f,points=p,normals=norm,sensors=sensors)
 rows.append(dict(e,input=str(f),eligible_for_pca16=e['build_points']>=16))
(S/'build_manifest.json').write_text(json.dumps(rows,indent=2));print('BUILD_EXPORT',len(rows),sum(x['eligible_for_pca16'] for x in rows),flush=True)
