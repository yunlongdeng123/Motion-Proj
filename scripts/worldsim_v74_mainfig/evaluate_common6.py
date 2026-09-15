"""额外观测与更多输出表面解耦：12图推理，固定读取前6图。明确为发现后的普通控制。"""
import pathlib,json,time,numpy as np,torch
from scipy.spatial import cKDTree
from geometry_contract import R,P,outputs,build_control
from evaluate_surfaces import D,RAW,query,mesh_for_case,measure
torch.set_num_threads(2)
scenes=torch.load(RAW,weights_only=False,map_location='cpu',mmap=True);cohort=json.loads((R/'cohort.json').read_text())
reg=R/'common6_registration.json'
if not reg.exists():reg.write_text(json.dumps({'registered_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'role':'POSTHOC_ORDINARY_CONTROL','motivation':'Full 12-map readout changes both evidence and surface count. Fix output maps to common first six to isolate prediction update.','input':'Existing frozen 12-image model outputs, no new forward','scale':'Same common BUILD sample3 background rule as main evaluation; all six anchors','selection':'All 6 scenes/5logs/75 actors, no case-specific tuning','mesh':'Same primary calibrated-depth grid, discontinuity and crop rules'},indent=2))
for f in sorted((R/'predictions').glob('*/*/twelve/result.json')):
 rec=json.loads(f.read_text());out=R/'evaluation_common6'/rec['method']/rec['scene']/'twelve_common6';out.mkdir(parents=True,exist_ok=True)
 if (out/'summary.json').exists():continue
 s=next(x for x in scenes if x['scene_id']==rec['scene']);ctrl=build_control(f.parent,s);rec,views,cam,conf,K,base,cv=outputs(f.parent);views=views[:6];cam=cam[:6];conf=conf[:6];K=K[:6];rows=[]
 for e in [x for x in cohort if x['scene']==rec['scene']]:
  c=torch.load(D/e['file'],weights_only=False,map_location='cpu');O,d,rg,off,frame=query(c);speed=e.get('translation_speed_mps');static=(speed<=.5) if speed is not None else None
  if len(rg)==0:rows.append({'owner':e['owner'],'status':'NO_OWNED_QUERY','static':static,'rays':0});continue
  target=O+d*rg[:,None];b=np.asarray(c['points_actor_m']);support=cKDTree(b).query(target)[0]<=.2 if len(b) else np.zeros(len(rg),bool)
  V,F,source=mesh_for_case(c,views,cam,conf,K,'cal_build_1.0',ctrl['build_global_scalar']);counts,t,ids,dist,good=measure(V,F,O,d,rg);delta=t-rg;sub={}
  for name,mask in [('build_supported',support),('query2',frame==2),('query5',frame==5)]:sub[name]={'rays':int(mask.sum()),'early':int((mask&(delta<-.2)).sum()),'hit':int((mask&np.isfinite(t)&(np.abs(delta)<=.2)).sum()),'miss':int((mask&~np.isfinite(t)).sum()),'near_vertices':int((mask&(dist<=.2)).sum())}
  rows.append({'owner':e['owner'],'category':e['category'],'static':static,'speed_mps':speed,'status':'EVALUATED','protocol':'cal_build_1.0','vertices':len(V),'faces':len(F),'counts':counts,'strata':sub})
  dest=out/e['owner'];dest.mkdir(exist_ok=True);np.savez_compressed(dest/'cal_build_1.0_rays.npz',origins=O,directions=d,ranges=rg,first=t,face_ids=ids,near_distance=dist,any_correct=good,build_supported=support,frame_ids=frame)
  if e['owner']=='204704542f8642dc8ab046ffbd70e0c5':np.savez_compressed(dest/'primary_surface.npz',vertices=V,faces=F,source_view_pixel=source,build_points=b,size=np.asarray(c['size_lwh_m']))
 (out/'summary.json').write_text(json.dumps({'method':rec['method'],'scene':rec['scene'],'log':s['log_id'],'variant':'twelve_common6','build_control':ctrl,'rows':rows,'role':'POSTHOC_ORDINARY_CONTROL'},indent=2));print(rec['method'],rec['scene'],'DONE',flush=True)
