import json,pathlib,argparse,numpy as np,torch
from scipy.spatial import cKDTree
from evaluate_surfaces import R,D,query,measure
S=R/'secondary';ap=argparse.ArgumentParser();ap.add_argument('--method',default='nksr');args=ap.parse_args();rows=[]
for e in json.loads((R/'cohort.json').read_text()):
 dest=S/args.method/(e['scene']+'__'+e['owner']);f=dest/'result.json'
 if not f.exists():rows.append(dict(e,method=args.method,status='NO_METHOD_RESULT'));continue
 rec=json.loads(f.read_text());c=torch.load(D/e['file'],weights_only=False,map_location='cpu');O,dirs,rg,offs,ids=query(c)
 if not len(rg):rows.append(dict(rec,query_status='NO_OWNED_QUERY',rays=0));continue
 if rec['status']!='DONE':rows.append(dict(rec,query_status='METHOD_UNAVAILABLE',rays=len(rg)));continue
 a=np.load(dest/'surface.npz');counts,t,face,dist,good=measure(a['vertices'].astype('float32'),a['faces'].astype('uint32'),O,dirs,rg)
 b=np.asarray(c['points_actor_m']);sup=cKDTree(b).query(O+dirs*rg[:,None])[0]<=.2;delta=t-rg
 strata={'rays':int(sup.sum()),'early':int((sup&(delta<-.2)).sum()),'hit':int((sup&np.isfinite(t)&(np.abs(delta)<=.2)).sum())}
 np.savez_compressed(dest/'query_rays.npz',origins=O,directions=dirs,ranges=rg,first=t,face_ids=face,near_distance=dist,any_correct=good,build_supported=sup,frame_ids=ids)
 row=dict(rec,query_status='EVALUATED',counts=counts,build_supported=strata);rows.append(row);(dest/'evaluation.json').write_text(json.dumps(row,indent=2))
(S/args.method/'evaluation.json').write_text(json.dumps({'method':args.method,'rows':rows,'evaluable':sum(r.get('query_status')=='EVALUATED' for r in rows),'information':'BUILD LiDAR + PCA16 normals + annotated actor poses; native mesh, no QUERY adjustment','notes':'Missing/insufficient inputs excluded from conditional method metrics and explicitly retained in 75-object table'},indent=2));print(args.method,'evaluated',sum(r.get('query_status')=='EVALUATED' for r in rows),flush=True)
