import pathlib,json,sys,numpy as np
B=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(B.parent/'v74_return'))
from audit import intersect
record=json.loads((B/'white_witness/audit.json').read_text());ids=record['common_early_indices'];rows=[]
for m in ['vggt','omega512','dvgt1','pi3x']:
 p=B/'official_case_assets'/record['owner']/m/'twelve';mesh=np.load(p/'primary_surface.npz');q=np.load(p/'cal_build_1.0_rays.npz');t,face,_,_=intersect(mesh['vertices'],mesh['faces'],q['origins'][ids],q['directions'][ids],q['ranges'][ids]);diff=np.abs(t-q['first'][ids]);assert diff.max()<1e-4
 rows.append({'method':m,'rays':len(ids),'max_float64_vs_open3d_m':float(diff.max()),'all_remain_early':bool(np.all(t<q['ranges'][ids]-.2))})
(B/'evidence/white_float64_qa.json').write_bytes(json.dumps({'status':'PASS','rows':rows},indent=2).encode());print(rows)
