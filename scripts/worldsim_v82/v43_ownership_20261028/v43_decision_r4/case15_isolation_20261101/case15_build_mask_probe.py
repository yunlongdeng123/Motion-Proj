import pathlib,json,numpy as np
W=pathlib.Path(__file__).resolve().parent;H=W/'case15';ss=json.loads((H/'shortlist.json').read_text(encoding='utf8'));M=pathlib.Path(r'C:\Users\dengyunlong\Documents\Codex\2026-10-31\g\work\actor_diagnostics_20261101\media\generated');out=[]
for r in ss:
 if r['category_code']!='B':continue
 d=json.loads((M/'scenes'/r['context_id']/'media.json').read_text(encoding='utf8'));w=r['native_window'];fs=[]
 for f in d['frames']:
  if not w['first_raw_frame']<=f['raw']<=w['last_raw_frame']:continue
  b=next(b for b in f['boxes'] if b['track']==r['track_id']);bb=(b.get('projection_comparison') or {}).get('official_box')
  if not bb:
   xy=np.array([e[:2] for e in b['edges']]+[e[2:] for e in b['edges']]);bb=np.r_[xy.min(0),xy.max(0)]
  fs.append({'raw':f['raw'],'bbox':(np.array(bb)*2).tolist()})
 out.append({'context_id':r['context_id'],'case':r['scene_key']+'_'+r['alias'],'track':r['track_id'],'bbox_role':'source official FRONT projection box; fallback clipped-edge AABB','frames':fs})
(H/'mask_probe_input_r2.json').write_text(json.dumps(out),encoding='utf8')
