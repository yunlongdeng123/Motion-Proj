from pathlib import Path
import json,numpy as np
RUN=Path(__file__).resolve().parent.parent
q=json.loads((RUN/'evaluation/all49/metrics.json').read_text());r=np.load(RUN/'inputs/road_reference.npz');c=np.load(RUN/'inputs/coarse_envelope.npz')
for name in ['fit','new_dark','original_shadow','yellow','left_road','outside','seam']:
    rows=[x['regions'][name] for x in q['rows'] if x['regions'][name]['samples']];n=sum(x['samples'] for x in rows)
    print(json.dumps(dict(region=name,samples=n,**{key:sum(x[key]*x['samples'] for x in rows)/n for key in ['before_MAE','after_MAE','native_srgb_change']})))
d=np.load(RUN/'inputs/views/105.npz')
for x,y in [(240,240),(240,290),(240,310),(270,318),(220,315),(200,300),(170,290)]:
    print(json.dumps(dict(x=x,y=y,**{key:np.asarray(d[key][y,x]).tolist() for key in ['unit_id','fit_weight','other_vehicle_guard','actor_labels','depth_m','new_shadow_proxy','protect_mask','reference_linear','recovery_reference_linear']})))
