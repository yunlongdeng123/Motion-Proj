import json
import numpy as np
from pathlib import Path
W=Path(__file__).resolve().parent
q=json.loads((W/'evidence_native_pilot/downstream_audit.json').read_text())
out=[]
for condition in sorted({r['condition'] for r in q['replay']}):
    rows=[r for r in q['replay'] if r['condition']==condition]
    delta=np.array([r['ADE_delta_vs_real_m'] for r in rows])
    det={t:{'matched':sum(r['detection'][t]['matched'] for r in rows),'eligible':sum(r['detection'][t]['eligible_visible_forward_vehicles'] for r in rows)} for t in ['0.25','0.5']}
    out.append({'condition':condition,'frames':len(rows),'mean_ADE_delta_m':float(delta.mean()),
        'min_max_ADE_delta_m':[float(delta.min()),float(delta.max())],
        'max_endpoint_change_m':max(r['execution_endpoint_change_vs_real_m'] for r in rows),
        'max_extra_braking_mps2':min(r['extra_minimum_acceleration_mps2'] for r in rows),
        'detection':det})
print(json.dumps(out,indent=2))
(W/'evidence_native_pilot/aggregate.json').write_text(json.dumps(out,indent=2))
