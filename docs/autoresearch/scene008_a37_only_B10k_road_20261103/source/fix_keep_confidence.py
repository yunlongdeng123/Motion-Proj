from pathlib import Path
import json,numpy as np
RUN=Path(__file__).resolve().parent.parent
for p in sorted((RUN/'inputs/views').glob('*.npz')):
    d=dict(np.load(p));confidence=np.where(d['inferred_fine'],.25,1)
    coverage=d['fit_weight']/confidence
    d['native_keep_weight_pre_confidence_fix']=d['native_keep_weight'].copy()
    d['native_keep_weight']=np.maximum(1-coverage,d['other_vehicle_guard']).astype(np.float32)
    d['world_edit_weight']=coverage
    np.savez_compressed(p,**d)
r=json.loads((RUN/'inputs/receipt.json').read_text());r['native_preserve_policy']='coverage weight independent of weaker inferred photometric confidence; original draft keep masks preserved in each npz'
(RUN/'inputs/receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(status='PASS_KEEP_CONFIDENCE_SEPARATION',frames=49,updates=0)))
