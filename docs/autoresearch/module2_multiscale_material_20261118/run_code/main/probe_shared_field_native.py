from pathlib import Path
import json,time,sys
import numpy as np
import torch

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
sys.path.insert(0,str(P/'code'))
from train_world_material import TrainingState

out=P/'root_native_field_probe_r1';out.mkdir(exist_ok=False)
with np.load(P/'stage0_native_r1/scope.npz') as z:s={k:z[k].copy() for k in z.files}
with np.load('/root/autodl-tmp/drivingeditor_repro/campaign/road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded/provenance.npz') as z:
    known=z['known'][s['active_world_rows']]
s['eligible'] &= ((s['actual_A_mass']>0)|(s['actual_S_mass']>0)|((s['actual_C_mass']>0)&~known))
np.savez_compressed(out/'scope.npz',**s)
started=time.monotonic();torch.cuda.reset_peak_memory_stats()
state=TrainingState(P/'cpu_exact_r1/field_initial.pt',out/'scope.npz',P/'stage0_native_r1',P/'execution_contract.json','road_asset')
print(json.dumps(dict(stage='state_loaded',seconds=time.monotonic()-started,eligible=int(s['eligible'].sum()),mask=state.mask_statistics)),flush=True)
rows=[]
state.model.set_stage('C')
for raw in (86,110):
    start=time.monotonic();loss,terms=state.objective(raw)
    gradient=torch.autograd.grad(loss,[state.model.low,state.model.high])
    torch.cuda.synchronize()
    record=dict(raw=raw,loss=float(loss.detach()),seconds=time.monotonic()-start,gradient={})
    for name,g in zip(('low','high'),gradient):
        record['gradient'][name]=dict(finite=bool(torch.isfinite(g).all()),nonzero=int((g!=0).sum()),max_abs=float(g.abs().max()))
        assert record['gradient'][name]['finite'] and record['gradient'][name]['nonzero']>0
    rows.append(record);print(json.dumps(record),flush=True)
    del gradient,loss,terms
state.assert_frozen()
receipt=dict(schema='actual_shared_world_grid_native_gradient.v1',GPU_executed=True,optimizer_updates=0,
 actual_shared_field_gradient=True,independent_color_leaf_used=False,rows=rows,
 scope_rows=int(s['eligible'].sum()),C_only_known_protected=True,mask=state.mask_statistics,
 fixed_S_photo=3131,fixed_S_full=3187,source='cpu_exact_r1 authoritative field plus current native renderer',
 native_calls=state.ctx.native_calls,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
 elapsed_seconds=time.monotonic()-started,estimated_physical1400_core_seconds=1400*np.mean([x['seconds'] for x in rows]))
state.ctx.close();(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
