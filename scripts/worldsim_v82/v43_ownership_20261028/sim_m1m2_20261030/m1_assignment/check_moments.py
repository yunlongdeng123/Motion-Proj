from pathlib import Path
import numpy as np,json
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/m1_assignment_r1")
rows=[]
for f in ("000000","000008","000015"):
 a=np.load(r/"recall_verified"/("selection_"+f+".npz")); b=np.load(r/"recall_moments"/("selection_"+f+".npz"))
 checks={k:bool(np.array_equal(a[k],b[k])) if k=="selected" else bool(np.allclose(a[k],b[k],rtol=2e-5,atol=2e-6)) for k in ("selected","body_alphaT","background_alphaT")}
 assert all(checks.values())
 rows.append(dict(frame=f,unchanged=checks,selected=int(b["selected"].sum()),selected_vehicle_target_invalid=int((b["selected"]&~b["vehicle_target_valid"]).sum())))
x=dict(all_passed=True,rows=rows)
(r/"moment_parity.json").write_text(json.dumps(x,indent=2));print(x)
