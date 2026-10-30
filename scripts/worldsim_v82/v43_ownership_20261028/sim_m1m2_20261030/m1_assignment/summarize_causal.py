import json,numpy as np
from pathlib import Path
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030")
out=r/"m1_assignment_r1"
rows=[]
for f in ("000000","000008","000015"):
 a=np.load(out/"recall"/("selection_"+f+".npz"))
 for policy in ("strict_r2","balanced","recall"):
  s=np.load(out/policy/("selection_"+f+".npz"))["selected"]
  metrics=json.loads((out/policy/"reference_metrics.json").read_text())
  x=next(x for x in metrics["rows"] if x["frame"]==f and x["edit"]=="delete")
  rows.append(dict(frame=f,policy=policy,selected=int(s.sum()),static_body_alphaT_recalled_fraction=float(a["body_alphaT"][s].sum()/a["body_alphaT"].sum()),background_alphaT_selected=float(a["background_alphaT"][s].sum()),delete_newroad_mae=x["regions"]["newly_exposed_road"]["clamped_mae"],delete_full_mae=x["regions"]["full"]["clamped_mae"]))
receipt=dict(status="COMPLETE_3_ARMS_36_NATIVE_RENDERS",rows=rows,training_updates=0,persistent_identity_solved=False,unique_owner_gt=False,thresholds_fixed_before_reference_scoring=True,recall="actual unified native raster features gradient gives sum alpha*T per Gaussian over independent body/nonvehicle-surface pixels; no near-depth gate")
(out/"causal_receipt.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
