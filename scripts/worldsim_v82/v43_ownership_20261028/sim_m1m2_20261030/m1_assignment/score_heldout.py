import json,numpy as np,sys
from pathlib import Path
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030")
sys.path.insert(0,str(r/"code/reconstruction"))
from evaluate_edits import rgb,region_error
p=r/"m1_assignment_r1/heldout_recall"
pred=np.load(p/"heldout_delete.npz")["final"]
ref=rgb(r/"heldout_edit_refs_r1/delete/000008.png")
m=json.loads((r/"dataset_r2/privileged/reference_manifest.json").read_text())
g=np.load(r/"dataset_r2/privileged"/m["heldout"][0]["geometry"])
mask=g["semantic_id"]==10
x=dict(status="SCORED_INDEPENDENT_HELDOUT_DELETE",full=region_error(pred,ref,np.ones_like(mask)),original_body_footprint_newly_exposed=region_error(pred,ref,mask),reference=str(r/"heldout_edit_refs_r1/delete/000008.png"),selection="fixed original train f8 recall; no heldout GT for selection",reference_generation_parity_caveat="independent none rerender differs51/460800 channels, max5/255; parity report preserved",training_updates=0)
(p/"reference_metrics.json").write_text(json.dumps(x,indent=2))
print(json.dumps(x,indent=2))
