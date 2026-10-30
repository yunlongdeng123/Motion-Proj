from pathlib import Path
import numpy as np,json,sys
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030")
sys.path.insert(0,str(r/"code/reconstruction"))
from evaluate_edits import rgb,region_error
ref=r/"dataset_r2/privileged";man=json.loads((ref/"reference_manifest.json").read_text());g=np.load(ref/man["heldout"][0]["geometry"]);body=g["semantic_id"]==10;rows=[]
for arm in ("m0","m1"):
 for edit in ("none","delete","shift","rotate"):
  p=r/"m1_assignment_r1"/("heldout_recall" if arm=="m1" and edit=="delete" else "heldout_"+arm+"_"+edit)
  a=np.load(p/("heldout_"+edit+".npz"))["final"];target=rgb(r/"heldout_edit_refs_r1"/edit/"000008.png")
  metrics=dict(arm=arm,edit=edit,full=region_error(a,target,np.ones_like(body)),original_body_footprint=region_error(a,target,body),reference=str(r/"heldout_edit_refs_r1"/edit/"000008.png"),selection="same original train f8 recall_verified; no heldout GT for selection",training_updates=0,raw_and_clamped=True)
  (p/"reference_metrics.json").write_text(json.dumps(metrics,indent=2));rows.append(metrics)
a=np.load(r/"m1_assignment_r1/heldout_m0_none/heldout_none.npz")["final"];b=np.load(r/"m1_assignment_r1/heldout_m1_none/heldout_none.npz")["final"]
x=dict(status="COMPLETE_M0_M1_FIXED_SELECTION_HELDOUT_8_RENDERS",rows=rows,none_baseline_vs_m1_max_abs=float(np.max(np.abs(a-b))),heldout_gt_for_selection=False,reference_parity_caveat="fresh independent none reference differs51/460800 uint8 channels max5/255 from original; not candidate-tuned",training_updates=0)
(r/"m1_assignment_r1/heldout_suite_metrics.json").write_text(json.dumps(x,indent=2));print(x["status"],x["none_baseline_vs_m1_max_abs"])
for x in rows: print(x["arm"],x["edit"],x["full"]["clamped_mae"],x["original_body_footprint"]["clamped_mae"])
