"""CPU-only uniform scoring from actual FP32 final arrays; no predicted PNG quantization."""
from pathlib import Path
import numpy as np,json,sys,hashlib
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030")
sys.path.insert(0,str(r/"code/reconstruction"))
from evaluate_edits import rgb,region_error
ref=r/"dataset_r2/privileged";manifest=json.loads((ref/"reference_manifest.json").read_text());held=manifest["heldout"][0]
body=np.load(ref/held["geometry"])["semantic_id"]==10
records=[]
for folder in sorted((r/"m2_repair_r1").glob("heldout_*")):
 p=folder/"render_manifest.json"
 if not p.exists():continue
 m=json.loads(p.read_text())
 for item in m.get("frames",[]): records.append((folder.name,item["edit"],Path(item["fp32"])))
for arm in ("m0","m1"):
 for edit in ("none","delete","shift","rotate"):
  folder=r/"m1_assignment_r1"/("heldout_recall" if arm=="m1" and edit=="delete" else "heldout_"+arm+"_"+edit)
  records.append((arm,edit,folder/("heldout_"+edit+".npz")))
rows=[]
for arm,edit,fp in records:
 if not fp.exists():raise FileNotFoundError(fp)
 prediction=np.load(fp)["final"]
 target_path=r/"heldout_edit_refs_r1"/edit/"000008.png"
 target=rgb(target_path)
 assert prediction.dtype==np.float32 and prediction.shape==target.shape and np.isfinite(prediction).all()
 rows.append(dict(arm=arm,edit=edit,fp32=str(fp),prediction_dtype=str(prediction.dtype),reference=str(target_path),full=region_error(prediction,target,np.ones_like(body)),original_body_footprint=region_error(prediction,target,body)))
x=dict(status="UNIFORM_ACTUAL_FP32_HELDOUT_RAW_AND_CLAMPED_SCORED",rows=rows,row_count=len(rows),arms=sorted(set(x["arm"] for x in rows)),prediction_source="actual native FP32 final; never rendered prediction PNG",regions="full and fixed original heldout none firsthit body footprint; mask is evaluation-only",reference_only_final_evaluation=True,no_gpu_or_weight_update=True,reference_parity_caveat="new independent none differs51/460800 channels max5/255 from original; all arms use same new refs")
p=r/"m2_repair_r1/heldout_uniform_metrics.json";p.write_text(json.dumps(x,indent=2));print(json.dumps(dict(path=str(p),rows=len(rows),arms=x["arms"]),indent=2))
