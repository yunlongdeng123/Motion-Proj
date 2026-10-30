import json
from pathlib import Path
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030")
out=r/"m1_assignment_r1"
base=json.loads((r/"edit_r3_10k/reference_metrics.json").read_text())
b={(x["frame"],x["edit"]):x for x in base["rows"]}
result=[]
for policy in ("strict_r2","balanced"):
    p=out/policy
    m=json.loads((p/"reference_metrics.json").read_text())
    s=json.loads((p/"responsibility_summary.json").read_text())
    for x in m["rows"]:
        if x["edit"]=="none": continue
        region="newly_exposed_road" if x["edit"]=="delete" else "full"
        result.append(dict(policy=policy,frame=x["frame"],edit=x["edit"],region=region,baseline_mae=b[x["frame"],x["edit"]]["regions"][region]["clamped_mae"],m1_mae=x["regions"][region]["clamped_mae"]))
    manifest=json.loads((p/"render_manifest.json").read_text())
    manifest["routing"]="truth-assisted conditional static selection plus entire dynamic branch; frozen parameters, no unique Gaussian GT"
    (p/"render_manifest.json").write_text(json.dumps(manifest,indent=2))
receipt=dict(status="COMPLETE_2_POLICIES_24_NATIVE_RENDERS",rows=result,training_updates=0,checkpoint_unchanged=True,failed_attempt_preserved="strict",failure="first attempt depth float64 versus native float32; fixed by explicit dtype",limitations=["finite13 sigma-point depth/surface association, not complete Gaussian footprint or alphaT responsibility","conditional per-frame selection; distinct policy and evidence uncertainty, not unique owner GT","wrong-depth/unobserved components deliberately remain, so this is not best possible deletion","no geometry/color/opacity optimization; no shadow/effect GT assignment"])
(out/"completion_receipt.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
