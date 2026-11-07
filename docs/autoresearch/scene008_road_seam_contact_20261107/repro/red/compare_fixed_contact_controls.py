"""Read-only native paired-control audit at preregistered contact ray sets.

No ray reselection, source edits, model updates, or GPU work.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return dict(n=len(a),min=float(a.min()) if len(a) else None,mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,max=float(a.max()) if len(a) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--name",required=True);args=p.parse_args()
    run=args.run;out=run/"red";target=out/f"{args.name}_fixed_contact_comparison.json"
    if target.exists():raise FileExistsError(target)
    old=run.parent/"scene008_road_r0_v44_20261107_r1"
    report128=json.loads((out/"raw128_contact_audit_v2.json").read_text())
    records128=[r for r in report128["evidence_pixels"] if r["baseline_assetW"]>.9]
    assert len(records128)==36
    report130=json.loads((old/"review/asset_ground_conflict/canonical_chart_35ray_first_hit_probe.json").read_text())
    reference={128:[dict(pixel_xy=r["pixel_xy"],ordering={1:"chart_first",2:"ground_first",3:"near_contact_005m",0:"uncertain"}[r["ordering_matte_001"]]) for r in records128],
        130:[dict(pixel_xy=r["pixel_xy"],ordering=r["matte_thresholds"]["0.01"]["ordering"]) for r in report130["pixels"]]}
    assert len(reference[130])==35
    cases=[];files={};allpixels=[]
    for raw,rays in reference.items():
        folder=run/f"results/{args.name}/{raw}"
        parent=load(folder/"parent_delete.npz");candidate=load(folder/"candidate_delete.npz")
        layer=load(run/f"diagnostic/layers_v1/{raw}/r0_delete.npz");v44=load(run/f"diagnostic/layers_v1/{raw}/v44_delete.npz")
        obs=load(run/f"diagnostic/layers_v1/{raw}/observations.npz")
        for q in (folder/"parent_delete.npz",folder/"candidate_delete.npz"):
            files[str(q.relative_to(run))]=sha(q)
        diffs={key:float(np.max(np.abs(parent[key]-layer[key]))) for key in ("rgb","alpha","group_alpha_T","newroad_alpha_T","depth_m")}
        # Diagnostic and pilot wrappers normalize expected depth independently;
        # accept <1e-4m rounding while requiring exact RGB/alpha parent identity.
        assert max(v for k,v in diffs.items() if k!="depth_m")==0, (raw,diffs)
        assert diffs["depth_m"]<1e-4, (raw,diffs)
        rows=[]
        for ray in rays:
            x,y=ray["pixel_xy"];v=float(v44["group_alpha_T"][0,y,x]);a=float(parent["group_alpha_T"][0,y,x]);b=float(candidate["group_alpha_T"][0,y,x])
            rows.append(dict(raw=raw,**ray,V44_assetW=v,parent_assetW=a,candidate_assetW=b,assetW_gain=b-a,
                parent_newroadW=float(parent["newroad_alpha_T"][y,x]),candidate_newroadW=float(candidate["newroad_alpha_T"][y,x]),
                RGB_abs_change=float(np.abs(candidate["rgb"][:,y,x]-parent["rgb"][:,y,x]).mean()),
                original_actor_label=int(obs["actor_labels"][y,x]),a6_visible=bool(obs["visible"][0,y,x]),a6_unknown=bool(obs["unknown"][0,y,x]),
                a6_occluded=bool(obs["occluded"][0,y,x]),original_road=bool(obs["road"][y,x])))
        for group in ("ALL","chart_first","ground_first","near_contact_005m","uncertain"):
            r=[a for a in rows if group=="ALL" or a["ordering"]==group]
            gains=np.array([a["assetW_gain"] for a in r]);losses=np.array([a["V44_assetW"]-a["candidate_assetW"] for a in r])
            cases.append(dict(raw=raw,group=group,rays=len(r),parent_assetW=stat([a["parent_assetW"] for a in r]),candidate_assetW=stat([a["candidate_assetW"] for a in r]),
                assetW_gain=stat(gains),gained_gt001=int((gains>.01).sum()),gained_gt01=int((gains>.1).sum()),gained_gt05=int((gains>.5).sum()),
                lost_gt001=int((gains<-.01).sum()),lost_gt01=int((gains<-.1).sum()),remaining_loss_to_V44_gt001=int((losses>.01).sum()),
                remaining_loss_to_V44_gt05=int((losses>.5).sum()),RGB_abs_change=stat([a["RGB_abs_change"] for a in r]),
                parent_matches_layers_v1_max_abs=diffs))
        allpixels.extend(rows)
    mrows=[]
    for raw in (105,113,128,130,137):
        m=json.loads((run/f"results/{args.name}/{raw}/metrics.json").read_text())
        for row in m:
            a=row["variants"]["parent"];b=row["variants"]["candidate"]
            assert a["asset_rows_hash"]==b["asset_rows_hash"]
            mrows.append(dict(raw=raw,state=row["state"],asset_rows_hash_equal=True,metrics={k:dict(parent=a[k]["mean"],candidate=b[k]["mean"],delta=b[k]["mean"]-a[k]["mean"]) for k in ("a6_RGB","protected_RGB","observed_road_RGB")}))
    report=dict(schema="FIXED_CONTACT_RAY_CONTROL_COMPARISON_V1",control=args.name,
        rayset_provenance={"128":"Fixed registered rectangle; original n1-V44 W>.9 and loss>.01: 23 chart-first/12 ground-first/1 near-contact; selected before q2/control output.",
            "130":"Immutable prior 35-ray canonical UV first-hit report: 24 chart-first/11 ground-first; no ray reselection."},
        files_sha256=files,cases=cases,pixels=allpixels,key5_metrics=mrows,
        limitations=["Internal open C2 chart ordering reference and measured-pool ground approximation, not hidden physical GT or complete MOG.",
            "AssetW recovery at chart-first rays tests the footprint mechanism; recovery at ground-first rays is not automatically an improvement.",
            "No hidden original C2 RGB truth. RGB_abs_change is an image-change measurement, not wheel accuracy.",
            "Fixed raysets are diagnostic cohorts, not all-image adoption gates. Blue seam and road coverage are separate."],
        GPU_used=False,model_updates=0)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(cases=cases,key5_metrics=mrows)),flush=True)


if __name__=="__main__":main()
