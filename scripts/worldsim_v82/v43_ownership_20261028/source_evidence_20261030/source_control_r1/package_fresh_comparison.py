from pathlib import Path
import json,hashlib,tarfile,shutil,numpy as np
root=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030");src=root/"source_control_r1";out=src/"fresh_comparison_delivery_r1";out.mkdir();dest=out/"source";dest.mkdir()
oldnative=json.loads((root/"edit_r3_10k/reference_metrics.json").read_text());newnative=json.loads((root/"evidence_reconstruction_r1/edit_common_10k/reference_metrics.json").read_text());rows=[]
for f in ("000000","000008","000015"):
 old=json.loads((src/"run_r2_color_maps"/("source_"+f)/"source_receipt.json").read_text());new=json.loads((src/"fresh_source_r1"/("source_"+f)/"source_receipt.json").read_text())
 on=next(x for x in oldnative["rows"] if x["frame"]==f and x["edit"]=="delete");nn=next(x for x in newnative["rows"] if x["frame"]==f and x["edit"]=="delete");om=next(x for x in old["cases"] if x["case"]=="baseline_delete");nm=next(x for x in new["cases"] if x["case"]=="baseline_delete")
 values={}
 for name,x,reg in (("old_native",on,"newly_exposed_road"),("new_native",nn,"newly_exposed_road"),("old_M1",om,"original_body_revealed_road"),("new_M1",nm,"original_body_revealed_road")):values[name]=x["regions"][reg]
 assert len({x["pixels"] for x in values.values()})==1
 rows.append(dict(frame=f,regions=values,new_M1=dict(static_rows=new["static_count"],selected_rows=new["selected_count"],original_body_selection_audit=new["selection_original_body_audit"],revealed_road_retained_pre_alphaT=new["retained_static_original_body_alphaT"],revealed_road_retained_post_alphaT=new["postdelete_body_alphaT"],combined_pre_ROI_alphaT=new["pre_conservation"]["body_alpha_sum"]),old_M1=dict(static_rows=old["static_count"],selected_rows=old["selected_count"],revealed_road_retained_pre_alphaT=old["retained_static_original_body_alphaT"],revealed_road_retained_post_alphaT=old["postdelete_body_alphaT"])))
 for name,folder,prefix in (("old_native",root/"edit_r3_10k/delete",f+"_"),("new_native",root/"evidence_reconstruction_r1/edit_common_10k/delete",f+"_"),("old_M1",src/"run_r2_color_maps"/("source_"+f),"baseline_delete_"),("new_M1",src/"fresh_source_r1"/("source_"+f),"baseline_delete_")):
  target=dest/name;target.mkdir(exist_ok=True)
  for display in ("final","white"):shutil.copy2(folder/(prefix+display+".png"),target/(f+"_"+display+".png"))
 # Independent target for visual verification, same for all four columns.
 reference=Path(on["reference"]);target=dest/"independent_reference";target.mkdir(exist_ok=True);shutil.copy2(reference,target/(f+".png"))
oldvis=json.loads((src/"allrow_observed16_r1/weighted_visibility_bins.json").read_text());newvis=json.loads((src/"fresh_weighted_visibility_r1/weighted_visibility_bins.json").read_text())
prov=json.loads((src/"fresh_query_bundle_r1/query_provenance.json").read_text());transform=Path(prov["original_query_scene"])/"transform_matrix.txt";assert hashlib.sha256(transform.read_bytes()).hexdigest()==prov["original_transform_sha256"];assert transform.stat().st_mtime_ns==prov["original_transform_mtime_ns"]
summary=dict(status="COMPLETE_MATCHED_NATIVE_AND_M1_FRESH_SOURCE_COMPARISON",frames=rows,ROI="original common-view body AND original independent delete firsthit road; identical pixels across four protocols",protocols=dict(native="delete entire learned dynamic branch only",M1="fresh per-checkpoint static selection by actual original-body alphaT>1e-4 plus all existing dynamic; no near-depth gate; frozen parameters; identities do not transfer"),old_new_budgets="independent fresh10000 updates each, inherited30000 LR horizon; different16-view camera trajectories and emergent static counts, not a fixed-capacity matched test",new_model_frozen_provenance=prov,visibility=dict(old=dict(alphaT_weak_lt01=sum(x["postdelete_body_alphaT_fraction"] for x in oldvis["bins"][:3]),errorweight_weak_lt01=sum(x["shared_pixel_error_weighted_alphaT_fraction"] for x in oldvis["bins"][:3]),bins=oldvis["bins"]),new=dict(alphaT_weak_lt01=sum(x["postdelete_body_alphaT_fraction"] for x in newvis["bins"][:3]),errorweight_weak_lt01=sum(x["shared_pixel_error_weighted_alphaT_fraction"] for x in newvis["bins"][:3]),bins=newvis["bins"])),visual_assessment="Assistant inspection of new frame8 M1 delete final: clear blurred red vehicle-shaped residual and road patches/streaks remain. Not a human acceptance verdict.",training_updates_during_diagnostics=0,quality_accepted=False,human_verdict=None)
(dest/"comparison_summary.json").write_text(json.dumps(summary,indent=2))
# Preserve raw fresh source PNG/CSV/JSON and whole-row16 aggregate; full FP32 render arrays stay remote.
for folder in (src/"fresh_source_r1",src/"fresh_allrow_observed16_r1",src/"fresh_weighted_visibility_r1"):
 for p in folder.rglob("*"):
  if not p.is_file() or "scene_loader" in p.parts:continue
  if p.suffix in (".json",".csv",".png") or (folder.name=="fresh_weighted_visibility_r1" and p.name=="allrow_visibility_aggregate.npz"):
   t=dest/folder.name/p.relative_to(folder);t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
for name,p in (("old_native_metrics.json",root/"edit_r3_10k/reference_metrics.json"),("new_native_metrics.json",root/"evidence_reconstruction_r1/edit_common_10k/reference_metrics.json"),("old_visibility_bins.json",src/"allrow_observed16_r1/weighted_visibility_bins.json"),("fresh_source.log",src/"fresh_source_r1.log"),("fresh_allrow.log",src/"fresh_allrow_observed16_r1.log")):shutil.copy2(p,dest/name)
for name in ("render_generic_source.py","generic_source_audit.py","render_generic_allrow_history.py","generic_allrow_history.py","finalize_generic_allrow.py","prepare_frozen_query.py"):
 target=dest/"code";target.mkdir(exist_ok=True);shutil.copy2(src/"code"/name,target/name)
files=sorted(p for p in dest.rglob("*") if p.is_file());manifest=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files];(dest/"delivery_manifest.json").write_text(json.dumps(manifest,indent=2))
with tarfile.open(out/"fresh_source_delivery.tar.gz","w:gz") as t:
 for p in files+[dest/"delivery_manifest.json"]:t.add(p,arcname=str(p.relative_to(out)))
print("files",len(files),"bytes",sum(p.stat().st_size for p in files))
for row in rows:print(row["frame"],{name:(x["raw_mae"],x["clamped_mae"]) for name,x in row["regions"].items()})
print(summary["visibility"]["new"]["alphaT_weak_lt01"],summary["visibility"]["new"]["errorweight_weak_lt01"])
