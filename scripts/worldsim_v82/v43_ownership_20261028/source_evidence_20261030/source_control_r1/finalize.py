from pathlib import Path
import json,numpy as np,hashlib,tarfile
root=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/source_control_r1");r=root/"run_r2_color_maps"
frames=[]
for folder in sorted(r.glob("source_*")):
 x=json.loads((folder/"source_receipt.json").read_text());a=np.load(folder/"per_row_sources.npz");idx=np.argsort(a["after_error"])[::-1][:2000]
 np.savez_compressed(folder/"top2000_sources.npz",**{k:v[idx] for k,v in a.items() if len(v)==x["static_count"]})
 top=x["groups"]["top_error_32"]
 frames.append(dict(frame=x["frame"],top32_ids=top,conservation=dict(before=x["pre_conservation"],after=x["post_conservation"]),retained_pre=x["retained_static_original_body_alphaT"],retained_post=x["postdelete_body_alphaT"],newly_revealed=x["postdelete_newly_revealed_body_alphaT"],native_color_parity=x["native_color_precompute_parity_max_abs"],cases=[dict(case=c["case"],body_raw_mae=c["regions"]["original_body_revealed_road"]["raw_mae"],body_clamped_mae=c["regions"]["original_body_revealed_road"]["clamped_mae"],outside_road_clamped_mae=c["regions"]["road_outside_original_body"]["clamped_mae"]) for c in x["cases"]]))
summary=dict(status="COMPLETE_FROZEN_ROW_SOURCE_CAUSAL_DIAGNOSTICS",frames=frames,original_source="frozen formal_r3_10k; recall_verified selection unchanged",ground_truth_scope="independent delete geometry/RGB only diagnostic evaluation and color sensitivity intervention, never identity or training",training_updates=0,replays="run_r1 first full diagnostic and run_r2_color_maps adds actual source maps and color-only causal control; both preserved",row_data="all3 complete per_row_sources.npz retained remote; midframe complete included delivery, top2000 allframes included delivery",no_identity_gt_claim=True)
(root/"completion_receipt.json").write_text(json.dumps(summary,indent=2))
files=[]
for p in root.rglob("*"):
 if not p.is_file() or "scene_loader" in p.parts:continue
 if p.suffix in (".json",".png",".csv",".py",".log") or p.name=="top2000_sources.npz" or (p.name=="per_row_sources.npz" and "run_r2_color_maps/source_000008" in p.as_posix()):files.append(p)
files=sorted(p for p in files if p.name!="delivery_manifest.json")
manifest=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(root/"delivery_manifest.json").write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/"delivery.tar.gz","w:gz") as t:
 for p in files+[root/"delivery_manifest.json"]:t.add(p,arcname=str(p.relative_to(root)))
print("files",len(files),"bytes",sum(p.stat().st_size for p in files))
for f in frames:
 print(f["frame"],"topids",f["top32_ids"][:8],"colorparity",f["native_color_parity"])
 for c in f["cases"]:
  if c["case"] in ("baseline_delete","remove_top_error_32","top128_color_only_reference_moment"):print(c)
