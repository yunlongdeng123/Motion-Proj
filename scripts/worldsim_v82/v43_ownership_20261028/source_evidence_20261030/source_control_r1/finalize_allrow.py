from pathlib import Path
import numpy as np,json,hashlib,tarfile
root=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/source_control_r1");run=root/"allrow_observed16_r1";p=run/"allrow_history/aggregate.npz";a=np.load(p);assert len(a["frames_seen"])==16
mid=np.load(root/"run_r2_color_maps/source_000008/per_row_sources.npz");receipt=json.loads((root/"run_r2_color_maps/source_000008/source_receipt.json").read_text());top128=np.array(receipt["groups"]["top_error_128"]);top8=top128[:8]
vis=a["total_alphaT_over_original_views"];cats=a["sum_alphaT_by_category"];retained=~mid["selected"];w=mid["after_body"].astype(float);ew=mid["after_error"].astype(float)
assert len(vis)==len(w) and np.all(w[~retained]==0)
labels=["<1e-4","[1e-4,.01)","[.01,.1)","[.1,1)",">=1"];edges=[0,1e-4,.01,.1,1,np.inf];bins=[]
for i,label in enumerate(labels):
 mask=retained&(vis>=edges[i])&(vis<edges[i+1]);bins.append(dict(bin=label,retained_rows=int(mask.sum()),postdelete_body_alphaT=float(w[mask].sum()),postdelete_body_alphaT_fraction=float(w[mask].sum()/w.sum()),shared_pixel_error_weighted_alphaT=float(ew[mask].sum()),shared_pixel_error_weighted_alphaT_fraction=float(ew[mask].sum()/ew.sum()),original16_category_alphaT_sum=cats[mask].sum(0).tolist()))
q=cats/np.maximum(vis[:,None],1e-30);observed=vis>=1e-4
composition={}
for name,weights in (("postdelete_body_alphaT",w),("shared_pixel_error_weighted_alphaT",ew)):
 composition[name]=dict(weak_original_visibility_fraction=float(weights[~observed].sum()/weights.sum()),original_category_mixture_fractions=((q[observed]*weights[observed,None]).sum(0)/weights.sum()).tolist(),definition="Each sufficiently visible row divides its original16 exposure across vehicle/road/other, then row fractions are weighted by its postdelete contribution; weak rows remain separate. Not identity labels.")
checks=[]
for j in sorted((run/"allrow_history").glob("*.json")):checks+=json.loads(j.read_text())["category_conservation"]
summary=dict(status="COMPLETE_ALL_STATIC_ROWS_ORIGINAL16_REPLAY_WEIGHTED_MID_POSTDELETE",static_rows=len(vis),retained_static_rows=int(retained.sum()),frames=16,total_postdelete_body_alphaT=float(w.sum()),total_shared_pixel_error_weighted_alphaT=float(ew.sum()),bins=bins,categories=["original_vehicle_firsthit","original_road_firsthit","other_including_nohit"],category_composition=composition,top_subsets={name:dict(rows=len(idx),postdelete_body_alphaT_fraction=float(w[idx].sum()/w.sum()),shared_pixel_error_weighted_alphaT_fraction=float(ew[idx].sum()/ew.sum()),original16_sum_alphaT=float(vis[idx].sum())) for name,idx in (("top8",top8),("top128",top128))},max_featuregradient_conservation_relative_error=max(x["relative_error"] for x in checks),scope="All retained static rows, fixed original M0 checkpoint; denominator is actual midframe original-body AND independent delete-road ROI after fixed recall deletion. Original16 visibility is frozen final-model replay, not optimizer history. Error-weighted alphaT is a ranking statistic, not per-row causal error.",no_identity_gt=True,training_updates=0)
(run/"weighted_visibility_bins.json").write_text(json.dumps(summary,indent=2))
np.savez_compressed(run/"allrow_visibility_aggregate.npz",**{k:a[k] for k in a.files},retained_mid=retained,mid_post_body_alphaT=w,mid_post_shared_error_weighted_alphaT=ew,top128_ids=top128)
files=[run/"weighted_visibility_bins.json",run/"allrow_visibility_aggregate.npz",run/"execution.json",run/"render_manifest.json",root/"allrow_observed16_r1.log",root/"code/allrow_history.py",root/"code/render_allrow_history.py"]+list((run/"allrow_history").glob("*.json"))
manifest=[dict(path=str(x.relative_to(root)),bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in files];(root/"allrow_delivery_manifest.json").write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/"allrow_delivery.tar.gz","w:gz") as t:
 for x in files+[root/"allrow_delivery_manifest.json"]:t.add(x,arcname=str(x.relative_to(root)))
print(json.dumps(summary,indent=2))
