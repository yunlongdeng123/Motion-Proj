from pathlib import Path
import numpy as np,json,hashlib,tarfile
import argparse
parser=argparse.ArgumentParser();parser.add_argument("--history-output",type=Path,required=True);parser.add_argument("--mid-source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
run=args.history_output;root=args.output;root.mkdir(parents=True,exist_ok=False);p=run/"allrow_history/aggregate.npz";a=np.load(p);assert len(a["frames_seen"])==16
history_manifest=json.loads((run/"render_manifest.json").read_text());source_manifest=json.loads((args.mid_source.parent/"render_manifest.json").read_text());
if history_manifest["checkpoint_event"] != source_manifest["checkpoint_event"]: raise ValueError("History and postdelete sources must come from the same frozen checkpoint; do not transfer row IDs")
mid=np.load(args.mid_source/"per_row_sources.npz");receipt=json.loads((args.mid_source/"source_receipt.json").read_text());top128=np.array(receipt["groups"]["top_error_128"]);top8=top128[:8]
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
summary=dict(status="COMPLETE_ALL_STATIC_ROWS_ORIGINAL16_REPLAY_WEIGHTED_MID_POSTDELETE",static_rows=len(vis),retained_static_rows=int(retained.sum()),frames=16,total_postdelete_body_alphaT=float(w.sum()),total_shared_pixel_error_weighted_alphaT=float(ew.sum()),bins=bins,categories=["original_vehicle_firsthit","original_road_firsthit","other_including_nohit"],category_composition=composition,top_subsets={name:dict(rows=len(idx),postdelete_body_alphaT_fraction=float(w[idx].sum()/w.sum()),shared_pixel_error_weighted_alphaT_fraction=float(ew[idx].sum()/ew.sum()),original16_sum_alphaT=float(vis[idx].sum())) for name,idx in (("top8",top8),("top128",top128))},max_featuregradient_conservation_relative_error=max(x["relative_error"] for x in checks),scope="All retained static rows, explicit frozen checkpoint of the supplied history and mid-source artifacts; denominator is actual midframe original-body AND independent delete-road ROI after fixed recall deletion. Original16 visibility is frozen final-model replay, not optimizer history. Error-weighted alphaT is a ranking statistic, not per-row causal error.",no_identity_gt=True,training_updates=0)
(root/"weighted_visibility_bins.json").write_text(json.dumps(summary,indent=2))
np.savez_compressed(root/"allrow_visibility_aggregate.npz",**{k:a[k] for k in a.files},retained_mid=retained,mid_post_body_alphaT=w,mid_post_shared_error_weighted_alphaT=ew,top128_ids=top128)
print(json.dumps(summary,indent=2))
