"""CPU bridge summary: same old weights/new legal queries, distinct from fresh reconstruction."""
from pathlib import Path
import json,numpy as np,hashlib,tarfile
from PIL import Image
root=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030");src=root/"source_control_r1";b=src/"bridge_oldmodel_newviews_r1"
old=np.load(src/"allrow_observed16_r1/allrow_visibility_aggregate.npz");new=np.load(b/"probe/allrow_history/aggregate.npz");mid=np.load(src/"run_r2_color_maps/source_000008/per_row_sources.npz");w=mid["after_body"].astype(float);ew=mid["after_error"].astype(float);ov=old["total_alphaT_over_original_views"];nv=new["total_alphaT_over_original_views"];nc=new["sum_alphaT_by_category"]
assert len(ov)==len(nv)==len(w) and len(new["frames_seen"])==16
receipt=json.loads((src/"run_r2_color_maps/source_000008/source_receipt.json").read_text());ids=np.array(receipt["groups"]["top_error_128"]);edges=np.array([0,1e-4,.01,.1,1,np.inf]);labels=["<1e-4","[1e-4,.01)","[.01,.1)","[.1,1)",">=1"]
ob=np.minimum(np.searchsorted(edges,ov,side="right")-1,4);nb=np.minimum(np.searchsorted(edges,nv,side="right")-1,4);bins=[]
for k,label in enumerate(labels):
 mask=nb==k;bins.append(dict(new_visibility=label,mid_post_alphaT_fraction=float(w[mask].sum()/w.sum()),shared_error_weighted_fraction=float(ew[mask].sum()/ew.sum())))
transition=np.zeros((5,5));transition_e=transition.copy()
for i in range(5):
 for j in range(5):
  mask=(ob==i)&(nb==j);transition[i,j]=w[mask].sum()/w.sum();transition_e[i,j]=ew[mask].sum()/ew.sum()
# Exact nearest distances to new legal sparse measurements for the SAME old rows and time.
prior=np.load(src/"constraint_audit_r2/top128_world_covariance_lidar.npz");assert np.array_equal(prior["static_index"],ids)
centers=prior["world_centers_by_frame"];inv=np.linalg.inv(prior["covariance_world"]);obs=root/"evidence_reconstruction_r1/dataset_hybrid8/observations";man=json.loads((obs/"manifest.json").read_text());dist=np.full((16,128,3),np.nan);mah=dist.copy()
for k,row in enumerate(man["frames"]):
 data=np.load(obs/row["lidar"]);pts=data["xyz_world"];uv=data["uv"];xx=np.floor(uv[:,0]).astype(int).clip(0,row["width"]-1);yy=np.floor(uv[:,1]).astype(int).clip(0,row["height"]-1)
 body=np.asarray(Image.open(obs/row["body_mask"]).convert("L"))>127;car=body[yy,xx];gt=np.load(root/"evidence_reconstruction_r1/dataset_hybrid8/privileged/original"/(str(row["id"]).zfill(6)+".npz"));road=np.isin(gt["semantic_id"][yy,xx],[1,2])
 for j,m in enumerate((np.ones(len(pts),bool),car,road)):
  p=pts[m]
  if not len(p):continue
  delta=p[None]-centers[k,:,None];dist[k,:,j]=np.sqrt((delta**2).sum(2).min(1));mah[k,:,j]=np.sqrt(np.einsum("npi,nij,npj->np",delta,inv,delta).min(1).clip(0))
q=nc/np.maximum(nv[:,None],1e-30);vis=nv>=1e-4
comp=dict(weak_fraction=float(w[~vis].sum()/w.sum()),vehicle_road_other_fraction=(q[vis]*w[vis,None]).sum(0).tolist());comp["vehicle_road_other_fraction"]=(np.asarray(comp["vehicle_road_other_fraction"])/w.sum()).tolist()
top=[]
for j,idx in enumerate(ids):top.append(dict(row_id=int(idx),old16_alphaT=float(ov[idx]),new16_alphaT=float(nv[idx]),new16_vehicle_road_other=nc[idx].tolist(),new_RGB_contribution_weighted_fit_rms=float(new["observed_rgb_fit_rms"][idx]),min_new_legal_lidar_center_m=float(np.nanmin(dist[:,j,0])),min_new_legal_lidar_mahal_sigma=float(np.nanmin(mah[:,j,0])),frames_new_point_inside3sigma=int((mah[:,j,0]<=3).sum()),old_frames_point_inside3sigma=int((prior["nearest_mahalanobis_sigma"][:,j,0]<=3).sum())))
transform=json.loads((b/"original_new_transform_receipt.json").read_text());tf=Path(transform["path"]);transform["sha_unchanged"]=hashlib.sha256(tf.read_bytes()).hexdigest()==transform["sha256"];transform["mtime_unchanged"]=tf.stat().st_mtime_ns==transform["mtime_ns"];assert transform["sha_unchanged"] and transform["mtime_unchanged"]
checks=[]
for p in (b/"probe/allrow_history").glob("*.json"):checks+=json.loads(p.read_text())["category_conservation"]
summary=dict(status="COMPLETE_FROZEN_OLD10K_NEW_OBSERVATION_BRIDGE_PROBE",weights="original formal_r3_10k unchanged; this is not fresh hybrid10K reconstruction",frame_count=16,new_trajectory="13 of16 queries changed; 0/8/15 retained; same times and shared legal baseline gauge",new_visibility_bins=bins,old_to_new_visibility_alphaT_fraction_matrix=transition.tolist(),old_to_new_visibility_shared_error_fraction_matrix=transition_e.tolist(),matrix_axis_labels=labels,new_category_composition_postweighted=comp,original_weak_lt01_exposed_now_ge01_post_alphaT_fraction=float(w[(ov<.1)&(nv>=.1)].sum()/w.sum()),original_weak_lt01_total_post_alphaT_fraction=float(w[ov<.1].sum()/w.sum()),top128=dict(original16_alphaT_sum=float(ov[ids].sum()),new16_alphaT_sum=float(nv[ids].sum()),new_road_alphaT_sum=float(nc[ids,1].sum()),rows_new_total_ge01=int((nv[ids]>=.1).sum()),rows_new_total_ge1=int((nv[ids]>=1).sum())),top8=top[:8],top128_rows=top,max_gradient_conservation_relative_error=max(x["relative_error"] for x in checks),new_shared_transform_protection=transform,interpretation="Actual old Gaussian native visibility, not only road-surface visibility. Pixel category contribution and row RGB mismatch are diagnostic evidence, not identity or per-row training loss. Sparse coordinates are actual legal inputs; road annotation alone is privileged firsthit at measured UV. Final frozen query probe does not reconstruct optimizer history.",training_updates=0)
(b/"bridge_receipt.json").write_text(json.dumps(summary,indent=2));np.savez_compressed(b/"bridge_allrow_aggregate.npz",static_index=np.arange(len(ov)),old_original16_visibility=ov,new_query16_visibility=nv,new_category_alphaT=nc,new_RGB_fit_rms=new["observed_rgb_fit_rms"],old_midpost_alphaT=w,old_midpost_shared_error_weighted_alphaT=ew,top128_ids=ids,new_lidar_distance_top128=dist,new_lidar_mahal_top128=mah)
files=[b/"bridge_receipt.json",b/"bridge_allrow_aggregate.npz",b/"probe_event.json",b/"cameras.json",b/"original_new_transform_receipt.json",b/"probe.log",b/"probe/execution.json",b/"probe/render_manifest.json",b/"probe/training_camera_restore_verification.json"]+list((b/"probe/allrow_history").glob("*.json"))+list((b/"probe/none").glob("*.png"))+[src/"code/render_bridge_history.py",src/"code/bridge_allrow_history.py",src/"code/prepare_bridge.py"]
m=[dict(path=str(p.relative_to(src)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files];(src/"bridge_delivery_manifest.json").write_text(json.dumps(m,indent=2))
with tarfile.open(src/"bridge_delivery.tar.gz","w:gz") as tar:
 for p in files+[src/"bridge_delivery_manifest.json"]:tar.add(p,arcname=str(p.relative_to(src)))
print(json.dumps({k:summary[k] for k in ("new_visibility_bins","original_weak_lt01_exposed_now_ge01_post_alphaT_fraction","top128","top8","max_gradient_conservation_relative_error")},indent=2))
