from pathlib import Path
import numpy as np,json,csv,tarfile,hashlib
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/source_control_r1");h=r/"training_history_r2/history"
paths=sorted(h.glob("*.npz"));assert len(paths)==16
samples=[np.load(p) for p in paths];ids=samples[0]["static_index"];weights=np.stack([x["alphaT_by_original_category"] for x in samples]);total=np.stack([x["total_alphaT"] for x in samples]);colors=np.stack([x["native_view_rgb"] for x in samples]);fit=np.stack([x["observed_rgb_fit_rms"] for x in samples]);mass=weights.sum(0)
rows=[]
for j,idx in enumerate(ids):
 supported=total[:,j]>1e-4
 row=dict(row_id=int(idx),sum_vehicle=float(mass[j,0]),sum_road=float(mass[j,1]),sum_other=float(mass[j,2]),sum_total=float(total[:,j].sum()),frames_total_gt1e4=int(supported.sum()),frames_vehicle_gt1e4=int((weights[:,j,0]>1e-4).sum()),frames_road_gt1e4=int((weights[:,j,1]>1e-4).sum()),frames_other_gt1e4=int((weights[:,j,2]>1e-4).sum()),native_rgb_min=colors[:,j].min(0).tolist(),native_rgb_max=colors[:,j].max(0).tolist(),hdr_view_count=int((colors[:,j]>1).any(1).sum()),contribution_weighted_input_rgb_rms=float((fit[:,j]*total[:,j]).sum()/max(1e-12,total[:,j].sum())),per_frame=[dict(frame=p.stem,vehicle=float(weights[k,j,0]),road=float(weights[k,j,1]),other=float(weights[k,j,2]),total=float(total[k,j]),native_rgb=colors[k,j].tolist()) for k,p in enumerate(paths)])
 rows.append(row)
checks=[json.loads(p.read_text())["category_conservation"] for p in sorted(h.glob("*.json"))]
x=dict(status="COMPLETE_FROZEN10K_ORIGINAL16_VIEW_SOURCE_TRACE",tracked_midframe_postdelete_top128=ids.tolist(),top8=rows[:8],all_rows=rows,frame_count=16,threshold_for_visibility_reporting=1e-4,max_category_conservation_relative_error=max(z["relative_error"] for c in checks for z in c),closure_category_vs_total_max_abs=float(np.abs(weights.sum(2)-total).max()),interpretation="Final frozen10K replay over original observed view/time with correct moving dynamic. This is not per-update training trajectory. Original firsthit categories describe pixels receiving actual alphaT, never Gaussian identity. Zero contribution can mean occlusion, out of frame, raster mask, small opacity or footprint; center depth is separate diagnostic.",failed_attempt="training_history_r1 CPU camera tensor vs CUDA; fixed explicit device, preserved",training_updates=0)
(r/"observed16_source_trace.json").write_text(json.dumps(x,indent=2))
with (r/"observed16_top128.csv").open("w",newline="") as f:
 writer=csv.writer(f);writer.writerow(["row_id","sum_vehicle","sum_road","sum_other","sum_total","visible_frames_gt1e4","vehicle_frames_gt1e4","road_frames_gt1e4","hdr_views","input_RGB_fit_RMS"])
 for row in rows:writer.writerow([row[k] for k in ("row_id","sum_vehicle","sum_road","sum_other","sum_total","frames_total_gt1e4","frames_vehicle_gt1e4","frames_road_gt1e4","hdr_view_count","contribution_weighted_input_rgb_rms")])
files=list(h.glob("*"))+[r/"observed16_source_trace.json",r/"observed16_top128.csv",r/"training_history_r2/execution.json",r/"training_history_r2/render_manifest.json",r/"training_history_r2.log",r/"training_history_r1.log",r/"code/training_history.py",r/"code/render_history.py"]
manifest=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files];(r/"history_delivery_manifest.json").write_text(json.dumps(manifest,indent=2))
with tarfile.open(r/"history_delivery.tar.gz","w:gz") as t:
 for p in files+[r/"history_delivery_manifest.json"]:t.add(p,arcname=str(p.relative_to(r)))
print("maxclosure",x["max_category_conservation_relative_error"],"total error",x["closure_category_vs_total_max_abs"])
for row in rows[:8]:print(row["row_id"],row["sum_vehicle"],row["sum_road"],row["sum_other"],row["frames_total_gt1e4"],row["hdr_view_count"])
