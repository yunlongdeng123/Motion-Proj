from pathlib import Path
import numpy as np,json
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030")
ref=r/"dataset_r2/privileged";m=json.loads((ref/"reference_manifest.json").read_text());scale=np.linalg.norm(np.loadtxt(r/"native_scene_r3/transform_matrix.txt")[:3,0]);rows=[]
for f in ("000000","000008","000015"):
 original=np.load(ref/"original"/(f+".npz"));body=original["semantic_id"]==10
 rr=next(x for x in m["references"] if str(x["frame_id"]).zfill(6)==f and x["edit"]=="delete" and x["vehicle_shadow"])
 target=np.load(ref/rr["geometry"]);d=target["depth_z_m"]
 for arm in ("strict_r2","balanced","recall_verified","recall_moments"):
  a=np.load(r/"m1_assignment_r1"/arm/"delete"/(f+".npz"));alpha=a["alpha"][0];z=a["depth_sum_model"][0]/np.maximum(alpha,1e-8)/scale
  valid=body&(alpha>1e-4)&(d>0)&np.isfinite(z);err=np.abs(z-d)
  rows.append(dict(frame=f,arm=arm,body_footprint_pixels=int(body.sum()),alpha_covered_pixels=int((body&(alpha>1e-4)).sum()),coverage_fraction=float((body&(alpha>1e-4)).sum()/body.sum()),reference_depth_valid_pixels=int(valid.sum()),conditional_expected_depth_mae_m=float(err[valid].mean()),conditional_expected_depth_median_error_m=float(np.median(err[valid])),near_road_depth_fraction_035m=float((err[valid]<=.35).mean()),far_depth_fraction_gt035m=float((err[valid]>.35).mean()),semantics="expected alpha-normalized composite depth, not per-row surface identity; near road depth may be normal revealed receiver, not ghost"))
x=dict(status="EVAL_ONLY_POST_DELETE_DEPTH_AUDIT",rows=rows,selection_changed=False,depth_gate_for_this_evaluation_only_m=.35,scope="body footprint; independent delete firsthit depth; alpha alone is not ghost evidence")
(r/"m1_assignment_r1/postdelete_depth_audit.json").write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))
