from pathlib import Path
import json,shutil,numpy as np,hashlib
root=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030")
out=root/"source_control_r1/bridge_oldmodel_newviews_r1";out.mkdir();native=root/"evidence_reconstruction_r1/native_scene_hybrid8";copy=out/"query_scene";copy.mkdir()
for p in native.iterdir():
 if p.name in ("transform_matrix.txt","preparation_manifest.json"):shutil.copy2(p,copy/p.name)
 else:(copy/p.name).symlink_to(p,target_is_directory=p.is_dir())
old_event=root/"formal_r3_10k/checkpoint_events/iteration_010000.json";event=json.loads(old_event.read_text());prep=json.loads((copy/"preparation_manifest.json").read_text());event["config"]["source_path"]=str(copy);event["config"]["followup_scene_spec"]=prep["followup_scene_spec"];event["config"]["followup_static_anchor_frame"]=prep["followup_scene_spec"]["anchor_frame"];event["config"]["model_path"]=str(out);event["model_path"]=str(out);event["bridge_frozen_weight_event"]=str(old_event);event["bridge_query_scene_original"]=str(native)
(out/"probe_event.json").write_text(json.dumps(event,indent=2))
obs=json.loads((root/"evidence_reconstruction_r1/dataset_hybrid8/observations/manifest.json").read_text());gauge=json.loads((root/"evidence_reconstruction_r1/code/baseline_gauge.json").read_text());T=np.asarray(gauge["world_to_model"]);scale=np.linalg.norm(T[:3,0]);cams=[]
for row in obs["frames"]:
 c=np.asarray(row["camera_to_world"]);K=np.asarray(row["K"]);cams.append(dict(img_name=str(row["id"]).zfill(6),rotation=(T[:3,:3]/scale@c[:3,:3]).tolist(),position=(T[:3,:3]@c[:3,3]+T[:3,3]).tolist(),fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],width=row["width"],height=row["height"]))
(out/"cameras.json").write_text(json.dumps(cams,indent=2))
f=native/"transform_matrix.txt";(out/"original_new_transform_receipt.json").write_text(json.dumps(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),mtime_ns=f.stat().st_mtime_ns,query_uses_private_scene_copy=True,frozen_weight_event=str(old_event)),indent=2))
print(out)
