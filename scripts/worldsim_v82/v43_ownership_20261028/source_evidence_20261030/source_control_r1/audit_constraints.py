"""CPU sparse-measurement/covariance audit; never substitutes hidden GT for legal measurements."""
from pathlib import Path
import json,numpy as np,torch,hashlib,tarfile
from scipy.spatial.transform import Rotation
from PIL import Image
root=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030");out=root/"source_control_r1/constraint_audit_r2";out.mkdir()
obs=root/"dataset_r2/observations";manifest=json.loads((obs/"manifest.json").read_text());history=root/"source_control_r1/training_history_r2/history"
ids=np.load(history/"000008.npz")["static_index"];event=json.loads((root/"formal_r3_10k/checkpoint_events/iteration_010000.json").read_text());payload,_=torch.load(event["checkpoints"]["static"]["path"],map_location="cpu",mmap=True,weights_only=False)
logscales=payload[4][ids].detach().numpy();q=payload[5][ids].detach().numpy();q/=np.linalg.norm(q,axis=1,keepdims=True);rotation=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
transform=np.loadtxt(root/"native_scene_r3/transform_matrix.txt");invl=np.linalg.inv(transform[:3,:3]);scales=np.exp(logscales);cov_model=np.einsum("nij,nj,nkj->nik",rotation,scales**2,rotation);cov_world=invl[None]@cov_model@invl.T[None];invcov=np.linalg.inv(cov_world);sigma_world=np.sqrt(np.linalg.eigvalsh(cov_world))
results=[];centers=[];euclid=[];mahal=[];categories=["all_legal_lidar","legal_bodymask_lidar","legal_bodymask_complement_lidar","privileged_annotated_road_lidar"]
for k,item in enumerate(manifest["frames"]):
 frame=str(item["id"]).zfill(6);h=np.load(history/(frame+".npz"));c2w=np.asarray(item["camera_to_world"]);K=np.asarray(item["K"]);z=h["center_z_world_m"];u=h["u"];v=h["v"]
 xyzcam=np.stack(((u-K[0,2])/K[0,0]*z,(v-K[1,2])/K[1,1]*z,z),1);world=xyzcam@c2w[:3,:3].T+c2w[:3,3];centers.append(world)
 data=np.load(obs/item["lidar"]);points=data["xyz_world"];uv=data["uv"];xx=np.floor(uv[:,0]).astype(int).clip(0,int(item["width"])-1);yy=np.floor(uv[:,1]).astype(int).clip(0,int(item["height"])-1)
 body=np.asarray(Image.open(obs/item["body_mask"]).convert("L"))>127;car=body[yy,xx]
 geometry=np.load(root/"dataset_r2/privileged/original"/(frame+".npz"));road=np.isin(geometry["semantic_id"][yy,xx],[1,2])
 sets=[np.ones(len(points),bool),car,~car,road];ed=np.full((len(ids),4),np.nan);md=ed.copy();nearest=np.full((len(ids),4,3),np.nan)
 for j,mask in enumerate(sets):
  p=points[mask]
  if len(p)==0:continue
  delta=p[None]-world[:,None];d2=(delta**2).sum(2);idx=d2.argmin(1);ed[:,j]=np.sqrt(d2[np.arange(len(ids)),idx]);nearest[:,j]=p[idx]
  q2=np.einsum("npi,nij,npj->np",delta,invcov,delta);md[:,j]=np.sqrt(q2.min(1).clip(0))
 euclid.append(ed);mahal.append(md)
 results.append(dict(frame=frame,measurement_counts={name:int(mask.sum()) for name,mask in zip(categories,sets)},top8=[dict(row_id=int(ids[i]),world_center=world[i].tolist(),sigma_world_axes_m=sigma_world[i].tolist(),nearest_euclidean_m={name:float(ed[i,j]) for j,name in enumerate(categories)},nearest_mahalanobis_sigma={name:float(md[i,j]) for j,name in enumerate(categories)},nearest_all_legal_point=nearest[i,0].tolist()) for i in range(8)]))
centers=np.array(centers);euclid=np.array(euclid);mahal=np.array(mahal)
np.savez_compressed(out/"top128_world_covariance_lidar.npz",static_index=ids,world_centers_by_frame=centers,covariance_world=cov_world,sigma_world_axes_m=sigma_world,nearest_euclidean_m=euclid,nearest_mahalanobis_sigma=mahal)
summary=dict(status="COMPLETE_LEGAL_SPARSE_MEASUREMENT_DISTANCE_AUDIT",tracked_ids=ids.tolist(),frames=results,categories=categories,scope="Every point coordinate is actual legal input sparse LiDAR. Car/complement uses legal bodymask. Road category alone uses privileged firsthit semantic annotation at measured UV solely for diagnosis; it is not a provided road sensor label. Exact min Mahalanobis is over all points in each set; Euclidean and Mahalanobis minima can be different points. No GT surface nearest-neighbor is substituted.",top8_min_over16=[dict(row_id=int(ids[i]),world_center_frame8=centers[8,i].tolist(),sigma_world_axes_m=sigma_world[i].tolist(),min_lidar_center_distance_m=float(np.nanmin(euclid[:,i,0])),min_lidar_mahalanobis_sigma=float(np.nanmin(mahal[:,i,0])),frames_with_any_legal_point_inside_3sigma=int((mahal[:,i,0]<=3).sum()),min_car_measured_distance_m=float(np.nanmin(euclid[:,i,1])),min_road_annotated_measured_distance_m=float(np.nanmin(euclid[:,i,3]))) for i in range(8)],covariance_not_surface_identity=True,no_training_history_claim=True,training_updates=0)
(out/"constraint_receipt.json").write_text(json.dumps(summary,indent=2))
# Snapshot exact source excerpts with absolute provenance and hash, never infer observed historical gradients.
source=Path(event["config"].get("source_path",""))
code=Path("/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137")
ranges={"train.py":[[300,410]],"utils/hybrid30k_losses.py":[[45,100],[125,180]],"utils/paper_losses.py":[[23,30],[61,102]],"scene/envlight.py":[[1,48]],"gaussian_renderer/__init__.py":[[207,220],[275,298],[350,375]],"scene/gaussian_model.py":[[140,185]]}
refs=[]
for name,spans in ranges.items():
 p=code/name;txt=p.read_text();lines=txt.splitlines();refs.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),excerpts=[dict(start=a,end=b,text="\n".join(str(i+1)+": "+lines[i] for i in range(a-1,min(b,len(lines))))) for a,b in spans]))
(out/"source_evidence.json").write_text(json.dumps(refs,indent=2))
print(json.dumps(summary["top8_min_over16"],indent=2))
