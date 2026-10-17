"""Fresh alpha*T mixed parents -> measured 3D boundary split, both children kept."""
from gate_common_003000 import *
from shadow_rebuild.ownership import split_plane,split_moments
import time
start=time.monotonic();saved,cfg,d,model,opt=load('main/latest.pt');field=model.field
audit=np.load(ROOT/'gates/003000/before/per_gaussian.npz');ev=audit['evidence'];nf=audit['frames'];owner=field.owner.numpy();xyz=field.xyz.detach().numpy();cov=covariance(field.quaternion,field.scales).detach().numpy();prec=np.linalg.inv(cov);basis=quaternion_matrix(field.quaternion).detach().numpy();scales=field.scales.detach().numpy()
assert len(ev)==len(xyz)
rows=[];directions=[];cuts=[];provenance=[]
for actor in TARGETS:
 body=body_bank(actor);road=road_bank(actor);build_only(body,d);build_only(road,d)
 if not len(body['xyz']) or not len(road['xyz']):continue
 bt=cKDTree(body['xyz']);rt=cKDTree(road['xyz']);mixed=np.flatnonzero((owner==actor)&(nf[:,0]>=3)&(ev[:,0]>=.5)&(nf[:,2]>=3)&(ev[:,2]>=.2))
 for row in mixed:
  radius=3*float(scales[row].max());bi=volume_points(body['xyz'],bt,xyz[row],prec[row],radius,3);ri=volume_points(road['xyz'],rt,xyz[row],prec[row],radius,3)
  if min(len(bi),len(ri))<3 or min(len(np.unique(body['source_frame'][bi])),len(np.unique(road['source_frame'][ri])))<3:continue
  wb=(body['xyz'][bi]-xyz[row])@basis[row]/scales[row];wr=(road['xyz'][ri]-xyz[row])@basis[row]/scales[row];normal=wb.mean(0)-wr.mean(0);length=np.linalg.norm(normal)
  if length<.25:continue
  normal/=length;pb=wb@normal;pr=wr@normal;cut=.5*(np.median(pb)+np.median(pr))
  if abs(cut)>1.5 or (pb>cut).mean()<.75 or (pr<cut).mean()<.75:continue
  _,_,variance=split_moments(torch.tensor([cut],dtype=torch.float32));v=basis[row]@(scales[row]*normal);eigen=np.linalg.eigvalsh(cov[row][None]+(variance.numpy()[0]-1)[:,None,None]*(v[:,None]*v[None,:])[None])
  if eigen.min()<cfg.scale_min_m**2 or eigen.max()>cfg.scale_max_m**2:continue
  rows.append(int(row));directions.append(normal);cuts.append(float(cut));provenance.append(dict(row=int(row),owner=actor,body_actual_source_frames=np.unique(body['source_frame'][bi]).tolist(),road_actual_source_frames=np.unique(road['source_frame'][ri]).tolist(),body_right_fraction=float((pb>cut).mean()),road_left_fraction=float((pr<cut).mean()),fullscene_evidence=ev[row].tolist(),contribution_frame_counts=nf[row].tolist()))
limit=max(0,cfg.max_gaussians-len(field.xyz));rows=rows[:limit];directions=directions[:limit];cuts=cuts[:limit];provenance=provenance[:limit]
receipt=split_plane(field,torch.tensor(rows,dtype=torch.long),torch.as_tensor(np.asarray(directions,np.float32).reshape(-1,3)),torch.as_tensor(cuts,dtype=torch.float32),opt)
receipt.update(parent_checkpoint='main/pre_owner_003000.pt',global_step=3000,training_updates=0,once_only=True,measured_boundary_provenance=provenance,no_owner_change_before_reaudit=True,no_height_crop=True,seconds=time.monotonic()-start)
out=ROOT/'gates/003000/split';out.mkdir(exist_ok=False);saved['state']=model.state_dict();saved['optimizer']=opt.state_dict();saved['supervision_lineage']={'kind':'C_measured_mixed_split_both_children','parent_checkpoint':str(ROOT/'main/pre_owner_003000.pt'),'previous':saved.get('supervision_lineage')};save(out/'latest.pt',saved);write(out/'split.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='measured_boundary_provenance'}),flush=True)
