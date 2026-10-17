"""Re-audited actual owner repair and certified measured-body pool recovery, zero SGD."""
from gate_common_009000 import *
from shadow_rebuild.identity import evidence_probability,reassign_rigid_owner,append_measured_candidates
from shadow_rebuild.ownership import reanchor_road
from shadow_rebuild.prepare import local_geometry,voxel_rows
from shadow_rebuild.references import refresh
import time
start=time.monotonic();saved,cfg,d,model,opt=load('gates/009000/split/latest.pt');field=model.field
audit=np.load(ROOT/'gates/009000/after_split/per_gaussian.npz');ids=np.load(ROOT/'gates/009000/after_split/identity_evidence.npz');ev=audit['evidence'];nf=audit['frames'];mass=ids['mass'];id_frames=ids['frames'];n=len(field.xyz);assert len(ev)==n
visibility=source_visibility(d);motion=motion_source_keys();tr=np.load(ROOT/'idsplat_frontend/data_IDSplat/trajectory.npz');pool=dict(np.load(ROOT/'gates/003000/owner/unresolved_candidate_pool.npz'));build_only(pool,d);receipts=[];admitted=[];road_rows=[];road_anchor=[];road_meta=[]
xyz=field.xyz.detach().numpy();cov=covariance(field.quaternion,field.scales).detach().numpy();prec=np.linalg.inv(cov);scales=field.scales.detach().numpy();owner=field.owner.numpy().copy();old_opacity=field.opacity_logit.detach().clone();neighbor_before=field.neighbor.clone()
for actor in TARGETS:
 body=body_bank(actor);road=road_bank(actor);bt=cKDTree(body['xyz']) if len(body['xyz']) else None;rt=cKDTree(road['xyz']) if len(road['xyz']) else None
 candidate=np.flatnonzero((owner==actor)&(nf[:,2]>=3)&(ev[:,2]>=.5)&(ev[:,2]/np.maximum(ev[:,:3].sum(-1),1e-9)>.8)&(ev[:,0]/np.maximum(ev[:,:3].sum(-1),1e-9)<.05)&~((nf[:,0]>=3)&(ev[:,0]>=.5)))
 for row in candidate:
  bi=np.empty(0,np.int64) if bt is None else volume_points(body['xyz'],bt,xyz[row],prec[row],4*float(scales[row].max()),4)
  if len(bi) or (nf[row,1]>=3 and ev[row,1]>=.5) or rt is None:continue
  ri=volume_points(road['xyz'],rt,xyz[row],prec[row],3*float(scales[row].max()),3)
  if len(np.unique(road['source_frame'][ri]))<3:continue
  near=int(ri[np.argmin(np.linalg.norm(road['xyz'][ri]-xyz[row],axis=-1))])
  if np.linalg.norm(road['xyz'][near]-xyz[row])>.06:continue
  road_rows.append(int(row));road_anchor.append(road['source_world'][near]);road_meta.append(dict(row=int(row),previous_owner=actor,new_owner=-1,frame=int(road['source_frame'][near]),raw_row=int(road['source_raw_row'][near]),world=road['source_world'][near].tolist(),support_source_frames=np.unique(road['source_frame'][ri]).tolist(),actual_body_in_4sigma_volume=False))
rr=torch.tensor(road_rows,dtype=torch.long);anchors=np.asarray(road_anchor,np.float32).reshape(-1,3)
if len(rr):
 road_seed=np.flatnonzero(d.seed['road']);near=cKDTree(d.seed['xyz'][road_seed]).query(anchors,workers=1)[1];reanchor_road(field,rr,torch.from_numpy(anchors),torch.from_numpy(d.seed['quaternion'][road_seed[near]]),opt)
 with torch.no_grad():
  field.log_scales[rr]=torch.from_numpy(d.seed['scales'][road_seed[near]]).log();field.anchor_log_scales[rr]=field.log_scales[rr]
  field.identity_probability[rr]=torch.as_tensor(evidence_probability(False,np.ones(len(rr),bool),False,False),dtype=torch.float32);field.identity_evidence_status[rr]=2
  for row,source in zip(road_rows,road_meta):field.source_frame[row]=source['frame'];field.source_raw_row[row]=source['raw_row'];field.source_world[row]=torch.tensor(source['world'])
# Existing static->Actor transfer needs independent measured3D motion, repeated
# rendered body contribution AND same-frame actual body/depth. Missing motion
# never transfers a parked instance to background or forces a conversion.
static_to_actor=[];worldroad_tree=cKDTree(np.load(ROOT/'actual_world_road_support.npz')['xyz'])
for actor in TARGETS:
 candidate=np.flatnonzero((field.owner.numpy()==-1)&(field.source_candidate_owner.numpy()==actor)&(id_frames[:,actor]>=3)&(mass[:,actor]>=.5)&(mass[:,actor]/np.maximum(mass.sum(-1),1e-9)>.8))
 for row in candidate:
  fi=int(field.source_frame[row]);raw=int(field.source_raw_row[row]);key=fi*1000000+raw
  if key not in motion[actor] or raw not in visibility[fi]['body_raw_rows'][actor] or not tr['valid'][fi,actor]:continue
  sourceworld=field.source_world[row].numpy()
  if worldroad_tree.query(sourceworld,workers=1)[0]<.06:continue
  reassign_rigid_owner(field,[int(row)],actor,np.eye(4),tr['poses'][fi,actor],opt)
  with torch.no_grad():field.identity_probability[row]=torch.tensor(evidence_probability(True,False,True,False),dtype=torch.float32);field.identity_evidence_status[row]=1
  static_to_actor.append(dict(row=int(row),previous_owner=-1,new_owner=actor,source_frame=fi,source_raw_row=raw,independent_motion_source_key=key))
# Restore real body points withheld by B, never all unknown points. The gate uses
# BUILD semantic/depth, positive multiframe parent alpha*T, actual3D volume and
# absence of positive world-road conflict. No QUERY selection or RGB-error gate.
old_count=len(field.xyz);pool_owner=pool['source_candidate_owner'];add_parts=[]
for actor in TARGETS:
 source=np.flatnonzero(pool_owner==actor);strong=np.flatnonzero((field.owner.numpy()==actor)&(id_frames[:,actor]>=3)&(mass[:,actor]>=.5));bank=dict(np.load(ROOT/f'source_candidate_evidence_a{actor:02d}.npz'));body=body_bank(actor)
 if not len(strong) or not len(source):receipts.append(dict(actor=actor,admitted=0,reason='no repeated rendered body parent or no unknown measurements'));continue
 body_tree=cKDTree(body['xyz']);mean=field.xyz.detach().numpy();current_scales=field.scales.detach().numpy();cov_now=covariance(field.quaternion,field.scales).detach().numpy();precision=np.linalg.inv(cov_now[strong]);tree=cKDTree(mean[strong]);radius=3*float(current_scales[strong].max())
 bankkeys=bank['source_frame']*1000000+bank['source_raw_row'];order=np.argsort(bankkeys);sortedkeys=bankkeys[order];existing=cKDTree(mean[field.owner.numpy()==actor]);accepted=[];parent_rows=[];reasons=[]
 for pi in source:
  pi=int(pi);fi=int(pool['source_frame'][pi]);raw=int(pool['source_raw_row'][pi]);key=fi*1000000+raw
  if raw not in visibility[fi]['front_raw_row']:continue
  bp=np.searchsorted(sortedkeys,key)
  if bp==len(sortedkeys) or sortedkeys[bp]!=key:raise ValueError('pending raw provenance not in frozen candidate evidence')
  bi=int(order[bp]);actual_body=raw in visibility[fi]['body_raw_rows'][actor];moving=key in motion[actor]
  if bool(bank['road_positive'][bi]) or bool(bank['motion_static'][bi]):continue
  if not (actual_body or (moving and int(bank['semantic_repeat'][bi])>=1)):continue
  if body_tree.query(pool['xyz'][pi],workers=1)[0]>.08:continue
  neighbors=tree.query_ball_point(pool['xyz'][pi],radius)
  if not neighbors:continue
  ni=np.asarray(neighbors,np.int64);delta=pool['xyz'][pi]-mean[strong[ni]];mahal=np.einsum('ni,nij,nj->n',delta,precision[ni],delta);pick=int(np.argmin(mahal))
  if mahal[pick]>9 or existing.query(pool['xyz'][pi],workers=1)[0]<.02:continue
  if not np.allclose(transform(pool['xyz'][pi:pi+1],tr['poses'][fi,actor])[0],pool['source_world'][pi],atol=1e-4):raise ValueError('raw local/world coordinates do not agree')
  accepted.append(pi);parent_rows.append(int(strong[ni[pick]]));reasons.append('actual source body+depth+repeated rendered3Dparent' if actual_body else 'independent3Dmotion+semantic neighbor+visible raw depth+repeated parent')
 if accepted:
  accepted=np.asarray(accepted,np.int64);chosen=voxel_rows(pool['xyz'][accepted],.04);accepted=accepted[chosen];parent_rows=np.asarray(parent_rows)[chosen];reasons=[reasons[j] for j in chosen]
  remaining=max(0,cfg.max_gaussians-old_count-sum(len(p['xyz']) for p in add_parts));accepted=accepted[:remaining];parent_rows=parent_rows[:remaining];reasons=reasons[:remaining]
  if len(accepted):
   seed={key:pool[key][accepted].copy() for key in ['xyz','rgb','source_frame','source_raw_row','source_world','source_candidate_owner']}
   context=body['xyz'][voxel_rows(body['xyz'],.04)];measured=np.concatenate((context,seed['xyz']));all_size,all_q,_=local_geometry(measured,.04);size=all_size[-len(accepted):];q=all_q[-len(accepted):]
   seed.update(scales=size,quaternion=q,neighbor=np.arange(len(accepted),dtype=np.int64),owner=np.full(len(accepted),actor,np.int64),road=np.zeros(len(accepted),bool),observed_rgb=seed['rgb'].copy(),reference_rgb=np.zeros_like(seed['rgb']),reference_confidence=np.zeros(len(accepted),np.float32),identity_probability=evidence_probability(np.ones(len(accepted),bool),False,False,False).astype(np.float32),identity_evidence_status=np.ones(len(accepted),np.int64));add_parts.append(seed)
   for pi,parent,reason in zip(accepted,parent_rows,reasons):admitted.append(dict(pool_row=int(pi),owner=actor,source_frame=int(pool['source_frame'][pi]),source_raw_row=int(pool['source_raw_row'][pi]),repeated_body_parent=int(parent),reason=reason))
 receipts.append(dict(actor=actor,unknown_candidates=len(source),repeated_body_parents=len(strong),admitted=sum(1 for p in admitted if p['owner']==actor)))
for part in add_parts:append_measured_candidates(field,part,opt)
if len(field.xyz)>cfg.max_gaussians:raise ValueError('C exceeds matched densification cap')
torch.testing.assert_close(field.opacity_logit[:old_count],old_opacity,rtol=0,atol=0)
# Only changed ownership groups need new adjacency. Restore untouched context
# neighbors after generic coordinate/append helpers, preserving their training.
touched=set([p['previous_owner'] for p in road_meta]+([-1] if road_meta or static_to_actor else [])+[p['new_owner'] for p in static_to_actor]+[p['owner'] for p in admitted])
untouched=torch.tensor([int(a) not in touched for a in field.owner[:old_count]],dtype=torch.bool);field.neighbor[:old_count][untouched]=neighbor_before[untouched]
for actor in sorted(touched):
 rows=np.flatnonzero(field.owner.numpy()==actor);near=cKDTree(field.xyz.detach().numpy()[rows]).query(field.xyz.detach().numpy()[rows],k=min(2,len(rows)),workers=1)[1];field.neighbor[rows]=torch.as_tensor(rows[near[:,1] if near.ndim==2 else near])
if len(road_rows):
 before_ref=field.reference_rgb.clone();before_conf=field.reference_confidence.clone();reference=refresh(field,d)
 keep=torch.ones(len(field.xyz),dtype=torch.bool);keep[rr]=False;field.reference_rgb[keep]=before_ref[keep];field.reference_confidence[keep]=before_conf[keep];reference['scope']='only new actual road-owner rows retain refreshed targets; existing targets preserved'
else:reference={'refreshed':False,'reason':'no road owner changes'}
if not bool((field.owner[field.neighbor]==field.owner).all()):raise ValueError('cross-owner neighbor remained after correction')
out=ROOT/'gates/009000/owner';out.mkdir(exist_ok=False)
remaining=~np.isin(np.arange(len(pool['xyz'])),np.array([x['pool_row'] for x in admitted],np.int64));np.savez_compressed(out/'unresolved_candidate_pool.npz',**{k:v[remaining] for k,v in pool.items()})
receipt=dict(status='C_ONCE_ONLY_EVIDENCE_INTERVENTION_COMPLETE_REQUIRES49FRAME_GATE',global_step=9000,training_updates=0,old_Gaussians=old_count,new_Gaussians=len(field.xyz),Actor_to_static=len(road_rows),static_to_Actor=len(static_to_actor),unknown_measured_body_admitted=len(admitted),unknown_pool_retained=int(remaining.sum()),probability='independent Dirichlet evidence scores, not calibrated truth and not opacity',existing_opacity_unchanged=True,existing_opacity_scope='relative to after-split checkpoint; mixed split itself redistributes optical density between retained children',all_inputs_BUILD_only=True,QUERY_used_for_selection=False,development_QUERY_already_seen=True,fresh_independent_test=False,body_preservation_not_yet_accepted=True,actors=receipts,road_provenance=road_meta,static_to_Actor_provenance=static_to_actor,pool_admission_provenance=admitted,reference=reference,seconds=time.monotonic()-start)
saved['state']=model.state_dict();saved['optimizer']=opt.state_dict();saved['supervision_lineage']={'kind':'C_staged_actual_owner_and_measured_unknown_pool','parent_checkpoint':str(ROOT/'gates/009000/split/latest.pt'),'previous':saved.get('supervision_lineage'),'receipt':str(out/'owner.json')};save(out/'latest.pt',saved);write(out/'owner.json',receipt);save(ROOT/'main/latest.pt',saved);print(json.dumps({k:v for k,v in receipt.items() if 'provenance' not in k}),flush=True)
