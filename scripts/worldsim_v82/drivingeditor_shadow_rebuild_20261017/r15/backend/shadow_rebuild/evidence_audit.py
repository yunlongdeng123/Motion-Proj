"""Periodic explicit ownership correction from current geometry and BUILD evidence.

Evidence buffers are NOT optimized logits. Repeated negative identity evidence
can revoke certainty, but cannot manufacture a static label. Owner changes need
both current visible contribution and actual repeated 3D surface measurements.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree


def audit_due(config,step):
    return step>=config.ownership_audit_first and (step-config.ownership_audit_first)%config.ownership_audit_interval==0


def classify_current(body,negative,road,body_frames,negative_frames,road_frames):
    """No positive supervision from unknown pixels; no static from complements."""
    b=(body_frames>=3)&(body>=.5)
    neg=(negative_frames>=3)&(negative>=.5)&(negative/np.maximum(body+negative,1e-9)>.95)
    r=(road_frames>=3)&(road>=.5)&(road/np.maximum(body+road,1e-9)>.8)
    return dict(body=b,contradicted=neg&~b,road=r&~b,mixed=b&(road_frames>=3)&(road>=.2))


def repeated_near(points,bank,coordinate='xyz',radius=.08,world_consistency=False):
    """Return >=3 distinct actual source frames and the nearest measured row."""
    points=np.asarray(points);count=np.zeros(len(points),np.int16);nearest=np.full(len(points),-1,np.int64)
    if not len(points) or not len(bank[coordinate]):return count,nearest
    xyz=bank[coordinate];tree=cKDTree(xyz)
    for start in range(0,len(points),2048):
        batch=points[start:start+2048]
        for j,ids in enumerate(tree.query_ball_point(batch,radius,workers=2)):
            if not ids:continue
            ids=np.asarray(ids,np.int64);pick=ids[np.argmin(np.linalg.norm(xyz[ids]-batch[j],axis=-1))]
            if world_consistency:
                ids=ids[np.linalg.norm(bank['source_world'][ids]-bank['source_world'][pick],axis=-1)<=radius]
            count[start+j]=len(np.unique(bank['source_frame'][ids]));nearest[start+j]=pick
    return count,nearest


def _build_only(bank,data):
    frame=bank['source_frame'];raw=bank['source_raw_row']
    if np.any(raw<0) or np.any(raw%5==0) or any(int(i) not in data.train_indices for i in np.unique(frame)):
        raise ValueError('non-BUILD measurement entered ownership correction')


def _save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)


def run_evidence_audit(model,data,optimizer,step,output):
    from .raster import cuda_raster
    from .losses import pixel_valid
    from .identity import reassign_rigid_owner
    from .ownership import reanchor_road
    from .repair_geometry import project_actor_constraints
    start=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    f=model.field;root=Path(model.config.ownership_measurement_root)
    # Input coverage is explicit; all points may enter the candidate set, without
    # a source_candidate_owner filter. Only actors with actual BUILD banks certify.
    targets=[a for a in range(len(data.actor_ids)) if (root/f'body_bank_a{a:02d}.npz').is_file()]
    if not targets:raise ValueError('no actual BUILD body banks for ownership audit')
    n=len(f.xyz);k=len(targets);mass=np.zeros((n,2*k+2),np.float32);frames=np.zeros_like(mass,dtype=np.uint8)
    _save(out/'progress.json',dict(state='RENDERING_CURRENT_EVIDENCE',step=step,targets=targets))
    for j,index in enumerate(data.train_indices):
        frame=data.frame(index)
        with torch.no_grad():
            g=f.evaluate(frame.poses,frame.valid,frame.camera.center)
            valid=pixel_valid(frame);body=frame.masks&valid
            road=frame.road_certified&(frame.depth>0)&~body.any(-1)
            domains=torch.cat((body[...,targets],(valid&~body)[...,targets],road[...,None],torch.ones_like(road)[...,None]),-1).float()
        feature=torch.ones(len(g['xyz']),2*k+2,device=f.xyz.device,requires_grad=True)
        image=cuda_raster(frame.camera,g['xyz'],g['cov'],g['opacity'],feature,model.config.max_features)
        contribution=torch.autograd.grad((image*domains).sum(),feature)[0].detach().cpu().numpy()
        rows=g['rows'].cpu().numpy();mass[rows]+=contribution;frames[rows]+=(contribution>.02).astype(np.uint8)
        del image,feature,contribution,g
        if (j+1)%10==0:_save(out/'progress.json',dict(state='RENDERING_CURRENT_EVIDENCE',step=step,frames=j+1,total=len(data.train_indices)))
    np.savez_compressed(out/'current_contributions.npz',mass=mass,frames=frames,targets=targets,
                        lineage=f.sgcm_lineage.cpu().numpy(),source_frame=f.source_frame.cpu().numpy(),source_raw_row=f.source_raw_row.cpu().numpy())
    owner=f.owner.cpu().numpy().copy();status=f.identity_evidence_status.cpu().numpy().copy()
    previous_status=status.copy();previous_owner=owner.copy();xyz=f.xyz.detach().cpu().numpy().copy()
    roadbank=dict(np.load(root/'actual_world_road_support.npz'));_build_only(roadbank,data)
    roadtree=cKDTree(roadbank['xyz']);bodybanks={a:dict(np.load(root/f'body_bank_a{a:02d}.npz')) for a in targets}
    for bank in bodybanks.values():_build_only(bank,data)
    decisions=[];funnel=[];transferred=set();touched=set();unknown_rows=[];body_cert=[];static_cert=[]
    # Move CUDA data once; explicit modifications below are no-grad and do not
    # rewrite immutable original measurement provenance or initializer status.
    for col,actor in enumerate(targets):
        bank=bodybanks[actor];own=owner==actor
        c=classify_current(mass[:,col],mass[:,k+col],mass[:,-2],frames[:,col],frames[:,k+col],frames[:,-2])
        revoke=np.flatnonzero(own&(c['contradicted']|c['mixed']))
        status[revoke]=0;unknown_rows.extend(revoke.tolist())
        candidates=np.flatnonzero(own&c['body']&~c['mixed'])
        hits,near=repeated_near(xyz[candidates],bank)
        confirmed=candidates[hits>=3];status[confirmed]=1;body_cert.extend(confirmed.tolist())
        # Current visible support discovers static candidates even if the old
        # source tag has no entry. Repeated WORLD body support prevents moving a
        # distant road Gaussian merely because its SH paints a car silhouette.
        static=np.flatnonzero((owner<0)&c['body']&(mass[:,col]/np.maximum(mass[:,:k].sum(-1),1e-9)>.8))
        shits,snear=repeated_near(xyz[static],bank,coordinate='source_world')
        distance,rnear=roadtree.query(xyz[static],workers=2)
        road_veto=(distance<.08)&(roadbank['support_frames'][rnear]>=3)
        # Matching the CENTER does not certify a large Gaussian's whole support.
        # Such mixed support needs decomposition, not whole-point reassignment.
        seed_sigma=data.seed['scales'][data.seed['owner']==actor].max()
        shape_ok=f.scales[torch.as_tensor(static,device=f.xyz.device)].detach().amax(-1).cpu().numpy()<=seed_sigma*model.config.actor_scale_growth_limit
        eligible=static[(shits>=3)&~road_veto&shape_ok]
        selected_near=snear[(shits>=3)&~road_veto&shape_ok]
        eligible_with_source=[]
        for row,measurement in zip(eligible,selected_near):
            row=int(row);fi=int(bank['source_frame'][measurement])
            if row in transferred or not data.valid[fi,actor]:continue
            local=(xyz[row]-data.poses[fi,actor,:3,3])@data.poses[fi,actor,:3,:3]
            if np.any(np.abs(local)>f.actor_box_half[actor].cpu().numpy()+model.config.actor_box_margin_m):continue
            eligible_with_source.append((row,int(measurement),fi))
        # Batch by real source pose, avoiding a full neighbor rebuild per point.
        for fi in sorted({v[2] for v in eligible_with_source}):
            rows=[v[0] for v in eligible_with_source if v[2]==fi]
            reassign_rigid_owner(f,rows,actor,np.eye(4),data.poses[fi,actor],optimizer,rebuild_neighbors=False)
            with torch.no_grad():
                f.geometry_origin[rows]=f.xyz[rows]
                f.geometry_origin_sigma[rows]=f.scales[rows].amax(-1)
            status[rows]=1;owner[rows]=actor;body_cert.extend(rows);transferred.update(rows);touched.update([-1,actor])
        for row,m,fi in eligible_with_source:
            decisions.append(dict(row=row,old_owner=-1,new_owner=actor,source_frame=fi,source_raw_row=int(bank['source_raw_row'][m]),measured_world=bank['source_world'][m].tolist(),reason='repeated actual world body + current visible identity, no positive world-road conflict'))
        # Actor -> static requires repeated measured road close in its actual
        # local coordinates AND the same measured world surface across frames.
        rr=np.flatnonzero(own&c['road']);localroad={'xyz':[], 'source_world':[], 'source_frame':[], 'source_raw_row':[]}
        for fi in np.unique(roadbank['source_frame']):
            fi=int(fi)
            if not data.valid[fi,actor]:continue
            take=roadbank['source_frame']==fi;world=roadbank['xyz'][take];pose=data.poses[fi,actor]
            localroad['xyz'].append((world-pose[:3,3])@pose[:3,:3]);localroad['source_world'].append(world)
            localroad['source_frame'].append(roadbank['source_frame'][take]);localroad['source_raw_row'].append(roadbank['source_raw_row'][take])
        localroad={key:np.concatenate(value) for key,value in localroad.items()}
        rhits,rsource=repeated_near(xyz[rr],localroad,radius=.06,world_consistency=True)
        bodydistance=cKDTree(bank['xyz']).query(xyz[rr],workers=2)[0]
        body_exclusion=np.maximum(.08,4*f.scales[torch.as_tensor(rr,device=f.xyz.device)].detach().amax(-1).cpu().numpy())
        approved=rr[(rhits>=3)&(bodydistance>body_exclusion)]
        sources=rsource[(rhits>=3)&(bodydistance>body_exclusion)]
        if len(approved):
            anchor=localroad['source_world'][sources];roadseed=np.flatnonzero(data.seed['road'])
            nearest=cKDTree(data.seed['xyz'][roadseed]).query(anchor,workers=2)[1];seedrow=roadseed[nearest]
            device=f.xyz.device;rows=torch.as_tensor(approved,device=device)
            reanchor_road(f,rows,torch.as_tensor(anchor,device=device),torch.as_tensor(data.seed['quaternion'][seedrow],device=device),optimizer)
            with torch.no_grad():
                f.log_scales[rows]=torch.as_tensor(data.seed['scales'][seedrow],device=device).log()
                f.geometry_origin_sigma[rows]=f.scales[rows].amax(-1)
            owner[approved]=-1;status[approved]=2;static_cert.extend(approved.tolist());touched.update([-1,actor])
            for row,m in zip(approved,sources):
                decisions.append(dict(row=int(row),old_owner=actor,new_owner=-1,source_frame=int(localroad['source_frame'][m]),source_raw_row=int(localroad['source_raw_row'][m]),measured_world=localroad['source_world'][m].tolist(),reason='repeated current road contribution + repeated same world road anchor + no measured body'))
        funnel.append(dict(actor=actor,owner_points=int(own.sum()),known_contradicted=int((own&c['contradicted']).sum()),mixed_held_unknown=int((own&c['mixed']).sum()),own_body_candidates=len(candidates),own_body_3d_confirmed=len(confirmed),
            static_body_candidates_without_source_tag=len(static),static_repeated_world_body=int((shits>=3).sum()),static_positive_road_veto=int(((shits>=3)&road_veto).sum()),static_large_support_veto=int(((shits>=3)&~shape_ok).sum()),static_to_actor=len(eligible_with_source),
            actor_road_candidates=len(rr),actor_repeated_same_world_road=int((rhits>=3).sum()),actor_to_static=len(approved)))
    with torch.no_grad():
        f.identity_evidence_status.copy_(torch.as_tensor(status,device=f.xyz.device))
        for rows,value in [(unknown_rows,[.1,.1,.8]),(body_cert,[.05,.9,.05]),(static_cert,[.9,.05,.05])]:
            if rows:f.identity_probability[rows]=f.identity_probability.new_tensor(value)
        # Only changed owner groups require neighborhood rebuild.
        for actor in touched:
            rows=np.flatnonzero(owner==actor)
            if not len(rows):continue
            points=f.xyz[torch.as_tensor(rows,device=f.xyz.device)].cpu().numpy()
            near=cKDTree(points).query(points,k=min(2,len(rows)),workers=2)[1]
            f.neighbor[torch.as_tensor(rows,device=f.xyz.device)]=torch.as_tensor(rows[near[:,1] if near.ndim==2 else near],device=f.xyz.device)
        if not bool((f.owner[f.neighbor]==f.owner).all()):raise ValueError('ownership audit left cross-owner neighbors')
        project_actor_constraints(f,optimizer)
    matrix=np.zeros((3,3),np.int64);np.add.at(matrix,(previous_status,status),1)
    receipt=dict(status='COMPLETE',step=step,frames=len(data.train_indices),actors_with_actual_3d_banks=targets,
        policy='explicit current-evidence update; not learned logits; unknown pixels ignored; negative evidence never becomes static',
        audit_before_opacity_reset=True,source_tag_required=False,initial_provenance_unchanged=True,
        probability='declared evidence scores, not calibrated truth',status_transition_raw_0U_1Actor_2Static=matrix.tolist(),
        actual_owner_changes=int((owner!=previous_owner).sum()),funnel=funnel,owner_changes=decisions,
        pool_admission=0,pool_policy='external measurements preserved; this repair audits existing geometry only',
        mixed_policy='mark uncertain; preserve geometry; no unsupported cut or owner reassignment',
        seconds=time.monotonic()-start,human_verdict=None)
    _save(out/'receipt.json',receipt);_save(out/'progress.json',dict(state='COMPLETE',step=step))
    return receipt
