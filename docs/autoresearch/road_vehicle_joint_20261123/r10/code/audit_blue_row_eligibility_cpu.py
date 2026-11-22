"""Read measured per-row alpha*T; distinguish blue contributors and exclusions.

No new raster, source selection, gate application, asset changes, or updates.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def pin(p):
    return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)


def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def mass_report(ids,mass,mask):
    return dict(rows=int(mask.sum()),positive_blue_rows_by_view=[int(((mass[:,j,5]>0)&mask).sum()) for j in range(mass.shape[1])],
        blue_alphaT_by_view=mass[mask,:,5].astype(np.float64).sum(0).tolist(),
        red_alphaT_by_view=mass[mask,:,4].astype(np.float64).sum(0).tolist(),
        body_alphaT_by_view=mass[mask,:,1].astype(np.float64).sum(0).tolist(),
        K_alphaT_by_view=mass[mask,:,0].astype(np.float64).sum(0).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.time();r=a.root
    assert not a.output.exists(),'Keep previous evidence'
    cfg=read(r/'local_measure_contract_v2.json');base=read(cfg['baseline_contract']);frames=cfg['frames']
    receipt=read(r/'measure_v1/receipt.json');assert receipt['complete']
    packet_paths=[r/f'measure_v1/{raw:03d}/all_context_rows.npz' for raw in frames]
    packets=[bank(path) for path in packet_paths]
    for z in packets:
        assert (np.diff(z['global_row_ids'])>0).all()
        assert z['ROI_alphaT_mass'].shape==(len(z['global_row_ids']),7)
        assert np.isfinite(z['ROI_alphaT_mass']).all() and (z['ROI_alphaT_mass']>=0).all()
    ids=np.unique(np.concatenate([z['global_row_ids'][z['native_context']&(z['ROI_alphaT_mass'][:,5]>0)] for z in packets]))
    mass=np.zeros((len(ids),len(frames),7),np.float32)
    present=np.zeros((len(ids),len(frames)),bool)
    owners=np.full((len(ids),len(frames)),np.iinfo(np.int32).min,np.int32)
    for j,z in enumerate(packets):
        where=np.searchsorted(z['global_row_ids'],ids);ok=where<len(z['global_row_ids'])
        ok[ok]&=z['global_row_ids'][where[ok]]==ids[ok]
        assert z['native_context'][where[ok]].all()
        mass[ok,j]=z['ROI_alphaT_mass'][where[ok]];owners[ok,j]=z['owner_ids'][where[ok]];present[ok,j]=True
    plan=read(base['context_plan']);permanent=np.asarray(plan['preserved_native_context_global_row_ids'],np.int64)
    deleted=np.asarray(cfg['deleted_target_owner_ids'],np.int32);assert np.array_equal(deleted,[10])
    allowed=(owners<=0)|np.isin(owners,deleted)
    allowed_all=((allowed|~present).all(1))
    selected_owner=(owners==21).any(1)
    other_positive=((owners>0)&~np.isin(owners,np.r_[deleted,21])&present).any(1)
    changed=np.any(np.where(present,owners,owners.max(1,keepdims=True))!=owners.max(1,keepdims=True),1)
    flags=dict(selected_native_owner21=selected_owner,other_retained_positive_owner=other_positive,
        explicitly_deleted_owner10=(owners==10).any(1),unowned_or_nonpositive=((owners<=0)|~present).all(1),
        any_five_K_positive=(mass[:,:,0]>0).any(1),any_five_body_positive=(mass[:,:,1]>0).any(1),
        permanent_old_guard=np.isin(ids,permanent),missing_from_any_view=~present.all(1),owner_inconsistent_across_present_views=changed)
    eligible=allowed_all&present.all(1)&~flags['any_five_K_positive']&~flags['permanent_old_guard']&~changed
    flags['scope_Kfree_oldguardfree_owner_allowed_body_overlap_diagnostic']=eligible
    flags['strict_also_body_zero']=eligible&~flags['any_five_body_positive']
    strict=bank(r/'measure_v1/native_candidates.npz')['global_row_ids']
    overlap=bank(r/'body_overlap_policy_v1/diagnostic_native_candidates.npz')['global_row_ids']
    flags['existing_strict_8816_set']=np.isin(ids,strict)
    flags['existing_red_based_body_overlap_5444_set']=np.isin(ids,overlap)
    flags['outside_both_existing_sets']=~(flags['existing_strict_8816_set']|flags['existing_red_based_body_overlap_5444_set'])
    # Independent flags above intentionally overlap; this priority partition sums exactly.
    order=['selected_native_owner21','other_retained_positive_owner','permanent_old_guard','any_five_K_positive','missing_from_any_view','owner_inconsistent_across_present_views','any_five_body_positive']
    partition={};remaining=np.ones(len(ids),bool)
    for name in order:
        subset=remaining&flags[name];partition[name]=mass_report(ids,mass,subset);remaining&=~subset
    partition['eligible_zero_body']=mass_report(ids,mass,remaining)
    partitions_per_owner=[];all_owner_values=np.unique(np.concatenate([z['owner_ids'][z['native_context']&(z['ROI_alphaT_mass'][:,5]>0)] for z in packets]))
    for owner in all_owner_values:
        per=[]
        for z in packets:
            m=z['native_context']&(z['owner_ids']==owner);v=z['ROI_alphaT_mass'][m,5]
            per.append(dict(rows=int((v>0).sum()),blue_alphaT_mass=float(v.astype(np.float64).sum())))
        partitions_per_owner.append(dict(owner=int(owner),views=per))
    direct=[]
    for raw,z in zip(frames,packets):
        ev=bank(Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz');baseline=bank(r/f'measure_v1/{raw:03d}/baseline_joint.npz');blue=ev['blue_cyan_diagnostic']
        direct.append(dict(raw=raw,fixed_blue_pixels=int(blue.sum()),
            all_native_rows_blue_alphaT=float(z['ROI_alphaT_mass'][z['native_context'],5].astype(np.float64).sum()),
            existing_road_rows_blue_alphaT=float(z['ROI_alphaT_mass'][~z['native_context'],5].astype(np.float64).sum()),
            direct_native_context_alphaT=float(baseline['context_alphaT'][blue].astype(np.float64).sum()),
            direct_existing_road_alphaT=float(baseline['road_alphaT'][blue].astype(np.float64).sum()),
            direct_canonical_alphaT=float(baseline['actor_alphaT'][blue].astype(np.float64).sum()),
            direct_scene_alpha_mass=float(baseline['alpha'][blue].astype(np.float64).sum())))
    top=[]
    score=mass[:,:,5].astype(np.float64).sum(1)
    for index in np.argsort(-score)[:30]:
        top.append(dict(global_row_id=int(ids[index]),owners_by_view=owners[index].tolist(),present_by_view=present[index].tolist(),blue_alphaT_by_view=mass[index,:,5].tolist(),K_alphaT_by_view=mass[index,:,0].tolist(),body_alphaT_by_view=mass[index,:,1].tolist(),permanent_old_guard=bool(flags['permanent_old_guard'][index])))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'blue_row_evidence.npz',global_row_ids=ids,frames=np.asarray(frames),owners_by_view=owners,present_by_view=present,ROI_alphaT_mass=mass,ROI_names=packets[0]['ROI_names'],**flags)
    report=dict(schema='R10.blue_native_row_qualification_CPU.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        measurement_receipt=pin(r/'measure_v1/receipt.json'),measurement_contract=pin(r/'local_measure_contract_v2.json'),baseline_contract=pin(Path(cfg['baseline_contract'])),context_plan=pin(Path(base['context_plan'])),input_packets=[pin(x) for x in packet_paths],
        frames=frames,blue_contributing_native_union_rows=len(ids),actual_fixed_pixel_direct_alpha=direct,owner_partition=partitions_per_owner,
        independent_overlapping_flags={name:mass_report(ids,mass,m) for name,m in flags.items()},
        mutually_exclusive_priority=order+['eligible_zero_body'],mutually_exclusive_partition=partition,largest_measured_rows=top,
        row_evidence=pin(a.output/'blue_row_evidence.npz'),
        interpretation='These are measured alpha*T integrals over the frozen blue diagnostic mask; the earlier 70% figure was premultiplied RGB share, a different quantity. Missing rows in other views mean unmeasured visibility, not certified zero contribution.',
        world_footprint_candidate_rule=dict(status='PENDING prepared geometry for blue contributors',native_owner_allowed='owner<=0 or explicitly deleted owner10 only; original canonical A20, native21 and all retained positive owners fixed',
            protection='actual all-five K=0, permanent-old-guard exclusion; body overlap only separate zero-step diagnostic, never an automatic production change',
            locality='all five view prepared mean/covariance; 3sigma ellipsoid footprint intersects registered non-colour ground band/world patch; road-relative height/depth interval reported against accepted local surface, not fitted to blue RGB',
            candidates_not_a_gate=True),
        current_native_identity_gate_remains_selected=True,retirement_or_optimization_authorized=False,seconds=time.time()-start)
    (a.output/'ledger.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,rows=len(ids),ledger=pin(a.output/'ledger.json'),per_view_direct=direct,independent_flags={k:v for k,v in report['independent_overlapping_flags'].items() if k in ['any_five_K_positive','any_five_body_positive','permanent_old_guard','missing_from_any_view','scope_Kfree_oldguardfree_owner_allowed_body_overlap_diagnostic','outside_both_existing_sets']}),ensure_ascii=False))


if __name__=='__main__':main()
