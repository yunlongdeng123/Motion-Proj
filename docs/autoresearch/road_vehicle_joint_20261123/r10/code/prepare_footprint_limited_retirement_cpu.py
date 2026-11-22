"""Strict whole-kernel diagnostic from measured full-image semantic footprints.

No RGB selection, no mass epsilon, no fitting, no source asset mutation.
Mixed-footprint rows remain intact, for separate splitting/reconstruction work.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def columns(z):
    names=[str(x) for x in z['ROI_names'].tolist()]
    aliases=[('full_image','full','FULL'),('outside_allowed_edit','outside_edit','outside'),('actor_protected','car','actual_car_protected'),('K_protected','K','other_K'),
        ('inside_allowed_edit','inside_edit'),('outside_replacement','outside_replacement_support'),('inside_replacement','inside_replacement_support')]
    chosen=[]
    for variants in aliases:
        hit=[names.index(name) for name in variants if name in names]
        assert len(hit)==1,(names,variants)
        chosen.append(hit[0])
    mass=z['ROI_alphaT_mass'][:,chosen]
    assert mass.shape==(len(z['global_row_ids']),7)
    assert np.isfinite(mass).all() and (mass>=0).all()
    return mass


def summarize(mask,mass):
    values=mass[mask].astype(np.float64)
    return dict(rows=int(mask.sum()),full_alphaT_by_view=values[:,:,0].sum(0).tolist(),outside_alphaT_by_view=values[:,:,1].sum(0).tolist(),
        car_alphaT_by_view=values[:,:,2].sum(0).tolist(),K_alphaT_by_view=values[:,:,3].sum(0).tolist(),
        inside_edit_alphaT_by_view=values[:,:,4].sum(0).tolist(),outside_replacement_alphaT_by_view=values[:,:,5].sum(0).tolist(),inside_replacement_alphaT_by_view=values[:,:,6].sum(0).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['contract','measurement','asset','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();start=time.time();assert not a.output.exists()
    cfg=read(a.contract);base=read(cfg['baseline_contract']);frames=cfg['frames'];receipt=read(a.measurement/'receipt.json')
    assert receipt['complete'] and not receipt.get('error')
    assert receipt['frames']==frames and receipt['baseline_contract_sha256']==sha(Path(cfg['baseline_contract']))
    assert receipt['measurement_contract']['sha256']==sha(a.contract)
    assert receipt['all_native_source_parameters_unchanged'] is True and receipt['all_native_retained'] is True
    assert max(int(receipt.get(k,0)) for k in ['nativeF','actual_nativeF'])>=len(frames)
    assert max(int(receipt.get(k,0)) for k in ['nativeB','actual_nativeB'])>=len(frames)
    asset_path=a.asset/'manifest.json';manifest=read(asset_path)
    source_ref=receipt.get('asset_manifest',receipt.get('new_local_asset_manifest'))
    assert source_ref and source_ref['sha256']==sha(asset_path),'Footprint must bind the actual current new0 asset'
    assert manifest['parent_contract']['sha256']==sha(Path(cfg['baseline_contract']))
    assert manifest['actor_parent_sha256']==cfg['accepted_actor_sha256']
    assert manifest['native_gate']['mode']=='identity_no_retirement'
    mask_ref=receipt['mask_receipt'];assert sha(Path(mask_ref['path']))==mask_ref['sha256']
    mask_receipt=read(mask_ref['path']);assert mask_receipt['complete'] and mask_receipt['frames']==frames
    assert mask_receipt['parent_asset_manifest']['sha256']==sha(asset_path)
    assert mask_receipt['measurement_contract']['sha256']==sha(a.contract)
    assert mask_receipt['source_colour_used_for_masks'] is False and mask_receipt['broad_canonical_alpha_not_used_as_car_mask'] is True
    surface_ref=mask_receipt['direct_surface_receipt'];assert sha(Path(surface_ref['path']))==surface_ref['sha256']
    surface_receipt=read(surface_ref['path']);assert surface_receipt['complete']
    assert surface_receipt['geometry_sha256']==manifest['geometry']['npz']['sha256'] and surface_receipt['final_atlas_sha256']==manifest['source']['field']['sha256']
    domain_checks=[]
    for raw in frames:
        row=next(x for x in mask_receipt['rows'] if x['raw']==raw)
        for key in ['masks','fixed_eval','rectified_masks','direct_surface_projection']:
            assert sha(Path(row[key]['path']))==row[key]['sha256']
        domain=bank(row['masks']['path']);ev=bank(row['fixed_eval']['path']);protected=bank(row['rectified_masks']['path']);surface=bank(row['direct_surface_projection']['path'])
        car=protected['actor_visible'] if row['car_mask_kind']=='opened_strict_real_V' else protected['actor_visible_proxy']
        assert np.array_equal(domain['K_protected'],protected['protected']) and np.array_equal(domain['K_protected'],ev['K_protected'])
        assert np.array_equal(domain['actor_protected'],car)
        expected=ev['full_original_band']&~protected['protected']&~car
        assert np.array_equal(domain['inside_allowed_edit'],expected)
        assert domain['full_image'].all() and np.array_equal(domain['outside_allowed_edit'],~expected)
        assert np.array_equal(domain['direct_surface_shown'],surface['shown'])
        replacement=expected&surface['shown']
        assert np.array_equal(domain['inside_replacement'],replacement) and np.array_equal(domain['outside_replacement'],~replacement)
        domain_checks.append(dict(raw=raw,car_mask_kind=row['car_mask_kind'],edit_pixels=int(expected.sum()),replacement_pixels=int(replacement.sum()),
            full_image_partition_exact=True,current_Gaussian_alpha_not_used=True,independently_reconstructed_from_pinned_inputs=True,masks=row['masks']))
    packet_paths=[a.measurement/f'{raw:03d}/all_context_rows.npz' for raw in frames]
    zero_receipt_ref=receipt['zero_receipt'];assert sha(Path(zero_receipt_ref['path']))==zero_receipt_ref['sha256']
    zero_root=Path(zero_receipt_ref['path']).parent
    per_frame_proofs=[]
    for raw,path in zip(frames,packet_paths):
        row=next(x for x in receipt['rows'] if x['raw']==raw)
        assert row['packet']['sha256']==sha(path) and row['full_RGB_alpha_byteexact_new0'] is True
        sentinel=next(x for x in row['conservation'] if x['region']=='single_pixel_sentinel')
        assert sentinel['pixels']==1 and sentinel['absolute_sum_error']<=1e-5
        measured=bank(a.measurement/f'{raw:03d}/measurement_joint.npz');zero=bank(zero_root/f'{raw:03d}/joint_render.npz')
        for key in ['rgb','alpha']:
            assert measured[key].dtype==zero[key].dtype and measured[key].shape==zero[key].shape
            assert np.isfinite(measured[key]).all() and measured[key].tobytes()==zero[key].tobytes()
        per_frame_proofs.append(dict(raw=raw,actual_RGBA_exact_new_zero=True,sentinel=sentinel))
    zs=[bank(path) for path in packet_paths];masses=[columns(z) for z in zs]
    initial=[]
    for z,m in zip(zs,masses):
        assert (np.diff(z['global_row_ids'])>0).all()
        assert np.array_equal(z['old_permanent_guard'],np.isin(z['global_row_ids'],read(base['context_plan'])['preserved_native_context_global_row_ids']))
        initial.append(z['global_row_ids'][z['native_context']&(z['owner_ids']<=0)&(m[:,0]>0)])
    ids=np.unique(np.concatenate(initial));mass=np.zeros((len(ids),len(frames),7),np.float32)
    owners=np.full((len(ids),len(frames)),-999,np.int32);present=np.zeros_like(owners,bool);native=np.zeros_like(owners,bool)
    for j,(z,m) in enumerate(zip(zs,masses)):
        positions=np.searchsorted(z['global_row_ids'],ids);ok=positions<len(z['global_row_ids']);ok[ok]&=z['global_row_ids'][positions[ok]]==ids[ok]
        present[ok,j]=True;native[ok,j]=z['native_context'][positions[ok]];owners[ok,j]=z['owner_ids'][positions[ok]];mass[ok,j]=m[positions[ok]]
    preserve=np.asarray(read(base['context_plan'])['preserved_native_context_global_row_ids'],np.int64)
    flags=dict(missing_from_any_view=~present.all(1),not_native_in_any_view=~native.all(1),retained_positive_owner=((owners>0)&present).any(1),
        permanent_old_guard=np.isin(ids,preserve),outside_edit_positive=(mass[:,:,1]>0).any(1),car_positive=(mass[:,:,2]>0).any(1),K_positive=(mass[:,:,3]>0).any(1),outside_replacement_positive=(mass[:,:,5]>0).any(1))
    policy_free=~np.logical_or.reduce([flags[k] for k in ['missing_from_any_view','not_native_in_any_view','retained_positive_owner','permanent_old_guard','car_positive','K_positive']])
    edit_keep=policy_free&~flags['outside_edit_positive']&(mass[:,:,4]>0).any(1)
    keep=policy_free&~flags['outside_replacement_positive']&(mass[:,:,6]>0).any(1)
    assert not (keep&~edit_keep).any()
    mixed=flags['outside_replacement_positive']&(mass[:,:,6]>0).any(1)
    assert (mass[keep][:,:,[1,2,3,5]]==0).all() and (owners[keep]<=0).all() and native[keep].all() and present[keep].all()
    # Independent full/inside/outside VJPs may sum in different float32 order.
    # This check never relaxes the exact-zero qualification above.
    roundoffs=[]
    for outside,inside in [(1,4),(5,6)]:
        d=mass[:,:,0].astype(np.float64)-mass[:,:,outside]-mass[:,:,inside]
        bound=64*np.finfo(np.float32).eps*(1+mass[:,:,0].astype(np.float64)+mass[:,:,outside]+mass[:,:,inside])
        roundoffs.append(dict(outside_column=outside,inside_column=inside,max_abs=float(abs(d).max(initial=0)),reference_64eps_fraction_max=float((abs(d)/bound).max(initial=0)),reference_64eps_exceed_count=int((abs(d)>bound).sum()),
            diagnostic_only_not_a_selector_or_new_acceptance_gate=True))
    selected=mass[keep];chosen=ids[keep]
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'candidate_native_rows.npz',global_row_ids=chosen,frames=np.asarray(frames),owners_by_view=owners[keep],
        allview_full_mass=selected[:,:,0],allview_outside_mass=selected[:,:,1],allview_car_mass=selected[:,:,2],allview_K_mass=selected[:,:,3],
        allview_outside_edit_mass=selected[:,:,1],allview_inside_edit_mass=selected[:,:,4],
        allview_inside_mass=selected[:,:,4],allview_outside_replacement_mass=selected[:,:,5],allview_inside_replacement_mass=selected[:,:,6],
        allview_present=present[keep],allview_native=native[keep])
    np.savez_compressed(a.output/'diagnostic_gate0.npz',global_row_ids=chosen,gate=np.zeros(len(chosen),np.float32))
    np.savez_compressed(a.output/'edit_contained_candidate_native_rows.npz',global_row_ids=ids[edit_keep],frames=np.asarray(frames),owners_by_view=owners[edit_keep],measured_mass=mass[edit_keep])
    np.savez_compressed(a.output/'edit_contained_diagnostic_gate0.npz',global_row_ids=ids[edit_keep],gate=np.zeros(edit_keep.sum(),np.float32))
    np.savez_compressed(a.output/'qualification_ledger.npz',global_row_ids=ids,frames=np.asarray(frames),owners_by_view=owners,present_by_view=present,native_by_view=native,
        measured_mass=mass,ROI_names=np.asarray(['full_image','outside_allowed_edit','actor_protected','K_protected','inside_allowed_edit','outside_replacement','inside_replacement']),selected=keep,edit_contained=edit_keep,replacement_mixed=mixed,**flags)
    order=['retained_positive_owner','permanent_old_guard','missing_from_any_view','not_native_in_any_view','K_positive','car_positive','outside_replacement_positive']
    remaining=np.ones(len(ids),bool);part={}
    for name in order:
        m=remaining&flags[name];part[name]=summarize(m,mass);remaining&=~m
    remaining&=(mass[:,:,6]>0).any(1)
    assert np.array_equal(remaining,keep)
    part['whole_kernel_candidate']=summarize(keep,mass)
    proof=dict(schema='R10.semantic_footprint_limited_whole_kernel_scope.v1',complete=True,decision='SCOPE_VALID_DIAGNOSTIC_ONLY',
        policy='full_image_outside_edit_and_replacement_actual_car_K_all_five_exact_zero',frames=frames,candidate_rows=len(chosen),edit_contained_candidate_rows=int(edit_keep.sum()),initial_unowned_visible_union_rows=len(ids),
        baseline_contract_sha256=sha(Path(cfg['baseline_contract'])),measurement_contract=pin(a.contract),actual_attribution_receipt=pin(a.measurement/'receipt.json'),actual_attribution_receipt_sha256=sha(a.measurement/'receipt.json'),
        asset_manifest=pin(asset_path),candidate_npz_sha256=sha(a.output/'candidate_native_rows.npz'),gate_npz_sha256=sha(a.output/'diagnostic_gate0.npz'),input_packets=[pin(x) for x in packet_paths],
        mask_receipt=mask_ref,independent_pixel_domain_checks=domain_checks,independent_new_zero_RGBA_and_single_pixel_sentinel=per_frame_proofs,
        allowed_edit_domain='registered full non-colour ground band minus actual strict visible actor mask at 86/110, repaired-instance proxy in evaluation-only views, and other protected K; RGB and broad body-alpha footprint are not domain selectors',
        all_five_outside_exact_zero=bool((selected[:,:,1]==0).all()),all_five_outside_replacement_exact_zero=bool((selected[:,:,5]==0).all()),all_five_car_K_exact_zero=bool((selected[:,:,2:4]==0).all()),
        zero_proof_is_vacuous=len(chosen)==0,mass_tolerance_for_selection=0.,all_five_rows_present=True,old_permanent_guard_preserved=True,
        native21_and_all_retained_positive_owners_preserved=True,selection_uses_RGB=False,selection_uses_residual=False,full_scene_contribution_required=True,
        domain_certificate='measured only in the five immutable cameras and current native visibility contract; not arbitrary-camera support certification',
        independent_partition_float32_roundoff=roundoffs,float32_partition_check_never_relaxes_exact_zero_selection=True,overlapping_rejection={k:summarize(m,mass) for k,m in flags.items()},
        exclusive_priority=order,exclusive_partition=part,edit_contained_candidates=summarize(edit_keep,mass),replacement_contained_candidates=summarize(keep,mass),
        mixed_footprint_rows=summarize(mixed,mass),entirely_outside_replacement_rows=summarize(flags['outside_replacement_positive']&~(mass[:,:,6]>0).any(1),mass),
        mixed_kernel_action='preserve; separate splitting/local reconstruction requires a new explicit asset and measured readout',
        original_native_identity_asset_not_modified=True,whole_kernel_retirement_not_production=True,actual_zero_step_required=True,
        actual_zero_step_acceptance='compare every view outside allowed edit, strict V/proxy, K, full original band, unknown geometry/material and scene alpha holes; no optimization',
        CPU_only=True,nativeF=0,nativeB=0,updates=0,optimization_allowed=False,quality_accepted=False,seconds=time.time()-start)
    (a.output/'scope.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,candidate_rows=len(chosen),scope=pin(a.output/'scope.json'),selected=part['whole_kernel_candidate'],mixed=proof['mixed_footprint_rows']),ensure_ascii=False))


if __name__=='__main__':main()
