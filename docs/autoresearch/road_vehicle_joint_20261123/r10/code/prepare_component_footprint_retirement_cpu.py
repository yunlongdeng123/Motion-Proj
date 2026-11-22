"""Strict typed native-unowned plus old-RoadRows whole-kernel diagnostic from fresh measured footprints.

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
    sentinel=names.index('single_pixel_sentinel');chosen.append(sentinel)
    mass=z['ROI_alphaT_mass'][:,chosen]
    assert mass.shape==(len(z['global_row_ids']),8)
    assert np.isfinite(mass).all() and (mass>=0).all()
    return mass


def summarize(mask,mass):
    values=mass[mask].astype(np.float64)
    return dict(rows=int(mask.sum()),full_alphaT_by_view=values[:,:,0].sum(0).tolist(),outside_alphaT_by_view=values[:,:,1].sum(0).tolist(),
        car_alphaT_by_view=values[:,:,2].sum(0).tolist(),K_alphaT_by_view=values[:,:,3].sum(0).tolist(),
        inside_edit_alphaT_by_view=values[:,:,4].sum(0).tolist(),outside_replacement_alphaT_by_view=values[:,:,5].sum(0).tolist(),inside_replacement_alphaT_by_view=values[:,:,6].sum(0).tolist())


def qualify_components(ids,mass,owners,present,native,guards,component,preserve):
    assert mass.shape==(len(ids),present.shape[1],8) and present.shape[1]==5
    assert np.isfinite(mass).all() and (mass>=0).all()
    expected_native=(component==0)[:,None]
    flags=dict(missing_from_any_view=~present.all(1),component_inconsistent_in_any_view=((native!=expected_native)&present).any(1),
        retained_positive_owner=((owners>0)&native&present).any(1),
        permanent_old_guard=guards.any(1)|((component==0)&np.isin(ids,preserve)),
        outside_edit_positive=(mass[:,:,1]>0).any(1),car_positive=(mass[:,:,2]>0).any(1),K_positive=(mass[:,:,3]>0).any(1),
        outside_replacement_positive=(mass[:,:,5]>0).any(1))
    policy_free=~np.logical_or.reduce([flags[k] for k in ['missing_from_any_view','component_inconsistent_in_any_view','retained_positive_owner','permanent_old_guard','car_positive','K_positive']])
    edit_keep=policy_free&~flags['outside_edit_positive']&(mass[:,:,4]>0).any(1)
    keep=edit_keep&~flags['outside_replacement_positive']&(mass[:,:,6]>0).any(1)
    mixed=flags['outside_replacement_positive']&(mass[:,:,6]>0).any(1)
    return flags,policy_free,edit_keep,keep,mixed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['contract','measurement','old-road-identity','output']:p.add_argument('--'+k,type=Path,required=True)
    assets=p.add_mutually_exclusive_group(required=True);assets.add_argument('--asset',type=Path);assets.add_argument('--asset-manifest',type=Path)
    p.add_argument('--masks',type=Path)
    p.add_argument('--old-road-identity-sha256',default='3bf0d887bc89b755ed81aa3315cf94f8f84dcb6ff3a2cf5c3fdc8213a94ce456')
    a=p.parse_args();start=time.time();assert not a.output.exists()
    cfg=read(a.contract);base=read(cfg['baseline_contract']);frames=cfg['frames'];receipt=read(a.measurement/'receipt.json')
    assert receipt['complete'] and not receipt.get('error')
    assert receipt['frames']==frames and receipt['baseline_contract_sha256']==sha(Path(cfg['baseline_contract']))
    assert receipt['measurement_contract']['sha256']==sha(a.contract)
    assert receipt['all_native_source_parameters_unchanged'] is True and receipt['all_native_retained'] is True
    assert max(int(receipt.get(k,0)) for k in ['nativeF','actual_nativeF'])>=len(frames)
    assert max(int(receipt.get(k,0)) for k in ['nativeB','actual_nativeB'])>=len(frames)
    asset_path=a.asset_manifest or a.asset/'manifest.json';manifest=read(asset_path)
    source_ref=receipt.get('asset_manifest',receipt.get('new_local_asset_manifest'))
    assert source_ref and source_ref['sha256']==sha(asset_path),'Footprint must bind the actual current new0 asset'
    assert manifest['parent_contract']['sha256']==sha(Path(cfg['baseline_contract']))
    assert manifest['actor_parent_sha256']==cfg['accepted_actor_sha256']
    assert manifest['native_gate']['mode']=='identity_no_retirement'
    mask_ref=receipt['mask_receipt'];assert sha(Path(mask_ref['path']))==mask_ref['sha256']
    if a.masks:
        given_mask_receipt=a.masks/'receipt.json' if a.masks.is_dir() else a.masks
        assert sha(given_mask_receipt)==mask_ref['sha256'],'Explicit masks must be the measured immutable masks'
    identity_path=a.old_road_identity/'receipt.json' if a.old_road_identity.is_dir() else a.old_road_identity
    assert sha(identity_path)==a.old_road_identity_sha256
    identity=read(identity_path);assert identity['complete'] and identity['schema']=='R10.old_road_world_row_identity_CPU.v1'
    assert identity['parent_contract']['sha256']==sha(Path(cfg['baseline_contract']))
    for key in ['parent_contract','road_asset','persistent_scene_manifest','persistent_scene_proof','global_identity']:
        assert sha(Path(identity[key]['path']))==identity[key]['sha256']
    road_identity=bank(identity['global_identity']['path']);offset=int(road_identity['native_global_row_count'])
    assert offset==identity['native_global_row_count']
    assert np.array_equal(road_identity['road_active_index'],np.arange(identity['road_active_rows'],dtype=np.int64))
    assert np.array_equal(road_identity['global_row_id'],offset+road_identity['road_active_index'])
    road_bank=bank(identity['road_asset']['path'])
    assert np.array_equal(road_identity['world_row_id'],np.flatnonzero(road_bank['active']))
    del road_bank
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
    raw_feature_proofs=[]
    for raw,z,m in zip(frames,zs,masses):
        raw_path=a.measurement/f'{raw:03d}/raw_feature_alphaT_vjp.npz';grad=bank(raw_path)
        assert int(grad['context_prefix_rows'])==len(z['global_row_ids'])
        assert np.array_equal(grad['context_global_row_ids'],z['global_row_ids'])
        for ci,name in enumerate(['full_image','outside_allowed_edit','actor_protected','K_protected','inside_allowed_edit','outside_replacement','inside_replacement','single_pixel_sentinel']):
            x=grad['alphaT_'+name][:len(z['global_row_ids'])]
            assert x.dtype==m[:,ci].dtype and x.tobytes()==m[:,ci].tobytes()
        raw_feature_proofs.append(dict(raw=raw,raw_feature_packet=pin(raw_path),context_prefix_rows_match=True,all_8_VJP_columns_byteexact=True))
        del grad
    initial=[]
    for z,m in zip(zs,masses):
        assert (np.diff(z['global_row_ids'])>0).all()
        assert np.array_equal(z['old_permanent_guard'],np.isin(z['global_row_ids'],read(base['context_plan'])['preserved_native_context_global_row_ids']))
        assert np.array_equal(z['native_context'],z['global_row_ids']<offset)
        road=~z['native_context'];index=z['global_row_ids'][road]-offset
        assert ((index>=0)&(index<len(road_identity['global_row_id']))).all()
        assert np.array_equal(z['global_row_ids'][road],road_identity['global_row_id'][index])
        assert (z['owner_ids'][road]==0).all() and not z['old_permanent_guard'][road].any()
        initial.append(z['global_row_ids'][((z['native_context']&(z['owner_ids']<=0))|road)&(m[:,0]>0)])
    ids=np.unique(np.concatenate(initial));mass=np.zeros((len(ids),len(frames),8),np.float32)
    owners=np.full((len(ids),len(frames)),-999,np.int32);present=np.zeros_like(owners,bool);native=np.zeros_like(owners,bool)
    guards=np.zeros_like(owners,bool);packet_position=np.full(owners.shape,-1,np.int64)
    component=(ids>=offset).astype(np.uint8);road_index=np.where(component==1,ids-offset,-1)
    component_id=ids.copy();component_id[component==1]=road_identity['world_row_id'][road_index[component==1]]
    for j,(z,m) in enumerate(zip(zs,masses)):
        positions=np.searchsorted(z['global_row_ids'],ids);ok=positions<len(z['global_row_ids']);ok[ok]&=z['global_row_ids'][positions[ok]]==ids[ok]
        present[ok,j]=True;native[ok,j]=z['native_context'][positions[ok]];owners[ok,j]=z['owner_ids'][positions[ok]];mass[ok,j]=m[positions[ok]]
        guards[ok,j]=z['old_permanent_guard'][positions[ok]];packet_position[ok,j]=positions[ok]
    preserve=np.asarray(read(base['context_plan'])['preserved_native_context_global_row_ids'],np.int64)
    flags,policy_free,edit_keep,keep,mixed=qualify_components(ids,mass,owners,present,native,guards,component,preserve)
    assert not (keep&~edit_keep).any()
    assert (mass[keep][:,:,[1,2,3,5]]==0).all() and present[keep].all()
    assert (owners[keep&(component==0)]<=0).all() and native[keep&(component==0)].all()
    assert (~native[keep&(component==1)]).all()
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
    roi_names=np.asarray(['full_image','outside_allowed_edit','actor_protected','K_protected','inside_allowed_edit','outside_replacement','inside_replacement','single_pixel_sentinel'])
    selected_component=component[keep];selected_component_id=component_id[keep];selected_road_index=road_index[keep]
    payload=dict(global_row_ids=chosen,component_kind=selected_component,component_row_id=selected_component_id,
        road_active_index=selected_road_index,packet_row_index_by_view=packet_position[keep],allview_mass=selected,
        owner_ids_by_view=owners[keep],old_permanent_guard=guards[keep],present=present[keep],ROI_names=roi_names,frames=np.asarray(frames),
        native_global_row_ids=np.where(selected_component==0,chosen,-1),oldroad_world_row_ids=np.where(selected_component==1,selected_component_id,-1))
    np.savez_compressed(a.output/'candidate_context_rows.npz',**payload)
    np.savez_compressed(a.output/'diagnostic_gate0.npz',global_row_ids=chosen,component_kind=selected_component,component_row_id=selected_component_id,
        road_active_index=selected_road_index,gate=np.zeros(len(chosen),np.float32))
    nm=selected_component==0;rm=selected_component==1
    np.savez_compressed(a.output/'native_gate0.npz',global_row_ids=chosen[nm],gate=np.zeros(int(nm.sum()),np.float32))
    np.savez_compressed(a.output/'oldroad_gate0.npz',global_row_ids=chosen[rm],road_active_index=selected_road_index[rm],world_row_id=selected_component_id[rm],gate=np.zeros(int(rm.sum()),np.float32))
    np.savez_compressed(a.output/'qualification_ledger.npz',global_row_ids=ids,component_kind=component,component_row_id=component_id,road_active_index=road_index,
        packet_row_index_by_view=packet_position,frames=np.asarray(frames),owner_ids_by_view=owners,present_by_view=present,native_by_view=native,old_permanent_guard=guards,
        allview_mass=mass,ROI_names=roi_names,selected=keep,edit_contained=edit_keep,replacement_mixed=mixed,**flags)
    order=['retained_positive_owner','permanent_old_guard','missing_from_any_view','component_inconsistent_in_any_view','K_positive','car_positive','outside_edit_positive','outside_replacement_positive']
    remaining=np.ones(len(ids),bool);part={}
    for name in order:
        m=remaining&flags[name];part[name]=summarize(m,mass);remaining&=~m
    remaining&=(mass[:,:,6]>0).any(1)
    assert np.array_equal(remaining,keep)
    part['whole_kernel_candidate']=summarize(keep,mass)
    proof=dict(schema='R10.component_semantic_footprint_limited_whole_kernel_scope.v1',complete=True,decision='SCOPE_VALID_DIAGNOSTIC_ONLY',
        policy='measured_component_replacement_contained_native_unowned_oldroad_all5_exact_zero',frames=frames,candidate_rows=len(chosen),edit_contained_candidate_rows=int(edit_keep.sum()),initial_unowned_visible_union_rows=len(ids),
        baseline_contract_sha256=sha(Path(cfg['baseline_contract'])),measurement_contract=pin(a.contract),actual_attribution_receipt=pin(a.measurement/'receipt.json'),actual_attribution_receipt_sha256=sha(a.measurement/'receipt.json'),
        asset_manifest=pin(asset_path),candidate_npz_sha256=sha(a.output/'candidate_context_rows.npz'),gate_npz_sha256=sha(a.output/'diagnostic_gate0.npz'),input_packets=[pin(x) for x in packet_paths],
        mask_receipt=mask_ref,zero_receipt=zero_receipt_ref,old_road_identity=pin(identity_path),old_road_identity_sha256=sha(identity_path),global_road_identity=identity['global_identity'],native_global_row_count=offset,
        candidate_npz=pin(a.output/'candidate_context_rows.npz'),diagnostic_gate=pin(a.output/'diagnostic_gate0.npz'),raw_feature_prefix_proofs=raw_feature_proofs,
        independent_pixel_domain_checks=domain_checks,independent_new_zero_RGBA_and_single_pixel_sentinel=per_frame_proofs,
        allowed_edit_domain='registered full non-colour ground band minus actual strict visible actor mask at 86/110, repaired-instance proxy in evaluation-only views, and other protected K; RGB and broad body-alpha footprint are not domain selectors',
        native_candidate_rows=int((selected_component==0).sum()),oldroad_candidate_rows=int((selected_component==1).sum()),
        component_namespace={'0':'original native global row ID','1':'old RoadRows world row ID; separate global packet ID=offset+active_index'},
        packet_row_index_namespace='all_context_rows.npz record index, independently checked to equal the actual raw-feature full-packet context prefix',
        old_road_world_id_roundtrip_exact=True,old_road_global_id_is_not_world_id=True,new_local_rows_never_selected=True,
        all_five_outside_edit_exact_zero=bool((selected[:,:,1]==0).all()),all_five_outside_exact_zero=bool((selected[:,:,1]==0).all()),all_five_outside_replacement_exact_zero=bool((selected[:,:,5]==0).all()),all_five_car_K_exact_zero=bool((selected[:,:,2:4]==0).all()),
        zero_proof_is_vacuous=len(chosen)==0,mass_tolerance_for_selection=0.,all_five_rows_present=True,old_permanent_guard_preserved=True,
        native21_and_all_retained_positive_owners_preserved=True,old_road_rows_qualified_by_actual_measured_footprints_only=True,selection_uses_RGB=False,selection_uses_residual=False,full_scene_contribution_required=True,
        domain_certificate='measured only in the five immutable cameras and current native visibility contract; not arbitrary-camera support certification',
        independent_partition_float32_roundoff=roundoffs,float32_partition_check_never_relaxes_exact_zero_selection=True,overlapping_rejection={k:summarize(m,mass) for k,m in flags.items()},
        exclusive_priority=order,exclusive_partition=part,edit_contained_candidates=summarize(edit_keep,mass),replacement_contained_candidates=summarize(keep,mass),
        mixed_footprint_rows=summarize(mixed,mass),entirely_outside_replacement_rows=summarize(flags['outside_replacement_positive']&~(mass[:,:,6]>0).any(1),mass),
        mixed_kernel_action='preserve; separate splitting/local reconstruction requires a new explicit asset and measured readout',
        original_native_identity_asset_not_modified=True,whole_kernel_retirement_not_production=True,actual_zero_step_required=True,
        actual_zero_step_acceptance='compare every view outside allowed edit, strict V/proxy, K, full original band, unknown geometry/material and scene alpha holes; no optimization',
        CPU_only=True,nativeF=0,nativeB=0,updates=0,optimization_allowed=False,quality_accepted=False,seconds=time.time()-start)
    (a.output/'scope.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,candidate_rows=len(chosen),scope=pin(a.output/'scope.json'),native_candidate_rows=proof['native_candidate_rows'],oldroad_candidate_rows=proof['oldroad_candidate_rows'],selected=part['whole_kernel_candidate'],mixed=proof['mixed_footprint_rows']),ensure_ascii=False))


if __name__=='__main__':main()
