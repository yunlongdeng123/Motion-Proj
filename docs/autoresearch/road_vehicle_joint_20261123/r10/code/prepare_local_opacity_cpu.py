"""Bind a final child, actual five-view footprints and fixed joint targets.

CPU only. No RGB/residual candidate selection and no broad body-alpha freeze.
The produced contract is reviewed by root before any smoke is executed.
"""
from pathlib import Path
import argparse,sys,time
import numpy as np
from local_gaussian_asset import bank,read,write,sha,load_local_asset
from local_opacity_gate import pin
from prepare_footprint_limited_retirement_cpu import columns
from fit_new_local_rgb import field_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['asset','zero','footprint','masks','surface-review','baseline-measurement','measurement-contract','method-contract','child-scope-review','output']:
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();started=time.time();assert not a.output.exists()
    method=read(a.method_contract);cfg=read(a.measurement_contract)
    assert method['formal_steps']==400 and method['train_frames']==[86,110]
    asset=load_local_asset(a.asset,verify_parent_pins=True);m=asset.manifest;frames=m['frames']
    assert frames==[86,110,134,179,198] and frames==cfg['frames']
    assert m['native_gate']['mode']=='identity_no_retirement' and len(asset.native_gate['global_row_ids'])==0
    receipt=read(a.footprint/'receipt.json');mr=read(a.masks/'receipt.json');zr=read(a.zero/'receipt.json')
    assert receipt['complete'] and mr['complete'] and zr['complete']
    assetsha=sha(a.asset/'manifest.json')
    assert receipt['asset_manifest']['sha256']==mr['parent_asset_manifest']['sha256']==zr['asset_manifest']['sha256']==assetsha
    assert receipt['zero_receipt']['sha256']==sha(a.zero/'receipt.json')
    assert receipt['mask_receipt']['sha256']==sha(a.masks/'receipt.json')
    assert receipt['measurement_contract']['sha256']==sha(a.measurement_contract)
    assert receipt['frames']==mr['frames']==frames
    assert receipt['all_native_source_parameters_unchanged'] and receipt['all_native_retained']
    assert receipt['nativeF']>=5 and receipt['nativeB']>=5
    scope=read(a.child_scope_review)
    assert scope['complete'] and scope['decision']=='PASS_LIMITED_BOUNDED_INITIALIZATION'
    assert scope['asset']==a.asset.name and scope['actual_zero']==a.zero.name
    assert scope['K_and_car_realV_or_proxy_RGBA_byteexact_all_five']
    assert mr['source_colour_used_for_masks'] is False and mr['broad_canonical_alpha_not_used_as_car_mask'] is True
    base=read(m['parent_contract']['path']);assert sha(cfg['baseline_contract'])==m['parent_contract']['sha256']
    oldguard=np.asarray(read(base['context_plan'])['preserved_native_context_global_row_ids'],np.int64)
    paths=[a.footprint/f'{raw:03d}/all_context_rows.npz' for raw in frames]
    initial=[]
    for raw,path in zip(frames,paths):
        z=bank(path);mass=columns(z);row=next(x for x in receipt['rows'] if x['raw']==raw)
        assert sha(path)==row['packet']['sha256'] and row['full_RGB_alpha_byteexact_new0']
        assert (np.diff(z['global_row_ids'])>0).all()
        assert np.array_equal(z['old_permanent_guard'],np.isin(z['global_row_ids'],oldguard))
        allowed=(~z['native_context'])|((z['owner_ids']<=0)&~z['old_permanent_guard'])
        initial.append(z['global_row_ids'][allowed&(mass[:,4]>0)])
    ids=np.unique(np.concatenate(initial));mass=np.zeros((len(ids),5,7),np.float32)
    present=np.zeros((len(ids),5),bool);kind=np.full((len(ids),5),-1,np.int8);owner=np.zeros((len(ids),5),np.int32)
    for j,path in enumerate(paths):
        z=bank(path);v=columns(z);ix=np.searchsorted(z['global_row_ids'],ids);ok=ix<len(z['global_row_ids'])
        ok[ok]&=z['global_row_ids'][ix[ok]]==ids[ok]
        present[ok,j]=True;mass[ok,j]=v[ix[ok]];kind[ok,j]=(~z['native_context'][ix[ok]]).astype(np.int8);owner[ok,j]=z['owner_ids'][ix[ok]]
    frozen=(~present.all(1))|(kind!=kind[:,0,None]).any(1)|((kind==0)&(owner>0)).any(1)|np.isin(ids,oldguard)
    frozen|=(mass[:,:,2:4]>0).any((1,2))  # exact K / actual car mass, no epsilon
    road=bank(Path(base['road_asset'])/'road.npz');active=np.flatnonzero(road['active'])
    topology=read(base['scene_proof'])['topology'];native_end=int(topology['static_rows']+topology['dynamic_rows'])
    provenance=bank(Path(base['road_asset'])/'provenance.npz')
    guarded=np.asarray(provenance.get('guarded_world_rows',np.zeros(len(road['active']),bool)),bool)
    roadpos=ids-native_end;isroad=(kind[:,0]==1)&(roadpos>=0)&(roadpos<len(active))
    badroad=(kind[:,0]==1)&~isroad
    frozen|=badroad
    frozen[isroad]|=guarded[active[roadpos[isroad]]]
    selected=~frozen&(mass[:,:,4]>0).any(1)
    assert selected.any() and (mass[selected,:,2:4]==0).all()
    assert (owner[selected][kind[selected]==0]<=0).all()
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'eligible_rows.npz',global_row_ids=ids[selected],group_kind=kind[selected,0],frames=np.asarray(frames),
        owners_by_view=owner[selected],allview_mass=mass[selected],allview_present=present[selected],
        native_global_row_count=np.asarray(native_end),road_active_world_rows=active)
    np.savez_compressed(a.output/'qualification_ledger.npz',global_row_ids=ids,owner=owner,group_kind=kind,present=present,measured_mass=mass,selected=selected)
    fieldpath=Path(m['source']['field']['path']);field=bank(fieldpath)
    assert sha(fieldpath)==m['source']['field']['sha256']
    review=read(a.surface_review/'receipt.json')
    assert review['complete'] and review['final_atlas_sha256']==sha(fieldpath)
    assert review['geometry_sha256']==m['geometry']['npz']['sha256']
    legal=field['permission_mask']&(field['source_kind']>0)&np.isfinite(field['atlas_linear']).all(-1)
    rows=[]
    for raw in frames:
        maskrow=next(x for x in mr['rows'] if x['raw']==raw)
        for key in ['masks','fixed_eval','rectified_masks','direct_surface_projection']:
            assert sha(maskrow[key]['path'])==maskrow[key]['sha256']
        domain=bank(a.masks/f'{raw:03d}/scope_masks.npz');surface=bank(a.surface_review/f'{raw:03d}/surface_review.npz')
        rect=bank(maskrow['rectified_masks']['path']);ev=bank(maskrow['fixed_eval']['path'])
        car=rect['actor_visible'] if raw in [86,110] else rect['actor_visible_proxy']
        inside=ev['full_original_band']&~rect['protected']&~car
        assert np.array_equal(domain['inside_allowed_edit'],inside)
        assert np.array_equal(domain['outside_allowed_edit'],~inside)
        assert np.array_equal(domain['actor_protected'],car) and np.array_equal(domain['K_protected'],rect['protected'])
        assert np.array_equal(domain['direct_surface_shown'],surface['shown'])
        target,valid=field_sample(field['atlas_linear'],surface['atlas_uv_global'],legal)
        support=inside&surface['shown']&surface['projection_valid']&valid&rect['RGB_valid']
        assert support.any() and np.isfinite(target[support]).all()
        original=bank(a.baseline_measurement/f'{raw:03d}/baseline_joint.npz');zero=bank(a.zero/f'{raw:03d}/joint_render.npz')
        measured=bank(a.footprint/f'{raw:03d}/measurement_joint.npz')
        for key in ['rgb','alpha']:
            assert measured[key].tobytes()==zero[key].tobytes()
        out=~inside;protect=rect['protected']|car
        changes=lambda d:dict(pixels=int(d.sum()),RGB_max=float(abs(zero['rgb']-original['rgb'])[:,d].max(initial=0)),
            alpha_max=float(abs(zero['alpha']-original['alpha'])[d].max(initial=0)),
            changed_RGB_pixels=int((d&(abs(zero['rgb']-original['rgb']).max(0)>0)).sum()))
        folder=a.output/f'{raw:03d}';folder.mkdir()
        np.savez_compressed(folder/'target.npz',target_common_linear=target.transpose(2,0,1),support=support,
            inside_allowed_edit=inside,outside_allowed_edit=out,K=rect['protected'],car=car,
            full_original_band=ev['full_original_band'],unknown_inside=inside&~support,
            alpha_floor_mask=ev['full_original_band']&(original['alpha']>=.95),
            original_rgb=original['rgb'],original_alpha=original['alpha'],
            parentzero_rgb=zero['rgb'],parentzero_alpha=zero['alpha'],gain=np.asarray(float(ev['exposure_gain'])),
            photo_srgb=bank(Path(cfg['camera_cache'])/f'{raw:03d}/view_cache.npz')['photo_pinhole_srgb'].transpose(2,0,1))
        assert changes(protect)['RGB_max']==0 and changes(protect)['alpha_max']==0
        assert changes(out)['RGB_max']<=method['initial_child_outside_RGB_max_abs_bound']
        assert changes(out)['alpha_max']<=method['outside_alpha_max_abs_bound']
        assert not (ev['full_original_band']&(original['alpha']>=.95)&(zero['alpha']<.95)).any()
        rows.append(dict(raw=raw,target=pin(folder/'target.npz'),support_pixels=int(support.sum()),unknown_inside_pixels=int((inside&~support).sum()),
            protected=changes(protect),outside_original=changes(out),strict_real_V_opened=raw in [86,110],
            all_domains_fixed_before_opacity_updates=True))
    contract=dict(method,complete=True,asset=str(a.asset),zero=str(a.zero),frames=frames,r8_code=m['r8_code'],
        source={'field':str(fieldpath),'sha256':sha(fieldpath)},rows=rows,eligible=pin(a.output/'eligible_rows.npz'),
        child_scope_review=pin(a.child_scope_review),fixed_eval=cfg['fixed_eval'],baseline_measurement=str(a.baseline_measurement),
        qualification_ledger=pin(a.output/'qualification_ledger.npz'),native_global_row_count=native_end,
        input_pins=[pin(x) for x in [a.asset/'manifest.json',a.zero/'receipt.json',a.footprint/'receipt.json',a.masks/'receipt.json',
            a.surface_review/'receipt.json',a.baseline_measurement/'receipt.json',a.measurement_contract,a.method_contract,a.child_scope_review,*paths]],
        hard_guard='All measured K/car-positive rows, native retained positive owners, native permanent guards and historical guarded RoadRows absent from trainable vector; their implicit gate is exactly one',
        selected_rows=int(selected.sum()),native_rows=int((kind[selected,0]==0).sum()),oldRoadRows=int((kind[selected,0]==1).sum()),
        mixed_outside_rows=int((mass[selected,:,1]>0).any(1).sum()),mass_epsilon=0.,
        source_colour_or_residual_used_for_eligibility=False,broad_body_alpha_used=False,
        initial_outside_parent_difference_is_recorded_not_silently_accepted=True,
        CPU_only=True,nativeF=0,nativeB=0,updates=0,seconds=time.time()-started)
    write(a.output/'contract.json',contract)
    print(dict(prepared=True,contract=pin(a.output/'contract.json'),eligible_rows=int(selected.sum()),
        initial_protection=[{'raw':r['raw'],**r['protected']} for r in rows],initial_outside=[{'raw':r['raw'],**r['outside_original']} for r in rows],GPU_calls=0),flush=True)


if __name__=='__main__':main()
