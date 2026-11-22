"""Qualify exact replacement-contained native and old-road row identities.

CPU only. This prepares a diagnostic scope; it does not retire any row or
approve a render. The current asset's actual five-view visibility is required.
"""
from pathlib import Path
import argparse
import numpy as np
from local_gaussian_asset import bank,read,write,sha,load_local_asset
from render_new_local_zero import pin

IDENTITY_SHA='3bf0d887bc89b755ed81aa3315cf94f8f84dcb6ff3a2cf5c3fdc8213a94ce456'
REGIONS=['full_image','outside_allowed_edit','actor_protected','K_protected','inside_allowed_edit','outside_replacement','inside_replacement','single_pixel_sentinel']


def verify_identity(path,asset):
    assert sha(path/'receipt.json')==IDENTITY_SHA
    r=read(path/'receipt.json');assert r['complete'] and r['decision']=='IDENTITY_MAPPING_ONLY_NO_RETIREMENT_QUALIFICATION'
    assert r['parent_contract']['sha256']==asset.manifest['parent_contract']['sha256']
    assert sha(r['road_asset']['path'])==r['road_asset']['sha256'] and sha(r['global_identity']['path'])==r['global_identity']['sha256']
    z=bank(r['global_identity']['path']);offset=r['native_global_row_count']
    with np.load(r['road_asset']['path'],allow_pickle=False) as road:active=np.flatnonzero(road['active']).astype(np.int64)
    assert np.array_equal(z['road_active_index'],np.arange(len(active))) and np.array_equal(z['world_row_id'],active)
    assert np.array_equal(z['global_row_id'],offset+np.arange(len(active)))
    return r,z


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['asset','measurement','old-road-identity','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--components',choices=['native','old_road','both'],default='both');a=p.parse_args()
    asset=load_local_asset(a.asset,verify_parent_pins=True);m=read(a.measurement/'receipt.json')
    assert m['complete'] and m['nativeF']==m['nativeB']==5 and m['asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    assert m['ROI_names']==REGIONS and m['all_native_retained'] and m['old_road400_frozen'] and m['local_asset_unchanged']
    identity,roadmap=verify_identity(a.old_road_identity,asset);offset=identity['native_global_row_count'];frames=m['frames']
    assert frames==asset.manifest['frames']==[86,110,134,179,198]
    mask=read(m['mask_receipt']['path']);assert mask['complete'] and sha(m['mask_receipt']['path'])==m['mask_receipt']['sha256']
    data=[];common=None;source=[]
    for r in m['rows']:
        raw=r['raw'];path=a.measurement/f'{raw:03d}/all_context_rows.npz';assert sha(path)==r['packet']['sha256']
        z=bank(path);ids=z['global_row_ids'];assert np.array_equal(ids,np.unique(ids)) and z['ROI_names'].tolist()==REGIONS
        assert np.array_equal(z['native_context'],ids<offset)
        assert z['ROI_alphaT_mass'].shape==(len(ids),len(REGIONS)) and np.isfinite(z['ROI_alphaT_mass']).all() and (z['ROI_alphaT_mass']>=0).all()
        assert r['full_RGB_alpha_byteexact_new0'] and r['actual_single_depth_sort']
        data.append(z);common=ids.copy() if common is None else np.intersect1d(common,ids,assume_unique=True)
        source.append(dict(raw=raw,rows=pin(path),receipt=pin(a.measurement/f'{raw:03d}/receipt.json')))
    kind=(common>=offset).astype(np.uint8);component_id=common.copy();road_active_index=np.full(len(common),-1,np.int64)
    road=kind==1;road_active_index[road]=common[road]-offset
    assert ((road_active_index[road]>=0)&(road_active_index[road]<len(roadmap['world_row_id']))).all()
    component_id[road]=roadmap['world_row_id'][road_active_index[road]]
    pos=np.stack([np.searchsorted(z['global_row_ids'],common) for z in data],1)
    mass=np.stack([z['ROI_alphaT_mass'][pos[:,j]] for j,z in enumerate(data)],1)
    owners=np.stack([z['owner_ids'][pos[:,j]] for j,z in enumerate(data)],1)
    guards=np.stack([z['old_permanent_guard'][pos[:,j]] for j,z in enumerate(data)],1)
    native_eligible=(owners<=0).all(1)&~(owners>0).any(1)
    eligible=(road|native_eligible)&~guards.any(1)
    if a.components=='native':eligible&=~road
    if a.components=='old_road':eligible&=road
    zero=(mass[:,:,1]==0).all(1)&(mass[:,:,2]==0).all(1)&(mass[:,:,3]==0).all(1)&(mass[:,:,5]==0).all(1)
    positive=(mass[:,:,6]>0).any(1);selected=eligible&zero&positive
    assert not a.output.exists();a.output.mkdir(parents=True)
    ids=common[selected];k=kind[selected];cid=component_id[selected];n=len(ids)
    np.savez_compressed(a.output/'candidate_context_rows.npz',global_row_ids=ids,component_kind=k,component_row_id=cid,
        road_active_index=road_active_index[selected],frames=np.asarray(frames),ROI_names=np.asarray(REGIONS),
        packet_row_index_by_view=pos[selected],allview_mass=mass[selected],owner_ids_by_view=owners[selected],
        old_permanent_guard=guards[selected],present=np.ones((n,5),bool))
    np.savez_compressed(a.output/'diagnostic_gate0.npz',global_row_ids=ids,component_kind=k,component_row_id=cid,gate=np.zeros(n,np.float32))
    counts={}
    for label,test in [('native_unowned',~road),('old_road',road)]:
        counts[label]=dict(all_five_present=int(test.sum()),permission_eligible=int((test&eligible).sum()),
            replacement_contained=int((test&selected).sum()),mixed_outside_support=int((test&eligible&positive&~zero).sum()))
    out=dict(schema='R10.native_and_old_road_replacement_scope_CPU.v1',complete=True,decision='SCOPE_VALID_DIAGNOSTIC_ONLY',
        policy='actual_all_five_replacement_contained_component_rows',components=a.components,candidate_rows=n,
        native_candidate_rows=int((k==0).sum()),old_road_candidate_rows=int((k==1).sum()),counts=counts,
        asset_manifest=pin(a.asset/'manifest.json'),actual_attribution_receipt=pin(a.measurement/'receipt.json'),
        actual_attribution_receipt_sha256=sha(a.measurement/'receipt.json'),mask_receipt=m['mask_receipt'],zero_receipt=m['zero_receipt'],
        baseline_contract_sha256=asset.manifest['parent_contract']['sha256'],old_road_identity=pin(a.old_road_identity/'receipt.json'),
        old_road_active_mapping=identity['global_identity'],old_road_asset=identity['road_asset'],native_global_row_count=offset,
        candidate_npz=pin(a.output/'candidate_context_rows.npz'),gate_npz=pin(a.output/'diagnostic_gate0.npz'),actual_sources=source,
        identity_rule='component_kind + component_row_id; native ID or mapped original old-road world_row_id',
        old_road_identity_is_not_gid_minus_offset=True,all_five_rows_present=True,all_five_outside_edit_replacement_car_K_exact_zero=True,
        inside_replacement_positive_in_at_least_one_view=True,old_permanent_guard_preserved=True,
        all_native_positive_owners_and_original_A20_preserved=True,new_local_asset_not_a_retirement_component=True,
        selection_uses_RGB=False,selection_uses_residual=False,numerical_zero_tolerance=0,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        diagnostic_only=True,production_asset_ready=False,optimization_allowed=False,whole_scene_quality_accepted=False)
    write(a.output/'scope.json',out);print(dict(complete=True,components=a.components,candidate_rows=n,native=out['native_candidate_rows'],old_road=out['old_road_candidate_rows'],counts=counts))

if __name__=='__main__':main()
