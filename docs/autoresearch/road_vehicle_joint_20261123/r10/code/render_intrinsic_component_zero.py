"""One zero-training control using independently rendered local coverage.

The asset, RGB, Gaussian parameters, cameras and original fixed evaluation
remain unchanged. Only the measured replacement-domain policy differs.
"""
from pathlib import Path
import argparse
import numpy as np
import render_component_replacement_zero as generic
import render_component_replacement_bounded_zero as bounded
from local_gaussian_asset import bank,read,sha,write,load_local_asset,ACTOR_SHA

IDENTITY_SHA='3bf0d887bc89b755ed81aa3315cf94f8f84dcb6ff3a2cf5c3fdc8213a94ce456'
POLICY='intrinsic_local_coverage_component_replacement_all5_exact_zero'


def qualify(a,asset,cfg):
    scope=generic.normalized_scope(a.policy/'scope.json');cpath=a.policy/'candidate_context_rows.npz';gpath=a.policy/'diagnostic_gate0.npz'
    candidate=bank(cpath);gate=bank(gpath);ids=gate['global_row_ids'];kind=gate['component_kind'];cid=gate['component_row_id']
    assert scope['complete'] and scope['decision']=='SCOPE_VALID_DIAGNOSTIC_ONLY' and scope['policy']==POLICY
    assert scope['baseline_contract_sha256']==asset.manifest['parent_contract']['sha256']==sha(cfg['baseline_contract'])
    assert scope['asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    assert scope['actual_attribution_receipt_sha256']==sha(a.measurement/'receipt.json')
    assert scope['candidate_npz_sha256']==sha(cpath) and scope['gate_npz_sha256']==sha(gpath)
    assert not scope['selection_uses_RGB'] and not scope['selection_uses_residual'] and not scope['optimization_allowed']
    assert ids.dtype==np.int64 and kind.dtype==np.uint8 and cid.dtype==np.int64 and np.array_equal(ids,np.unique(ids))
    assert len(ids)>0 and len(ids)==scope['candidate_rows'] and np.isin(kind,[0,1]).all()
    assert gate['gate'].dtype==np.float32 and gate['gate'].shape==ids.shape and (gate['gate']==0).all()
    for k in ['global_row_ids','component_kind','component_row_id']:assert np.array_equal(gate[k],candidate[k])
    assert candidate['frames'].tolist()==cfg['frames'] and candidate['present'].all()
    mass=candidate['allview_mass'];names=candidate['ROI_names'].tolist();assert mass.shape[:2]==(len(ids),5)
    assert np.isfinite(mass).all() and (mass>=0).all() and not candidate['old_permanent_guard'].any()
    for name in ['outside_allowed_edit','actor_protected','K_protected','outside_replacement']:assert (mass[:,:,names.index(name)]==0).all()
    assert (mass[:,:,names.index('inside_replacement')]>0).any(1).all()
    native=kind==0;road=~native;assert (candidate['owner_ids_by_view'][native]<=0).all()
    assert np.array_equal(cid[native],ids[native])
    identity_ref=scope['old_road_identity'];assert identity_ref['sha256']==sha(identity_ref['path'])==IDENTITY_SHA
    identity=read(identity_ref['path']);mapping=bank(identity['global_identity']['path'])
    assert sha(identity['global_identity']['path'])==identity['global_identity']['sha256']
    assert sha(identity['road_asset']['path'])==identity['road_asset']['sha256']
    offset=identity['native_global_row_count'];ri=ids[road]-offset
    assert (ids[native]<offset).all() and ((ri>=0)&(ri<len(mapping['world_row_id']))).all()
    assert np.array_equal(candidate['road_active_index'][road],ri) and np.array_equal(cid[road],mapping['world_row_id'][ri])
    assert np.array_equal(mapping['global_row_id'][ri],ids[road])
    m=read(a.measurement/'receipt.json');assert m['complete'] and m['nativeF']==m['nativeB']==5
    assert m['asset_manifest']['sha256']==sha(a.asset/'manifest.json') and m['zero_receipt']['sha256']==sha(a.zero/'receipt.json')
    assert m['mask_receipt']['sha256']==scope['mask_receipt']['sha256']==sha(scope['mask_receipt']['path'])
    mask_receipt=read(scope['mask_receipt']['path'])
    assert mask_receipt['schema']=='R10.intrinsic_local_coverage_edit_masks.v1' and mask_receipt['complete']
    assert mask_receipt['intrinsic_alpha_positive_threshold']==0.0 and mask_receipt['replacement_alpha_threshold']==.95
    intrinsic_ref=mask_receipt['intrinsic_coverage_receipt'];assert intrinsic_ref['sha256']==sha(intrinsic_ref['path'])
    assert scope['intrinsic_coverage_receipt']['sha256']==intrinsic_ref['sha256']
    intrinsic=read(intrinsic_ref['path']);assert intrinsic['complete'] and intrinsic['nativeF']==5 and intrinsic['nativeB']==0
    assert intrinsic['asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    assert intrinsic['actual_component']=='local_only' and intrinsic['intrinsic_alpha_from_actual_native_output'] and intrinsic['joint_local_alphaT_not_used']
    assert intrinsic['all_asset_fields_unchanged'] and not intrinsic['source_or_geometry_changed'] and intrinsic['updates']==0
    assert intrinsic['source_field_sha256']==asset.manifest['source']['field']['sha256'] and intrinsic['geometry_sha256']==asset.manifest['geometry']['npz']['sha256']
    for j,raw in enumerate(cfg['frames']):
        r=next(r for r in m['rows'] if r['raw']==raw);path=a.measurement/f'{raw:03d}/all_context_rows.npz';assert sha(path)==r['packet']['sha256']
        z=bank(path);position=candidate['packet_row_index_by_view'][:,j]
        assert np.array_equal(z['global_row_ids'][position],ids) and np.array_equal((~z['native_context'][position]).astype(np.uint8),kind)
        assert np.array_equal(z['owner_ids'][position],candidate['owner_ids_by_view'][:,j])
        assert np.array_equal(z['old_permanent_guard'][position],candidate['old_permanent_guard'][:,j])
        actual=z['ROI_alphaT_mass'][position][:,[z['ROI_names'].tolist().index(k) for k in names]]
        assert actual.tobytes()==mass[:,j].tobytes()
        mr=next(r for r in mask_receipt['rows'] if r['raw']==raw)
        for k in ['masks','fixed_eval','rectified_masks','intrinsic_alpha_npz']:assert sha(mr[k]['path'])==mr[k]['sha256']
        d=bank(mr['masks']['path']);ev=bank(mr['fixed_eval']['path']);semantic=bank(mr['rectified_masks']['path'])
        alpha=bank(mr['intrinsic_alpha_npz']['path'])['alpha'].squeeze()
        ir=next(r for r in intrinsic['rows'] if r['raw']==raw)
        assert ir['render']['sha256']==mr['intrinsic_alpha_npz']['sha256'] and ir['actual_component']=='local' and not ir['environment_added']
        assert alpha.shape==(768,1152) and np.isfinite(alpha).all() and ((alpha>=0)&(alpha<=1)).all()
        car=semantic['actor_visible'] if raw in [86,110] else semantic['actor_visible_proxy'];K=ev['K_protected']
        allowed=(ev['full_original_band']|(alpha>0))&~K&~car;replacement=allowed&(alpha>=.95)
        assert np.array_equal(d['inside_allowed_edit'],allowed) and np.array_equal(d['outside_allowed_edit'],~allowed)
        assert np.array_equal(d['inside_replacement'],replacement) and np.array_equal(d['outside_replacement'],~replacement)
        assert np.array_equal(d['actor_protected'],car) and np.array_equal(d['K_protected'],K)
        assert np.array_equal(d['full_original_band'],ev['full_original_band']),'Original evaluation denominator changed'
    return scope,gate,candidate


def main():
    # Bounded rendering and exact car/K readout are reused without changing
    # old qualification code or any existing experiment's source files.
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--preflight-scope-only',action='store_true')
    args,_=parser.parse_known_args();bounded.OriginalQualify=qualify
    if args.preflight_scope_only:
        p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
        for k in ['asset','measurement-contract','measurement','policy','zero','baseline-measurement','output']:p.add_argument('--'+k,type=Path,required=True)
        p.add_argument('--preflight-scope-only',action='store_true');a=p.parse_args()
        asset=load_local_asset(a.asset,verify_parent_pins=True);cfg=read(a.measurement_contract)
        assert cfg['accepted_actor_sha256']==asset.manifest['actor_parent_sha256']==ACTOR_SHA
        scope,gate,candidate=qualify(a,asset,cfg)
        for raw in cfg['frames']:
            for path in [a.baseline_measurement/f'{raw:03d}/baseline_joint.npz',a.zero/f'{raw:03d}/joint_render.npz']:
                z=bank(path);assert z['rgb'].shape==(3,768,1152) and z['alpha'].shape==(768,1152)
        print(dict(prepared=True,scope_and_all_five_inputs_verified=True,candidate_rows=len(gate['global_row_ids']),
            native_rows=scope['native_candidate_rows'],old_road_rows=scope['old_road_candidate_rows'],nativeF=0,nativeB=0,updates=0,
            root_readout_contract_still_required=True,expected_nativeF=5));return
    try:bounded.main()
    finally:
        if args.execute and (args.output/'receipt.json').is_file():
            r=read(args.output/'receipt.json');scope=read(r['diagnostic_scope']['path'])
            r.update(schema='R10.intrinsic_local_support_component_replacement_zero.v1',control_kind='independent_local_only_alpha_support',
                intrinsic_coverage_receipt=scope['intrinsic_coverage_receipt'],intrinsic_alpha_positive_threshold=0.0,replacement_alpha_threshold=.95,
                source_asset_camera_and_original_eval_unchanged=True,additional_training_updates=0,
                support_domain_only_changed=True,scientific_whole_scene_quality_accepted=False)
            write(args.output/'receipt.json',r)

if __name__=='__main__':main()
