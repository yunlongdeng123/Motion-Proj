"""Temporary native-unowned and old-road replacement-contained retirement.

The original vehicles and persistent native/old-road files remain immutable.
This separate diagnostic uses one actual joint depth sort per fixed view.
"""
from pathlib import Path
import argparse
import numpy as np
import render_new_local_background_zero as base
import render_replacement_contained_zero as contained
from local_gaussian_asset import bank,read,write,sha
from local_joint_adapter import LocalJointAdapter

IDENTITY_SHA='3bf0d887bc89b755ed81aa3315cf94f8f84dcb6ff3a2cf5c3fdc8213a94ce456'
ARGS=None

def normalized_scope(path):
    r=read(path);r=dict(r);r['old_road_candidate_rows']=r.get('old_road_candidate_rows',r.get('oldroad_candidate_rows'))
    assert r['old_road_candidate_rows'] is not None
    return r


def qualify(a,asset,cfg):
    scope=normalized_scope(a.policy/'scope.json');cpath=a.policy/'candidate_context_rows.npz';gpath=a.policy/'diagnostic_gate0.npz'
    candidate=bank(cpath);gate=bank(gpath);ids=gate['global_row_ids'];kind=gate['component_kind'];cid=gate['component_row_id']
    assert scope['complete'] and scope['decision']=='SCOPE_VALID_DIAGNOSTIC_ONLY'
    assert scope['policy']=='measured_component_replacement_contained_native_unowned_oldroad_all5_exact_zero'
    assert scope['baseline_contract_sha256']==asset.manifest['parent_contract']['sha256']==sha(cfg['baseline_contract'])
    assert scope['asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    assert scope['actual_attribution_receipt_sha256']==sha(a.measurement/'receipt.json')
    candidate_sha=scope.get('candidate_npz_sha256',scope.get('candidate_npz',{}).get('sha256'))
    gate_sha=scope.get('gate_npz_sha256',scope.get('gate_npz',{}).get('sha256'))
    assert candidate_sha==sha(cpath) and gate_sha==sha(gpath)
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
    for j,raw in enumerate(cfg['frames']):
        r=next(r for r in m['rows'] if r['raw']==raw);path=a.measurement/f'{raw:03d}/all_context_rows.npz';assert sha(path)==r['packet']['sha256']
        z=bank(path);position=candidate['packet_row_index_by_view'][:,j]
        assert np.array_equal(z['global_row_ids'][position],ids) and np.array_equal((~z['native_context'][position]).astype(np.uint8),kind)
        assert np.array_equal(z['owner_ids'][position],candidate['owner_ids_by_view'][:,j])
        assert np.array_equal(z['old_permanent_guard'][position],candidate['old_permanent_guard'][:,j])
        actual=z['ROI_alphaT_mass'][position][:,[z['ROI_names'].tolist().index(k) for k in names]]
        assert actual.tobytes()==mass[:,j].tobytes(),'Candidate mass differs from actual current visibility'
    return scope,gate,candidate


class ComponentAdapter(contained.ContainedAdapter):
    def __init__(self,asset_directory,output,gate,proof):
        LocalJointAdapter.__init__(self,asset_directory,output,enable_gpu=True)
        assert self.asset.manifest['native_gate']['mode']=='identity_no_retirement' and not len(self.gate_ids)
        torch=self.torch;self.diagnostic_ids=torch.as_tensor(gate['global_row_ids'],device='cuda',dtype=torch.long)
        self.diagnostic_kind=torch.as_tensor(gate['component_kind'],device='cuda',dtype=torch.long)
        self.diagnostic_proof=proof;self.applied=[]

    def _native_gate(self,raw,packet):
        packet=LocalJointAdapter._native_gate(self,raw,packet);torch=self.torch
        _,group,owner,gids,offset=self.base.context[raw]
        positions=torch.searchsorted(self.diagnostic_ids,gids);safe=positions.clamp_max(len(self.diagnostic_ids)-1)
        selected=(positions<len(self.diagnostic_ids))&(self.diagnostic_ids[safe]==gids)
        assert int(selected.sum())==len(self.diagnostic_ids)
        assert torch.equal(group[selected].long(),self.diagnostic_kind[safe[selected]])
        native=selected&(group==0);road=selected&(group==1)
        assert (owner[native]<=0).all() and not torch.isin(gids[selected],self.base.preserve).any()
        assert int(road.sum())==self.diagnostic_proof['old_road_candidate_rows']
        assert int(native.sum())==self.diagnostic_proof['native_candidate_rows']
        # The integer indices are in the context prefix. Canonical actor/local
        # suffixes are never addressed by this temporary packet-only override.
        positions=torch.nonzero(selected).reshape(-1);out=dict(packet);opacity=packet['opacities'].clone();opacity[positions]=0;out['opacities']=opacity
        self.applied.append(dict(raw=raw,temporary_zero_rows=len(positions),native_unowned_zero_rows=int(native.sum()),old_road_zero_rows=int(road.sum()),
            original_A20_and_native_positive_owners_untouched=True,persistent_source_arrays_unchanged=True))
        return out


def main():
    global ARGS
    p=argparse.ArgumentParser(add_help=False)
    for name in ['asset','measurement-contract','measurement','policy','zero','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute',action='store_true');ARGS,_=p.parse_known_args()
    scope=normalized_scope(ARGS.policy/'scope.json')
    if not scope['candidate_rows']:print(dict(not_run=True,reason='No whole replacement-contained native or old-road rows',nativeF=0,nativeB=0,updates=0));return
    contained.ARGS=ARGS;contained.CHECKS.clear();base.qualify=qualify;base.DiagnosticBackgroundAdapter=ComponentAdapter
    try:base.main()
    finally:
        if ARGS.execute and (ARGS.output/'receipt.json').is_file():
            r=read(ARGS.output/'receipt.json');r.update(schema='R10.component_replacement_contained_joint_zero.v1',
                outside_RGBA_invariance_checks=contained.CHECKS,
                all_five_actual_outside_edit_replacement_car_K_RGBA_exact=len(contained.CHECKS)==5 and all(x['all_outside_and_protection_RGBA_exact'] for x in contained.CHECKS),
                native_unowned_temporary_retirement_rows=scope['native_candidate_rows'],old_road_temporary_retirement_rows=scope['old_road_candidate_rows'],
                old_road_packet_opacity_override=bool(scope['old_road_candidate_rows']),old_road_persistent_parameters_unchanged=True,
                local_asset_and_all_parent_assets_unchanged=True,new_local_asset_not_retired=True,mask_receipt=scope['mask_receipt'],
                old_road_identity=scope['old_road_identity'],component_identity='kind + native global ID or original road world_row_id',
                root_scope_approval_required_for_execution=True,production_gate_adopted=False,whole_scene_quality_accepted=False)
            write(ARGS.output/'receipt.json',r)

if __name__=='__main__':main()
