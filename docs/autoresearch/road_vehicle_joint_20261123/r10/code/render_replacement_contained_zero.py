"""Only retire whole unowned kernels contained in the measured replacement.

The original identity asset stays unchanged. This is a temporary zero-step
diagnostic with actual full-image outside-edit/replacement RGBA invariance.
"""
from pathlib import Path
import argparse
import numpy as np
import render_new_local_background_zero as base
from local_gaussian_asset import bank,read,write,sha
from render_new_local_zero import npy,pin,png

ARGS=None
CHECKS=[]
OriginalAdapter=base.DiagnosticBackgroundAdapter

def qualify(a,asset,cfg):
    scope=read(a.policy/'scope.json');candidate=bank(a.policy/'candidate_native_rows.npz');gate=bank(a.policy/'diagnostic_gate0.npz')
    assert scope['complete'] and scope['decision']=='SCOPE_VALID_DIAGNOSTIC_ONLY'
    assert scope['policy']=='full_image_outside_edit_and_replacement_actual_car_K_all_five_exact_zero'
    assert scope['baseline_contract_sha256']==asset.manifest['parent_contract']['sha256']==sha(cfg['baseline_contract'])
    assert scope['actual_attribution_receipt_sha256']==sha(a.measurement/'receipt.json')
    assert scope['asset_manifest']['sha256']==sha(a.asset/'manifest.json')
    assert scope['candidate_npz_sha256']==sha(a.policy/'candidate_native_rows.npz') and scope['gate_npz_sha256']==sha(a.policy/'diagnostic_gate0.npz')
    assert scope['all_five_outside_exact_zero'] and scope['all_five_outside_replacement_exact_zero'] and scope['all_five_car_K_exact_zero']
    assert scope['all_five_rows_present'] and scope['old_permanent_guard_preserved'] and scope['native21_and_all_retained_positive_owners_preserved']
    assert not scope['selection_uses_RGB'] and not scope['selection_uses_residual'] and not scope['optimization_allowed']
    assert scope['original_native_identity_asset_not_modified'] and scope['whole_kernel_retirement_not_production']
    ids=gate['global_row_ids'];assert ids.dtype==np.int64 and ids.ndim==1 and np.array_equal(ids,np.unique(ids)) and np.array_equal(ids,candidate['global_row_ids'])
    assert len(ids)==scope['candidate_rows'] and gate['gate'].dtype==np.float32 and gate['gate'].shape==ids.shape and (gate['gate']==0).all()
    assert candidate['frames'].tolist()==cfg['frames'] and (candidate['owners_by_view']<=0).all()
    assert candidate['allview_present'].all() and candidate['allview_native'].all()
    for name in ['outside_edit','outside_replacement','car','K']:assert (candidate['allview_'+name+'_mass']==0).all()
    assert (candidate['allview_inside_replacement_mass'].sum(1)>0).all()
    m=read(a.measurement/'receipt.json');assert m['complete'] and m['nativeF']==m['nativeB']==5
    assert m['mask_receipt']['sha256']==scope['mask_receipt']['sha256']==sha(scope['mask_receipt']['path'])
    assert m['zero_receipt']['sha256']==sha(a.zero/'receipt.json')
    for row in read(scope['mask_receipt']['path'])['rows']:assert sha(row['masks']['path'])==row['masks']['sha256']
    return scope,gate,candidate

class ContainedAdapter(OriginalAdapter):
    def render(self,raw,component='shared'):
        out=super().render(raw,component)
        if component!='shared':return out
        rgb=npy(out['rgb']);alpha=npy(out['alpha']).squeeze();ref=bank(ARGS.zero/f'{raw:03d}/joint_render.npz')
        maskroot=Path(self.diagnostic_proof['mask_receipt']['path']).parent;m=bank(maskroot/f'{raw:03d}/scope_masks.npz')
        row=dict(raw=raw,checks={})
        for name in ['outside_allowed_edit','outside_replacement','actor_protected','K_protected']:
            d=m[name];delta=abs(rgb-ref['rgb'])[:,d];ad=abs(alpha-ref['alpha'])[d]
            row['checks'][name]=dict(pixels=int(d.sum()),RGB_byteexact=rgb[:,d].tobytes()==ref['rgb'][:,d].tobytes(),alpha_byteexact=alpha[d].tobytes()==ref['alpha'][d].tobytes(),
                RGB_max_abs=float(delta.max()) if d.any() else 0.,alpha_max_abs=float(ad.max()) if d.any() else 0.)
        row['all_outside_and_protection_RGBA_exact']=all(r['RGB_byteexact'] and r['alpha_byteexact'] for r in row['checks'].values())
        CHECKS.append(row);write(ARGS.output/f'{raw:03d}_outside_invariance.json',row)
        if not row['all_outside_and_protection_RGBA_exact']:
            np.savez_compressed(ARGS.output/f'{raw:03d}_failed_RGBA.npz',rgb=rgb,alpha=alpha)
            png(ARGS.output/f'{raw:03d}_failed_outside_delta.png',np.clip(abs(rgb-ref['rgb'])*4,0,1))
        assert row['all_outside_and_protection_RGBA_exact'],'Measured contained kernels altered outside edit/replacement or protected car/K'
        return out

def main():
    global ARGS
    p=argparse.ArgumentParser(add_help=False)
    for name in ['asset','measurement-contract','measurement','policy','zero','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute',action='store_true');ARGS,_=p.parse_known_args()
    count=read(ARGS.policy/'scope.json')['candidate_rows']
    if not count:print(dict(not_run=True,reason='No Replacementcontained whole kernels',nativeF=0,nativeB=0,updates=0));return
    assert not (ARGS.execute and ARGS.output.exists()),'Preserve the previous diagnostic run'
    base.qualify=qualify;base.DiagnosticBackgroundAdapter=ContainedAdapter
    try:base.main()
    finally:
        if ARGS.execute and (ARGS.output/'receipt.json').is_file():
            receipt=read(ARGS.output/'receipt.json');receipt['schema']='R10.replacement_contained_background_zero_diagnostic.v1'
            receipt['outside_RGBA_invariance_checks']=CHECKS
            receipt['all_five_actual_outside_edit_replacement_car_K_RGBA_exact']=len(CHECKS)==5 and all(r['all_outside_and_protection_RGBA_exact'] for r in CHECKS)
            receipt['candidate_source']='footprint_limited_policy_v1 Replacementcontained; not prior22816 hard-retire selection'
            receipt['mask_receipt']=read(ARGS.policy/'scope.json')['mask_receipt']
            write(ARGS.output/'receipt.json',receipt)

if __name__=='__main__':main()
