"""Extend only M2's original zero RGB columns; keep observed coefficients exact."""
import json,sys,hashlib,time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import torch
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign');OLD=CAM/'scene008_a6_asset_controls_20261105';RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(OLD/'source'))
from sh_asset import SHAsset,C0
from fit_material import graph,METHODS
from surface_material import extend_material,verify_train_invariance
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    start=time.time();dest=RUN/'real';dest.mkdir(exist_ok=False);g=graph()
    source=CAM/'scene008_a6_io_audit_20261105/contract_asset/m2_with_io_contract.npz'
    arrays=dict(np.load(source));base=SHAsset.from_geometry(source);coeff=base.coefficients.numpy().copy();rgb=base.rgb.numpy().copy()
    A=load_npz(OLD/'m2/design.npz');A.eliminate_zeros();unknown=A.getnnz(axis=0)==0
    assert int(unknown.sum())==10557 and bool(base.colour_inferred.numpy()[unknown].all())
    # The current zero-column material is exactly one RGB fallback.
    global_rgb=rgb[unknown][0].copy();assert np.array_equal(rgb[unknown],np.tile(global_rgb,(unknown.sum(),1)))
    reports=[]
    for name in METHODS:
        topology='parent' if name.startswith('parent_') else 'child';rule=name.split('_')[-1] if name!='median' else 'median'
        result=extend_material(g[topology],rgb[~unknown],unknown,global_rgb,method=rule)
        current=coeff.copy();change=np.asarray(result.propagated,bool)
        if name!='median':current[change,0]=(result.rgb[change]-.5)/C0
        assert np.array_equal(current[~unknown],coeff[~unknown])
        payload={k:v.copy() for k,v in arrays.items()};payload.pop('provenance_hash')
        payload['appearance_extension_unknown']=unknown;payload['appearance_extension_propagated']=change
        payload['appearance_extension_unreachable']=np.asarray(result.unreachable,bool)
        payload['canonical_sh']=current
        metadata=json.loads(str(payload['metadata_json']));metadata['appearance_extension']=dict(method=name,
            source_sha256=sha(source),scope='Only original BUILD strictly zero columns; all nonzero SH coefficients preserved byte-for-byte',
            appearance_inferred=True,not_texture_truth=True,topology=topology,graph_geometry_hash=g[topology].geometry_hash)
        payload['metadata_json']=np.asarray(json.dumps(metadata,sort_keys=True))
        asset=SHAsset(payload,metadata);asset.set_coefficients(current,base.colour_inferred.numpy())
        exported=asset.export(dest/f'{name}_asset.npz');fresh=SHAsset.from_geometry(dest/f'{name}_asset.npz')
        assert fresh.geometry_hash==base.geometry_hash and torch.equal(fresh.coefficients[~unknown],base.coefficients[~unknown])
        invariant=verify_train_invariance(A,rgb,fresh.rgb.numpy())
        if name=='median':assert torch.equal(fresh.coefficients,base.coefficients)
        reports.append(dict(method=name,export=exported,extension=result.audit,train_invariance=invariant,
            full_nonzero_SH_coefficients_bitwise_unchanged=True,original_inferred_flags_unchanged=True))
        print(json.dumps(dict(method=name,propagated=int(change.sum()))),flush=True)
    write(dest/'receipt.json',dict(status='PASS_INTERFACE_AND_DATA_PRESERVATION_ONLY',methods=reports,
        actual_unknown_zero_columns=int(unknown.sum()),old_M2_sha256=sha(OLD/'m2/complete_a6.npz'),
        child_graph=g['child'].audit,parent_graph=g['parent'].audit,seconds=time.time()-start,
        colour_fit_steps=0,model_training_steps=0,scene_adoption=False))

if __name__=='__main__':main()
