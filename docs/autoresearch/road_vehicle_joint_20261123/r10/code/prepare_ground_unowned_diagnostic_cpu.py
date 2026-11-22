"""Non-colour ground-band selection for one explicit zero-step diagnostic.

All native rows, even useful body contributors, are measured before selection.
This writes a separate diagnostic sidecar and never edits the identity asset.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.time();r=a.root;assert not a.output.exists()
    cfg=read(r/'local_measure_contract_v2.json');base=read(cfg['baseline_contract']);frames=cfg['frames']
    measurement=r/'measure_v1/receipt.json';assert read(measurement)['complete']
    paths=[r/f'measure_v1/{raw:03d}/all_context_rows.npz' for raw in frames];z=[bank(p) for p in paths]
    initial=[]
    for value in z:
        assert (np.diff(value['global_row_ids'])>0).all()
        initial.append(value['global_row_ids'][value['native_context']&(value['owner_ids']<=0)&(value['ROI_alphaT_mass'][:,2]>0)])
    union=np.unique(np.concatenate(initial));mass=np.zeros((len(union),len(frames),7),np.float32)
    owners=np.full((len(union),len(frames)),-999,np.int32);present=np.zeros_like(owners,bool);native=np.zeros_like(owners,bool)
    for j,value in enumerate(z):
        ids=value['global_row_ids'];where=np.searchsorted(ids,union);ok=where<len(ids);ok[ok]&=ids[where[ok]]==union[ok]
        present[ok,j]=True;native[ok,j]=value['native_context'][where[ok]];owners[ok,j]=value['owner_ids'][where[ok]];mass[ok,j]=value['ROI_alphaT_mass'][where[ok]]
    assert np.isfinite(mass).all() and (mass>=0).all()
    permanent=np.asarray(read(base['context_plan'])['preserved_native_context_global_row_ids'],np.int64)
    flags=dict(missing_from_any_view=~present.all(1),not_native_in_any_view=~native.all(1),retained_positive_owner=((owners>0)&present).any(1),
        any_K_positive=(mass[:,:,0]>0).any(1),permanent_old_guard=np.isin(union,permanent))
    keep=~np.logical_or.reduce(list(flags.values()))
    assert (mass[keep,:,2].sum((1))>0).all() and (owners[keep]<=0).all() and native[keep].all() and present[keep].all()
    a.output.mkdir(parents=True);chosen=union[keep];selected=mass[keep];selected_owners=owners[keep]
    names=['K','body','ground','certified_ground','red','blue','dark']
    payload=dict(global_row_ids=chosen,frames=np.asarray(frames),owners_by_view=selected_owners)
    for j,name in enumerate(names):payload['allview_'+name+'_mass']=selected[:,:,j]
    np.savez_compressed(a.output/'diagnostic_native_candidates.npz',**payload)
    np.savez_compressed(a.output/'diagnostic_gate0.npz',global_row_ids=chosen,gate=np.zeros(len(chosen),np.float32))
    rows=[]
    for j,raw in enumerate(frames):
        rows.append(dict(raw=raw,**{name+'_alphaT_mass':float(selected[:,j,k].astype(np.float64).sum()) for k,name in enumerate(names)},
            body_touch_rows=int((selected[:,j,1]>0).sum()),ground_touch_rows=int((selected[:,j,2]>0).sum())))
    reject={k:dict(rows=int(v.sum()),ground_alphaT_by_view=mass[v,:,2].astype(np.float64).sum(0).tolist(),blue_alphaT_by_view=mass[v,:,5].astype(np.float64).sum(0).tolist()) for k,v in flags.items()}
    scope=dict(schema='R10.noncolour_ground_unowned_zero_step_scope.v1',complete=True,decision='SCOPE_VALID_DIAGNOSTIC_ONLY',policy='ground_band_all_unowned_Kfree_oldguardfree_body_overlap_allowed',
        baseline_contract_sha256=sha(Path(cfg['baseline_contract'])),measurement_contract=pin(r/'local_measure_contract_v2.json'),actual_attribution_receipt=pin(measurement),actual_attribution_receipt_sha256=sha(measurement),
        context_plan=pin(Path(base['context_plan'])),diagnostic_candidates=pin(a.output/'diagnostic_native_candidates.npz'),candidate_npz_sha256=sha(a.output/'diagnostic_native_candidates.npz'),
        diagnostic_gate=pin(a.output/'diagnostic_gate0.npz'),gate_npz_sha256=sha(a.output/'diagnostic_gate0.npz'),input_packets=[pin(p) for p in paths],
        frames=frames,candidate_rows=len(chosen),initial_noncolour_union_rows=len(union),native_owner_values=np.unique(selected_owners).tolist(),
        rule='native prefix; present all 5 views; every measured owner<=0; exact all-5 K mass zero; permanent old guard excluded; any positive contribution to registered full non-colour ground band; body overlap allowed solely for actual zero-step readout',
        ground_band_definition=cfg['ground_band_definition'],selection_uses_RGB_or_red_blue_masks=False,selection_uses_source_target_residual=False,
        K_exact_zero=True,old_permanent_guard_preserved=True,body_overlap_allowed_only_for_counterfactual=True,canonical_native21_and_other_positive_owners_fixed=True,
        rows=rows,overlapping_rejection_ledger=reject,CPU_only=True,nativeF=0,nativeB=0,updates=0,new_geometry=False,
        production_asset_ready=False,quality_accepted=False,optimization_allowed=False,original_native_identity_gate_not_modified=True,
        actual_zero_step_all_five_K_strictV_body_unknown_coverage_required=True,seconds=time.time()-started)
    (a.output/'scope.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,candidate_rows=len(chosen),scope=pin(a.output/'scope.json'),rows=rows),ensure_ascii=False))


if __name__=='__main__':main()
