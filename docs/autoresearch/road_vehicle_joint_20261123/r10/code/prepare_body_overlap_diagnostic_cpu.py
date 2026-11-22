"""One separate red-row counterfactual policy, without changing strict results.

Body overlap may change occlusion, never the accepted actor/native21 assets.
This is a diagnostic gate, not a production/local-reconstruction certificate.
"""
from pathlib import Path
import argparse
import numpy as np
from r10_common import contracts,read,load,write,sha


def pin(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--measurement',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg,base=contracts(a.contract);receipt=read(a.measurement/'receipt.json');assert receipt['complete'] and receipt['actual_nativeF']==5 and receipt['nativeB']==5
    a.output.mkdir(parents=True,exist_ok=False);z={raw:load(a.measurement/f'{raw:03d}/all_context_rows.npz') for raw in cfg['frames']}
    initial=[]
    for raw,value in z.items():
        allowed=value['owner_ids']<=0
        for owner in cfg['deleted_target_owner_ids']:allowed|=value['owner_ids']==owner
        initial.append(value['global_row_ids'][value['native_context']&allowed&(value['ROI_alphaT_mass'][:,4]>0)])
    union=np.unique(np.concatenate(initial));mass=np.zeros((len(union),len(cfg['frames']),7),np.float32);owner=np.full(mass.shape[:2],-999,np.int32);present=np.zeros(mass.shape[:2],bool)
    for column,raw in enumerate(cfg['frames']):
        value=z[raw];ids=value['global_row_ids'];where=np.searchsorted(ids,union);ok=where<len(ids);ok[ok]&=ids[where[ok]]==union[ok]
        present[:,column]=ok;mass[ok,column]=value['ROI_alphaT_mass'][where[ok]];owner[ok,column]=value['owner_ids'][where[ok]]
    permanent=np.asarray(read(base['context_plan'])['preserved_native_context_global_row_ids'],np.int64)
    allowed=owner<=0
    for value in cfg['deleted_target_owner_ids']:allowed|=owner==value
    keep=present.all(1)&allowed.all(1)&(owner==owner[:,0:1]).all(1)&(mass[:,:,0]==0).all(1)&~np.isin(union,permanent)
    chosen=union[keep];selected=mass[keep];owners=owner[keep]
    np.savez_compressed(a.output/'diagnostic_native_candidates.npz',global_row_ids=chosen,frames=np.asarray(cfg['frames']),owners_by_view=owners,
        allview_K_mass=selected[:,:,0],allview_body_mass=selected[:,:,1],allview_ground_mass=selected[:,:,2],
        allview_certified_ground_mass=selected[:,:,3],allview_red_mass=selected[:,:,4],allview_blue_mass=selected[:,:,5],allview_dark_mass=selected[:,:,6])
    np.savez_compressed(a.output/'diagnostic_gate0.npz',global_row_ids=chosen,gate=np.zeros(len(chosen),np.float32))
    frame=[]
    for index,raw in enumerate(cfg['frames']):
        frame.append(dict(raw=raw,selected_red_alphaT_mass=float(selected[:,index,4].astype(float).sum()),
            selected_blue_alphaT_mass=float(selected[:,index,5].astype(float).sum()),selected_body_alphaT_mass=float(selected[:,index,1].astype(float).sum()),
            selected_ground_alphaT_mass=float(selected[:,index,2].astype(float).sum()),selected_K_alphaT_mass=float(selected[:,index,0].astype(float).sum())))
    report=dict(schema='R10.body_overlap_diagnostic_scope.v1',complete=True,decision='SCOPE_VALID_DIAGNOSTIC_ONLY',policy='body_overlap_diagnostic',
        source_measurement=pin(a.measurement/'receipt.json'),baseline_contract_sha256=sha(cfg['baseline_contract']),source_field_sha256=cfg['source']['sha256'],
        diagnostic_candidates=pin(a.output/'diagnostic_native_candidates.npz'),diagnostic_gate=pin(a.output/'diagnostic_gate0.npz'),
        old_strict_candidates_unchanged=pin(a.measurement/'native_candidates.npz'),old_strict_gate_unchanged=pin(a.measurement/'native_gate0.npz'),
        positive_red_allowed_union_rows=len(union),candidate_rows=len(chosen),rejected_missing_view=int((~present.all(1)).sum()),
        rejected_Ktouch=int((mass[:,:,0]>0).any(1).sum()),rejected_old_guard=int(np.isin(union,permanent).sum()),
        rows_with_body_overlap=int((selected[:,:,1]>0).any(1).sum()),owners=np.unique(owners).tolist(),rows=frame,
        all_five_K_exact_zero=True,old_permanent_guard_preserved=True,body_overlap_allowed_only_for_counterfactual=True,
        canonical_native21_and_known_vehicles_assets_fixed=True,known_retained_owners_not_reclassified_as_A9=True,
        formal_updates=0,new_geometry=False,production_asset_ready=False,quality_accepted=False,optimization_allowed=False,
        interpretation='Frozen original red pixels select actual native contributors; these are not automatically bad/A9 Gaussian labels. Only a zero-step counterfactual is authorized.')
    write(a.output/'scope.json',report);print(dict(complete=True,candidates=len(chosen),body_overlap_rows=report['rows_with_body_overlap'],rows=frame,scope=pin(a.output/'scope.json')))


if __name__=='__main__':main()
