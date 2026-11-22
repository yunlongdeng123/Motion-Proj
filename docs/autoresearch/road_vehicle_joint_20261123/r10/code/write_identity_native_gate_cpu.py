"""Explicit no-op native binding for a separately qualified new local field.

Empty IDs mean preserve every old native Gaussian. This scope certificate
does not qualify new geometry, appearance, K/body visibility or a final asset.
"""
from pathlib import Path
import argparse
import numpy as np
from r10_common import contracts,read,write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--measurement',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg,base=contracts(a.contract)
    measured=read(a.measurement/'receipt.json');assert measured['complete'] and measured['actual_nativeF']==5 and measured['nativeB']==5 and measured['updates']==0
    assert measured['baseline_contract_sha256']==sha(cfg['baseline_contract'])
    parent=read(base['scene_proof']);a.output.mkdir(parents=True,exist_ok=False)
    ids=np.empty(0,np.int64);zero=np.empty((0,len(cfg['frames'])),np.float32)
    np.savez_compressed(a.output/'native_candidates.npz',global_row_ids=ids,frames=np.asarray(cfg['frames']),allview_K_mass=zero,allview_body_mass=zero,
        allview_ground_mass=zero,allview_certified_ground_mass=zero,allview_red_mass=zero,allview_blue_mass=zero,allview_dark_mass=zero,
        owners_by_view=np.empty((0,len(cfg['frames'])),np.int32))
    np.savez_compressed(a.output/'native_gate0.npz',global_row_ids=ids,gate=np.empty(0,np.float32))
    record=dict(schema='R10.native_global_opacity_gate.v1',complete=True,decision='PASS',policy='identity_native_noop',action='preserve_all_native_rows',candidate_rows=0,
        baseline_contract_sha256=sha(cfg['baseline_contract']),candidate_npz_sha256=sha(a.output/'native_candidates.npz'),gate_npz_sha256=sha(a.output/'native_gate0.npz'),
        actual_attribution_receipt_sha256=sha(a.measurement/'receipt.json'),fixed_eval_receipt_sha256=measured['fixed_eval_receipt_sha256'],
        all_five_K_body_exact_zero=True,K_body_condition_is_vacuous_for_empty_selection=True,
        old_permanent_guard_preserved=True,old_context_plan_sha256=sha(base['context_plan']),native_source_pins=parent['native_sources'],native_topology=parent['topology'],
        native_scene_proof_sha256=sha(base['scene_proof']),accepted_actor_sha256=cfg['accepted_actor_sha256'],source_field_sha256=cfg['source']['sha256'],
        native21_and_other_retained_actors_frozen=True,all_old_native_parameters_unmodified=True,gate_only_decrease=True,new_geometry=False,
        old_RoadRows_retirement_not_authorized_by_this_scope=True,
        new_field_geometry_and_appearance_acceptance_required=True,new_field_actual_K_and_body_measurement_required=True,
        new_field_can_naturally_occlude_distant_native_tail_only_after_validation=True,
        diagnostic_5444_and_8816_gates_not_adopted=True,scope_PASS_does_not_mean_quality_PASS=True,
        production_new_local_asset_ready=False,optimization_enabled=False,CPU_only=True,nativeF=0,nativeB=0,updates=0)
    write(a.output/'native_gate0.proof.json',record);print(dict(complete=True,identity=True,gate_sha256=record['gate_npz_sha256'],proof_sha256=sha(a.output/'native_gate0.proof.json')))


if __name__=='__main__':main()
