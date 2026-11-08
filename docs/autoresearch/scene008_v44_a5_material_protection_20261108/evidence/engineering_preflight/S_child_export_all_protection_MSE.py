"""S child all-protection MSE: exact diagnostic export without seam RGB guard."""
from pathlib import Path
import argparse,copy,json,shutil,sys
sys.path.insert(0,'/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1/M4_protected_supervision_v1/source')
import numpy as np
from M4_extract_operator import R,N,M,SCENE,write,sha
C=R/'M4_child_material_capacity_v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--certificate',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--solver-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cert=json.loads(a.certificate.read_text());protocol=json.loads(a.protocol.read_text())
    assert cert['diagnostic'] is True and cert['export_integrity_pass'] is True
    assert cert['status']=='FINITE_BOUNDED_CHILD_ALL_PROTECTION_PALETTE_READY_NATIVE8F'
    assert cert['free_child_count']==20725 and cert['seam_frozen_child_count']==0
    assert cert['material_basis']=='child_RGB'
    assert cert['protocol_sha256']==sha(a.protocol)
    assert cert['source_sha256']==sha(a.solver_source)
    assert cert['candidate']['sha256']==sha(a.candidate)
    assert protocol['status'].startswith('SEALED_BEFORE')
    role='S_child_all_protection_MSE_diagnostic'
    with np.load(C/'operator_v1/children.npz',allow_pickle=False) as z:
        ids=z['variable_child_ids'].copy();c0=z['full_child_RGB'].copy();mapping=z['child_to_parent_ids'].copy();source_parent_rgb=z['full_parent_RGB'].copy()
    with np.load(a.candidate,allow_pickle=False) as z:
        full=z['full_child_RGB'].copy();candidate_ids=z['variable_child_ids'].copy();candidate_mapping=z['child_to_parent_ids'].copy()
    assert full.shape==(40392,3) and full.dtype==np.float32 and np.isfinite(full).all() and full.min()>=0 and full.max()<=1
    assert np.array_equal(ids,candidate_ids) and ids.shape==(20725,) and np.array_equal(mapping,candidate_mapping)
    fixed=np.ones(40392,bool);fixed[ids]=False
    assert np.array_equal(full[fixed],c0[fixed])
    parent=json.loads(SCENE.read_text());asset=SCENE.parent/parent['new_road']['file'];assert sha(asset)==parent['new_road']['sha256']
    with np.load(asset,allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    assert np.array_equal(c0,old['render_colours_srgb']) and np.array_equal(source_parent_rgb,old['parent_render_colours_srgb']) and np.array_equal(mapping,old['child_to_parent_ids'])
    new={k:v.copy() for k,v in old.items()}
    new['render_colours_srgb']=full
    out=a.output;out.mkdir(exist_ok=False);target=out/'new_road_M4_independent_child_RGB.npz';np.savez_compressed(target,**new)
    allowed={'render_colours_srgb'};changed=[]
    with np.load(target,allow_pickle=False) as z:
        assert set(z.files)==set(old)
        for k in old:
            assert z[k].dtype==old[k].dtype and z[k].shape==old[k].shape
            if z[k].tobytes()!=old[k].tobytes():changed.append(k)
            assert k in allowed or z[k].tobytes()==old[k].tobytes(),k
            assert z[k].tobytes()==new[k].tobytes(),k
    manifest=copy.deepcopy(parent)
    for item in parent['replay_sources'].values():
        source=SCENE.parent/item['file'];assert sha(source)==item['sha256'];shutil.copyfile(source,out/item['file'])
    provenance=out/'provenance';provenance.mkdir()
    for source,name in [(SCENE,'formal2k_scene_manifest.json'),(a.candidate,'S_all_protection_MSE_candidate_child_RGB.npz'),(a.certificate,'S_child_CPU_result.json'),(a.protocol,'S_child_MSE_protocol.json'),(a.solver_source,'S_child_CPU_solver_source.py'),(C/'operator_v1/operator_manifest.json','M4_child_operator_manifest.json'),(Path(__file__),'S_child_export_all_protection_MSE.py')]:shutil.copyfile(source,provenance/name)
    changedids=np.flatnonzero(np.any(full!=c0,axis=1)).astype(np.int64)
    uniform=bool(np.array_equal(full.reshape(10098,4,3),np.repeat(full.reshape(10098,4,3)[:,:1],4,axis=1)))
    details=dict(formal2k_parent_asset_sha256=sha(asset),candidate_file=str(a.candidate),candidate_sha256=sha(a.candidate),CPU_certificate_file=str(a.certificate),CPU_certificate_sha256=sha(a.certificate),LP_protocol_file=str(a.protocol),LP_protocol_sha256=sha(a.protocol),CPU_solver_source_file=str(a.solver_source),CPU_solver_source_sha256=sha(a.solver_source),CPU_certificate_status=cert.get('status'),CPU_LS_calls=cert.get('actual_solver_calls',cert.get('lsq_calls')),engineering_role=role,diagnostic=True,material_basis='child_RGB',original_quality_adoption_claim=False,variable_child_count=len(ids),fixed_child_count=int(fixed.sum()),actual_changed_child_count=len(changedids),actual_changed_child_ids=changedids.tolist(),fixed19667_child_RGB_byte_exact=True,all10098_parent_RGB_byte_exact=True,parent_RGB_array_role='Unchanged historical source provenance; active material reads independent child RGB',same_colour_within_each_parent=uniform,actual_changed_array_keys=changed,all_other_asset_arrays_dtype_shape_bytes_exact=True,no_resolve_no_clip_no_reprojection=True,Adam_updates=0,original_model_updates=0,source_sha256=sha(__file__))
    details.update(extra_seam_RGB_guard=False,only_ablation='Remove the added seam RGB freezing; all20725 existing child RGB variables are available. User geometry/opacity/actor/identity freezes stay exact.')
    manifest['new_road'].update(file=target.name,sha256=sha(target),material_variant='S_child_all_protection_MSE_fixed_H2')
    manifest['lineage']=dict(formal2k_parent_manifest_file='provenance/formal2k_scene_manifest.json',formal2k_parent_manifest_sha256=sha(SCENE),M4_child_material_capacity=details)
    manifest['material_writeback'].update(M4_Adam_updates=0,M4_original_model_updates=0,current_colour_binding='independent_existing_child_RGB',current_child_colour_key='render_colours_srgb',parent_RGB_array_role=details['parent_RGB_array_role'],same_colour_within_each_parent=uniform)
    manifest['known_unresolved']+=['M4 child RGB is a four BUILD frame candidate; human verdict remains unset.','Use the original fixed H2 child visibility; no new culling or image patch is included.','Parent RGB array is historical provenance; child colours may differ within each parent.']
    manifest.update(exporter_sha256=sha(__file__),GPU_parity_state_at_export='M4_REAL8F_PENDING',GPU_parity_receipt='native_validation_pending.json',adoption_claim=False,human_verdict=None)
    write(out/'scene_manifest.json',manifest);write(out/'native_validation_pending.json',dict(status='NOT_RUN_BY_CPU_EXPORTER',native_forwards=0))
    write(out/'export_receipt.json',dict(status='PASS_M4_CHILD_EXACT_CPU_RGB_EXPORT_PENDING_NATIVE8F',scene_manifest_sha256=sha(out/'scene_manifest.json'),asset_sha256=sha(target),**details))
    print(json.dumps(dict(scene_manifest=str(out/'scene_manifest.json'),changed_arrays=changed,changed_children=len(changedids),fixed19667_exact=True,all_parent_RGB_exact=True,native_forwards=0)),flush=True)

if __name__=='__main__':main()
