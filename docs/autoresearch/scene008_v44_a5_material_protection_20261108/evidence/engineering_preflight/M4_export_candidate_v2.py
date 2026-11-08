"""M4 exact CPU material export. No solver, clipping, or model execution."""
from pathlib import Path
import argparse,copy,json,shutil,sys
import numpy as np
from M4_extract_operator import R,N,M,SCENE,write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--certificate',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--solver-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cert=json.loads(a.certificate.read_text());protocol=json.loads(a.protocol.read_text())
    assert cert['status'] in ['MAIN_PROTECTION_CERTIFIED_PENDING_NATIVE_REVIEW','FALLBACK_MINIMAX_CERTIFIED_PENDING_NATIVE_REVIEW','PHASE_I_DIAGNOSTIC_CERTIFIED_PENDING_NATIVE_REVIEW','PRIMARY_DIAGNOSTIC_NO_TIEBREAK']
    diagnostic=cert['status'] in ['PHASE_I_DIAGNOSTIC_CERTIFIED_PENDING_NATIVE_REVIEW','PRIMARY_DIAGNOSTIC_NO_TIEBREAK']
    if not diagnostic:
        assert cert['float32_certificate_pass'] and cert['float64_certificate_pass'],cert.get('status')
    else:
        # 诊断palette允许原门失败；完整box/冻结检查在下方按实际NPZ重做。
        assert cert['export_integrity_pass'] is True
    assert cert['protocol_sha256']==sha(a.protocol)
    assert cert['source_sha256']==sha(a.solver_source)
    assert cert['candidate']['sha256']==sha(a.candidate)
    assert protocol['status'].startswith('SEALED_BEFORE')
    if diagnostic:
        assert cert['phase_I_diagnostic'] is True
        assert np.isfinite(cert['photo_slack_t']) and cert['photo_slack_t']>=0
    role='phase_I_diagnostic' if diagnostic else 'registered_parent_candidate'
    with np.load(M/'operator_v1/parents.npz',allow_pickle=False) as z:
        ids=z['variable_parent_ids'].copy();c0=z['full_parent_RGB'].copy()
    with np.load(a.candidate,allow_pickle=False) as z:
        full=z['full_parent_RGB'].copy();candidate_ids=z['variable_parent_ids'].copy()
    assert full.shape==(10098,3) and full.dtype==np.float32 and np.isfinite(full).all() and full.min()>=0 and full.max()<=1
    assert np.array_equal(ids,candidate_ids) and ids.shape==(5246,)
    fixed=np.ones(10098,bool);fixed[ids]=False
    assert np.array_equal(full[fixed],c0[fixed])
    parent=json.loads(SCENE.read_text());asset=SCENE.parent/parent['new_road']['file'];assert sha(asset)==parent['new_road']['sha256']
    with np.load(asset,allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    assert np.array_equal(c0,old['parent_render_colours_srgb'])
    new={k:v.copy() for k,v in old.items()}
    new['parent_render_colours_srgb']=full;new['render_colours_srgb']=full[old['child_to_parent_ids']]
    out=a.output;out.mkdir(exist_ok=False);target=out/'new_road_M4_protected_supervision.npz';np.savez_compressed(target,**new)
    allowed={'parent_render_colours_srgb','render_colours_srgb'};changed=[]
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
    for source,name in [(SCENE,'formal2k_scene_manifest.json'),(a.candidate,'M4_candidate_parent_RGB.npz'),(a.certificate,'M4_CPU_certificate.json'),(a.protocol,'M4_LP_protocol.json'),(a.solver_source,'M4_CPU_solver_source.py'),(M/'operator_v1/operator_manifest.json','M4_operator_manifest.json'),(Path(__file__),'M4_export_candidate_v2.py')]:shutil.copyfile(source,provenance/name)
    changedids=np.flatnonzero(np.any(full!=c0,axis=1)).astype(np.int64)
    details=dict(formal2k_parent_asset_sha256=sha(asset),candidate_file=str(a.candidate),candidate_sha256=sha(a.candidate),CPU_certificate_file=str(a.certificate),CPU_certificate_sha256=sha(a.certificate),LP_protocol_file=str(a.protocol),LP_protocol_sha256=sha(a.protocol),CPU_solver_source_file=str(a.solver_source),CPU_solver_source_sha256=sha(a.solver_source),CPU_certificate_status=cert.get('status'),float32_engineering_certificate_pass=cert['float32_certificate_pass'],float64_certificate_pass=cert.get('float64_certificate_pass'),engineering_role=role,phase_I_diagnostic=diagnostic,photo_slack_t=cert.get('photo_slack_t'),original_photo_gates_pass=cert.get('original_photo_gates_pass'),original_quality_adoption_claim=False,variable_parent_count=len(ids),fixed_parent_count=int(fixed.sum()),actual_changed_parent_count=len(changedids),actual_changed_parent_ids=changedids.tolist(),fixed4852_parent_RGB_byte_exact=True,actual_changed_array_keys=changed,all_other_asset_arrays_dtype_shape_bytes_exact=True,no_resolve_no_clip_no_reprojection=True,Adam_updates=0,original_model_updates=0,source_sha256=sha(__file__))
    manifest['new_road'].update(file=target.name,sha256=sha(target),material_variant='M4_fixed_H2_'+role+'_shared_parent_RGB')
    manifest['lineage']=dict(formal2k_parent_manifest_file='provenance/formal2k_scene_manifest.json',formal2k_parent_manifest_sha256=sha(SCENE),M4_protected_supervision=details)
    manifest['material_writeback'].update(M4_Adam_updates=0,M4_original_model_updates=0,current_parent_colour_key='parent_render_colours_srgb',current_child_colour_key='render_colours_srgb')
    manifest['known_unresolved']+=['M4 is a four BUILD frame candidate; human verdict remains unset.','Use the original fixed H2 child visibility; no new culling or image patch is included.']
    manifest.update(exporter_sha256=sha(__file__),GPU_parity_state_at_export='M4_REAL8F_PENDING',GPU_parity_receipt='native_validation_pending.json',adoption_claim=False,human_verdict=None)
    write(out/'scene_manifest.json',manifest);write(out/'native_validation_pending.json',dict(status='NOT_RUN_BY_CPU_EXPORTER',native_forwards=0))
    write(out/'export_receipt.json',dict(status='PASS_M4_EXACT_CPU_RGB_EXPORT_PENDING_NATIVE8F',scene_manifest_sha256=sha(out/'scene_manifest.json'),asset_sha256=sha(target),**details))
    print(json.dumps(dict(scene_manifest=str(out/'scene_manifest.json'),changed_arrays=changed,changed_parents=len(changedids),fixed4852_exact=True,native_forwards=0)),flush=True)

if __name__=='__main__':main()
