"""Native8F of independent existing child RGB; geometry/identity stay frozen."""
from pathlib import Path
import argparse,json,os,sys,time
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,'/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1/M4_protected_supervision_v1/source')
import numpy as np
from M4_extract_operator import B,R,N,S,M,REPO,H2,REF,RAWS,STATES,write,sha,ident
C=R/'M4_child_material_capacity_v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    scene_manifest=json.loads(a.scene_manifest.read_text());asset=a.scene_manifest.parent/scene_manifest['new_road']['file'];assert sha(asset)==scene_manifest['new_road']['sha256']
    export=json.loads((a.scene_manifest.parent/'export_receipt.json').read_text());assert export['status']=='PASS_M4_CHILD_EXACT_CPU_RGB_EXPORT_PENDING_NATIVE8F' and export['fixed19667_child_RGB_byte_exact'] and export['all10098_parent_RGB_byte_exact'] and export['all_other_asset_arrays_dtype_shape_bytes_exact']
    role=export['engineering_role'];assert role=='registered_child_candidate' and export['material_basis']=='child_RGB'
    with np.load(C/'operator_v1/children.npz',allow_pickle=False) as z:ids=z['variable_child_ids'].copy();c0=z['full_child_RGB'].copy();source_parent_rgb=z['full_parent_RGB'].copy()
    with np.load(asset,allow_pickle=False) as z:palette=z['render_colours_srgb'].copy();parent_rgb=z['parent_render_colours_srgb'].copy()
    assert palette.dtype==np.float32 and palette.shape==(40392,3) and np.isfinite(palette).all() and palette.min()>=0 and palette.max()<=1
    assert np.array_equal(source_parent_rgb,parent_rgb)
    fixed=np.ones(40392,bool);fixed[ids]=False;assert np.array_equal(palette[fixed],c0[fixed])
    delta=palette[ids].astype(np.float64)-c0[ids].astype(np.float64)
    out=a.output;out.mkdir(exist_ok=False);(out/'frames').mkdir();start=time.time()
    guarded=[Path(__file__),a.scene_manifest,asset,a.scene_manifest.parent/'export_receipt.json',R/'source/foreground_road_culling_H4_v2.py',Path(export['CPU_certificate_file']),Path(export['LP_protocol_file']),Path(export['CPU_solver_source_file']),Path(export['candidate_file'])]
    assert sha(export['CPU_certificate_file'])==export['CPU_certificate_sha256']
    assert sha(export['LP_protocol_file'])==export['LP_protocol_sha256']
    assert sha(export['CPU_solver_source_file'])==export['CPU_solver_source_sha256']
    assert sha(export['candidate_file'])==export['candidate_sha256']
    before={str(x):ident(x) for x in guarded};write(out/'frozen_inputs_before.json',before)
    write(out/'protocol.json',dict(status='SEALED_BEFORE_M4_CHILD_CANDIDATE_NATIVE8F',engineering_role=role,material_basis='child_RGB',parent_RGB_array_role=export['parent_RGB_array_role'],same_colour_within_each_parent=export['same_colour_within_each_parent'],original_quality_adoption_claim=False,planned_native_forwards=8,planned_backwards=0,optimizer_updates=0,original_model_updates=0,candidate_manifest=str(a.scene_manifest),candidate_manifest_sha256=sha(a.scene_manifest),asset_sha256=sha(asset),source_sha256=sha(__file__),fixed_geometry='original H2 visibility plus21K v44+a5 and complete C geometry',shutdown=False,automatic49=False))
    import torch
    from scipy import sparse
    from PIL import Image
    sys.path[:0]=[str(R/'source'),str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
    from v44_runtime import V44Runtime
    from foreground_road_culling_H4_v2 import ForegroundRoadCulling
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.baseline_road_appearance import baseline_appearance_binding
    torch.set_num_threads(1);base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
    sem=SemanticActor5V44Integration(base,semantic_sidecar_path=S/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=R/'contract/full5_transform_matrix.txt')
    binding=baseline_appearance_binding(sem);road=RoadFusionV44Integration(sem,a.scene_manifest);h2=ForegroundRoadCulling(road,max_iterations=64)
    for item in json.loads((H2/'receipt.json').read_text())['records']:
        raw,state=item['raw'],item['state'];record=item['meta']['foreground_road_culling']
        with np.load(H2/f'frames/{raw:03d}/{state}_visibility_mask.npz',allow_pickle=False) as z:mask=z['selected_road_child_rows'].copy();guard=z['model_foreground_guard'].copy()
        assert record['converged'] and int(mask.sum())==record['active_rows']
        h2.cache[(raw,state=='delete')]=dict(mask=torch.as_tensor(mask,device='cuda'),guard=torch.as_tensor(guard,device='cuda'),record=record,baseline=None)
    records=[]
    for raw in RAWS:
        folder=out/f'frames/{raw:03d}';folder.mkdir()
        for state in STATES:
            with torch.no_grad():native=h2.render_scene(raw,delete=state=='delete')
            write(out/'kernel_progress.json',dict(actual_native_forwards=h2.actual_forwards,actual_backwards=0,optimizer_updates=0,raw=raw,state=state))
            rgb=native['rgb'].cpu().numpy();alpha=native['alpha'].cpu().numpy()[0];ds=native['depth_sum'].cpu().numpy()[0]
            values=dict(rgb=rgb,alpha=alpha,depth_sum=ds,depth_m=ds/np.maximum(alpha,1e-8)/sem.scale,group_alpha_T=native['group_alpha_T'].cpu().numpy(),added_owner_rgb_T=native['added_owner_rgb_T'].cpu().numpy(),newroad_rgb_T=native['newroad_rgb_T'].cpu().numpy())
            with np.load(REF/f'frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
            errors={}
            for k in ['alpha','depth_sum','depth_m','group_alpha_T','added_owner_rgb_T']:
                errors[k]=float(np.abs(values[k]-ref[k]).max());assert np.array_equal(values[k],ref[k]),(raw,state,k,errors[k])
            op=C/f'operator_v1/{raw:03d}_{state}';A=sparse.load_npz(op/'A_child_rgb_csr.npz').astype(np.float64)
            with np.load(op/'pixel_domains_baseline.npz',allow_pickle=False) as z:px=z['pixel_index'].copy()
            expected=ref['rgb'].reshape(3,-1)[:,px].T.astype(np.float64)+A@delta
            actual=rgb.reshape(3,-1)[:,px].T.astype(np.float64);affinity=float(np.abs(expected-actual).max(initial=0))
            np.savez_compressed(folder/(state+'_float.npz'),**values)
            Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(folder/(state+'.png'))
            rec=dict(raw=raw,state=state,formal2k_frozen_geometry_8groups_identityRGB_max_abs=errors,pixel_affinity_max_abs=affinity,pixel_affinity_original_1e6_gate_pass=affinity<=1e-6,road_mass_on_original_model_foreground=native['meta']['public_foreground_road_alpha_mass'])
            records.append(rec);print(json.dumps(rec),flush=True);del native
        base._data.clear();torch.cuda.empty_cache()
    assert h2.actual_forwards==8 and h2.actual_dummy_backwards==0
    assert baseline_appearance_binding(sem)==binding
    assert {str(x):ident(x) for x in guarded}==before
    receipt=h2.receipt();write(out/'frozen_inputs_after.json',before)
    write(out/'receipt.json',dict(status='COMPLETE_M4_CHILD_EXACT_NATIVE8F_PENDING_REGISTERED_GATES',engineering_role=role,material_basis='child_RGB',parent_RGB_array_role=export['parent_RGB_array_role'],same_colour_within_each_parent=export['same_colour_within_each_parent'],original_quality_adoption_claim=False,actual_native_forwards=8,actual_backwards=0,optimizer_updates=0,original_model_updates=0,records=records,pixel_affinity_original_1e6_all_pass=all(x['pixel_affinity_original_1e6_gate_pass'] for x in records),all_nonRGB_asset_arrays_exact=True,fixed19667_child_RGB_exact=True,all10098_parent_RGB_exact=True,oldH2_geometry_8groups_identityRGB_exact=True,source_and_candidate_unchanged=True,runtime_receipt=receipt,seconds=time.time()-start,max_reserved_cuda_bytes=int(torch.cuda.max_memory_reserved()),quality_accepted=False,human_verdict=None,automatic49=False,shutdown=False))

if __name__=='__main__':main()
