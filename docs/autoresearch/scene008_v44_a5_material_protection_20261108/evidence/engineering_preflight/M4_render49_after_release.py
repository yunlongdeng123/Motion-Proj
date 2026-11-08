"""Prepared49-frame M renderer. A four-frame engineering release is required."""
from pathlib import Path
import argparse,json,os,sys,time
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from M4_extract_operator import B,R,S,M,REPO,H2,RAWS,STATES,OBS,write,sha

def dependencies(out):
    out.mkdir(exist_ok=False);base49=S/'candidate_semantic_a5_49';records=[]
    for raw in range(89,138):
        assert (OBS/f'{raw:03d}.npz').exists()
        for state in STATES:
            reference=base49/f'frames/{raw:03d}/{state}_float.npz';assert reference.exists()
            common=None
            if raw in RAWS:
                with np.load(reference,allow_pickle=False) as z:old={k:z[k].copy() for k in ['rgb','alpha','depth_sum']}
                with np.load(R/f'baseline4/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
                    common={k:bool(np.array_equal(v,z[k])) for k,v in old.items()};assert all(common.values())
            records.append(dict(raw=raw,state=state,original21k_reference=str(reference),baseline4_overlap_byte_exact=common))
    write(out/'receipt.json',dict(status='PREPARED49_ENTRY_ONLY_NOT_LAUNCHED',source=str(Path(__file__)),source_sha256=sha(__file__),raw_ids=list(range(89,138)),states=STATES,reference21k_root=str(base49),records=records,planned_public_forwards=98,additional_H2_qualification_forwards_and_dummy_backwards='Required for90 uncached raw/states, registered original64iter policy; count separately at runtime',existing_cached_rawstates=8,actual_native_forwards=0,actual_backwards=0,optimizer_updates=0,release_required=True,automatic49=False,shutdown=False))
    print(json.dumps(dict(status='PREPARED49_ONLY_NO_GPU',output=str(out),reference98_files_present=True,baseline4_common8_RGB_alpha_depth_byte_exact=True)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--scene-manifest',type=Path);p.add_argument('--release-json',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.prepare_only:
        dependencies(a.output);return
    assert a.scene_manifest and a.release_json,'Four-frame release must be concrete before49 launch'
    release=json.loads(a.release_json.read_text())
    assert release['status']=='SEALED_AFTER_M4_FOURFRAME_NATIVE_AND_ROOT_REVIEW'
    assert release['four_frame_registered_hard_gates_pass'] and release['root_visual_review_pass']
    assert release['raw_ids']==list(range(89,138)) and release['states']==STATES
    assert sha(a.scene_manifest)==release['scene_manifest_sha256']
    receipt=Path(release['four_frame_native_receipt']);assert sha(receipt)==release['four_frame_native_receipt_sha256']
    four=json.loads(receipt.read_text());assert four['status']=='COMPLETE_M4_EXACT_NATIVE8F_PENDING_REGISTERED_GATES'
    assert four['actual_native_forwards']==8 and four['oldH2_geometry_8groups_identityRGB_exact']
    start=time.time();out=a.output;out.mkdir(exist_ok=False);(out/'frames').mkdir()
    write(out/'protocol.json',dict(status='SEALED_BEFORE_RELEASED49_NATIVE_RENDER',release=str(a.release_json),release_sha256=sha(a.release_json),source_sha256=sha(__file__),planned_public_forwards=98,qualification_policy='Original H2 model-foreground conservative whole-child culling; max_iterations64 unchanged; no observed RGB/mask visibility choice',qualification_F_B='Additional exact counts reported separately',optimizer_updates=0,geometry_updates=0,material_updates=0,shutdown=False))
    import torch
    from PIL import Image
    sys.path[:0]=[str(R/'source'),str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
    from v44_runtime import V44Runtime
    from foreground_road_culling_H4_v2 import ForegroundRoadCulling
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration
    torch.set_num_threads(1);base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
    sem=SemanticActor5V44Integration(base,semantic_sidecar_path=S/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=R/'contract/full5_transform_matrix.txt')
    road=RoadFusionV44Integration(sem,a.scene_manifest);h2=ForegroundRoadCulling(road,max_iterations=64)
    for item in json.loads((H2/'receipt.json').read_text())['records']:
        raw,state=item['raw'],item['state'];record=item['meta']['foreground_road_culling']
        with np.load(H2/f'frames/{raw:03d}/{state}_visibility_mask.npz',allow_pickle=False) as z:mask=z['selected_road_child_rows'].copy();guard=z['model_foreground_guard'].copy()
        assert record['converged'];h2.cache[(raw,state=='delete')]=dict(mask=torch.as_tensor(mask,device='cuda'),guard=torch.as_tensor(guard,device='cuda'),record=record,baseline=None)
    rows=[]
    for raw in range(89,138):
        folder=out/f'frames/{raw:03d}';folder.mkdir()
        for state in STATES:
            # Missing cache qualification intentionally enables only dummy-feature autograd.
            native=h2.render_scene(raw,delete=state=='delete')
            alpha=native['alpha'].detach().cpu().numpy()[0];ds=native['depth_sum'].detach().cpu().numpy()[0];rgb=native['rgb'].detach().cpu().numpy()
            values=dict(rgb=rgb,alpha=alpha,depth_sum=ds,depth_m=ds/np.maximum(alpha,1e-8)/sem.scale,group_alpha_T=native['group_alpha_T'].detach().cpu().numpy(),added_owner_rgb_T=native['added_owner_rgb_T'].detach().cpu().numpy(),newroad_rgb_T=native['newroad_rgb_T'].detach().cpu().numpy())
            np.savez_compressed(folder/(state+'_float.npz'),**values);Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(folder/(state+'.png'))
            cached=h2.cache[(raw,state=='delete')];np.savez_compressed(folder/(state+'_visibility_mask.npz'),selected_road_child_rows=cached['mask'].cpu().numpy(),model_foreground_guard=cached['guard'].cpu().numpy());write(folder/(state+'_visibility_record.json'),cached['record'])
            checks={}
            if raw in RAWS:
                with np.load(receipt.parent/f'frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
                    checks={k:bool(np.array_equal(v,z[k])) for k,v in values.items()};assert all(checks.values())
            row=dict(raw=raw,state=state,foreground_road_mass=native['meta']['public_foreground_road_alpha_mass'],four_frame_overlap_all_fields_byte_exact=checks)
            rows.append(row);write(out/'kernel_progress.json',dict(completed_public_forwards=len(rows),actual_total_native_forwards=h2.actual_forwards,actual_qualification_dummy_backwards=h2.actual_dummy_backwards,optimizer_updates=0,seconds=time.time()-start));print(json.dumps(row),flush=True);del native
        base._data.clear();torch.cuda.empty_cache()
    rt=h2.receipt();assert h2.actual_public_forwards==98
    write(out/'receipt.json',dict(status='COMPLETE_RELEASED_M49_NATIVE_RENDER_PENDING_FULL_GATES',actual_public_forwards=98,actual_qualification_forwards=h2.actual_forwards-98,actual_qualification_dummy_backwards=h2.actual_dummy_backwards,actual_total_native_forwards=h2.actual_forwards,optimizer_updates=0,original_model_updates=0,records=rows,runtime_receipt=rt,seconds=time.time()-start,quality_accepted=False,human_verdict=None,shutdown=False))

if __name__=='__main__':main()
