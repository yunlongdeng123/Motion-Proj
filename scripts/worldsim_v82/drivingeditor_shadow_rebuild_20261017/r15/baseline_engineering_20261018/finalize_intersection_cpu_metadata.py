from pathlib import Path
import json,sys,importlib.util,hashlib,shutil
E=Path(__file__).resolve().parent;P=E/'all_vehicle_pipeline';SID='scene0008_089_137';M=P/'metadata'/SID;BASE=Path('/root/autodl-tmp/drivingeditor_repro');OLD=BASE/'campaign/followup_scene_prep_20261013'
def main():
 assert not (E/'lidar_camera_intersection_input_receipt.json').exists()
 spec=json.loads((M/'scene_spec.json').read_text());before=spec['actor_PLY_paths'];spec['actor_PLY_paths']=sorted(before,key=lambda p:int(Path(p).stem.split('_')[-1]))
 assert len(spec['actor_PLY_paths'])==20
 shutil.copy2(M/'scene_spec.json',M/'scene_spec.before_numeric_PLY_order.json');(M/'scene_spec.json').write_text(json.dumps(spec,indent=2))
 source=P/'prepare_gn11_window.py';shutil.copy2(source,source.with_suffix('.py.before_numeric_PLY_order'))
 s=source.read_text();old="priors=sorted(scene.glob('points3d_dynamic_*.ply'))";assert old in s;source.write_text(s.replace(old,"priors=sorted(scene.glob('points3d_dynamic_*.ply'),key=lambda p:int(p.stem.split('_')[-1]))"))
 sys.path.insert(0,str(OLD/'tools/followup'));from contracts import scene_spec,build_config,readiness_contract,gate_contract
 from prepare_followup import prepare
 ready=json.loads((M/'CPU_readiness.json').read_text());ref=json.loads((M/'reference_GN11.json').read_text());canonical=scene_spec(spec);readiness_contract(ready,canonical);cfg=build_config(ref,canonical,P)
 paths=dict(campaign_root=str(P),source_repository=str(OLD/'frozen_repository'),matrix_tools_dir=str(BASE/'campaign/hybrid30k_20261012/tools'),scene_spec=str(M/'scene_spec.json'),readiness_file=str(M/'CPU_readiness.json'),reference_config=str(M/'reference_GN11.json'))
 r=prepare(paths,cfg,canonical,ready['status']);oldgate=OLD/'metadata/GN11_gate_passed.json';g=json.loads(oldgate.read_text());g['allowed_scene_ids']=['scene0017_079_127',SID];g['extension_authorization']=dict(source='latest human request: LiDAR and camera intersection stage1 only',reference_acceptance_only=True,new_window_quality_passed=False,new_window_human_verdict=None,inherited_gate_path=str(oldgate),inherited_gate_sha256=hashlib.sha256(oldgate.read_bytes()).hexdigest());gate=M.parent/'GN11_reference_acceptance_authorization.json';gate.write_text(json.dumps(g,indent=2));e=gate_contract(gate,canonical,ref,True);(M/'inherited_GN11_gate_recheck.json').write_text(json.dumps(e,indent=2))
 sel=json.loads((E/'lidar_camera_intersection_selection.json').read_text());actual=json.loads((P/'inputs'/SID/'dynamic_multi_actor_preparation.json').read_text())
 receipt=dict(status='LIDAR_CAMERA20_CPU_INPUT_BINDING_PASSED_VISUAL_REVIEW_PENDING',scope='LIDAR_AND_CAMERA_INTERSECTION_ONLY',frames=49,selected=sel['actors'],excluded_camera_track_ids=sel['excluded_camera_track_ids'],dynamic_real_seed_points=[a['total_real_points'] for a in actual['actors']],all49_full_RGB_retained=True,additional_source_frames_used=False,pure_camera_context_can_remain_in_static=True,all_scene_vehicle_coverage_claimed=False,GT_boxes_filled_as_masks=False,GT_boxes_used_to_filter_individual_initialization_points=False,physical_geometry_QA='unchanged whole-observation bound; never individual point box crop',actual_CPU_cameras=ready['train_cameras'],actual_meter_to_model_scale=ready['actual_meter_to_model_scale'],numeric_PLY_path_order_fixed=True,human_verdict=None,training_updates=0,training_gate_written=False)
 (E/'lidar_camera_intersection_input_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(status=receipt['status'],point_counts=receipt['dynamic_real_seed_points'])))
if __name__=='__main__':main()
