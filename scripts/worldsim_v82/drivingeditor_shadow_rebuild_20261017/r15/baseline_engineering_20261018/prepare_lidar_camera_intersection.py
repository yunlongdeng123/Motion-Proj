"""Separate GN11 inputs for the user's LiDAR/camera intersection scope.

No training gate is written here. Pure-camera vehicles remain outside the edit
target set, with full RGB context retained. Original input data stays read-only.
"""
from pathlib import Path
import os,sys,json,shutil,subprocess
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from PIL import Image
E=Path(__file__).resolve().parent;ROOT=E.parent;NEW=E/'all_vehicle_pipeline';SID='scene0008_089_137';C=E/'all_vehicle_camera33_candidates'
def main():
 assert not NEW.exists(),'preserve prepared data; do not blindly restart'
 selection=json.loads((E/'lidar_camera_intersection_selection.json').read_text());actors=selection['actors'];assert len(actors)==20 and len({a['raw_camera_track_id'] for a in actors})==20
 NEW.mkdir();src=ROOT/'inputs'/SID;dest=NEW/'inputs'/SID;dest.mkdir(parents=True)
 for name in ['image','calib','pose','velodyne','sky','timestamp','annotations','mask','points3d.ply']:
  os.symlink(src/name,dest/name,target_is_directory=(src/name).is_dir())
 prep=json.loads((src/'preparation_manifest.json').read_text());prep['selected_real_track_ids']=[a['raw_laser_track_id'] for a in actors]
 prep['editable_actor_scope']='LiDAR and FRONT camera jointly observed in original49-frame window'
 prep['selected_camera_track_ids']=[a['raw_camera_track_id'] for a in actors]
 prep['excluded_camera_track_ids']=selection['excluded_camera_track_ids'];prep['original_source_scene_read_only']=str(src)
 prep['camera_box_prompts_used_for_some_native_instance_masks']=True;prep['camera_rectangles_filled_as_body_masks']=False
 prep['additional_observation_frames_used']=False;prep['all_scene_vehicle_coverage_claimed']=False
 (dest/'preparation_manifest.json').write_text(json.dumps(prep,indent=2));(dest/'intersection_labels').mkdir();overlaps=[]
 for frame in prep['frames']:
  fid=frame['id'];ann={a['id']:a for a in json.loads((src/'annotations'/(fid+'.json')).read_text())['laser_labels']}
  cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (src/'calib'/(fid+'.txt')).read_text().splitlines() if ':' in s};tr=cal['Tr_velo_to_cam_0'].reshape(3,4)
  order=[]
  for i,a in enumerate(actors):
   b=ann.get(a['raw_laser_track_id'],{}).get('box');depth=float((tr@np.array([b['center_x'],b['center_y'],b['center_z'],1]))[2]) if b else float('inf');order.append((depth,i,a))
  labels=np.zeros((1280,1920),np.uint8);union=np.zeros_like(labels,bool);rows=[]
  for depth,i,a in sorted(order,key=lambda x:x[0]):
   p=C/f"actors/actor_{a['camera_actor_index']}/body_masks"/(fid+'.png')
   body=np.array(Image.open(p))>0 if p.exists() else np.zeros_like(union)
   overlap=body&union;visible=body&~union;union|=body;labels[visible]=i+1
   rows.append(dict(actor=a['R15_actor'],camera_actor_index=a['camera_actor_index'],original_pixels=int(body.sum()),visible_assigned_pixels=int(visible.sum()),overlap_resolved_pixels=int(overlap.sum()),missing_camera_annotation=not p.exists(),raw3D_annotation_missing=a['raw_laser_track_id'] not in ann,depth_m=depth if np.isfinite(depth) else None))
  Image.fromarray(labels).save(dest/'intersection_labels'/(fid+'.png'));overlaps.append(dict(frame_id=fid,actors=rows,union_pixels=int(union.sum())))
 (NEW/'reviewed_selections').mkdir()
 sel=dict(status='prepared_lidar_camera_intersection_pending_visual_review',GT_boxes_used_to_prompt_native_instance_masks=True,GT_boxes_filled_as_masks=False,GT_boxes_used_to_create_masks=True,scope='user-selected LiDAR and camera intersection; pure-camera context excluded',human_verdict=None,actors=[dict(raw_track_id=a['raw_laser_track_id'],raw_camera_track_id=a['raw_camera_track_id'],R15_actor=a['R15_actor'],description=f"a{a['R15_actor']} / camera c{a['camera_actor_index']}",frames=[dict(frame_id=f['id'],training_slot=i) for f in prep['frames']]) for i,a in enumerate(actors)])
 (NEW/'reviewed_selections'/(SID+'.json')).write_text(json.dumps(sel,indent=2));(NEW/'mask_overlap_resolution.json').write_text(json.dumps(dict(policy='resolve only overlapping actual model masks front-to-back by measured3D center depth; preserve union; no GT-box clipping',frames=overlaps),indent=2))
 shutil.copytree(E/'full5_pipeline/infra',NEW/'infra',ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(ROOT/'waymo_numpy_cpu.py',NEW/'waymo_numpy_cpu.py')
 builder=(ROOT/'build_selected_priors.py').read_text()
 builder=builder.replace('from data import _actual_body_from_selection',"""def _actual_body_from_selection(mask_scene, frame_id, labels, row):
    return np.asarray(Image.open(mask_scene/'intersection_labels'/(frame_id+'.png')))==row['training_slot']+1""")
 builder=builder.replace("sel['status']=='visually_reviewed' and sel['GT_boxes_used_to_create_masks'] is False","sel['status']=='prepared_lidar_camera_intersection_pending_visual_review' and sel['GT_boxes_filled_as_masks'] is False")
 old="gt=next(x for x in ann['laser_labels'] if x['id']==a['raw_track_id']);b=gt['box'];center=(pose@np.array([b['center_x'],b['center_y'],b['center_z'],1]))[:3]"
 new="""gt=next((x for x in ann['laser_labels'] if x['id']==a['raw_track_id']),None)
        if gt is None:
            # No synthetic trajectory or point crop. Keep the real image/mask,
            # reject this observation's entire initialization prior.
            retained=retained[:0];world=world[:0];clouds[i][-1]=world;times[i][-1]=times[i][-1][:0];source[i][fid]=retained.astype(np.int32)
            rejection='entire_frame_prior_rejected_missing_raw3D_annotation';b=dict(center_x=0,center_y=0,center_z=0,length=0,width=0,height=0)
        else:b=gt['box']
        center=(pose@np.array([b['center_x'],b['center_y'],b['center_z'],1]))[:3]"""
 assert builder.count(old)==1;builder=builder.replace(old,new)
 (NEW/'build_selected_priors.py').write_text(builder);shutil.copy2(ROOT/'prepare_gn11_window.py',NEW/'prepare_gn11_window.py')
 subprocess.run([sys.executable,'-B',str(NEW/'build_selected_priors.py'),SID,'--reject-unreliable-frame-priors'],check=True)
 actual=json.loads((dest/'dynamic_multi_actor_preparation.json').read_text());assert len(actual['actors'])==20 and all(a['total_real_points']>0 for a in actual['actors'])
 subprocess.run([sys.executable,'-B',str(NEW/'prepare_gn11_window.py'),SID],check=True)
 receipt=dict(status='LIDAR_CAMERA20_INPUTS_PREPARED_CPU_BINDING_PASSED_VISUAL_REVIEW_PENDING',scope='LIDAR_AND_CAMERA_INTERSECTION_ONLY',frames=49,selected=actors,excluded_camera_track_ids=selection['excluded_camera_track_ids'],dynamic_real_seed_points=[a['total_real_points'] for a in actual['actors']],all49_full_RGB_retained=True,additional_source_frames_used=False,pure_camera_context_can_remain_in_static=True,GT_boxes_filled_as_masks=False,GT_boxes_used_to_filter_individual_initialization_points=False,native_mask_overlap_policy='measured3D center depth only resolves overlapping prediction pixels; union unchanged',human_verdict=None,training_updates=0,training_gate_written=False)
 (E/'lidar_camera_intersection_input_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(status=receipt['status'],points=receipt['dynamic_real_seed_points'])))
if __name__=='__main__':main()
