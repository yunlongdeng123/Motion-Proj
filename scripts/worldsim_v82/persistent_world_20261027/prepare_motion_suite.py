"""Input-only metric edit requests for each single-target native scene.

These are protocols, not completed edits. All requests share the same central31
camera frames and the same eventual30K checkpoint. Labels condition requested
poses / collision diagnostics only; they are never hidden-background RGB GT.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')

def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
def rz(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
def ry(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def apply(T,xyz):return np.asarray(xyz)@T[:3,:3].T+T[:3,3]
def calibration(p):return {l.split(':')[0]:np.fromstring(l.split(':')[1],sep=' ') for l in p.read_text().splitlines()}

def footprint(center,size,R):
    return np.array([[x,y,0] for x,y in [(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)]])*np.array([size[0],size[1],1.])@R.T+center

def overlap(a,b):
    # Separating-axis overlap, retaining boundary contact as a collision warning.
    for poly in (a,b):
        edges=np.roll(poly[:,:2],-1,axis=0)-poly[:,:2]
        for e in edges:
            axis=np.array([-e[1],e[0]])
            aa=a[:,:2]@axis;bb=b[:,:2]@axis
            if aa.max()<bb.min()-1e-8 or bb.max()<aa.min()-1e-8:return False
    return True

def prepare(context):
    source=ROOT/'motion_scene_inputs_v1'/context;dest=ROOT/'motion_edit_suite_v1'/context
    dest.mkdir(parents=True,exist_ok=False)
    sensor=read(source/'sensor_preparation.json');selection=read(source/'native_preparation/selection_reviewed.json')
    assert selection['status']=='agent_visually_reviewed' and len(sensor['frames'])==49
    original=read(ROOT/'evaluation_set_v1/scenes'/context/'scene.json');raws=original['raw_annotations'];track=selection['track']
    frames=sensor['frames'];stamps=np.arange(49)*.02-.48;selected=np.arange(9,40);physical=(np.array([x['timestamp_micros'] for x in frames])-frames[0]['timestamp_micros'])/1e6
    assert len(selected)==31 and np.all(np.diff(physical)>0)
    pose0=np.loadtxt(source/'pose'/(frames[0]['frame_id']+'.txt'));world_to_anchor=np.linalg.inv(pose0)
    objects=[next(x for x in raws[r['raw']]['front_vehicles'] if x['track']==track) for r in frames]
    centers_world=np.array([x['center_world'] for x in objects]);centers_anchor=apply(world_to_anchor,centers_world)
    sizes=np.median(np.array([x['size_m'] for x in objects]),axis=0)
    specs=[
        dict(id='no_edit',family='no_edit'),dict(id='delete_target',family='delete'),
        dict(id='translate_lateral_1m',family='translate',lateral_m=1.),
        dict(id='insert_reused_lateral_2m',family='insert_reused',lateral_m=2.,asset_source='same checkpoint target, not external novel asset'),
        dict(id='insert_reused_lateral_35m',family='insert_reused',lateral_m=3.5,asset_source='same checkpoint target, not external novel asset'),
        dict(id='speed_075',family='speed',time_scale=.75),dict(id='speed_125',family='speed',time_scale=1.25),
        dict(id='turn_left15',family='turn',final_yaw_deg=15.),dict(id='turn_right15',family='turn',final_yaw_deg=-15.),
        dict(id='camera_lateral_05m',family='camera_lateral',camera_lateral_m=.5),
        dict(id='camera_yaw3',family='camera_yaw',camera_yaw_deg=3.)]
    prepared=[]
    for spec in specs:
        family=spec['family'];query=stamps[selected]*spec.get('time_scale',1.)
        assert query.min()>=stamps[0] and query.max()<=stamps[-1], 'No clamping or unsupported extrapolation'
        center_q=np.column_stack([np.interp(query,stamps,centers_anchor[:,i]) for i in range(3)])
        target=center_q.copy();angles=np.deg2rad(spec.get('final_yaw_deg',0.))*np.linspace(0,1,31)
        if family in ('translate','insert_reused'):target[:,1]+=spec['lateral_m']
        if family=='turn':
            target[0]=center_q[0]
            for i in range(1,31):target[i]=target[i-1]+rz((angles[i]+angles[i-1])/2)@(center_q[i]-center_q[i-1])
        rows=[]
        for i,index in enumerate(selected):
            frame=frames[index];name=frame['frame_id'];pose=np.loadtxt(source/'pose'/(name+'.txt'))
            labels=read(source/'annotations'/(name+'.json'))['laser_labels'];target_label=next(x for x in labels if x['id']==track)
            b=target_label['box'];center_label=apply(pose,np.array([[b['center_x'],b['center_y'],b['center_z']]]))[0]
            assert np.linalg.norm(center_label-centers_world[index])<1e-4,'Track-to-sensor world mapping mismatch'
            base_world=apply(pose0,center_q[i:i+1])[0];edited_world=apply(pose0,target[i:i+1])[0]
            Rworld=pose0[:3,:3]@rz(angles[i])@pose0[:3,:3].T
            assert np.allclose(Rworld.T@Rworld,np.eye(3),atol=1e-9) and abs(np.linalg.det(Rworld)-1)<1e-9
            translation=edited_world-Rworld@base_world
            assert np.linalg.norm(Rworld@base_world+translation-edited_world)<1e-8
            # Native centers and annotated boxes are separate: this is only a
            # request-space feasibility warning, not rendered-asset correctness.
            vehicle_to_anchor=world_to_anchor@pose;heading=float(target_label['box']['heading'])
            target_axes=vehicle_to_anchor[:3,:3]@rz(heading)
            poly=footprint(target[i],sizes,rz(angles[i])@target_axes)
            collisions=[]
            if family=='insert_reused':
                original_poly=footprint(centers_anchor[index],sizes,target_axes)
                if overlap(poly,original_poly):collisions.append(track)
            for other in labels:
                if other['id']==track or other.get('type') not in ('TYPE_VEHICLE',1):continue
                ob=other['box'];cc=apply(vehicle_to_anchor,np.array([[ob['center_x'],ob['center_y'],ob['center_z']]]))[0]
                if abs(cc[2]-target[i,2])>(float(ob['height'])+sizes[2])/2:continue
                axes=vehicle_to_anchor[:3,:3]@rz(float(ob['heading']))
                other_poly=footprint(cc,[ob['length'],ob['width'],ob['height']],axes)
                if overlap(poly,other_poly):collisions.append(other['id'])
            calib=calibration(source/'calib'/(name+'.txt'));vehicle_to_cam=np.eye(4);vehicle_to_cam[:3]=calib['Tr_velo_to_cam_0'].reshape(3,4)
            c2w=pose@np.linalg.inv(vehicle_to_cam)
            c2w[:3,3]+=c2w[:3,0]*spec.get('camera_lateral_m',0.)
            c2w[:3,:3]=c2w[:3,:3]@ry(np.deg2rad(spec.get('camera_yaw_deg',0.)))
            row=dict(frame_id=name,raw=frame['raw'],camera_index=int(index),camera_model_time=float(stamps[index]),dynamic_query_model_time=float(query[i]),dynamic_query_physical_seconds=float(np.interp(query[i],stamps,physical)),supported_time=True,target_original_center_world=base_world.tolist(),target_requested_center_world=edited_world.tolist(),target_rotation_delta_world=Rworld.tolist(),target_affine_translation_world=translation.tolist(),requested_yaw_delta_deg=float(np.rad2deg(angles[i])),camera_c2w_world=c2w.tolist(),request_box_collision_tracks=collisions)
            rows.append(row)
        if family=='no_edit':
            assert all(np.allclose(x['target_rotation_delta_world'],np.eye(3)) and np.linalg.norm(x['target_affine_translation_world'])<1e-8 for x in rows)
        collision_frames=sum(bool(x['request_box_collision_tracks']) for x in rows)
        spec.update(frames=rows,status='input_request_prepared_not_rendered',new_training_updates=0,checkpoint='formal30k/chkpnt30000.pth',full_target_and_camera_space_QA='pending_native_geometry_and_render',collision_diagnostic_frames=collision_frames,request_stratum='annotated_box_collision_stress' if collision_frames else 'no_annotated_box_collision_other_geometry_pending',background_truth='unknown under real occlusions; no counterfactual RGB GT',generation_policy='only disoccluded old target support for object edits; new object pixels protected; camera completion mask remains pending')
        write(dest/(spec['id']+'.json'),spec);prepared.append({k:v for k,v in spec.items() if k!='frames'})
    sequence=dict(id='continuous_revision_revisit',family='continuous_combination',status='request_sequence_prepared_not_executed',same_checkpoint=True,
        requests=[dict(revision='delete_original',parent='native',operation='delete',target=track),dict(revision='insert_reused',parent='delete_original',operation='insert_reused',new_instance='reused_asset_01',lateral_m=2.),dict(revision='move_inserted',parent='insert_reused',operation='translate',target='reused_asset_01',absolute_lateral_m=1.),dict(revision='turn_inserted',parent='move_inserted',operation='turn',target='reused_asset_01',final_yaw_deg=15.),dict(revision='camera_revisit',parent='turn_inserted',operation='camera_lateral',camera_lateral_m=.5),dict(revision='revisit_original_camera',parent='turn_inserted',operation='camera_native')],
        distinctions=['independent requests versus retained revision state must both be rendered','cache/reload equality is engineering only, not memory benefit','one original actor plus reused clone, not two independent learned actor assets'])
    write(dest/'continuous_revision_revisit.json',sequence)
    summary=dict(context=context,track=track,source_sensor=str(source),single_physical_dynamic_actor=True,original_raw_window=[frames[0]['raw'],frames[-1]['raw']],common_camera_indices=selected.tolist(),common_raw_frames=[frames[i]['raw'] for i in selected],physical_time_span_seconds=float(physical[selected[-1]]-physical[selected[0]]),normalization='model_time index*0.02-.48; actual microsecond map retained separately',world_frame='native source world meters; first-vehicle frame for planar edit directions',prepared_single_requests=prepared,continuous_sequence=sequence,completed_edit_cases=0,formal_benchmark_admitted=False,labels_used_for_request_and_collision_only=True,hidden_background_GT_available=False,quality_accepted=False,human_verdict=None)
    write(dest/'summary.json',summary)
    return dict(context=context,single_request_specs=len(prepared),sequence_request_specs=1,completed_edit_cases=0,collisions={x['id']:x['collision_diagnostic_frames'] for x in prepared})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',action='append');a=p.parse_args()
    contexts=a.context or sorted(p.name for p in (ROOT/'motion_scene_inputs_v1').iterdir() if p.is_dir())
    (ROOT/'motion_edit_suite_v1').mkdir(exist_ok=True)
    results=[prepare(context) for context in contexts]
    write(ROOT/'motion_edit_suite_v1/status.json',dict(phase='input_only_motion_suite_prepared',scenes=results,completed_edit_cases=0))
    print(json.dumps(results,indent=2))
