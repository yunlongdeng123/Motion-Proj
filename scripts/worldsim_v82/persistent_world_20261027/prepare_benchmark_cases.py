"""Materialize eight task families from real sensor windows; no edit score claims.

Case = one scene/target/physical request, not one frame, random seed or backend.
Frozen reserved cases are not used for method choice. Source boxes condition
requests and flag collisions, never replace semantic masks or background GT.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp
import screen_eval_inputs as qa
from prepare_motion_suite import rz,apply,footprint,overlap
from resource_control import require_start
ROOT=qa.ROOT;OUT=ROOT/'benchmark_cases_v1'
FAMILIES=['ego_speed','ego_lane_change','ego_direction','vehicle_speed','vehicle_lane_change','vehicle_direction','vehicle_insertion','vehicle_removal']
BANK=['14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000','473735159277431842_630_095_650_095']

def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)
def interp(times,values,query):return np.column_stack([np.interp(query,times,np.array(values)[:,i]) for i in range(np.array(values).shape[1])])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--context',required=True);a=parser.parse_args();require_start()
    source=ROOT/'motion_scene_inputs_v1'/a.context;sensor=read(source/'sensor_preparation.json')
    dest=OUT/'scenes'/a.context;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():return
    original=read(ROOT/'evaluation_set_v1/scenes'/a.context/'scene.json');scene=dict(original)
    selection=source/'native_preparation/selection_reviewed.json'
    track=read(selection)['track'] if selection.exists() else sensor['target_track']
    scene.update(selected_window=[f['raw'] for f in sensor['frames']],selected_track=track)
    write(dest/'scene.json',scene)
    records=[];pose_rows=[];cameras=[];images=[];labels=[]
    for row in sensor['frames']:
        fid=row['frame_id'];raw=row['raw'];pose=np.loadtxt(source/'pose'/(fid+'.txt'));pose_rows.append(pose)
        calibration=row.get('front_camera_calibration')
        if calibration:
            ext=np.array(calibration['extrinsic']).reshape(4,4);intr=np.array(calibration['intrinsic']);K=np.array([[intr[0]/4,0,intr[2]/4],[0,intr[1]/4,intr[3]/4],[0,0,1.]]);dist=intr[4:9]
            optical=np.eye(4);optical[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]];camera=pose@ext@np.linalg.inv(optical)
        else:
            old=read(ROOT/'evaluation_admission_v2/sensors'/(a.context+'.json'));previous=next(x for x in old['frames'] if x['raw']==raw)
            ext=np.array(previous['extrinsic']);K=np.array(previous['K']);dist=previous['distortion'];camera=np.array(previous['camera_world_from_optical'])
        cameras.append(camera);lab=read(source/'annotations'/(fid+'.json'))['laser_labels'];labels.append(lab)
        target=next((x for x in lab if x['id']==track),None);heading=None
        if target:
            h=float(target['box']['heading']);axis=pose[:3,:3]@np.array([np.cos(h),np.sin(h),0]);heading=float(np.arctan2(axis[1],axis[0]))
        records.append(dict(raw=raw,timestamp_micros=row['timestamp_micros'],pose=pose.tolist(),extrinsic=ext.tolist(),camera_world_from_optical=camera.tolist(),K=K.tolist(),distortion=list(dist),world_heading=heading,image_width=1920,image_height=1280))
        im=Image.open(source/'image/image_0'/(fid+'.png')).convert('RGB').resize((480,320),Image.Resampling.LANCZOS);images.append(im)
        (dest/'gt_front').mkdir(exist_ok=True);im.save(dest/'gt_front'/f'{raw:03d}.png')
    assert len(records)==49
    qa.DATA=OUT;qa.OUT=ROOT/'benchmark_window_admission_v1'
    write(qa.OUT/'sensors'/(a.context+'.json'),dict(context=a.context,frames=records,selected_record_crc_verified=True,source_sensor_receipt=str(source/'sensor_preparation.json')))
    screen=qa.quality_worker(scene);write(dest/'input_quality.json',screen)
    if a.context.startswith('3363533094480067586_'):
        screen['review_flags']=sorted(set(screen['review_flags']+['visually_confirmed_wet_lens_robustness_stratum']));screen['raw_input_status']='review_required';write(dest/'input_quality.json',screen)
    model=np.arange(49)*.02-.48;ids=np.arange(9,40);qtime=model[ids];physical=(np.array([x['timestamp_micros'] for x in sensor['frames']])-sensor['frames'][0]['timestamp_micros'])/1e6
    assert np.all(np.diff(physical)>0)
    poses=np.array(pose_rows);cameras=np.array(cameras);anchor=poses[0];inv=np.linalg.inv(anchor)
    origin=np.array([x[:3,3] for x in cameras]);ego=apply(inv,origin)
    target_rows=[next((x for x in original['raw_annotations'][r['raw']]['front_vehicles'] if x['track']==track and x.get('center_world') is not None),None) for r in sensor['frames']]
    complete=all(x is not None for x in target_rows);target=np.array([x['center_world'] for x in target_rows]) if complete else None
    sizes=np.median([x['size_m'] for x in target_rows if x is not None],axis=0) if any(x is not None for x in target_rows) else np.array([4.8,2.,1.8])
    parity=int(hashlib.sha256(a.context.encode()).hexdigest()[:2],16)%2;side=1 if parity else -1;factor=.75 if parity else 1.25
    prepared=[];story=[]
    for family in FAMILIES:
        query=qtime*factor if family.endswith('_speed') else qtime.copy();assert query.min()>=model[0] and query.max()<=model[-1]
        requested_camera=cameras[ids].copy();delta_R=np.repeat(np.eye(3)[None],31,axis=0);translation=np.zeros((31,3));requested_target=None
        u=np.linspace(0,1,31);smooth=u*u*(3-2*u);yaw=np.deg2rad(side*10)*smooth
        if family=='ego_speed':
            requested_camera[:,:3,3]=interp(model,origin,query);requested_camera[:,:3,:3]=Slerp(model,Rotation.from_matrix(cameras[:,:3,:3]))(query).as_matrix()
        if family=='ego_lane_change':requested_camera[:,:3,3]+=(anchor[:3,1][None]*side*3.*smooth[:,None])
        if family=='ego_direction':
            edited=ego[ids].copy()
            for j in range(1,31):edited[j]=edited[j-1]+rz((yaw[j]+yaw[j-1])/2)@(ego[ids[j]]-ego[ids[j-1]])
            requested_camera[:,:3,3]=apply(anchor,edited)
            for j in range(31):requested_camera[j,:3,:3]=anchor[:3,:3]@rz(yaw[j])@anchor[:3,:3].T@requested_camera[j,:3,:3]
        if family.startswith('vehicle_') and complete:
            center=interp(model,target,query);edited=apply(inv,center);original_center=edited.copy()
            if family=='vehicle_lane_change':edited[:,1]+=side*3.*smooth
            if family=='vehicle_direction':
                yaw=np.deg2rad(side*15)*smooth
                for j in range(1,31):edited[j]=edited[j-1]+rz((yaw[j]+yaw[j-1])/2)@(original_center[j]-original_center[j-1])
            if family=='vehicle_insertion':edited[:,1]+=side*4.
            requested_target=apply(anchor,edited)
            if family=='vehicle_direction':
                for j in range(31):delta_R[j]=anchor[:3,:3]@rz(yaw[j])@anchor[:3,:3].T
            translation=requested_target-np.einsum('nij,nj->ni',delta_R,center)
        eligible=screen['raw_input_status']=='raw_checks_passed';reasons=[]
        if family.startswith('ego_') and np.linalg.norm(np.diff(origin[ids],axis=0),axis=1).sum()<1:eligible=False;reasons.append('insufficient_ego_motion_for_trajectory_edit')
        if family.startswith('vehicle_') and not complete:eligible=False;reasons.append('incomplete_target_3D_track_window')
        if family in ('vehicle_speed','vehicle_lane_change','vehicle_direction') and not screen['operations']['change_speed']['raw_data_eligible']:eligible=False;reasons.append('stationary_or_unreliable_target_not_a_moving_trajectory_case')
        if family in ('vehicle_lane_change','vehicle_direction') and not screen['operations']['translate_vehicle']['raw_data_eligible']:eligible=False;reasons.append('target_motion_input_not_eligible')
        if screen['raw_input_status']!='raw_checks_passed':reasons+=screen['review_flags'] or ['input_structure_review']
        rows=[];collisions=0
        for j,index in enumerate(ids):
            current=[]
            if requested_target is not None and family not in ('vehicle_removal','vehicle_speed'):
                obj=next(x for x in labels[index] if x['id']==track);axes=poses[index][:3,:3]@rz(float(obj['box']['heading']));poly=footprint(requested_target[j],sizes,delta_R[j]@axes)
                for other in labels[index]:
                    if other.get('type') not in ('TYPE_VEHICLE',1):continue
                    if other['id']==track and family!='vehicle_insertion':continue
                    b=other['box'];cc=apply(poses[index],np.array([[b['center_x'],b['center_y'],b['center_z']]]))[0]
                    if abs(cc[2]-requested_target[j,2])>(float(b['height'])+sizes[2])/2:continue
                    if overlap(poly,footprint(cc,[b['length'],b['width'],b['height']],poses[index][:3,:3]@rz(float(b['heading'])))):current.append(other['id'])
            collisions+=bool(current)
            rows.append(dict(frame_id=sensor['frames'][index]['frame_id'],raw=sensor['frames'][index]['raw'],camera_index=int(index),model_time=float(model[index]),physical_seconds=float(physical[index]),query_model_time=float(query[j]),query_physical_seconds=float(np.interp(query[j],model,physical)),camera_c2w_world=requested_camera[j].tolist(),target_rotation_delta_world=delta_R[j].tolist(),target_affine_translation_world=translation[j].tolist(),target_requested_center_world=requested_target[j].tolist() if requested_target is not None else None,request_box_collision_tracks=current))
        asset=next(k for k in BANK if k!=a.context) if family=='vehicle_insertion' else None
        request=dict(case_id=a.context+'__'+family,context=a.context,split=original['split'],family=family,target_track=track,frames=rows,physical_unit='meter',speed_factor=factor if family.endswith('_speed') else None,side=side,source_asset_context=asset,asset_kind='different_scene_predeclared_development_asset_bank' if asset else None,source_asset_alignment_QA='pending' if asset else None,input_screen_eligible=bool(eligible),input_screen_reasons=reasons,collision_frames=collisions,request_stratum='collision_stress' if collisions else 'space_support_pending',formal_ready=False,completed=False,Gaussian_base='same scene once, all cases reuse',pending=['semantic_instance_binding','native_asset_background_QA','measured_request_space_support','unknown_reveal_domains','native_and_fixed_backend_execution'],counterfactual_RGB_GT=False,uses_edit_outputs_for_selection=False)
        write(dest/'requests'/(family+'.json'),request);prepared.append({k:request[k] for k in ['case_id','family','input_screen_eligible','input_screen_reasons','collision_frames','formal_ready','completed']})
        tile=Image.new('RGB',(480,350),'#122231');tile.paste(images[24],(0,30));ImageDraw.Draw(tile).text((8,8),family+': '+('input screened' if eligible else 'input review'),fill='white');story.append(tile)
    sheet=Image.new('RGB',(1920,700),'#122231')
    for j,tile in enumerate(story):sheet.paste(tile,((j%4)*480,(j//4)*350))
    sheet.save(dest/'case_inputs.jpg',quality=90)
    result=dict(context=a.context,split=original['split'],frames=49,trajectory_frames=31,real_sensor_inputs=str(source),actual_source_points=sum(x['lidar_points'] for x in sensor['frames']),families=prepared,input_screened_cases=sum(x['input_screen_eligible'] for x in prepared),Gaussian_cases_executed=0,formal_cases_admitted=0,specifications_only=False,description='49 real source RGB/LiDAR/calibration/pose frames and concrete physical requests, not edited outputs',quality_accepted=False,human_verdict=None)
    write(dest/'summary.json',result);print(json.dumps(result))

if __name__=='__main__':main()
