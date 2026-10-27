"""Evaluate only speed-query cameras exactly matching an existing input pose.

Arbitrary novel lane/turn cameras have no RGB GT here. Foreground exclusion is
an inferred DEVA proxy, not dense static semantics or counterfactual object GT.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation
from benchmark_geometry import ROOT,model_camera,sensor_camera

def read(path):return json.loads(path.read_text())

def measured_ground_domain(scene,run,source,native_camera,output_source):
    # Reuse the *actual* baseline SGCM fit; it classifies measured points only.
    # The plane itself is not a dense road target and generates no new geometry.
    cache=scene/'native_preparation/sgcm_geometry_from_native_log.json'
    if cache.exists():metadata=read(cache)['metadata']
    else:
        log=run.parent/'formal30k.log';metadata=None
        with log.open(errors='replace') as stream:
            for line in stream:
                if 'SGCM_INIT ' in line:
                    metadata=json.JSONDecoder().raw_decode(line.split('SGCM_INIT ',1)[1])[0];break
        assert metadata is not None
        cache.write_text(json.dumps(dict(source=str(log),metadata=metadata,native_fit_not_a_GT_surface=True),indent=2))
    raw=np.fromfile(scene/'velodyne'/(source['frame_id']+'.bin'),dtype=np.float32).reshape(-1,6)[:,:3].astype(np.float64)
    pose=np.loadtxt(scene/'pose'/(source['frame_id']+'.txt'));anchor=np.loadtxt(metadata['anchor_pose'])
    T=np.linalg.inv(anchor)@pose;local=raw@T[:3,:3].T+T[:3,3]
    plane=np.asarray(metadata['ground_fit']['plane_anchor']);near=np.abs(local@plane[:3]+plane[3])<=.15
    region=np.zeros(len(raw),bool)
    for bounds in metadata['regions_anchor']:
        bounds=np.asarray(bounds);region|=((local[:,:2]>=bounds[0,:2])&(local[:,:2]<=bounds[1,:2])).all(1)
    near&=region
    camera_world=sensor_camera(scene,source,scene.name);T=np.linalg.inv(camera_world)@pose
    cam=raw@T[:3,:3].T+T[:3,3]
    K=np.array([[native_camera['fx']/4,0,native_camera['cx']/4],[0,native_camera['fy']/4,native_camera['cy']/4],[0,0,1.]])
    uv=(cam@K.T)[:,:2]/np.maximum(cam[:,2:],1e-9)-.5;ij=np.floor(uv+.5).astype(np.int64)
    inside=(cam[:,2]>.01)&(ij[:,0]>=0)&(ij[:,0]<480)&(ij[:,1]>=0)&(ij[:,1]<320)
    nearest=np.full((320,480),np.inf);np.minimum.at(nearest,(ij[inside,1],ij[inside,0]),cam[inside,2])
    selected=np.flatnonzero(near&inside);xy=ij[selected]
    visible=np.abs(cam[selected,2]-nearest[xy[:,1],xy[:,0]])<=.3
    selected=selected[visible];xy=ij[selected];mask=np.zeros((320,480),bool);mask[xy[:,1],xy[:,0]]=True
    source_time_pixels=int(mask.sum())
    # Exclude confirmed occlusion at the fixed world timestamp, and require an
    # actual near-depth return there too. Unmeasured rays remain unscored.
    current_raw=np.fromfile(scene/'velodyne'/(output_source['frame_id']+'.bin'),dtype=np.float32).reshape(-1,6)[:,:3].astype(float)
    current_pose=np.loadtxt(scene/'pose'/(output_source['frame_id']+'.txt'));current_T=np.linalg.inv(camera_world)@current_pose
    current_cam=current_raw@current_T[:3,:3].T+current_T[:3,3]
    current_uv=(current_cam@K.T)[:,:2]/np.maximum(current_cam[:,2:],1e-9)-.5;current_ij=np.floor(current_uv+.5).astype(np.int64)
    current_inside=(current_cam[:,2]>.01)&(current_ij[:,0]>=0)&(current_ij[:,0]<480)&(current_ij[:,1]>=0)&(current_ij[:,1]<320)
    current_nearest=np.full((320,480),np.inf);np.minimum.at(current_nearest,(current_ij[current_inside,1],current_ij[current_inside,0]),current_cam[current_inside,2])
    comparable=np.isfinite(nearest)&np.isfinite(current_nearest)
    matches=np.zeros((320,480),bool);matches[comparable]=np.abs(nearest[comparable]-current_nearest[comparable])<=.3
    mask&=matches
    return mask,dict(measured_near_ground_points=int(near.sum()),visible_ground_returns=len(selected),source_time_visible_ground_pixels=source_time_pixels,dual_time_depth_supported_ground_pixels=int(mask.sum()),ground_plane_selection_tolerance_m=.15,nearest_same_frame_depth_tolerance_m=.3,current_time_projected_nearest_depth_tolerance_m=.3,fit_source=str(cache),semantic_ground_GT=False,current_view_visibility_is_sparse_LiDAR_proxy_not_camera_GT=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);p.add_argument('--fixed',type=Path,required=True);p.add_argument('--global-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-domain',choices=['all_DEVA_exclusion','visible_measured_ground'],default='visible_measured_ground');a=p.parse_args()
    assert read(a.fixed/'summary.json')['ego_speed_clock_policy']=='fixed_world'
    assert read(a.global_reference/'summary.json')['ego_speed_clock_policy']=='global_retime_control'
    a.output.mkdir(parents=True,exist_ok=False)
    scene=ROOT/'motion_scene_inputs_v1'/a.context;run=ROOT/'native_scene_training_v1'/a.context/'formal30k'
    sensor=read(scene/'sensor_preparation.json');request=read(ROOT/'benchmark_cases_v1/scenes'/a.context/'requests/ego_speed.json')
    cameras=read(run/'cameras.json');T=np.loadtxt(scene/'transform_matrix.txt')
    event=read(run/'checkpoint_events/iteration_030000.json');gts={row['image_name']:Path(row['channels']['gt']) for row in event['frames']}
    records=[];tiles=[]
    for row in request['frames']:
        query_index=(row['query_model_time']+.48)/.02;nearest=int(round(query_index))
        if abs(query_index-nearest)>1e-7:continue
        assert 0<=nearest<49
        source=sensor['frames'][nearest];camera=model_camera(T,np.asarray(row['camera_c2w_world']))
        native=cameras[nearest];expected=np.eye(4);expected[:3,:3]=native['rotation'];expected[:3,3]=native['position']
        error=float(np.max(np.abs(camera-expected)));assert error<=2e-6,'Do not call novel/interpolated cameras measured-pose GT'
        world=sensor_camera(scene,source,a.context);world_error=float(np.max(np.abs(world-np.asarray(row['camera_c2w_world']))));assert world_error<=2e-6
        stem=f'{row["raw"]:03d}';target=Image.open(gts[source['frame_id']]).convert('RGB');truth=np.asarray(target,dtype=np.float64)/255
        with np.load(a.fixed/'float'/(stem+'.npz')) as payload:fixed=np.clip(payload['rgb'].transpose(1,2,0),0,1);fixed_dynamic=payload['dynamic_weight'][0].copy()
        with np.load(a.global_reference/'float'/(stem+'.npz')) as payload:global_rgb=np.clip(payload['rgb'].transpose(1,2,0),0,1);global_dynamic=payload['dynamic_weight'][0].copy()
        excluded=np.zeros((320,480),bool);ground_record=None
        for name in [source['frame_id'],row['frame_id']]:
            mask_path=scene/('dynamic_masks/dynamic_mask_0' if a.reference_domain=='visible_measured_ground' else 'mask/mask_0/Annotations')/(name+'.png')
            mask=Image.open(mask_path).resize((480,320),Image.Resampling.NEAREST)
            excluded|=np.asarray(mask)>0
        excluded=binary_dilation(excluded,iterations=8)
        excluded|=binary_dilation((fixed_dynamic>.001)|(global_dynamic>.001),iterations=3)
        valid=~excluded
        if a.reference_domain=='visible_measured_ground':
            ground,ground_record=measured_ground_domain(scene,run,source,native,sensor['frames'][row['camera_index']]);valid&=ground
        n=int(valid.sum());assert n>0
        def metrics(rgb):
            delta=rgb-truth;mse=float(np.square(delta[valid]).mean())
            return dict(MAE=float(np.abs(delta[valid]).mean()),PSNR=-10*math.log10(mse) if mse>0 else None)
        f=metrics(fixed);g=metrics(global_rgb)
        record=dict(output_raw=row['raw'],reference_raw=source['raw'],reference_frame_id=source['frame_id'],source_index=nearest,
                    same_model_camera_max_difference=error,same_sensor_camera_max_difference=world_error,
                    full_frame_pixels=320*480,evaluation_pixels=n,excluded_pixels=int(excluded.sum()),
                    output_world_time=row['model_time'],reference_camera_time=row['query_model_time'],identity_anchor=abs(row['model_time']-row['query_model_time'])<1e-7,
                    fixed_world=f,global_playback=g,delta_MAE=f['MAE']-g['MAE'],delta_PSNR=f['PSNR']-g['PSNR'],measured_ground_reference=ground_record)
        records.append(record)
        if not record['identity_anchor']:
            tile=Image.new('RGB',(1920,350),'#142332');draw=ImageDraw.Draw(tile)
            for col,(label,array) in enumerate([('measured input camera RGB',truth),('global playback / original world time',global_rgb),('ego-only speed / fixed world time',fixed),('sparse measured evaluation domain',np.repeat(valid[...,None],3,axis=2))]):
                tile.paste(Image.fromarray(np.round(np.clip(array,0,1)*255).astype('uint8')),(col*480,30));draw.text((col*480+4,8),label+f' / out{row["raw"]} / ref{source["raw"]}',fill='white')
            tiles.append(tile)
    assert len(records)>=3
    rows=[r for r in records if not r['identity_anchor']];assert rows
    summary=dict(context=a.context,matched_source_pose_frames=len(records),nonanchor_frames=len(rows),rows=records,
                 nonanchor_evaluation_frame_pixels=sum(r['evaluation_pixels'] for r in rows),
                 fixed_world_MAE_macro=float(np.mean([r['fixed_world']['MAE'] for r in rows])),
                 global_playback_MAE_macro=float(np.mean([r['global_playback']['MAE'] for r in rows])),
                 fixed_world_PSNR_macro=float(np.mean([r['fixed_world']['PSNR'] for r in rows])),
                 global_playback_PSNR_macro=float(np.mean([r['global_playback']['PSNR'] for r in rows])),
                 reference='native exported 8-bit source RGB at exactly measured camera; prediction native float clamp[0,1]. Source time and fixed world time differ: colour consistency diagnostic, not counterfactual world-time RGB GT.',
                 reference_domain=a.reference_domain,
                 exclusion='Selected actor body masks at source/output timestamps plus8px, rendered target support plus3px. Near-ground measured returns classified by actual native coarse plane and checked against same-frame nearest LiDAR. No box-to-mask conversion.' if a.reference_domain=='visible_measured_ground' else 'All DEVA instance pixels at source/output timestamps plus8px and rendered target support plus3px.',
                 exclusion_is_inferred_static_proxy_not_dense_static_GT=True,sparse_reference_not_full_road_GT=True,unknown_novel_camera_GT=False,source_colour_consistency_not_counterfactual_quality=True,
                 no_counterfactual_vehicle_RGB_GT=True,observed_pose_diagnostic_only=True,new_training_updates=0,new_edit_case_count=0,
                 quality_accepted=False,human_verdict=None)
    sheet=Image.new('RGB',(1920,350*len(tiles)),'#142332')
    for index,tile in enumerate(tiles):sheet.paste(tile,(0,index*350))
    sheet.save(a.output/'comparison.jpg',quality=94);(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))

if __name__=='__main__':main()
