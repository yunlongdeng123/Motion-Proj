"""Read-only body-frame viewing-direction audit; not visibility or RGB truth.

Uses actual native camera centres, tracked metric box axes and frozen requests.
No checkpoint tensor, segmentation, training input or editing method is changed.
"""
import argparse, json
from pathlib import Path
import numpy as np
from benchmark_geometry import ROOT, ActorTrack, sensor_camera, asset_transfer, read
from prepare_motion_suite import apply


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def angles(x):
    x = unit(np.asarray(x))
    return np.rad2deg(np.arctan2(x[..., 1], x[..., 0])), np.rad2deg(np.arctan2(x[..., 2], np.linalg.norm(x[..., :2], axis=-1)))


def profile(track):
    run = ROOT/'native_scene_training_v1'/track.context/'formal30k'
    returned = read(run/'returned.json')
    assert returned['iteration'] == 30000, 'Only completed development bases'
    request = read(ROOT/'benchmark_cases_v1/scenes'/track.context/'summary.json')
    assert request['split'] != 'heldout_reserved', 'Do not use heldout assets for method selection'
    T = np.loadtxt(track.source/'transform_matrix.txt'); inv = np.linalg.inv(T)
    cameras = read(run/'cameras.json'); rays, positions = [], []
    closure = 0.
    for i, camera in enumerate(cameras):
        assert camera['img_name'] == track.frames[i]['frame_id']
        position = apply(inv, np.array([camera['position']]))[0]
        original = sensor_camera(track.source, track.frames[i], track.context)
        closure = max(closure, float(np.abs(original[:3, 3]-position).max()))
        center, axes, _ = track.at(track.times[i])
        rays.append(unit((position-center) @ axes)); positions.append(position)
    assert closure < 2e-6
    rays = np.asarray(rays); az, el = angles(rays)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(az)))
    cross = np.rad2deg(np.arccos(np.clip(rays @ rays.T, -1, 1)))
    np.fill_diagonal(cross, np.inf)
    preparation = read(track.source/'dynamic_multi_actor_preparation.json')['actors'][0]['frames']
    return dict(track=track, T=T, inverse=inv, rays=rays, positions=np.array(positions),
                json=dict(context=track.context, track=track.track, original_training_frames=49,
                          camera_sensor_closure_max_abs_m=closure,
                          azimuth_unwrapped_min_deg=float(unwrapped.min()), azimuth_unwrapped_max_deg=float(unwrapped.max()),
                          azimuth_span_deg=float(np.ptp(unwrapped)), elevation_min_deg=float(el.min()), elevation_max_deg=float(el.max()),
                          nearest_different_frame_direction_degrees_median=float(np.median(cross.min(1))),
                          maximum_training_pair_direction_degrees=float(np.max(cross[np.isfinite(cross)])),
                          observed_body_mask_pixels_native_resolution=[r['body_pixels'] for r in preparation],
                          retained_real_LiDAR_per_frame=[r['retained_real_points'] for r in preparation]))


def evaluate(p, name, rows, cameras, centers, axes, origin):
    rays = unit(np.einsum('ni,nij->nj', np.asarray(cameras)-np.asarray(centers), np.asarray(axes)))
    distances = np.rad2deg(np.arccos(np.clip(rays @ p['rays'].T, -1, 1)))
    closest = distances.argmin(1); nearest = distances.min(1)
    az, el = angles(rays); records = []
    for i, row in enumerate(rows):
        records.append(dict(raw=row['raw'], native_camera_index=row['camera_index'],
                            body_frame_view_direction=rays[i].tolist(), azimuth_deg=float(az[i]), elevation_deg=float(el[i]),
                            closest_training_frame_id=p['track'].frames[int(closest[i])]['frame_id'],
                            nearest_training_direction_deg=float(nearest[i])))
    result = dict(case=name, family=origin, frames=len(rows), source_context=p['track'].context,
                  nearest_direction_degrees_mean=float(nearest.mean()), nearest_direction_degrees_max=float(nearest.max()),
                  frames_over_direction_distance={str(t):int((nearest>t).sum()) for t in [1, 3, 5, 10]},
                  threshold_semantics='descriptive sensitivity, not admission or correctness thresholds', rows=records)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--context', action='append', required=True); parser.add_argument('--output',type=Path); a=parser.parse_args()
    destination=a.output or ROOT/'actor_view_support_v1'; destination.mkdir(exist_ok=False)
    profiles={c:profile(ActorTrack(c)) for c in a.context}; reports=[]
    for context,p in profiles.items():
        track=p['track']; cases=[]
        suite=read(ROOT/'motion_edit_suite_v1'/context/'summary.json')
        for spec in suite['prepared_single_requests']:
            case=read(ROOT/'motion_edit_suite_v1'/context/(spec['id']+'.json'))
            if case['family']=='delete': continue
            cameras=[]; centers=[]; axes=[]
            for row in case['frames']:
                i=row['camera_index']; camera=p['positions'][i].copy()
                if case.get('camera_lateral_m'):
                    native=read(ROOT/'native_scene_training_v1'/context/'formal30k/cameras.json')[i]
                    world_right=p['inverse'][:3,:3] @ np.array(native['rotation'])[:,0]
                    camera += unit(world_right)*case['camera_lateral_m']
                center,rotation,_=track.at(row['dynamic_query_model_time'])
                if case['family'] in ['translate','insert_reused','turn']:
                    delta=np.array(row['target_rotation_delta_world']);center=delta@center+row['target_affine_translation_world'];rotation=delta@rotation
                cameras.append(camera);centers.append(center);axes.append(rotation)
            result=evaluate(p,case['id'],case['frames'],cameras,centers,axes,'legacy_motion_request')
            if case['family']=='no_edit': assert result['nearest_direction_degrees_max']<2e-5
            cases.append(result)
        # Source observation coverage for genuine cross-scene insertion. No new GPU inference.
        folder=ROOT/'benchmark_cases_v1/scenes'/context/'requests'
        request=read(folder/'vehicle_insertion.json')
        source=request['source_asset_context']
        if source in profiles:
            donor=profiles[source];cameras=[];centers=[];axes=[];rows=request['frames']
            for row in rows:
                _,_,_,audit=asset_transfer(donor['track'],track,row,donor['T'],p['T'])
                _,rotation,_=track.at(row['query_model_time'])
                i=row['camera_index'];cameras.append(p['positions'][i]);centers.append(audit['target_center_world']);axes.append(rotation)
            cases.append(evaluate(donor,'foreign_asset_insertion',rows,cameras,centers,axes,'cross_scene_asset_request'))
        report=dict(**p['json'], cases=cases, interpretation='Object-centre direction coverage is an input proxy, not proof of visible surface, geometry accuracy or cause of render defects.',
                    rigid_box_track_proxy_not_exact_deformable_gaussian_frame=True, point_or_pixel_visibility_measured=False,
                    pooling_training_directions_over_time_can_overstate_per_time_support=True,
                    camera_rotation_at_fixed_center_does_not_change_this_proxy_but_can_change_FOV=True,
                    renderer_or_operator_modified=False, new_training_updates=0, new_edit_case_count=0, quality_accepted=False,human_verdict=None)
        (destination/(context+'.json')).write_text(json.dumps(report,indent=2));reports.append(report)
    plot(destination,profiles,reports)
    result=dict(contexts=[{k:v for k,v in r.items() if k!='cases'}|{'cases':[{k:v for k,v in c.items() if k!='rows'} for c in r['cases']]} for r in reports],
                additional_GPU_seconds=0,new_training_updates=0,new_edit_case_count=0,diagnostic_only=True,quality_accepted=False,human_verdict=None)
    (destination/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


def plot(destination,profiles,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(len(reports),2,figsize=(12,4.4*len(reports)),squeeze=False)
    for i,r in enumerate(reports):
        p=profiles[r['context']];az,el=angles(p['rays']);branch=float(np.median(az))
        same_branch=lambda value: (np.asarray(value)-branch+180)%360-180+branch
        axes[i,0].scatter(same_branch(az),el,s=18,label='Original49',c='black')
        for name,color in [('turn_left15','#006fa7'),('turn_right15','#d2761c'),('foreign_asset_insertion','#1b9868')]:
            case=next((x for x in r['cases'] if x['case']==name),None)
            if not case:continue
            rows=case['rows']
            # Foreign insertion uses a different donor's training directions;
            # do not plot it against the host's original-direction scatter.
            if name!='foreign_asset_insertion':axes[i,0].plot(same_branch([x['azimuth_deg'] for x in rows]),[x['elevation_deg'] for x in rows],'.-',label=name,c=color,markersize=3)
            axes[i,1].plot([x['raw'] for x in rows],[x['nearest_training_direction_deg'] for x in rows],label=name,c=color)
        axes[i,0].set(xlabel='Body-frame azimuth (degrees)',ylabel='Elevation (degrees)',title=r['context'].split('_')[0]+' observed and requested views')
        axes[i,1].set(xlabel='Raw camera frame',ylabel='Nearest original direction (degrees)',title='Direction distance, not surface visibility or quality')
        for ax in axes[i]:ax.legend(fontsize=8);ax.grid(alpha=.2)
    figure.tight_layout();figure.savefig(destination/'view_directions.png',dpi=160);plt.close(figure)


if __name__=='__main__':main()
