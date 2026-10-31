"""实际 OBB 量测的空间描述统计；不是 owner/P/完整车体判定。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,json,pathlib,time
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent

def spatial(root):
    media=json.loads((root/'media.json').read_text(encoding='utf8'));context=media['context_id']
    ext=np.array(media['projection']['source_calibration']['extrinsic']).reshape(4,4)
    frames={x['raw']:x for x in media['frames']};rows=[];rays={}
    for track,a in media['actors'].items():
        p=root/'lidar'/('actor_'+a['alias']+'.npz')
        with np.load(p) as f:xyz=f['actor_local_xyz'];raw=f['raw_frame'];cv=f['front_color_valid']
        dims=np.median([t['dimensions'] for t in a['trajectory']],axis=0)
        if len(xyz):
            voxels=len(np.unique(np.floor(xyz/.06).astype(np.int32),axis=0));bounds=np.quantile(xyz,[.01,.99],axis=0)
            extent=bounds[1]-bounds[0];normalized=extent/dims
        else:voxels=0;bounds=None;extent=None;normalized=None
        sectors=np.zeros(8,dtype=int);angles=[];vectors=[];supported20=0;frontsupported20=0
        for t in a['trajectory']:
            r=t['raw'];lo,hi=np.searchsorted(raw,[r,r+1]);n=int(hi-lo);supported20+=n>=20
            frame=frames[r];box=next(b for b in frame['boxes'] if b['track']==track)
            if n>=20 and box['official_FRONT_visible']:frontsupported20+=1
            if n:
                pose=np.array(frame['camera_pose_world']).reshape(4,4);camera=pose[:3,:3]@ext[:3,3]+pose[:3,3]
                R=np.array(t['orientation_world']).reshape(3,3);v=(camera-t['center_world'])@R;distance=float(np.linalg.norm(v))
                direction=v/distance if distance>0 else v;azimuth=float(np.arctan2(v[1],v[0]));sector=int(np.floor((azimuth+np.pi)/(2*np.pi)*8))%8
                if box['official_FRONT_visible']:sectors[sector]+=1;angles.append(azimuth)
                vectors.append(dict(raw=r,camera_direction_actor=direction.round(7).tolist(),camera_distance_m=round(distance,5),
                                    official_FRONT_visible=box['official_FRONT_visible'],actual_measured_box_points=n))
        rows.append(dict(context_id=context,scene_key=media['scene_key'],track_id=track,actor_alias=a['alias'],
            measured_box_rows=len(xyz),aggregate_unique_6cm_voxels=voxels,
            actor_local_q01_q99_bounds_m=None if bounds is None else bounds.round(6).tolist(),
            actor_local_q01_q99_extent_m=None if extent is None else extent.round(6).tolist(),
            median_annotation_box_dimensions_m=dims.round(6).tolist(),
            measured_extent_over_median_box=None if normalized is None else normalized.round(6).tolist(),
            frames_with_at_least_20_measured_points=supported20,official_FRONT_frames_with_at_least_20_measured_points=frontsupported20,
            labelled_frame_denominator=len(a['trajectory']),raw_scene_frame_denominator=media['frame_count'],
            FRONT_visible_supported_camera_azimuth_sector_count=int(np.count_nonzero(sectors)),
            FRONT_visible_supported_camera_azimuth_sector_frame_counts=sectors.tolist(),
            trusted_finite_FOV_color_rows=int(cv.sum()),view_vector_count=len(vectors),
            complete_body=None,closed_surface=None,edit_capability=None,human_verdict=None))
        rays[context+'|'+track]=vectors
    return rows,rays

def main(watch=False):
    cfg=json.loads((BASE/'scenes.json').read_text(encoding='utf8'));out=BASE/'spatial';out.mkdir(exist_ok=True);done={};allrays={};start=time.monotonic()
    while True:
        changed=False
        for scene in cfg['scenes']:
            key=scene['scene_key'];root=BASE/'generated/scenes'/scene['context_id']
            if key in done or not (root/'frame_point_views.json').exists():continue
            rows,rays=spatial(root);done[key]=rows;allrays.update(rays);changed=True
            print(json.dumps(dict(scene=key,actors=len(rows),status='spatial_stats_complete')),flush=True)
        if changed:
            rows=[r for s in cfg['scenes'] for r in done.get(s['scene_key'],[])]
            summary=dict(schema='actual_spatial_diagnostics_v1',complete_scenes=len(done),scene_denominator=len(cfg['scenes']),
                actors_with_actual_stats=len(rows),actor_denominator=sum(len(s['actors']) for s in cfg['scenes']),
                definition='Measured box-contained point statistics and calibrated FRONT camera pose bearings; descriptive input diagnostics only.',
                limitations=['OBB may contain road or occluders; extent/voxel occupancy does not prove vehicle surface coverage.',
                    'Camera direction is a pose bearing, not verified photometric visibility or measured surface normal.',
                    'TOP pixel ego motion compensated; within-scan actor motion not compensated.','No calibrated thresholds or formal edit eligibility.'],
                scene_states=[dict(scene_key=s['scene_key'],context_id=s['context_id'],state='complete' if s['scene_key'] in done else 'pending',
                                   actor_denominator=len(s['actors'])) for s in cfg['scenes']],rows=rows)
            (out/'actual_spatial_stats.json').write_text(json.dumps(summary,ensure_ascii=False,separators=(',',':')),encoding='utf8')
            (out/'actual_spatial_view_vectors.json').write_text(json.dumps(allrays,separators=(',',':')),encoding='utf8')
            with (out/'actual_spatial_stats.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
                for r in rows:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()})
        if len(done)==len(cfg['scenes']) or not watch:break
        if time.monotonic()-start>7200:raise TimeoutError('not all scenes completed in 2h')
        time.sleep(10)
    print(json.dumps(dict(status='complete' if len(done)==15 else 'partial',scenes=len(done),wall_seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();main(a.watch)
