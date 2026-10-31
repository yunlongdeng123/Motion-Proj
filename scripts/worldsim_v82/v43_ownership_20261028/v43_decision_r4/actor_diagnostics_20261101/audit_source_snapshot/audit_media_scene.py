"""Local stored-media audit plus three original-record geometry controls."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,importlib.util,json,math,sys,time
from pathlib import Path
DEST=Path(__file__).resolve().parent;BASE=DEST.parent/'media'
sys.path.insert(0,str(BASE/'vendor'))
import numpy as np

load=lambda p:json.loads(p.read_text(encoding='utf-8'))
def js_data(path):return json.JSONDecoder().raw_decode(path.read_text(encoding='utf-8').split(']=',1)[1])[0]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--context',required=True);parser.add_argument('--control-raw',type=Path,action='append',default=[]);args=parser.parse_args()
    start=time.perf_counter();out=BASE/'generated/scenes'/args.context;m=load(out/'media.json');manifest=load(out/'manifest.json')
    n=m['full_raw_frame_denominator'];assert m['frame_count']==manifest['raw_frame_count']==n
    assert manifest['full_sequence'] and len(m['actors'])==manifest['actors']
    assert m.get('media_state')!='RGB_and_3D_boxes_complete_LiDAR_pending'
    assert [f['raw'] for f in m['frames']]==list(range(n))
    timestamp_to_raw={f['timestamp_micros']:f['raw'] for f in m['frames']};assert len(timestamp_to_raw)==n
    requested={0,n//2,n-1};samples={};allrows=aggrows=frame_display_rows=frame_observations=0;maxbound=maxround=0.
    media_root=BASE/'generated'
    for track,actor in m['actors'].items():
        lidar=actor['lidar'];poses={p['raw']:p for p in actor['trajectory']};assert len(poses)==len(actor['trajectory'])==lidar['labelled_frames']
        with np.load(media_root/lidar['npz']) as p:
            xyz=p['actor_local_xyz'];raw=p['raw_frame'];src=p['source_cloud_row'];rgb=p['rgb'];cv=p['front_color_valid']
            assert xyz.shape==rgb.shape==(len(raw),3) and len(src)==len(cv)==len(raw)
            assert np.isfinite(xyz).all() and (src>=0).all() and (raw<n).all()
            keys=(raw.astype(np.uint64)<<32)|src.astype(np.uint64);order=np.argsort(keys);ordered=keys[order]
            assert len(np.unique(keys))==len(keys)==lidar['measured_points_before_voxel']
            assert np.all(rgb[~cv]==np.array([155,180,205],np.uint8))
            def check_display(d,frame=None):
                nonlocal maxround
                count=len(d['xyz']);assert d['display_points']==count==len(d['source_cloud_row'])==len(d['rgb'])
                rawids=np.array(d['raw_frame'],np.uint64) if frame is None else np.full(count,frame,np.uint64)
                dk=(rawids<<32)|np.array(d['source_cloud_row'],np.uint64);loc=np.searchsorted(ordered,dk)
                assert (loc<len(order)).all() and np.array_equal(ordered[loc],dk)
                if count:
                    error=float(np.max(np.abs(xyz[order[loc]]-np.array(d['xyz']))));assert error<=5.2e-5;maxround=max(maxround,error)
                    assert np.array_equal(rgb[order[loc]],np.array(d['rgb'],np.uint8))
                if 'front_color_valid' in d:assert np.array_equal(cv[order[loc]],np.array(d['front_color_valid'],bool))
                return count
            aggregate=js_data(media_root/lidar['data_js']);assert aggregate['track_id']==track and aggregate['context_id']==args.context
            assert check_display(aggregate)==lidar['display_points']<=12000;aggrows+=lidar['display_points']
            frame_path=media_root/lidar['frame_data_js'];frame_data=js_data(frame_path)
            assert set(map(int,frame_data))==set(poses)
            positive=0
            for frame,pose in poses.items():
                q=raw==frame;count=int(q.sum());assert count==pose['measured_lidar_points']
                rot=np.array(pose['orientation_world']).reshape(3,3);assert np.allclose(rot.T@rot,np.eye(3),atol=3e-9)
                if count:
                    positive+=1;excess=float(np.max(np.abs(xyz[q])-np.array(pose['dimensions'])/2));assert excess<=2e-5;maxbound=max(maxbound,excess)
                    if frame in requested:samples[(track,frame)]=(xyz[q].copy(),src[q].copy(),cv[q].copy())
                d=frame_data[str(frame)];assert d['measured_points_before_voxel']==count
                shown=check_display(d,frame);assert shown<=512
                assert (shown>0)==(count>0)
                if count:
                    _,keep=np.unique(np.floor(xyz[q]/.06).astype(np.int32),axis=0,return_index=True);keep.sort()
                    if len(keep)>512:keep=keep[np.linspace(0,len(keep)-1,512,dtype=int)]
                    assert np.array_equal(np.array(d['source_cloud_row']),src[q][keep])
                frame_display_rows+=shown;frame_observations+=1
            assert positive==lidar['source_frames'];allrows+=len(xyz)
    assert allrows==manifest['measured_box_rows']
    spec=importlib.util.spec_from_file_location('media_control_source',BASE/'build_media.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    controls=list(args.control_raw);full_original=False
    rawpath=Path(manifest.get('raw_path',''))
    if not controls and rawpath.is_file() and rawpath.name.startswith('segment-'):
        controls=[rawpath];full_original=True
    checked=[];seen=set();max_identity=0.;controlled_rows=0;crc_records=0;control_errors=[];color_flag_mismatches=0;controlled_tracks=set()
    for path in controls:
        for _,frame,receipt in builder.records(path):
            crc_records+=1;assert frame.context.name==args.context and frame.timestamp_micros in timestamp_to_raw
            raw=timestamp_to_raw[frame.timestamp_micros]
            if raw not in requested:continue
            assert raw not in seen;seen.add(raw)
            cloud,details=builder.decoder.lidar(frame,builder.pb);labels={l.id:l for l in frame.laser_labels if l.type==1}
            ego=np.array(frame.pose.transform).reshape(4,4);front=next(im for im in frame.images if im.name==1);camera=builder.camera_model(frame,front)
            cam=builder.transform(builder.transform(cloud[:,:3],ego),camera['camera_from_world']);uv,_=builder.project_camera(cam,camera)
            valid=builder.pinhole_frustum_valid(cam,camera)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<builder.W)&(uv[:,1]>=0)&(uv[:,1]<builder.H)
            matched=0
            for (track,f),(saved,source_rows,saved_cv) in samples.items():
                if f!=raw:continue
                assert (source_rows<len(cloud)).all();b=labels[track].box;c,s=math.cos(b.heading),math.sin(b.heading);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                reconstructed=np.einsum('ij,nj->ni',R.T,cloud[source_rows,:3]-np.array([b.center_x,b.center_y,b.center_z]))
                delta=np.max(np.abs(reconstructed-saved),axis=1);err=float(delta.max());max_identity=max(max_identity,err);control_errors.append(delta)
                color_flag_mismatches+=int(np.count_nonzero(saved_cv!=valid[source_rows]));matched+=len(saved)
                controlled_tracks.add(track)
            controlled_rows+=matched;checked.append(dict(raw=raw,decoded_sensor_points=len(cloud),reconstructed_actor_rows=matched))
    control_delta=np.concatenate(control_errors) if control_errors else np.array([])
    if full_original:assert crc_records==n, ('original_file_record_denominator',crc_records,n)
    strict_parity=seen==requested and max_identity<=2e-5 and color_flag_mismatches==0
    status=('PASS_STORED_POINT_IDENTITIES_AND_THREE_ORIGINAL_RECORD_CONTROLS' if strict_parity else
            'STORED_IDENTITIES_PASS_STRICT_COORDINATE_PARITY_FAIL' if seen==requested else 'PASS_STORED_POINT_IDENTITIES_ORIGINAL_CONTROLS_PENDING')
    result=dict(schema='independent_actor_media_scene_audit_v1',status=status,context_id=args.context,scene_key=m['scene_key'],
        full_frames=n,actors=len(m['actors']),box_filtered_measured_rows=allrows,aggregate_display_rows=aggrows,
        independent_frame_display_rows=frame_display_rows,labelled_actor_frame_observations=frame_observations,
        all_nonzero_measured_frames_have_nonempty_independent_display=True,per_frame_voxel_and_cap_recomputed=True,
        source_identity_tuple_unique_within_actor=True,max_OBB_excess_m=maxbound,max_display_rounding_m=maxround,
        original_control_records=checked,original_control_frames_required=sorted(requested),original_control_frames_checked=sorted(seen),
        original_control_actor_rows=controlled_rows,original_control_local_reconstruction_max_abs_m=max_identity,
        original_control_actors_represented=len(controlled_tracks),
        strict_original_control_tolerance_m=2e-5,strict_original_control_coordinate_and_color_parity_pass=strict_parity,
        original_control_coordinate_median_m=float(np.median(control_delta)) if len(control_delta) else None,
        original_control_coordinate_p99_m=float(np.quantile(control_delta,.99)) if len(control_delta) else None,
        original_control_rows_over_strict_tolerance=int((control_delta>2e-5).sum()),
        original_control_color_flag_mismatches=color_flag_mismatches,
        source_row_reassignment_or_nearest_neighbor_used=False,
        original_full_record_CRC_context_reread=full_original,CRC_checked_records_local=crc_records,
        controls_source_paths=[str(p) for p in controls],GPU_queries=0,remote_queries=0,human_quality_verdict=None,
        limits=['algebra/identity consistency is not sensor absolute accuracy','box-filtered rows are not instance segmentation or complete surface',
                'TOP ego-motion compensation does not compensate within-scan actor motion','color has FOV guard but no occlusion zbuffer',
                'projection is image-pose approximation without row rolling shutter/actor exposure motion correction',
                'three-control subset reread is distinguished from full-original TFRecord reread','formal geometry/view/background/motion eligibility remains unknown'],
        source_files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [BASE/'build_media.py',out/'media.json',out/'manifest.json']],
        wall_seconds=time.perf_counter()-start)
    directory=DEST/'media_scene_audits';directory.mkdir(exist_ok=True)
    (directory/(args.context+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','scene_key','actors','box_filtered_measured_rows','independent_frame_display_rows','original_control_frames_checked','wall_seconds']}))
    if seen==requested and not strict_parity:sys.exit(3)
if __name__=='__main__':main()
