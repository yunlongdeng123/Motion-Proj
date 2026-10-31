"""Local-only source/media identity audit; no model loading or GPU calls."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,importlib.util,json,math,sys
from pathlib import Path
DEST=Path(__file__).resolve().parent
BASE=DEST.parent/'media'
sys.path.insert(0,str(BASE/'vendor'))
import numpy as np
CONTEXT='13299463771883949918_4240_000_4260_000'
OUT=BASE/'generated/scenes'/CONTEXT
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
media=load(OUT/'media.json');manifest=load(OUT/'manifest.json')
receipts=load(OUT/'source_receipts.json');decode=load(OUT/'lidar_decode_receipts.json')
assert media['frame_count']==media['full_raw_frame_denominator']==manifest['raw_frame_count']==198
assert len(media['actors'])==manifest['actors']==83
assert len(receipts)==len(decode)==198
assert [x['raw'] for x in receipts]==list(range(198))
assert all(x['context']==CONTEXT and x['length_crc32c']==x['payload_crc32c']=='passed' for x in receipts)
assert sum(x['retained_points'] for x in decode)==manifest['measured_sensor_points']==35211590
for d in decode:
    assert len(d['sensors'])==10
    assert {(v['laser_name'],v['return_index']) for v in d['sensors']}=={(i,j) for i in range(1,6) for j in range(2)}
    assert sum(v['retained_points'] for v in d['sensors'])==d['retained_points']
allrows=displayrows=0;max_bound_excess=0.;max_display_error=0.;sample_points={};old_valid=new_valid=grayed=0
for track,actor in media['actors'].items():
    path=BASE/'generated'/actor['lidar']['npz']
    with np.load(path) as a:
        xyz=a['actor_local_xyz'];raw=a['raw_frame'];src=a['source_cloud_row'];rgb=a['rgb']
        colorvalid=a['front_color_valid']
        backup=OUT/'color_diagnostics_r1'/path.name
        with np.load(backup) as old:
            for key in ['actor_local_xyz','raw_frame','source_cloud_row']:assert np.array_equal(a[key],old[key])
            assert not np.any(colorvalid & ~old['front_color_valid'])
            old_valid+=int(old['front_color_valid'].sum());new_valid+=int(colorvalid.sum())
            grayed+=int((old['front_color_valid']&~colorvalid).sum())
        assert np.all(rgb[~colorvalid]==np.array([155,180,205],np.uint8))
        assert xyz.shape==(len(raw),3) and rgb.shape==xyz.shape and len(src)==len(raw)
        assert np.isfinite(xyz).all() and (src>=0).all()
        keys=(raw.astype(np.uint64)<<32)|src.astype(np.uint64)
        assert len(np.unique(keys))==len(keys)
        assert len(xyz)==actor['lidar']['measured_points_before_voxel']
        assert len(actor['trajectory'])==actor['lidar']['labelled_frames']
        positive=0
        for pose in actor['trajectory']:
            q=raw==pose['raw'];assert int(q.sum())==pose['measured_lidar_points']
            assert np.all(src[q]<decode[pose['raw']]['retained_points'])
            if q.any():
                positive+=1
                excess=float(np.max(np.abs(xyz[q])-np.array(pose['dimensions'])/2))
                max_bound_excess=max(max_bound_excess,excess)
                assert excess<=2e-5
            rot=np.array(pose['orientation_world']).reshape(3,3)
            assert np.allclose(rot.T@rot,np.eye(3),atol=3e-9)
            if pose['raw'] in [0,99,197] and q.any():sample_points[(track,pose['raw'])]=(xyz[q].copy(),src[q].copy(),colorvalid[q].copy())
        assert positive==actor['lidar']['source_frames']
        s=(BASE/'generated'/actor['lidar']['data_js']).read_text(encoding='utf-8')
        display=json.JSONDecoder().raw_decode(s.split(']=',1)[1])[0]
        assert display['context_id']==CONTEXT and display['track_id']==track
        assert display['display_points']==len(display['xyz'])==actor['lidar']['display_points']<=12000
        dk=(np.array(display['raw_frame'],np.uint64)<<32)|np.array(display['source_cloud_row'],np.uint64)
        order=np.argsort(keys);loc=np.searchsorted(keys[order],dk)
        assert (loc<len(order)).all() and np.array_equal(keys[order][loc],dk)
        if len(dk):
            error=float(np.max(np.abs(xyz[order[loc]]-np.array(display['xyz']))));max_display_error=max(max_display_error,error)
            assert error<=5.1e-5
            assert np.array_equal(np.array(display['front_color_valid'],bool),colorvalid[order[loc]])
        allrows+=len(xyz);displayrows+=len(dk)
assert allrows==manifest['measured_box_rows']==2231948

spec=importlib.util.spec_from_file_location('media_source_audit',BASE/'build_media.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
sampled=[];max_identity_error=0.;pointrows=0
for index,frame,receipt in builder.records(Path(manifest['raw_path'])):
    assert frame.context.name==CONTEXT and frame.timestamp_micros==receipts[index]['timestamp_micros']
    if index not in [0,99,197]:continue
    cloud,details=builder.decoder.lidar(frame,builder.pb)
    assert len(cloud)==decode[index]['retained_points']
    labels={label.id:label for label in frame.laser_labels if label.type==1}
    ego=np.array(frame.pose.transform).reshape(4,4);front=next(x for x in frame.images if x.name==1)
    camera=builder.camera_model(frame,front);cam=builder.transform(builder.transform(cloud[:,:3],ego),camera['camera_from_world'])
    uv,dep=builder.project_camera(cam,camera)
    valid=builder.pinhole_frustum_valid(cam,camera)&np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<builder.W)&(uv[:,1]>=0)&(uv[:,1]<builder.H)
    local_rows=0
    for (track,raw),(saved,source_rows,saved_cv) in sample_points.items():
        if raw!=index:continue
        b=labels[track].box;c,s=math.cos(b.heading),math.sin(b.heading)
        R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        independent=np.einsum('ij,nj->ni',R.T,cloud[source_rows,:3]-np.array([b.center_x,b.center_y,b.center_z]))
        error=float(np.max(np.abs(independent-saved)))
        assert error<2e-5
        assert np.array_equal(saved_cv,valid[source_rows])
        max_identity_error=max(max_identity_error,error);pointrows+=len(saved);local_rows+=len(saved)
    sampled.append(dict(raw=index,decoded_sensor_points=len(cloud),actor_rows_reconstructed=local_rows))
assert len(sampled)==3
assert (old_valid,new_valid,grayed)==(173837,97509,76328)
result=dict(schema='independent_source_media035_audit_v1',status='PASS_SOURCE_IDENTITY_AND_COORDINATE_CHECKS',
    context_id=CONTEXT,full_raw_frames=198,actors=83,all_sensor_measurements=35211590,
    box_filtered_actor_rows_checked=allrows,display_rows_checked=displayrows,
    duplicate_tuple_within_actor=0,point_identity_key='context + raw_frame + deterministic decoder source_cloud_row; not physical surface ID',
    OBB_max_excess_m=max_bound_excess,display_rounding_max_abs_m=max_display_error,
    sampled_source_redecode=sampled,reconstructed_actor_rows=pointrows,reconstructed_local_coordinate_max_abs_m=max_identity_error,
    full_original_TFRecord_CRC_and_context_reread=True,
    color_projection_fix=dict(status='PASS_INDEPENDENT_RECEIPT_AND_POINTWISE_CHECK',old_valid=old_valid,new_valid=new_valid,
        invalid_color_rows_grayed=grayed,all83_actor_XYZ_raw_source_exactly_unchanged=True,
        sampled_color_flags_recomputed_for_3_raw_frames=True),
    camera_projection_review='source math reviewed: inverse(image_pose@camera_extrinsic), camera xforward and negative y/z image axes, nine-intrinsic Brown-Conrady, cuboid near-plane/image clipping',
    original_native_model_or_GPU_calls=0,human_quality_verdict=None,
    limitations=['one completed035scene audited; other scenes require matching receipts',
       'TOP per-pixel ego pose compensation does not compensate actor motion within the range scan',
       'actor local aggregation uses each frame original box pose, not a completed or rigidly exact vehicle surface',
       'box returns can include ground/occluders, and the same sensor return can belong to overlapping box diagnostics',
       'projected FRONT color has no visibility zbuffer, so color validity means image bounds, not true visible surface color',
       'image-mid-pose cuboid projection does not implement row rolling shutter or actor motion compensation; no exact official projection parity asserted',
       'no complete-surface, f_geom, f_view, f_bg or edit-success inference'],
    source_files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [BASE/'build_media.py',BASE/'waymo_numpy_cpu.py',BASE/'fast_range_matrix.py',OUT/'media.json',OUT/'manifest.json',OUT/'source_receipts.json',OUT/'lidar_decode_receipts.json']])
(DEST/'media035_independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
counterexample_path=DEST/'media035_color_frustum_counterexample.json'
counterexample=load(counterexample_path);counterexample['status']='RESOLVED_SOURCE_COLOR_FRUSTUM_CHECK_PASS'
counterexample['resolution']=result['color_projection_fix'];counterexample['resolution_evidence']='media035_independent_audit.json'
counterexample_path.write_text(json.dumps(counterexample,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','box_filtered_actor_rows_checked','display_rows_checked','reconstructed_actor_rows','reconstructed_local_coordinate_max_abs_m']}))
