"""Targeted original-record / TOP pixel-pose / official camera-projection audit.

Walks record headers with seek, reads only five requested payloads. The original
export arithmetic is replayed, then independently compared with float64 TOP
sensor-to-world math using SciPy Euler rotations without frame roundtrip.
"""
import argparse,hashlib,importlib.util,json,shutil,struct,time,zlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats

TARGETS=[60,70,110,145,165]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def selected_records(path,decoder):
    crc=decoder.crc_function();last=max(TARGETS)
    with Path(path).open('rb') as f:
        for raw in range(last+1):
            offset=f.tell();header=f.read(12);assert len(header)==12
            length,checksum=struct.unpack('<QI',header);assert 0<length<96*1024**2 and crc(header[:8])==checksum
            if raw not in TARGETS:
                f.seek(length+4,1);continue
            payload=f.read(length);trailer=f.read(4);assert len(payload)==length and len(trailer)==4
            assert crc(payload)==struct.unpack('<I',trailer)[0]
            yield raw,payload,dict(header_offset=offset,payload_length=length,header_CRC32C=True,payload_CRC32C=True,
                payload_sha256=hashlib.sha256(payload).hexdigest())


def intmatrix(compressed,pb):
    m=pb.MatrixInt32();m.ParseFromString(zlib.decompress(compressed));return np.asarray(m.data,np.int32).reshape(tuple(m.shape.dims))


def top_world64(image,pixel_pose,calibration):
    H,W=image.shape[:2];E=np.asarray(calibration.extrinsic.transform,np.float64).reshape(4,4)
    if calibration.beam_inclinations:inc=np.asarray(calibration.beam_inclinations,np.float64)
    else:inc=(np.arange(H,dtype=float)+.5)/H*(calibration.beam_inclination_max-calibration.beam_inclination_min)+calibration.beam_inclination_min
    inc=inc[::-1]
    az=((np.arange(W,0,-1,dtype=float)-.5)/W*2-1)*np.pi-np.arctan2(E[1,0],E[0,0])
    r=image[...,0].astype(float)
    sensor=np.stack([np.cos(az)[None]*np.cos(inc)[:,None]*r,
                     np.sin(az)[None]*np.cos(inc)[:,None]*r,
                     np.broadcast_to(np.sin(inc)[:,None],(H,W))*r],-1)
    vehicle=sensor@E[:3,:3].T+E[:3,3]
    # Independent rotation implementation, not decoder.rotation().
    R=Rotation.from_euler('xyz',pixel_pose[...,:3].reshape(-1,3).astype(float)).as_matrix().reshape(H,W,3,3)
    world=np.einsum('hwij,hwj->hwi',R,vehicle)+pixel_pose[...,3:].astype(float)
    return world,vehicle


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time();R=C/'road_underbody_local_20261123_r10'
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs');receipt=read(bank/'receipt.json')
    droot=Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101');dp=droot/'waymo_numpy_cpu.py';mp=droot/'fast_range_matrix.py'
    assert pin(dp)['sha256']==receipt['decoder_sha256'] and pin(mp)['sha256']==receipt['matrix_decoder_sha256']
    dec=module('anomalous_raw_decoder',dp);dec.matrix=module('anomalous_fast_matrix',mp).matrix;pb=dec.protobuf()
    src=out/'source_snapshots';src.mkdir();sources=[]
    official=Path('/root/autodl-tmp/envs/dggt_data_cpu/lib/python3.10/site-packages/waymo_open_dataset/utils')
    for path in [dp,mp,official/'range_image_utils.py',official/'transform_utils.py',official/'frame_utils.py']:
        if path.exists():shutil.copy2(path,src/path.name);sources.append(pin(path))
    rows=[]
    for raw,payload,record in selected_records(receipt['raw'],dec):
        folder=out/f'{raw:03d}';folder.mkdir();frame=pb.Frame.FromString(payload)
        op=bank/f'observations/{raw:03d}.json';obs=read(op);lp=bank/obs['lidar_path'];original=np.fromfile(lp,dtype='<f4').reshape(-1,6)
        regenerated,details=dec.lidar(frame,pb);assert regenerated.shape==original.shape
        exact=regenerated.astype('<f4').tobytes()==original.tobytes()
        E=np.asarray(frame.pose.transform).reshape(4,4);obsE=np.asarray(obs['ego_world']).reshape(4,4)
        assert np.array_equal(E,obsE)
        timestamp_file=int((bank/f'timestamp/0065{raw:03d}.txt').read_text().strip());fr=next(x for x in receipt['frames'] if x['raw']==raw)
        assert frame.timestamp_micros==obs['timestamp_micros']==timestamp_file==fr['timestamp_micros']
        front=next(x for x in frame.images if x.name==1);camcal=next(x for x in frame.context.camera_calibrations if x.name==1)
        times={}
        for key in ['pose_timestamp','camera_trigger_time','camera_readout_done_time']:
            if hasattr(front,key):times[key]=float(getattr(front,key));times[key+'_minus_frame_seconds']=times[key]-frame.timestamp_micros*1e-6
        top=next(x for x in frame.lasers if x.name==1);cal=next(x for x in frame.context.laser_calibrations if x.name==1)
        pp=dec.matrix(top.ri_return1.range_image_pose_compressed,pb)
        data=[];return_rows=[]
        for ret in [0,1]:
            ri=getattr(top,f'ri_return{ret+1}');image=dec.matrix(ri.range_image_compressed,pb);H,W=image.shape[:2]
            valid=(image[...,0]>0)&(image[...,3]!=1);yy,xx=np.where(valid);encoded=((ret*H+yy)*W+xx).astype(np.int64)
            select=(original[:,5]>=ret*H*W)&(original[:,5]<(ret+1)*H*W);lidarids=np.flatnonzero(select)
            assert np.array_equal(original[select,5],encoded.astype(np.float32))
            world,vehicle=top_world64(image,pp,cal);reference=world[valid]
            exported_world=original[select,:3].astype(float)@E[:3,:3].T+E[:3,3]
            exported_to_reference=exported_world-reference
            reference_ego=(reference-E[:3,3])@E[:3,:3]
            cp=intmatrix(ri.camera_projection_compressed,pb);assert cp.shape==(H,W,6)
            labels=intmatrix(ri.segmentation_label_compressed,pb) if ri.segmentation_label_compressed else np.full((H,W,2),-1,np.int32)
            cam=cp[valid];front_uv=np.full((len(cam),2),-1,np.int32);front_present=(cam[:,0]==1)|(cam[:,3]==1)
            hit=cam[:,0]==1;front_uv[hit]=cam[hit,1:3];hit2=(cam[:,3]==1)&~hit;front_uv[hit2]=cam[hit2,4:6]
            data.append(dict(lidar_row_id=lidarids.astype(np.int32),TOP_encoded=encoded,return_index=np.full(len(encoded),ret,np.int8),
                range_row=yy.astype(np.int16),range_column=xx.astype(np.int32),official_camera_projection=cam,
                official_front_uv=front_uv,official_front_present=front_present,official_semantic=labels[valid,1],
                exported_worldXYZ=exported_world,independent_float64_pixel_worldXYZ=reference,
                point_pixel_pose_RPYXYZ=pp[valid],exported_to_float64_reference_deltaXYZ=exported_to_reference))
            return_rows.append(dict(return_index=ret,points=len(encoded),official_front_projected_points=int(front_present.sum()),
                independent_world_max_abs_xyz_m=abs(exported_to_reference).max(0).tolist(),
                independent_world_distance_m=stats(np.linalg.norm(exported_to_reference,axis=-1)),
                independent_ego_max_abs_xyz_m=abs(original[select,:3]-reference_ego).max(0).tolist(),
                world_Z_from_pixel_pose_minus_exported_m=stats(-exported_to_reference[:,2]),
                no_pixelPose_frame_world_minus_corrected_world_Z_m=stats((vehicle[valid]@E[:3,:3].T+E[:3,3]-reference)[:,2])))
        np.savez_compressed(folder/'official_TOP_camera_projection_and_world.npz',**{k:np.concatenate([x[k] for x in data]) for k in data[0]},
            frame_pose_world_from_vehicle=E,camera_pose_world=np.asarray(front.pose.transform).reshape(4,4),
            camera_extrinsic=np.asarray(camcal.extrinsic.transform).reshape(4,4),camera_intrinsic_Brown=np.asarray(camcal.intrinsic),
            timestamp_micros=np.asarray(frame.timestamp_micros),camera_pose_timestamp=np.asarray(front.pose_timestamp))
        row=dict(raw=raw,timestamp_micros=int(frame.timestamp_micros),original_record=record,
            all_timestamp_bindings_exact=True,exported_bin_redecode_all6columns_byteexact=exact,
            per_column_max_abs_difference=abs(regenerated-original).max(0).tolist(),all_sensor_return_details=details,
            frame_pose_vs_obs_max_abs=float(abs(E-obsE).max()),
            frame_pose_vs_posefile_max_abs=float(abs(E-np.loadtxt(bank/f'pose/0065{raw:03d}.txt')).max()),
            camera_pose_vs_obs_max_abs=float(abs(np.asarray(front.pose.transform)-obs['camera_pose_world']).max()),
            camera_extrinsic_vs_obs_max_abs=float(abs(np.asarray(camcal.extrinsic.transform)-obs['camera_extrinsic']).max()),
            camera_intrinsics_vs_obs_max_abs=float(abs(np.asarray(camcal.intrinsic)-obs['camera_intrinsic_Brown']).max()),
            camera_timing=times,TOP_pixelPose_RPY_range=[pp[...,:3].min((0,1)).tolist(),pp[...,:3].max((0,1)).tolist()],
            TOP_pixelPose_worldXYZ_range=[pp[...,3:].min((0,1)).tolist(),pp[...,3:].max((0,1)).tolist()],
            independent_TOP_returns=return_rows,observation=pin(op),original_bin=pin(lp),
            cache=pin(folder/'official_TOP_camera_projection_and_world.npz'))
        rows.append(row);write(folder/'frame_chain_receipt.json',row)
        print(json.dumps(dict(stage='targeted_record_complete',raw=raw,byteexact=exact,returns=return_rows,cache=row['cache'])),flush=True)
    assert [x['raw'] for x in rows]==TARGETS
    maxerror=max(max(x['independent_world_max_abs_xyz_m']) for r in rows for x in r['independent_TOP_returns'])
    result=dict(schema='R10.anomalous_frames_original_point_chain_audit.v1',complete=True,
        decision='PASS_EXPORT_BINDING_AND_INDEPENDENT_TOP_ARITHMETIC' if all(x['exported_bin_redecode_all6columns_byteexact'] for x in rows) and maxerror<.002 else 'REVIEW_REQUIRED',
        targets=TARGETS,rows=rows,source_code=sources,raw_reference=dict(path=receipt['raw'],bytes=receipt['raw_bytes'],sha256=receipt['raw_sha256'],whole_file_SHA_recomputed_here=False),
        selected_payload_CRC_and_SHA_verified=True,non_target_payloads_seeked_not_read_or_parsed=True,
        independent_math='float64 sensor polar + calibration extrinsic + SciPy extrinsic xyz Euler(pixelPose) -> world directly; no frame inverse/roundtrip and no decoder.rotation call',
        arithmetic_check_is_not_physical_pose_accuracy_validation=True,
        official_CP_is_original_range_camera_projection_not_recomputed_Brown=True,
        curb_or_pose_cause_still_undetermined=True,no_pose_or_point_changed=True,no_heldout_read=True,
        GPU_forward=0,backward=0,updates=0,seconds=time.time()-start)
    write(out/'point_chain_summary.json',result);print(json.dumps(dict(complete=True,receipt=pin(out/'point_chain_summary.json'),max_independent_world_abs_m=maxerror)),flush=True)


if __name__=='__main__':main()
