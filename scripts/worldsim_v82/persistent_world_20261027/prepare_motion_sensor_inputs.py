"""筛查后优先准备三个运动编辑开发路段；复用既有真实Waymo解码数学。

只提取传感器/标注与投影QA，不生成占位mask，不训练，不修改①②③算法。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import concurrent.futures as futures
import json,struct,sys
import numpy as np
from PIL import Image
# 先加载data_cpu环境中的快速protobuf；后续只复用原有NumPy LiDAR数学。
from waymo_open_dataset import dataset_pb2 as pb
sys.path.insert(0,'/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
import waymo_numpy_cpu as decoder
FAST=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v42_20261024/fullclip_20261026')
sys.path.insert(0,str(FAST))
from fast_range_matrix import matrix
decoder.matrix=matrix

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
CHOSEN=['14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000','473735159277431842_630_095_650_095']


def write(p,d):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)


def worker(scene,local_index):
    key=scene['context'];folder=ROOT/'motion_scene_inputs_v1'/key;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'sensor_preparation.json').exists():return json.loads((folder/'sensor_preparation.json').read_text())
    for name in ['image/image_0','velodyne','calib','pose','timestamp','annotations','projection_QA','frame_receipts']:(folder/name).mkdir(parents=True,exist_ok=True)
    selected=set(scene['selected_window']);rows=[];crc=decoder.crc_function()
    from google.protobuf.json_format import MessageToDict
    with Path(scene['raw_path']).open('rb') as stream:
        for raw in range(max(selected)+1):
            header=stream.read(12);length,check=struct.unpack('<QI',header);assert crc(header[:8])==check
            if raw not in selected:stream.seek(length+4,1);continue
            payload=stream.read(length);assert crc(payload)==struct.unpack('<I',stream.read(4))[0]
            frame=pb.Frame.FromString(payload);assert frame.context.name==key
            fid=f'{local_index:04d}{raw:03d}';receipt=folder/'frame_receipts'/(fid+'.json')
            if receipt.exists():rows.append(json.loads(receipt.read_text()));continue
            rgb,_=decoder.front_rgb(frame);Image.fromarray(rgb).save(folder/'image/image_0'/(fid+'.png'))
            cloud,details=decoder.lidar(frame,pb);cloud.tofile(folder/'velodyne'/(fid+'.bin'))
            calibration=decoder.calibration_text(frame);(folder/'calib'/(fid+'.txt')).write_text(calibration)
            pose=np.array(frame.pose.transform).reshape(4,4);np.savetxt(folder/'pose'/(fid+'.txt'),pose)
            (folder/'timestamp'/(fid+'.txt')).write_text(str(frame.timestamp_micros))
            labels=dict(frame_pose_world_from_vehicle=list(frame.pose.transform),
                        laser_labels=[MessageToDict(x,preserving_proto_field_name=True) for x in frame.laser_labels],
                        selected_track=scene['selected_track'],target_box_used_for_initialization=False)
            write(folder/'annotations'/(fid+'.json'),labels)
            fields={x.split(':')[0]:np.fromstring(x.split(':')[1],sep=' ') for x in calibration.splitlines()}
            K=fields['P0'].reshape(3,4)[:,:3].copy();K[:2]/=4;T=np.eye(4);T[:3]=fields['Tr_velo_to_cam_0'].reshape(3,4)
            cam=cloud[:,:3]@T[:3,:3].T+T[:3,3];uv=(cam@K.T)[:,:2]/np.maximum(cam[:,2:],1e-8)-.5
            ij=np.floor(uv+.5).astype(np.int64);valid=(cam[:,2]>.1)&(ij[:,0]>=0)&(ij[:,0]<480)&(ij[:,1]>=0)&(ij[:,1]<320)
            if raw in [scene['selected_window'][i] for i in [0,24,48]]:
                base=np.array(Image.fromarray(rgb).resize((480,320),Image.Resampling.LANCZOS));ids=np.flatnonzero(valid)
                depth=cam[ids,2];order=np.argsort(depth)[::-1];ids=ids[order];v=np.clip(cam[ids,2]/60,0,1)
                colors=np.c_[(1-v)*255,100*np.ones(len(ids)),v*255].astype(np.uint8)
                base[ij[ids,1],ij[ids,0]]=colors
                Image.fromarray(base).save(folder/'projection_QA'/f'{raw:03d}.png')
            data=dict(frame_id=fid,raw=raw,timestamp_micros=frame.timestamp_micros,lidar_points=len(cloud),
                      front_projected_returns=int(valid.sum()),image_size=[rgb.shape[1],rgb.shape[0]],returns=details,crc_verified=True)
            write(receipt,data);rows.append(data)
            print('SENSOR_INPUT',local_index,raw,len(cloud),flush=True)
    result=dict(context=key,local_scene_index=local_index,index_provenance='assigned evaluation-local ID, not author scene mapping',
                source_raw=scene['raw_path'],frames=rows,target_track=scene['selected_track'],
                selection='new diagnostic development; passed raw checks; target continuous49, moving, calibrated correspondences; no editing outputs used',
                outputs='real RGB, five-LiDAR/two-return points with TOP motion compensation, calibration, poses, timestamps, labels',
                pending=['pixel masks and independent target association','native observed-region reconstruction QA','asset orientation/time coverage','destination/camera 3D support','source/evaluation split and long-run smoke'],
                placeholders_created=False,target_GT_boxes_filter_initialization=False,training_ready=False,
                new_training_updates=0,GPU_used=False,quality_accepted=False,human_verdict=None)
    write(folder/'sensor_preparation.json',result);return result


def main():
    candidates=json.loads((ROOT/'evaluation_set_v1/manifest.json').read_text())['scenes']
    registry=json.loads((ROOT/'evaluation_admission_v1/registry.json').read_text())
    screens={r['context']:r for r in registry['scenes']};selected=[next(s for s in candidates if s['context']==k) for k in CHOSEN]
    for s in selected:
        q=screens[s['context']];assert q['split']=='new_diagnostic_development' and q['raw_input_status']=='raw_checks_passed' and q['operations']['change_speed']['raw_data_eligible']
    destination=ROOT/'motion_scene_inputs_v1';destination.mkdir(exist_ok=True)
    results=[];errors=[]
    with futures.ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(worker,s,9001+i) for i,s in enumerate(selected)]
        for future in futures.as_completed(jobs):
            try:results.append(future.result())
            except Exception as error:errors.append(repr(error))
            write(destination/'status.json',dict(phase='sensor_extraction',completed=len(results),total=3,errors=errors))
    assert not errors,errors
    write(destination/'status.json',dict(phase='sensor_inputs_prepared',completed=3,total=3,training_ready=False))


if __name__=='__main__':main()
