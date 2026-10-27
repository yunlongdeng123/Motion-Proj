"""First scene only, all available raw frames, CRC verified. CPU extraction.

Adds evidence; never changes original49 cameras/checkpoints or experiment split.
Full clip evidence is transductive, not independent held-out reconstruction.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,struct,sys,time
import numpy as np
from PIL import Image
from waymo_open_dataset import dataset_pb2 as pb
from google.protobuf.json_format import MessageToDict
sys.path.insert(0,'/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
import waymo_numpy_cpu as decoder
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v42_20261024/fullclip_20261026')
from fast_range_matrix import matrix
decoder.matrix=matrix
CTX='14705303724557273004_3105_000_3125_000';OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027');R=OLD.parent/'v43_ownership_20261028';D=R/'target_unknown_v2/fullclip_inputs'

def main():
    D.mkdir(exist_ok=False)
    for name in ['rgb','velodyne','calib','pose','annotations']:(D/name).mkdir()
    original=json.loads((OLD/'motion_scene_inputs_v1'/CTX/'sensor_preparation.json').read_text());crc=decoder.crc_function();records=[];started=time.monotonic()
    with Path(original['source_raw']).open('rb') as stream:
        raw=0
        while True:
            header=stream.read(12)
            if not header:break
            size,check=struct.unpack('<QI',header);assert crc(header[:8])==check;payload=stream.read(size);assert crc(payload)==struct.unpack('<I',stream.read(4))[0];f=pb.Frame.FromString(payload);assert f.context.name==CTX
            fid='9001%03d'%raw;rgb,_=decoder.front_rgb(f);Image.fromarray(rgb).resize((480,320),Image.Resampling.LANCZOS).save(D/'rgb'/('%03d.png'%raw))
            points,detail=decoder.lidar(f,pb);points.tofile(D/'velodyne'/(fid+'.bin'));(D/'calib'/(fid+'.txt')).write_text(decoder.calibration_text(f));np.savetxt(D/'pose'/(fid+'.txt'),np.array(f.pose.transform).reshape(4,4))
            labels=[MessageToDict(x,preserving_proto_field_name=True) for x in f.laser_labels];(D/'annotations'/(fid+'.json')).write_text(json.dumps(labels))
            records.append(dict(raw=raw,frame_id=fid,timestamp_micros=f.timestamp_micros,lidar_points=len(points),CRC_verified=True));raw+=1
            (D/'status.json').write_text(json.dumps(dict(phase='extracting',frames=raw,pid=os.getpid())));print('FULLCLIP',raw,flush=True)
    (D/'manifest.json').write_text(json.dumps(dict(context=CTX,frames=records,source_raw=original['source_raw'],target_track=original['target_track'],observed_RGB_size=[480,320],calibration_image_size=[1920,1280],world_pose_unchanged=True,transductive_evidence_not_new_test_split=True,wall_seconds=time.monotonic()-started),indent=2));(D/'status.json').write_text(json.dumps(dict(phase='complete',frames=len(records))))

if __name__=='__main__':main()
