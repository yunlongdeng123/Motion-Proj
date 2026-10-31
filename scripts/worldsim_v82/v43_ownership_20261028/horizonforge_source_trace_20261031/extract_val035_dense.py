"""从已存在Waymo035抽取原FRONT的稠密小图及原JPEG稀疏锚点。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import csv
import importlib.util
import io
import json
from pathlib import Path
import resource
import struct
import sys
import time
from PIL import Image
sys.dont_write_bytecode=True
source=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_debug/followup_20261013/red_preparation/waymo_numpy_cpu.py')
spec=importlib.util.spec_from_file_location('original_waymo_numpy_cpu',source)
decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
pb=decoder.protobuf();crc=decoder.crc_function()
context='13299463771883949918_4240_000_4260_000'
raw=Path('/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files/validation')/('segment-'+context+'_with_camera_labels.tfrecord')
output=Path('/root/autodl-tmp/codex_tmp/horizon_waymo_inventory_20261031_r1/sorted_val035_dense')
output.mkdir(exist_ok=False);(output/'front_576x384').mkdir();(output/'original_jpeg_anchors').mkdir()
records=[];start=time.monotonic();index=0;last_jpeg=None
with raw.open('rb') as stream:
    while True:
        offset=stream.tell();header=stream.read(12)
        if not header:break
        size,check=struct.unpack('<QI',header)
        if crc(header[:8])!=check or not 0<size<96*1024**2:raise ValueError('header CRC/layout')
        payload=stream.read(size);trailer=stream.read(4)
        if len(payload)!=size or len(trailer)!=4 or crc(payload)!=struct.unpack('<I',trailer)[0]:raise ValueError('payload CRC')
        frame=pb.Frame.FromString(payload)
        if frame.context.name!=context:raise ValueError('context identity mismatch')
        front=next(image for image in frame.images if image.name==pb.CameraName.FRONT)
        jpeg=bytes(front.image);last_jpeg=jpeg
        image=Image.open(io.BytesIO(jpeg)).convert('RGB')
        image.resize((576,384),Image.Resampling.LANCZOS).save(output/'front_576x384'/('front_%03d.jpg'%index),quality=93,subsampling=0)
        if index%20==0:(output/'original_jpeg_anchors'/('front_%03d.jpg'%index)).write_bytes(jpeg)
        records.append(dict(frame_index=index,timestamp_micros=frame.timestamp_micros,
            elapsed_seconds=(frame.timestamp_micros-records[0]['timestamp_micros'])/1e6 if records else 0.,
            front_pose_timestamp=front.pose_timestamp,source_header_offset=offset,
            source_front_jpeg_bytes=len(jpeg),original_wh=list(image.size),analysis_wh=[576,384],
            context_id=context,crc32c='passed'))
        index+=1
if last_jpeg:(output/'original_jpeg_anchors'/('front_%03d.jpg'%(index-1))).write_bytes(last_jpeg)
with (output/'frame_timestamps.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
manifest=dict(schema='waymo_original_front_dense_matching_copy_v1',candidate_list='root_waymo_val_list_zero_based_035',
    context_id=context,raw_path=str(raw),raw_file_bytes=raw.stat().st_size,frames=records,
    frame_count=len(records),last_frame_index=index-1,
    representation=dict(analysis_images='original FRONT RGB, Lanczos resize576x384, JPEG quality93 subsampling0',
        anchors='exact original JPEG bytes every20 frames plus final',no_generated_or_reconstructed_RGB=True),
    wall_seconds=time.monotonic()-start,max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    gpu_calls=0,training_updates=0,source_parser=str(source),
    horizonforge_identity='candidate_under_root_visual_comparison_not_auto_confirmed')
(output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(output=str(output),frames=len(records),last_frame_index=index-1,
    wall_seconds=manifest['wall_seconds'],bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())),ensure_ascii=False))
