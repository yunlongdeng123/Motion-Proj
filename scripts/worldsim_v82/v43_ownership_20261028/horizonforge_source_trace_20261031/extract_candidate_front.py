"""仅从已有TFRecord抽取少量原始FRONT JPEG；跳过其他payload，不读LiDAR。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import time
from PIL import Image
sys.dont_write_bytecode=True
decoder_path=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_debug/followup_20261013/red_preparation/waymo_numpy_cpu.py')
spec=importlib.util.spec_from_file_location('original_waymo_numpy_cpu',decoder_path)
decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
pb=decoder.protobuf();crc=decoder.crc_function()
root=Path('/root/autodl-tmp')
output=root/'codex_tmp/horizon_waymo_inventory_20261031_r1/raw_candidates'
output.mkdir(exist_ok=False)
scenes=[('dggt_example_086','training','4295449061847708198_3769_000_3789_000'),
        ('dggt_val_079','validation','13469905891836363794_4429_660_4449_660'),
        ('sorted_val_035','validation','13299463771883949918_4240_000_4260_000')]
requested={0,45,89,120,198};summary=[]
for label,split,context in scenes:
    path=root/'datasets/waymo_v1_4_3/individual_files'/split/('segment-'+context+'_with_camera_labels.tfrecord')
    folder=output/label;folder.mkdir();records=[];start=time.monotonic();skipped=0;read_bytes=0
    with path.open('rb') as stream:
        for index in range(max(requested)+1):
            offset=stream.tell();header=stream.read(12)
            if not header:break
            if len(header)!=12:raise ValueError('partial TFRecord header')
            length,check=struct.unpack('<QI',header)
            if crc(header[:8])!=check or not 0<length<96*1024**2:raise ValueError('TFRecord header CRC/layout')
            if index not in requested:
                stream.seek(length+4,1);skipped+=length;continue
            payload=stream.read(length);trailer=stream.read(4);read_bytes+=length
            if len(payload)!=length or len(trailer)!=4 or crc(payload)!=struct.unpack('<I',trailer)[0]:raise ValueError('selected TFRecord CRC')
            frame=pb.Frame.FromString(payload)
            if frame.context.name!=context:raise ValueError('context mismatch')
            front=next(image for image in frame.images if image.name==pb.CameraName.FRONT)
            jpeg=bytes(front.image);image=Image.open(io.BytesIO(jpeg))
            destination=folder/('front_%03d.jpg'%index);destination.write_bytes(jpeg)
            records.append(dict(frame_index=index,path=str(destination),jpeg_bytes=len(jpeg),
                original_wh=list(image.size),frame_timestamp_micros=frame.timestamp_micros,
                front_pose_timestamp=front.pose_timestamp,source_header_offset=offset,
                source_payload_bytes=length,header_crc32c='passed',payload_crc32c='passed',
                original_front_jpeg_bytes_unchanged=True))
    receipt=dict(schema='waymo_original_front_candidate_samples_v1',candidate_label=label,
        context_id=context,raw_path=str(path),raw_file_bytes=path.stat().st_size,
        source_camera='FRONT',requested_frames=sorted(requested),frames=records,
        bytes_read_selected_payload=read_bytes,bytes_skipped_by_seek=skipped,
        wall_seconds=time.monotonic()-start,source_parser=str(decoder_path),
        horizonforge_identity='unverified_candidate_requires_visual_match',
        gpu_calls=0,training_updates=0)
    (folder/'manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    summary.append(receipt);print(json.dumps(dict(label=label,frames=len(records),wall_seconds=receipt['wall_seconds']),ensure_ascii=False),flush=True)
(output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
