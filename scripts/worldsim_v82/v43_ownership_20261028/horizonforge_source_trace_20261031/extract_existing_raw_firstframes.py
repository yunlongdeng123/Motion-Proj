import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
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
ROOT=Path('/root/autodl-tmp/codex_tmp/horizon_waymo_inventory_20261031_r1')
files=json.loads((ROOT/'all_raw_inventory.json').read_text())['files']
files=sorted((r for r in files if r['suffix'] in ('.tfrecord','.tfrecords') and r['context_id']),key=lambda r:r['context_id'])
OUT=ROOT/'existing_raw_firstframes';OUT.mkdir(exist_ok=False)
(OUT/'front_576x384').mkdir();(OUT/'original_front_jpeg').mkdir()
records=[];seen=set();started=time.monotonic();bytes_read=0
for row in files:
    context=row['context_id']
    if context in seen:continue
    seen.add(context);path=Path(row['path'])
    with path.open('rb') as stream:
        header=stream.read(12);size,check=struct.unpack('<QI',header)
        if crc(header[:8])!=check or not 0<size<96*1024**2:raise ValueError('header CRC/layout')
        payload=stream.read(size);trailer=stream.read(4);bytes_read+=len(header)+len(payload)+len(trailer)
        if len(payload)!=size or crc(payload)!=struct.unpack('<I',trailer)[0]:raise ValueError('payload CRC')
    frame=pb.Frame.FromString(payload)
    if frame.context.name!=context:raise ValueError('context identity mismatch')
    front=next(image for image in frame.images if image.name==pb.CameraName.FRONT)
    jpeg=bytes(front.image);image=Image.open(io.BytesIO(jpeg)).convert('RGB')
    original=OUT/'original_front_jpeg'/(context+'.jpg');original.write_bytes(jpeg)
    analysis=OUT/'front_576x384'/(context+'.jpg')
    image.resize((576,384),Image.Resampling.LANCZOS).save(analysis,quality=93,subsampling=0)
    records.append(dict(context_id=context,source_raw=str(path),raw_file_bytes=row['bytes'],
        frame_index=0,source_header_offset=0,timestamp_micros=frame.timestamp_micros,front_pose_timestamp=front.pose_timestamp,
        header_crc32c='passed',payload_crc32c='passed',protobuf_context='passed',first_record_bytes=size+16,
        original_front_jpeg=str(original),analysis_front_jpeg=str(analysis),original_wh=list(image.size),analysis_wh=[576,384],
        raw_firstframe_sensor_fields=dict(cameras=len(frame.images),camera_calibrations=len(frame.context.camera_calibrations),
            ego_pose_matrix_values=len(frame.pose.transform),lasers=len(frame.lasers),laser_labels=len(frame.laser_labels)),
        identity_role='deduplicated_existing_raw_context_not_list_number'))
    print(json.dumps(dict(completed=len(records),context_id=context),ensure_ascii=False),flush=True)
summary=dict(schema='existing_waymo_raw_firstframe_search_inputs_v1',context_count=len(records),frames=records,
    local_bytes_read=bytes_read,new_network_bytes=0,frame_sampling='one FRONT raw0 per existing unique TFRecord context only',
    wall_seconds=time.monotonic()-started,max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    gpu_calls=0,training_updates=0,model_or_GS_outputs_used=False)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(contexts=len(records),wall_seconds=summary['wall_seconds'],new_network_bytes=0),ensure_ascii=False))
