"""两条已授权候选的首记录Range，只读、累计≤50MB。"""
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
import urllib.request
import urllib.parse
from PIL import Image
sys.dont_write_bytecode=True
source=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_debug/followup_20261013/red_preparation/waymo_numpy_cpu.py')
spec=importlib.util.spec_from_file_location('original_waymo_numpy_cpu',source)
decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
pb=decoder.protobuf();crc=decoder.crc_function()
OUT=Path('/root/autodl-tmp/codex_tmp/horizon_waymo_inventory_20261031_r1/range_sorted_val')
OUT.mkdir(exist_ok=False)
REPO='AnnaZhang/waymo_open_dataset_v_1_4_3';REV='8a6fccd2b022077308c167a4d3c6f646d6e5e65b'
CAP=50*1024*1024;reserved=0;actual=0;requests=[];results=[]
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def get_range(url,start,end,total=None):
    global reserved,actual
    amount=end-start+1
    if reserved+amount>CAP:raise RuntimeError('50MiB cumulative budget would be exceeded')
    reserved+=amount;began=time.monotonic()
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}',
        'Accept-Encoding':'identity','User-Agent':'Motion-Proj-HorizonForge-source-trace/1.0'})
    row=dict(source_object_uri=url,range=[start,end],requested_bytes=amount,actual_body_bytes=0)
    requests.append(row)
    try:
        with opener.open(request,timeout=25) as response:
            value=response.headers.get('Content-Range','')
            row.update(http_status=response.status,content_range=value,
                final_host=urllib.parse.urlparse(response.geturl()).hostname)
            prefix=f'bytes {start}-{end}/'
            if response.status!=206 or not value.startswith(prefix):raise RuntimeError('Range response rejected before body read')
            size=int(value[len(prefix):])
            if total is not None and size!=total:raise RuntimeError('Full object size changed across ranges')
            if response.headers.get('Content-Length') and int(response.headers['Content-Length'])!=amount:raise RuntimeError('Range Content-Length mismatch')
            if response.headers.get('Content-Encoding','identity')!='identity':raise RuntimeError('Encoded response rejected')
            data=response.read(amount);actual+=len(data);row['actual_body_bytes']=len(data)
            if len(data)!=amount:raise RuntimeError('Short range')
            return data,size
    finally:row['wall_seconds']=time.monotonic()-began

for scene,context in [('086','17152649515605309595_3440_000_3460_000'),
                      ('004','10335539493577748957_1372_870_1392_870')]:
    relative='individual_files/validation/segment-'+context+'_with_camera_labels.tfrecord'
    url=f'https://hf-mirror.com/datasets/{REPO}/resolve/{REV}/{relative}'
    row=dict(scene_number=scene,context_id=context,source_repository=REPO,pinned_revision=REV,
        source_object_uri=url,source_relative_path=relative,credential_role='existing_public_mirror_access_no_new_credentials',
        full_raw_downloaded=False,full_raw_sha256_verified_here=False)
    started=time.monotonic()
    try:
        header,total=get_range(url,0,11)
        length,check=struct.unpack('<QI',header)
        if crc(header[:8])!=check:raise RuntimeError('Length CRC32C failed')
        if length+16>16*1024*1024:raise RuntimeError('First record exceeds per-candidate16MiB cap')
        rest,again=get_range(url,12,length+15,total)
        payload=rest[:-4]
        if crc(payload)!=struct.unpack('<I',rest[-4:])[0]:raise RuntimeError('Payload CRC32C failed')
        frame=pb.Frame.FromString(payload)
        if frame.context.name!=context:raise RuntimeError('protobuf context mismatch')
        front=next(image for image in frame.images if image.name==pb.CameraName.FRONT)
        folder=OUT/('sorted_val_'+scene);folder.mkdir()
        (folder/'front_000_original.jpg').write_bytes(bytes(front.image))
        image=Image.open(io.BytesIO(front.image)).convert('RGB')
        image.resize((576,384),Image.Resampling.LANCZOS).save(folder/'front_000_576x384.jpg',quality=93,subsampling=0)
        row.update(status='CRC_VERIFIED_FIRST_FRONT_RECORD',full_raw_expected_bytes=total,
            first_record_bytes=length+16,frame_index=0,timestamp_micros=frame.timestamp_micros,
            front_pose_timestamp=front.pose_timestamp,original_wh=list(image.size),
            length_crc32c='passed',payload_crc32c='passed',protobuf_context='passed',
            original_front_jpeg_bytes=len(front.image),original_jpeg=str(folder/'front_000_original.jpg'),
            analysis_jpeg=str(folder/'front_000_576x384.jpg'),horizonforge_match='pending_root_image_comparison')
        (folder/'source_receipt.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    except Exception as error:
        row.update(status='READ_FAILED_NO_FULL_DOWNLOAD',error_type=type(error).__name__,error=str(error).split('?')[0])
    row['wall_seconds']=time.monotonic()-started;results.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
    (OUT/'network_receipt.json').write_text(json.dumps(dict(schema='waymo_bounded_range_source_probe_v1',
        payload_budget_bytes=CAP,network_payload_bytes_reserved=reserved,network_payload_bytes_read=actual,
        candidates=results,requests=requests,full_segments_downloaded=0,gpu_calls=0,training_updates=0),ensure_ascii=False,indent=2)+'\n')
