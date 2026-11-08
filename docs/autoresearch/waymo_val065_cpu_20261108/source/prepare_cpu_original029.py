"""Export CRC-checked non-held-out Waymo observations, with all CUDA access disabled.

Reuses the previously validated five-LiDAR NumPy decoder and official-format
calibration writer. Produces source data only. Instance/sky segmentation and
static/dynamic initialization remain explicit later stages.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
os.environ['PYTHONDONTWRITEBYTECODE']='1'

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--media',type=Path,required=True)
    ap.add_argument('--decoder-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    selection=json.loads(args.selection.read_text(encoding='utf8'))
    media=json.loads(args.media.read_text(encoding='utf8'))
    frames={f['raw']:f for f in media['frames']}
    build=set(selection['build']);holdout=set(selection['holdout'])
    assert not build&holdout
    assert build|holdout==set(frames)
    assert media['context_id']==selection['context_id']
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['calib','pose','timestamp','image/image_0','rgb_preview','velodyne','observations','prompts']:
        (args.output/name).mkdir(parents=True)
    sys.path.insert(0,str(args.decoder_root))
    import waymo_numpy_cpu as decoder
    from fast_range_matrix import matrix
    decoder.matrix=matrix
    pb=decoder.protobuf()
    start=time.monotonic();stat_before=args.raw.stat()
    receipt=dict(schema='waymo.generic.cpu_observations.v1',context_id=selection['context_id'],
        status='EXTRACTING_CPU',raw=str(args.raw),raw_bytes=stat_before.st_size,
        raw_sha256=digest(args.raw),source_frame_count=len(frames),build_raws=sorted(build),holdout_raws=sorted(holdout),
        source_code_sha256=digest(Path(__file__)),selection_sha256=digest(args.selection),media_sha256=digest(args.media),
        decoder_sha256=digest(args.decoder_root/'waymo_numpy_cpu.py'),
        matrix_decoder_sha256=digest(args.decoder_root/'fast_range_matrix.py'),
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CPU_threads=1,GPU_queries=0,
        training_updates=0,render_calls=0,frames=[],skipped_holdout=[],readiness='CPU observations only',
        native_file_prefix='0029',model_frame_interval=.02,
        time_policy='model_time=(original_raw-(source_frame_count-1)/2)*0.02; holdout gaps preserved',
        limitations=['Official-format pinhole calibration ignores Brown distortion; full Brown coefficients and image pose saved separately.',
            'Within-scan actor motion is not compensated. TOP per-pixel ego-motion compensation retained.',
            'Projected boxes are prompts, not instance masks. No masks or point-cloud initialization are fabricated.',
            'Current scene008 runtime requires generic input/asset binding before new-scene GPU training.'])
    write(args.output/'receipt.json',receipt)
    crc_records=[];prompt_total=0;point_total=0
    for raw,payload,record in decoder.records(args.raw,len(frames)-1):
        if raw in holdout:
            receipt['skipped_holdout'].append(raw)
            continue
        frame=pb.Frame.FromString(payload)
        assert frame.context.name==selection['context_id']
        assert frame.timestamp_micros==frames[raw]['timestamp_micros']
        name=f'0029{raw:03d}'
        cloud,sensors=decoder.lidar(frame,pb)
        assert cloud.shape[1]==6 and np.isfinite(cloud).all() and len(cloud)>0
        cloud.astype('<f4').tofile(args.output/'velodyne'/f'{name}.bin')
        front=next(i for i in frame.images if i.name==1)
        with Image.open(io.BytesIO(front.image)) as im:
            im.convert('RGB').save(args.output/'image/image_0'/f'{name}.png')
            im.convert('RGB').resize((480,320),Image.Resampling.BILINEAR).save(args.output/'rgb_preview'/f'{raw:03d}.jpg',quality=94)
        (args.output/'calib'/f'{name}.txt').write_text(decoder.calibration_text(frame),encoding='ascii')
        pose=np.array(frame.pose.transform).reshape(4,4)
        np.savetxt(args.output/'pose'/f'{name}.txt',pose,fmt='%.15e')
        (args.output/'timestamp'/f'{name}.txt').write_text(str(frame.timestamp_micros)+'\n',encoding='ascii')
        cal=next(c for c in frame.context.camera_calibrations if c.name==1)
        observation=dict(raw=raw,timestamp_micros=frame.timestamp_micros,
            ego_world=pose.ravel().tolist(),camera_pose_world=list(front.pose.transform),
            camera_extrinsic=list(cal.extrinsic.transform),camera_intrinsic_Brown=list(cal.intrinsic),
            camera_size=[cal.width,cal.height],boxes=frames[raw]['boxes'],
            lidar_path=f'velodyne/{name}.bin',image_path=f'image/image_0/{name}.png',
            lidar_coordinates='same-frame ego vehicle coordinates, five sensors/two returns, NLZ filtered')
        write(args.output/'observations'/f'{raw:03d}.json',observation)
        prompts=[]
        for box in frames[raw]['boxes']:
            points=np.array([p for p in box['corners2D'] if p is not None])
            if not box['official_FRONT_visible'] or len(points)<3:
                continue
            lo=np.maximum(points.min(axis=0),[0,0]);hi=np.minimum(points.max(axis=0),[959,639])
            if np.any(hi<=lo):
                continue
            prompts.append(dict(track_id=box['track'],alias=media['actors'][box['track']]['alias'],
                bbox_xyxy_960x640=np.r_[lo,hi].round(3).tolist(),
                bbox_xyxy_native=(np.r_[lo,hi]*np.array([cal.width/960,cal.height/640]*2)).round(3).tolist(),
                world_from_actor=box['world_from_actor'],dimensions=box['dimensions'],
                role='projected annotation box prompt; target silhouette and identity need segmentation QA'))
        write(args.output/'prompts'/f'{raw:03d}.json',prompts)
        paths=[args.output/'velodyne'/f'{name}.bin',args.output/'image/image_0'/f'{name}.png',
               args.output/'calib'/f'{name}.txt',args.output/'pose'/f'{name}.txt']
        row=dict(raw=raw,id=name,timestamp_micros=frame.timestamp_micros,
            model_time=(raw-(len(frames)-1)/2)*.02,lidar_points=len(cloud),prompt_count=len(prompts),
            sensor_return_count=len(sensors),files={str(p.relative_to(args.output)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths})
        receipt['frames'].append(row);prompt_total+=len(prompts);point_total+=len(cloud)
        crc_records.append(dict(raw=raw,timestamp_micros=frame.timestamp_micros,**record))
        if len(receipt['frames'])%10==0:
            write(args.output/'receipt.json',receipt)
            print(json.dumps(dict(completed=len(receipt['frames']),build_total=len(build),raw=raw,seconds=round(time.monotonic()-start,1))),flush=True)
    after=args.raw.stat()
    assert (after.st_size,after.st_mtime_ns)==(stat_before.st_size,stat_before.st_mtime_ns)
    assert {r['raw'] for r in receipt['frames']}==build
    assert set(receipt['skipped_holdout'])==holdout
    assert not any(k=='torch' or k.startswith('tensorflow') for k in sys.modules)
    receipt.update(status='PASS_CPU_OBSERVATIONS',build_frame_count=len(build),holdout_count=len(holdout),
        measured_points=point_total,prompt_count=prompt_total,raw_CRC='passed for all records including skipped timestamps',
        source_unchanged=True,wall_seconds=time.monotonic()-start,
        next_requires_GPU='instance segmentation and sky masks, followed by real-environment lifecycle smoke')
    write(args.output/'receipt.json',receipt)
    write(args.output/'source_records.json',crc_records)
    write(args.output/'track_registry.json',dict(context_id=selection['context_id'],actors={t:dict(alias=a['alias']) for t,a in media['actors'].items()}))
    print(json.dumps({k:receipt[k] for k in ['status','build_frame_count','holdout_count','measured_points','prompt_count','wall_seconds','GPU_queries']}),flush=True)


if __name__=='__main__':
    main()
