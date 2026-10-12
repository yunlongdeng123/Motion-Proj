#!/usr/bin/env python3
"""Opt-in CPU exporter for FRONT/FRONT_LEFT/FRONT_RIGHT on a prepared paper scene.

Does not generate masks, change LiDAR, or launch training. Existing FRONT files
must exactly equal the corresponding raw RGB. New camera files are no-clobber.
"""
from pathlib import Path
import argparse,os,json,io,struct,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image
from waymo_open_dataset import dataset_pb2

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
a=p.parse_args();scene=a.scene.resolve()
assert scene.is_relative_to(Path('/root/autodl-tmp/drivingeditor_repro/data/waymo_paper_scenes'))
manifest=json.loads((scene/'preparation_manifest.json').read_text())
out=scene/'multicamera_preparation_manifest.json'
assert not out.exists(),'Multicamera preparation already exists; inspect rather than overwrite'
for i in (1,2):
    assert not (scene/f'image/image_{i}').exists()
    (scene/f'image/image_{i}').mkdir()
frames_by_index={row['source_index']:row for row in manifest['frames']}
report={'status':'exporting','scene':scene.name,'camera_indices':[0,1,2],
    'camera_enum_names':['FRONT','FRONT_LEFT','FRONT_RIGHT'],
    'camera_choice':'Declared reproduction choice; paperTableII says3cams but names are unspecified.',
    'raw_path':manifest['raw_path'],'raw_sha256':manifest.get('raw_sha256'),
    'model_time_rule':'Shared per-frame original preparation time; original per-camera exposure timestamp retained separately.',
    'loader_image_name':'frameid_cam{index}; source image filenames stay frameid.png in image_{index}',
    'frames':[],
    'pending':['Real sky masks for added cameras','Actual per-camera dynamic objects and shadow masks','Multi-view consistent physical actor point priors','CPU fullmulticamera loader and GPUintegration validation']}
out.write_text(json.dumps(report,indent=2))
with Path(manifest['raw_path']).open('rb') as stream:
    idx=0
    while True:
        header=stream.read(12)
        if not header:break
        size=struct.unpack('<Q',header[:8])[0]
        if idx not in frames_by_index:
            stream.seek(size+4,1);idx+=1;continue
        frame=dataset_pb2.Frame();frame.ParseFromString(stream.read(size));stream.read(4)
        row=frames_by_index[idx];record={'frame_id':row['id'],'source_index':idx,'timestamp_micros':frame.timestamp_micros,'model_time':row['model_time'],'cameras':[]}
        assert frame.timestamp_micros==row['timestamp_micros']
        for camera_index in (0,1,2):
            camera=next(x for x in frame.images if x.name==camera_index+1)
            image=Image.open(io.BytesIO(camera.image)).convert('RGB')
            target=scene/f'image/image_{camera_index}'/(row['id']+'.png')
            if camera_index==0:
                assert np.array_equal(np.array(image),np.array(Image.open(target).convert('RGB')))
            else:
                with target.open('xb') as h:image.save(h,format='PNG')
            record['cameras'].append({'camera_index':camera_index,'Waymo_enum':camera_index+1,'camera_pose_timestamp_seconds':camera.pose_timestamp,'image_size':list(image.size),'image_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
        report['frames'].append(record)
        print(json.dumps({'frame':row['id'],'cameras':3}),flush=True)
        idx+=1
        if len(report['frames'])==len(frames_by_index):break
assert len(report['frames'])==len(frames_by_index)
report['status']='rgb_calibration_pose_lidar_prepared; actual_extra_camera_masks_pending'
out.write_text(json.dumps(report,indent=2))
print(json.dumps({'status':report['status'],'frames':len(report['frames']),'cameras':3}),flush=True)
