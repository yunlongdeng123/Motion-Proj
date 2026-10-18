"""Finite CPU raw annotation inventory; no extra observations enter training.

Check camera labels, projected labels and genuine LiDAR availability for all
candidate identities, especially the previously seedless track. Reading wider
source annotations does not itself authorize pretending they are49-frame seeds.
"""
from pathlib import Path
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
E=Path(__file__).resolve().parent;ROOT=E.parent;OUT=E/'full_segment_vehicle_measurements.json'
def main():
 assert not OUT.exists(),'preserve the completed inventory'
 sys.path.insert(0,str(ROOT));from waymo_numpy_cpu import protobuf,records
 pb=protobuf();prep=json.loads((ROOT/'inputs/scene0008_089_137/preparation_manifest.json').read_text());raw=prep['raw_path']
 cov=json.loads((E/'all_vehicle_fullres_coverage/coverage.json').read_text());tracks={v['track_id']:dict(actor=v['R15_actor'],laser_frames=[],projected_front_frames=[]) for v in cov['vehicles']}
 camera_frames=[]
 for index,payload,crc in records(raw,198):
  f=pb.Frame.FromString(payload)
  for lab in f.laser_labels:
   if lab.id in tracks:
    tracks[lab.id]['laser_frames'].append(dict(raw_frame=index,num_lidar_points=int(lab.num_lidar_points_in_box),box=dict(center_x=lab.box.center_x,center_y=lab.box.center_y,center_z=lab.box.center_z,length=lab.box.length,width=lab.box.width,height=lab.box.height,heading=lab.box.heading)))
  for cam in f.projected_lidar_labels:
   if cam.name==1:
    for lab in cam.labels:
     t=lab.id.removesuffix('_FRONT')
     if t in tracks:tracks[t]['projected_front_frames'].append(dict(raw_frame=index,box_xywh=[lab.box.center_x,lab.box.center_y,lab.box.length,lab.box.width]))
  if 89<=index<=137:
   labels=[]
   for cam in f.camera_labels:
    if cam.name!=1:continue
    for lab in cam.labels:
     if lab.type==1:labels.append(dict(raw_camera_track_id=lab.id,box_xywh=[lab.box.center_x,lab.box.center_y,lab.box.length,lab.box.width],laser_detection_difficulty=int(lab.detection_difficulty_level),tracking_difficulty=int(lab.tracking_difficulty_level)))
   camera_frames.append(dict(raw_frame=index,vehicle_camera_labels=labels))
  if index%25==0:print('raw',index,flush=True)
 result=dict(status='RAW_SOURCE_ANNOTATION_INVENTORY_ONLY',raw_path=raw,raw_frames_read=199,training_frames_unchanged=list(range(89,138)),additional_training_views=0,GPU_used=False,tracks=tracks,camera_frames=camera_frames,human_verdict=None)
 OUT.write_text(json.dumps(result,indent=2));print(json.dumps({t:dict(actor=v['actor'],positive_lidar_frames=sum(x['num_lidar_points']>0 for x in v['laser_frames']),front_projected_frames=len(v['projected_front_frames'])) for t,v in tracks.items() if v['actor'] is None}))
if __name__=='__main__':main()
