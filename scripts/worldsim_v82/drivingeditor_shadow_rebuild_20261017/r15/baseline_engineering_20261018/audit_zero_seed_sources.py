"""Read-only wider-source evidence for zero-seed camera vehicles.

Records real camera-track persistence and potential measured LiDAR aliases.
No extra frame, point or pose is added to training by this script.
"""
from pathlib import Path
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
E=Path(__file__).resolve().parent;ROOT=E.parent;OUT=E/'zero_seed_source_inventory.json'
def xyxy(b):return [b.center_x-b.length/2,b.center_y-b.width/2,b.center_x+b.length/2,b.center_y+b.width/2]
def iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-12)
def main():
 assert not OUT.exists(),'preserve completed source inventory'
 d=json.loads((E/'all_vehicle_camera33_candidates/inventory.json').read_text());missing={v['raw_camera_track_id']:dict(camera_actor_index=v['camera_actor_index'],observations=[]) for v in d['vehicles'] if not v['total_real_seed_points']}
 sys.path.insert(0,str(ROOT));from waymo_numpy_cpu import protobuf,records
 pb=protobuf();raw=json.loads((ROOT/'inputs/scene0008_089_137/preparation_manifest.json').read_text())['raw_path']
 for idx,payload,_ in records(raw,198):
  f=pb.Frame.FromString(payload);laser={l.id:l for l in f.laser_labels};projected=[l for c in f.projected_lidar_labels if c.name==1 for l in c.labels]
  for c in f.camera_labels:
   if c.name!=1:continue
   for l in c.labels:
    if l.id not in missing:continue
    bb=xyxy(l.box);matches=sorted([dict(raw_laser_track_id=p.id.removesuffix('_FRONT'),projected_box_iou=iou(bb,xyxy(p.box)),num_lidar_points=int(laser[p.id.removesuffix('_FRONT')].num_lidar_points_in_box) if p.id.removesuffix('_FRONT') in laser else None) for p in projected],key=lambda x:-x['projected_box_iou'])[:3]
    missing[l.id]['observations'].append(dict(raw_frame=idx,in_training_window=89<=idx<=137,actual_camera_box_xyxy=bb,possible_laser_aliases=matches))
 result=dict(status='ZERO_SEED_REAL_SOURCE_CANDIDATES_ONLY',training_frames_unchanged=True,additional_training_views=0,additional_initialization_points=0,vehicles=missing,human_verdict=None)
 OUT.write_text(json.dumps(result,indent=2));print(json.dumps({t:dict(camera_actor_index=v['camera_actor_index'],frames=len(v['observations']),best_positive_candidates=[r for r in v['observations'] if any(x['projected_box_iou']>.3 and x['num_lidar_points'] for x in r['possible_laser_aliases'])][:3]) for t,v in missing.items()}))
if __name__=='__main__':main()
