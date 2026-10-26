"""Read-only source-index/world/time/annotated-box consistency; no GT filtering."""
from pathlib import Path
import json,numpy as np
from plyfile import PlyData
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
rows=[]
for scene in sorted((ROOT/'motion_scene_inputs_v1').iterdir()):
    if not scene.is_dir():continue
    if not (scene/'native_preparation/selection_reviewed.json').exists() or not (scene/'dynamic_actor_assets/actor_0/source_lidar_indices.npz').exists():continue
    chosen=json.loads((scene/'native_preparation/selection_reviewed.json').read_text());track=chosen['track']
    sensor=json.loads((scene/'sensor_preparation.json').read_text());indices=np.load(scene/'dynamic_actor_assets/actor_0/source_lidar_indices.npz')
    ply=PlyData.read(str(scene/'points3d_dynamic_0.ply'))['vertex'];offset=0;frames=[];inside={str(m):0 for m in [0.,.1,.25]};total=0;max_error=0.
    for index,frame in enumerate(sensor['frames']):
        fid=frame['frame_id'];raw=np.fromfile(scene/'velodyne'/(fid+'.bin'),dtype=np.float32).reshape(-1,6)
        ids=indices[fid];assert ids.dtype==np.int32 and len(ids)==len(np.unique(ids)) and (ids>=0).all() and (ids<len(raw)).all()
        xyz=raw[ids,:3].astype(np.float64);pose=np.loadtxt(scene/'pose'/(fid+'.txt'))
        expected=(np.column_stack((xyz,np.ones(len(xyz))))@pose.T)[:,:3].astype(np.float32)
        actual=np.column_stack([ply[k][offset:offset+len(ids)] for k in ['x','y','z']]);error=float(np.max(np.abs(expected-actual))) if len(ids) else 0.;max_error=max(max_error,error);assert error==0.
        t=np.float32(index*.02-.48)
        for key in ['nx','ny','nz','time']:assert np.all(ply[key][offset:offset+len(ids)]==t)
        labels=json.loads((scene/'annotations'/(fid+'.json')).read_text())['laser_labels'];b=next(x for x in labels if x['id']==track)['box']
        c,s=np.cos(b['heading']),np.sin(b['heading']);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        q=(xyz-np.array([b['center_x'],b['center_y'],b['center_z']]))@R;half=np.array([b['length'],b['width'],b['height']])/2
        counts={str(m):int((np.abs(q)<=half+m).all(1).sum()) for m in [0.,.1,.25]}
        for m,n in counts.items():inside[m]+=n
        frames.append(dict(frame_id=fid,points=len(ids),inside_annotated_box_counts=counts,world_replay_max_abs_m=error,below_box_bottom_01m=int((q[:,2]<-half[2]-.1).sum())))
        total+=len(ids);offset+=len(ids)
    assert offset==len(ply) and total>0
    result=dict(context=scene.name,frames=frames,total_dynamic_points=total,all_source_indices_valid_unique=True,world_replay_max_abs_m=max_error,model_time_channels_match=True,inside_annotated_box_fraction={k:v/total for k,v in inside.items()},ground_or_background_semantic_purity_proven=False,labels_used_to_filter_points=False,training_modified=False,interpretation='Annotated-box agreement is a geometric consistency diagnostic, not semantic surface truth or segmentation GT.',quality_accepted=False,human_verdict=None)
    (scene/'native_preparation/actor_source_audit.json').write_text(json.dumps(result,indent=2));rows.append({k:v for k,v in result.items() if k!='frames'})
(ROOT/'native_actor_source_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
