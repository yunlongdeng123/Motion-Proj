"""仅修正颜色FOV外推有效性；原量测坐标与来源行逐项不变。"""
import json,pathlib,shutil,time
import build_media as m
import numpy as np
from add_frame_point_views import run as frame_views

def run(root):
    root=pathlib.Path(root);media=json.loads((root/'media.json').read_text(encoding='utf8'));context=media['context_id'];backup=root/'color_diagnostics_r1';backup.mkdir(exist_ok=True)
    model=media['projection']['source_calibration'].copy();ext=np.array(model['extrinsic']).reshape(4,4);frames={f['raw']:f for f in media['frames']}
    for f in frames.values():f['cam_from_world']=np.linalg.inv(np.array(f['camera_pose_world']).reshape(4,4)@ext)
    changed=0;oldvalid=0;newvalid=0;counts=[];start=time.perf_counter()
    for track,actor in media['actors'].items():
        base='actor_'+actor['alias'];path=root/'lidar'/(base+'.npz');original=np.load(path);data={k:original[k].copy() for k in original.files};original.close()
        oldvalid+=int(data['front_color_valid'].sum());removed=0
        for pose in actor['trajectory']:
            raw=pose['raw'];lo,hi=np.searchsorted(data['raw_frame'],[raw,raw+1]);xyz=data['actor_local_xyz'][lo:hi]
            world=xyz@np.array(pose['orientation_world']).reshape(3,3).T+pose['center_world'];cam=m.transform(world,frames[raw]['cam_from_world'])
            valid=m.pinhole_frustum_valid(cam,model);bad=data['front_color_valid'][lo:hi]&~valid;removed+=int(bad.sum())
            data['front_color_valid'][lo:hi][bad]=False;data['rgb'][lo:hi][bad]=np.array([155,180,205],np.uint8)
        changed+=removed;newvalid+=int(data['front_color_valid'].sum());counts.append(dict(track=track,alias=actor['alias'],invalid_color_rows_grayed=removed))
        for suffix in ('.npz','.js','.jpg','_frames.js'):
            source=root/'lidar'/(base+suffix)
            if source.exists() and not (backup/source.name).exists():shutil.copy2(source,backup/source.name)
        np.savez_compressed(path,**data)
        # Bytewise equality of all geometry and original row identities is an explicit invariant.
        with np.load(backup/path.name) as old:
            for k in ('actor_local_xyz','raw_frame','source_cloud_row'):assert np.array_equal(data[k],old[k])
        xyz=data['actor_local_xyz'];_,keep=np.unique(np.floor(xyz/.06).astype(np.int32),axis=0,return_index=True) if len(xyz) else (None,np.empty(0,int));keep.sort()
        if len(keep)>12000:keep=keep[np.linspace(0,len(keep)-1,12000,dtype=int)]
        script=(root/'lidar'/(base+'.js')).read_text(encoding='utf8');pt=json.loads(script.split(']=',1)[1].rstrip(';'))
        pt['rgb']=data['rgb'][keep].tolist();pt['front_color_valid']=data['front_color_valid'][keep].astype(int).tolist()
        pt['color_role']='projected_FRONT_RGB_with_finite_pinhole_frustum_without_visibility_zbuffer_else_gray'
        (root/'lidar'/(base+'.js')).write_text('window.ACTOR_LIDAR=window.ACTOR_LIDAR||{};window.ACTOR_LIDAR['+json.dumps(context+'|'+track)+']='+json.dumps(pt,separators=(',',':'))+';',encoding='utf8')
        m.cloud_preview(root/'lidar'/(base+'.jpg'),xyz[keep],data['rgb'][keep],f"{media['scene_key']} {actor['alias']} | {len(xyz)} measured rows / {len(keep)} display voxels | actor local meters",None)
    for f in frames.values():del f['cam_from_world']
    media['lidar_contract']['color_projection']='finite pinhole frustum before Brown distortion; no visibility zbuffer; invalid colors gray'
    m.dump(root/'media.json',media);(root/'media.js').write_text('window.ACTOR_MEDIA=window.ACTOR_MEDIA||{};window.ACTOR_MEDIA['+json.dumps(context)+']='+json.dumps(media,ensure_ascii=False,separators=(',',':'))+';',encoding='utf8')
    frame_views(root)
    summary=dict(schema='color_frustum_fix_v1',old_valid_colors=oldvalid,new_valid_colors=newvalid,invalid_color_rows_grayed=changed,
                 point_geometry_source_identity_unchanged=True,actors=counts,wall_seconds=time.perf_counter()-start)
    m.dump(root/'color_frustum_regression.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='actors'}))
if __name__=='__main__':run(m.BASE/'generated/scenes/13299463771883949918_4240_000_4260_000')
