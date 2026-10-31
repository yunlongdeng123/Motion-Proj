"""原始 Waymo 片段的 CPU 媒体与实测 LiDAR 诊断；不加载 GS、不写源文件。

车体坐标 x向前/y向左/z向上，原点为各帧原始3D框中心。点云是五LiDAR、
双return、TOP逐像素运动补偿及NLZ过滤后的真实量测；框内筛点不等同语义分割。
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import argparse,io,json,math,pathlib,struct,subprocess,sys,time,hashlib
BASE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'vendor'),str(BASE/'pb'),str(BASE)]
import numpy as np
import google_crc32c
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from waymo_open_dataset import dataset_pb2 as pb
import waymo_numpy_cpu as decoder
from fast_range_matrix import matrix
decoder.matrix=matrix

EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
SIGNS=np.array([[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1],[1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1]],dtype=np.float64)
W,H=960,640

def dump(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf8')

def masked(data):
    value=google_crc32c.value(data)
    return (((value>>15)|(value<<17))+0xa282ead8)&0xffffffff

def records(path,limit=None):
    with pathlib.Path(path).open('rb') as f:
        index=0
        while limit is None or index<limit:
            offset=f.tell();header=f.read(12)
            if not header:break
            if len(header)!=12:raise ValueError('truncated TFRecord header')
            length,crc=struct.unpack('<QI',header)
            if masked(header[:8])!=crc or not 0<length<96*1024**2:raise ValueError('length CRC')
            payload=f.read(length);tail=f.read(4)
            if len(payload)!=length or len(tail)!=4:raise ValueError('truncated TFRecord payload')
            if masked(payload)!=struct.unpack('<I',tail)[0]:raise ValueError('payload CRC')
            yield index,pb.Frame.FromString(payload),dict(offset=offset,payload_bytes=length,length_crc32c='passed',payload_crc32c='passed')
            index+=1

def rz(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float64)

def transform(points,T):return points@T[:3,:3].T+T[:3,3]

def camera_model(frame,front):
    c=next(x for x in frame.context.camera_calibrations if x.name==1)
    ego=np.array(frame.pose.transform).reshape(4,4)
    image_pose=np.array(front.pose.transform).reshape(4,4) if len(front.pose.transform)==16 else ego
    ext=np.array(c.extrinsic.transform).reshape(4,4)
    return dict(intrinsic=list(c.intrinsic),width=c.width,height=c.height,extrinsic=ext,image_pose=image_pose,
                camera_from_world=np.linalg.inv(image_pose@ext),rolling_shutter_direction=int(c.rolling_shutter_direction),
                pose_timestamp=float(front.pose_timestamp),trigger=float(front.camera_trigger_time),readout=float(front.camera_readout_done_time))

def project_camera(cam,m):
    # Waymo camera: x forward, y left, z up. Brown-Conrady intrinsic order is
    # fx,fy,cx,cy,k1,k2,p1,p2,k3. Pixel centers retain original calibration convention.
    depth=cam[:,0];safe=np.maximum(depth,1e-5)
    x=-cam[:,1]/safe;y=-cam[:,2]/safe
    fx,fy,cx,cy,k1,k2,p1,p2,k3=m['intrinsic']
    r2=x*x+y*y;rad=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    xd=x*rad+2*p1*x*y+p2*(r2+2*x*x)
    yd=y*rad+p1*(r2+2*y*y)+2*p2*x*y
    uv=np.column_stack(((fx*xd+cx)*W/m['width'],(fy*yd+cy)*H/m['height']))
    return uv,depth

def pinhole_frustum_valid(cam,m):
    dep=cam[:,0];safe=np.maximum(dep,1e-5);fx,fy,cx,cy=m['intrinsic'][:4]
    u=-fx*cam[:,1]/safe+cx;v=-fy*cam[:,2]/safe+cy
    return (dep>.1)&(u>=-.15*m['width'])&(u<=1.15*m['width'])&(v>=-.15*m['height'])&(v<=1.15*m['height'])

def clip_segment(p0,p1):
    # Liang-Barsky image clipping avoids unbounded near-plane projections in JSON/PIL.
    dx,dy=p1-p0;t0,t1=0.,1.
    for p,q in ((-dx,p0[0]),(dx,W-1-p0[0]),(-dy,p0[1]),(dy,H-1-p0[1])):
        if abs(p)<1e-12:
            if q<0:return None
        else:
            r=q/p
            if p<0:t0=max(t0,r)
            else:t1=min(t1,r)
            if t0>t1:return None
    a=p0+t0*(p1-p0);b=p0+t1*(p1-p0)
    return np.r_[a,b].round(3).tolist()

def clip_camera_segment(a,b,m):
    # Clip BEFORE Brown distortion. Near-plane-only clipping can turn side/behind
    # rays back into the image when a high-degree radial polynomial is extrapolated.
    fx,fy,cx,cy=m['intrinsic'][:4];w,h=m['width'],m['height'];mx,my=w*.15,h*.15
    planes=[(np.array([1.,0,0]),-.1),(np.array([cx+mx,-fx,0]),0.),
            (np.array([w+mx-cx,fx,0]),0.),(np.array([cy+my,0,-fy]),0.),(np.array([h+my-cy,0,fy]),0.)]
    t0,t1=0.,1.;delta=b-a
    for normal,bias in planes:
        q=float(normal@a+bias);p=float(normal@delta)
        if abs(p)<1e-12:
            if q<0:return None
        elif p>0:t0=max(t0,-q/p)
        else:t1=min(t1,-q/p)
        if t0>t1:return None
    return a+t0*delta,a+t1*delta

def front_visibility(frame,m):
    visible={}
    for camera in frame.projected_lidar_labels:
        if camera.name!=1:continue
        for label in camera.labels:
            track=label.id[:-6] if label.id.endswith('_FRONT') else label.id
            b=label.box
            visible[track]=[(b.center_x-b.length/2)*W/m['width'],(b.center_y-b.width/2)*H/m['height'],
                            (b.center_x+b.length/2)*W/m['width'],(b.center_y+b.width/2)*H/m['height']]
    return visible

def projection_comparison(edges,official):
    if not edges:return dict(official_box=official,projected_edge_AABB=None,IoU=None,center_distance_px=None)
    q=np.array(edges).reshape(-1,2);a=np.r_[q.min(axis=0),q.max(axis=0)];b=np.array(official)
    b[[0,2]]=np.clip(b[[0,2]],0,W-1);b[[1,3]]=np.clip(b[[1,3]],0,H-1)
    wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));inter=float(np.prod(wh))
    union=float(np.prod(np.maximum(0,a[2:]-a[:2]))+np.prod(np.maximum(0,b[2:]-b[:2]))-inter)
    return dict(official_box=np.array(official).round(3).tolist(),projected_edge_AABB=a.round(3).tolist(),
                IoU=inter/union if union>0 else None,center_distance_px=float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)))

def box_geometry(label,ego,m):
    b=label.box;center=np.array([b.center_x,b.center_y,b.center_z])
    dimensions=np.array([b.length,b.width,b.height]);R=rz(b.heading)
    local=SIGNS*dimensions/2;vehicle=local@R.T+center
    world=transform(vehicle,ego);cam=transform(world,m['camera_from_world']);uv,depth=project_camera(cam,m)
    corner_valid=pinhole_frustum_valid(cam,m)
    corners=[v.round(3).tolist() if ok and np.max(np.abs(v))<1e6 else None for v,ok in zip(uv,corner_valid)]
    edges=[]
    for ia,ib in EDGES:
        a,b=cam[ia].copy(),cam[ib].copy()
        clipped=clip_camera_segment(a,b,m)
        if clipped is None:continue
        a,b=clipped
        q,_=project_camera(np.stack([a,b]),m)
        if np.isfinite(q).all():
            seg=clip_segment(q[0],q[1])
            if seg:edges.append(seg)
    return center,dimensions,R,corners,edges,transform(center[None],ego)[0],ego[:3,:3]@R

def color(track):
    b=hashlib.sha256(track.encode()).digest()
    return tuple(80+int(v)*175//255 for v in b[:3])

def encoder(path):
    return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}',
                             '-framerate','10','-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)

def cloud_preview(path,xyz,rgb,title,dims):
    im=Image.new('RGB',(1050,400),'#0d1422');d=ImageDraw.Draw(im)
    views=[(0,1,'top x/y'),(0,2,'side x/z'),(1,2,'front y/z')]
    if len(xyz):
        lo=np.quantile(xyz,.005,axis=0);hi=np.quantile(xyz,.995,axis=0)
        for j,(a,b,label) in enumerate(views):
            extent=max(float(hi[a]-lo[a]),float(hi[b]-lo[b]),1.)*1.15
            mid=(lo+hi)/2;xy=(xyz[:,[a,b]]-mid[[a,b]])/extent*290
            px=(xy[:,0]+j*350+175).astype(int);py=(210-xy[:,1]).astype(int)
            for x,y,c in zip(px,py,rgb):
                if j*350<x<(j+1)*350 and 30<y<370:d.ellipse((x-1,y-1,x+1,y+1),fill=tuple(map(int,c)))
            d.text((j*350+15,16),label,fill='white')
    d.text((15,380),title[:130],fill='white');im.save(path,quality=88)

def build(scene_key,limit=None,sample=False,front_only=False):
    cfg=json.loads((BASE/'scenes.json').read_text(encoding='utf8'))
    scene=next(s for s in cfg['scenes'] if s['scene_key']==scene_key)
    inv=next(s for s in json.loads((BASE/'raw_inventory.json').read_text()) if s['scene_key']==scene_key)
    context=scene['context_id'];raw=BASE/'raw'/pathlib.Path(inv['path']).name
    full_original=raw.exists() and raw.stat().st_size==inv['bytes']
    if not full_original:raw=BASE/'raw'/f'{scene_key}_front.tfrecord'
    lidar_stream=None;cache_dir=BASE/'cache'/scene_key;point_cache=None;cache_details=None
    if not front_only and (cache_dir/'manifest.json').exists():
        cache_manifest=json.loads((cache_dir/'manifest.json').read_text())
        if cache_manifest['context_id']!=context or cache_manifest['frame_count']!=scene['expected_frames']:raise ValueError('point cache identity')
        cache_actors=json.loads((cache_dir/'actors.json').read_text());point_cache={}
        for track,a in cache_actors.items():
            with np.load(cache_dir/a['file']) as f:point_cache[track]={k:f[k] for k in f.files}
        cache_details=json.loads((cache_dir/'lidar_decode_receipts.json').read_text())
    elif not full_original and not front_only:lidar_stream=iter(records(BASE/'raw'/f'{scene_key}_lidar.tfrecord',limit))
    out=BASE/('sample' if sample else 'front_generated' if front_only else 'generated')/'scenes'/context
    out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True);(out/'lidar').mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError(f'completed output already exists {out}')
    start=time.perf_counter();actors={};frame_rows=[];receipts=[];cloud_details=[];rawst=raw.stat();enc1=encoder(out/'front_rgb.mp4');enc2=encoder(out/'front_boxes.mp4')
    total_points=0
    try:
        for index,frame,receipt in records(raw,limit):
            if frame.context.name!=context:raise ValueError('context mismatch')
            if lidar_stream is not None:
                li,lf,lr=next(lidar_stream)
                if li!=index or lf.context.name!=context or lf.timestamp_micros!=frame.timestamp_micros:raise ValueError('split LiDAR/frame identity mismatch')
                frame.lasers.extend(lf.lasers);receipt['derived_lidar_record']=lr
            front=next(x for x in frame.images if x.name==1)
            rgb=Image.open(io.BytesIO(front.image)).convert('RGB').resize((W,H),Image.Resampling.LANCZOS)
            rgbarr=np.asarray(rgb);rgb.save(out/'frames'/f'{index:06d}.jpg',quality=88)
            overlay=rgb.copy();draw=ImageDraw.Draw(overlay)
            ego=np.array(frame.pose.transform).reshape(4,4);model=camera_model(frame,front)
            t0=time.perf_counter()
            cloud,details=(np.empty((0,6),np.float32),[]) if front_only or point_cache is not None else decoder.lidar(frame,pb)
            elapsed=time.perf_counter()-t0
            xyz=cloud[:,:3];tree=cKDTree(xyz);total_points+=cache_details[index]['retained_points'] if point_cache is not None else len(xyz)
            # Color is an optional image projection. Gray points are real LiDAR outside FRONT.
            worldxyz=transform(xyz,ego);cam=transform(worldxyz,model['camera_from_world']);uv,dep=project_camera(cam,model)
            valid=pinhole_frustum_valid(cam,model)&np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
            colors=np.tile(np.array([155,180,205],dtype=np.uint8),(len(xyz),1))
            pix=uv[valid].astype(int);colors[valid]=rgbarr[pix[:,1],pix[:,0]]
            boxes=[];visible_tracks=front_visibility(frame,model)
            for label in frame.laser_labels:
                if label.type!=1:continue
                track=label.id
                if track not in scene['actors']:raise ValueError(f'unregistered vehicle {track}')
                center,dimensions,R,corners,edges,cworld,Rworld=box_geometry(label,ego,model)
                official=visible_tracks.get(track)
                if official is None:edges=[]
                if point_cache is not None:
                    data=point_cache[track];lo,hi=np.searchsorted(data['raw_frame'],[index,index+1])
                    local=data['actor_local_xyz'][lo:hi];indices=data['source_cloud_row'][lo:hi]
                    pvehicle=local@R.T+center;pworld=transform(pvehicle,ego);pcam=transform(pworld,model['camera_from_world'])
                    puv,pdep=project_camera(pcam,model)
                    pvalid=pinhole_frustum_valid(pcam,model)&np.isfinite(puv).all(axis=1)&(puv[:,0]>=0)&(puv[:,0]<W)&(puv[:,1]>=0)&(puv[:,1]<H)
                    pcolors=np.tile(np.array([155,180,205],np.uint8),(len(local),1));px=puv[pvalid].astype(int);pcolors[pvalid]=rgbarr[px[:,1],px[:,0]]
                else:
                    indices=np.asarray(tree.query_ball_point(center,float(np.linalg.norm(dimensions)/2+1e-4)),dtype=np.int32)
                    local=(xyz[indices]-center)@R if len(indices) else np.empty((0,3))
                    inside=np.all(np.abs(local)<=dimensions/2+1e-5,axis=1)
                    indices=indices[inside];local=local[inside].astype(np.float32);pcolors=colors[indices];pvalid=valid[indices]
                actor=actors.setdefault(track,dict(alias=scene['actors'][track]['alias'],trajectory=[],clouds=[],colors=[],raws=[],indices=[],color_valid=[]))
                actor['trajectory'].append(dict(raw=index,timestamp_micros=int(frame.timestamp_micros),center_world=cworld.round(8).tolist(),
                    orientation_world=Rworld.round(10).ravel().tolist(),dimensions=dimensions.round(5).tolist(),center_vehicle=center.round(6).tolist(),
                    heading_vehicle=float(label.box.heading),measured_lidar_points=None if front_only else len(local)))
                if len(local):
                    actor['clouds'].append(local);actor['colors'].append(pcolors);actor['indices'].append(indices)
                    actor['raws'].append(np.full(len(local),index,dtype=np.uint16));actor['color_valid'].append(pvalid)
                row=dict(track=track,corners2D=corners,edges=edges,center_world=cworld.round(6).tolist(),dimensions=dimensions.round(4).tolist(),
                         measured_lidar_points=None if front_only else len(local),world_from_actor=np.r_[np.c_[Rworld,cworld],[[0,0,0,1]]].round(10).ravel().tolist(),
                         official_FRONT_visible=official is not None,projection_comparison=projection_comparison(edges,official) if official is not None else None)
                boxes.append(row)
                if edges:
                    c=color(track)
                    for edge in edges:draw.line(edge,fill=c,width=2)
                    minx=min(e[0] for e in edges);miny=min(e[1] for e in edges)
                    draw.text((minx+2,miny+2),actor['alias'],fill=c,stroke_width=1,stroke_fill='black')
            draw.text((9,9),f'{scene_key} raw {index:03d} | original RGB + projected 3D boxes',fill='white',stroke_width=1,stroke_fill='black')
            enc1.stdin.write(rgbarr.tobytes());enc2.stdin.write(np.asarray(overlay).tobytes())
            frame_rows.append(dict(raw=index,timestamp_micros=int(frame.timestamp_micros),image=f'scenes/{context}/frames/{index:06d}.jpg',
                ego_world=ego.ravel().tolist(),camera_pose_world=model['image_pose'].ravel().tolist(),boxes=boxes))
            receipts.append(dict(raw=index,context=context,timestamp_micros=int(frame.timestamp_micros),**receipt))
            cloud_details.append(cache_details[index] if point_cache is not None else dict(raw=index,retained_points=len(xyz),decode_seconds=round(elapsed,4),sensors=details))
            if index%20==0:print(json.dumps(dict(scene=scene_key,raw=index,elapsed=round(time.perf_counter()-start,2),points=len(xyz))),flush=True)
        enc1.stdin.close();enc2.stdin.close()
        if enc1.wait()!=0 or enc2.wait()!=0:raise RuntimeError('ffmpeg failed')
    except BaseException:
        for enc in (enc1,enc2):
            if enc.poll() is None:enc.kill()
        raise
    if not sample and len(frame_rows)!=scene['expected_frames']:raise ValueError(('frame denominator mismatch',len(frame_rows),scene['expected_frames']))
    actor_rows={};selected={q['track_id'] for q in cfg['queue'] if q['context_id']==context}
    for track,actor in actors.items():
        alias=actor['alias'];base=f'actor_{alias}'
        if front_only:
            actor_rows[track]=dict(alias=alias,trajectory=actor['trajectory'],lidar=dict(status='pending_actual_range_image_decode',
                measured_points_before_voxel=None,display_points=None,source_frames=None,labelled_frames=len(actor['trajectory']),
                frame_count_denominator=len(frame_rows),selected_preparation_queue=track in selected),human_verdict=None)
            continue
        points=np.concatenate(actor.pop('clouds')) if actor['clouds'] else np.empty((0,3),np.float32)
        colors=np.concatenate(actor.pop('colors')) if actor['colors'] else np.empty((0,3),np.uint8)
        raws=np.concatenate(actor.pop('raws')) if actor['raws'] else np.empty(0,np.uint16)
        indices=np.concatenate(actor.pop('indices')) if actor['indices'] else np.empty(0,np.int32)
        cv=np.concatenate(actor.pop('color_valid')) if actor['color_valid'] else np.empty(0,bool)
        # Preserve all measured box-filtered rows in NPZ; voxelization affects display only.
        np.savez_compressed(out/'lidar'/f'{base}.npz',actor_local_xyz=points,rgb=colors,raw_frame=raws,source_cloud_row=indices,front_color_valid=cv)
        if len(points):
            vox=np.floor(points/.06).astype(np.int32);_,keep=np.unique(vox,axis=0,return_index=True);keep.sort()
            if len(keep)>12000:keep=keep[np.linspace(0,len(keep)-1,12000,dtype=int)]
        else:keep=np.empty(0,dtype=int)
        visible=points[keep];vcolors=colors[keep]
        pt=dict(schema='actor_measured_lidar_v1',context_id=context,track_id=track,alias=alias,coordinates='actor_local_x_forward_y_left_z_up_box_center',
                xyz=visible.round(4).tolist(),rgb=vcolors.tolist(),raw_frame=raws[keep].astype(int).tolist(),source_cloud_row=indices[keep].astype(int).tolist(),front_color_valid=cv[keep].astype(int).tolist(),
                measured_points_before_voxel=len(points),voxel_m=.06,display_point_limit=12000,display_points=len(keep),
                color_role='projected_FRONT_RGB_with_finite_pinhole_frustum_without_visibility_zbuffer_else_gray',selection='measured_points_within_raw_3D_annotation_box_not_instance_segmentation')
        (out/'lidar'/f'{base}.js').write_text('window.ACTOR_LIDAR=window.ACTOR_LIDAR||{};window.ACTOR_LIDAR['+json.dumps(context+'|'+track)+']='+json.dumps(pt,separators=(',',':'))+';',encoding='utf8')
        cloud_preview(out/'lidar'/f'{base}.jpg',visible,vcolors,f'{scene_key} {alias} | {len(points)} measured rows / {len(keep)} display voxels | actor local meters',None)
        counts=[t['measured_lidar_points'] for t in actor['trajectory']]
        actor_rows[track]=dict(alias=alias,trajectory=actor['trajectory'],lidar=dict(data_js=f'scenes/{context}/lidar/{base}.js',preview=f'scenes/{context}/lidar/{base}.jpg',
                npz=f'scenes/{context}/lidar/{base}.npz',measured_points_before_voxel=len(points),display_points=len(keep),source_frames=int(np.count_nonzero(counts)),
                labelled_frames=len(counts),frame_count_denominator=len(frame_rows),median_measured_points=float(np.median(counts)),
                selected_preparation_queue=track in selected),human_verdict=None)
    if set(actors)!=set(scene['actors']) and not sample:raise ValueError(('actor denominator mismatch',len(actors),len(scene['actors'])))
    model_receipt={k:(v.ravel().tolist() if isinstance(v,np.ndarray) else v) for k,v in model.items()}
    media=dict(schema='actor_media_v1',context_id=context,scene_key=scene_key,display_name=scene['display_name'],imageWH=[W,H],fps=10,frame_count=len(frame_rows),
            full_raw_frame_denominator=scene['expected_frames'],video=f'scenes/{context}/front_rgb.mp4',box_video=f'scenes/{context}/front_boxes.mp4',
            frames=frame_rows,actors=actor_rows,box_edges_corner_indices=EDGES,
            projection=dict(model='Waymo camera calibration + image pose + Brown-Conrady 9 intrinsics; raw annotated 3D cuboid corners',
                rolling_shutter='provided image pose approximation; no per-row rolling shutter or object-motion compensation',
                drawing_visibility='raw projected_lidar_labels FRONT track whitelist; finite pinhole frustum before distortion',
                coordinates='resized FRONT pixel top-left origin; camera x forward y left z up',source_calibration=model_receipt),
            lidar_contract=dict(decoder='waymo_numpy_cpu + fast_range_matrix',sensors=5,returns=2,TOP_pixel_motion_compensation=True,NLZ_filter=True,
                box_selection='3D annotation OBB in frame vehicle coordinates; may contain ground/occluder points',coordinates='per-frame actor local x forward y left z up',
                aggregation='each measured row transformed with its own same-frame box pose; no learned completion',annotation_num_points_used_for_surface=False),
            media_state='RGB_and_3D_boxes_complete_LiDAR_pending' if front_only else 'complete_measured_LiDAR',human_verdict=None)
    media['lidar_contract']['motion_limit']='TOP per-pixel ego-motion compensation only; within-scan actor motion is not compensated'
    media['lidar_contract']['color_projection']='finite pinhole frustum before Brown distortion; no visibility zbuffer; invalid colors gray'
    dump(out/'media.json',media)
    (out/'media.js').write_text('window.ACTOR_MEDIA=window.ACTOR_MEDIA||{};window.ACTOR_MEDIA['+json.dumps(context)+']='+json.dumps(media,ensure_ascii=False,separators=(',',':'))+';',encoding='utf8')
    dump(out/'source_receipts.json',receipts);dump(out/'lidar_decode_receipts.json',cloud_details)
    finish=raw.stat()
    if finish.st_size!=rawst.st_size and not sample:raise ValueError('raw changed during processing')
    summary=dict(schema='actor_media_build_v1',scene_key=scene_key,context_id=context,raw_path=str(raw),remote_raw_path=inv['path'],
        raw_bytes=rawst.st_size,raw_frame_count=len(frame_rows),expected_frames=scene['expected_frames'],full_sequence=not sample,
        actors=len(actor_rows),selected_queue_actors=len(selected),selected_actors_with_actual_points=None if front_only else sum(a['lidar']['measured_points_before_voxel']>0 for k,a in actor_rows.items() if k in selected),
        measured_sensor_points=None if front_only else total_points,measured_box_rows=None if front_only else sum(a['lidar']['measured_points_before_voxel'] for a in actor_rows.values()),
        wall_seconds=round(time.perf_counter()-start,3),threads_per_process=1,training_updates=0,GPU_queries=0,input_role='complete_original_raw_TFRecord' if full_original else 'complete_sequence_from_exact_field_subset_with_source_record_CRC_receipts',
        original_source_bytes=inv['bytes'],annotation_role='raw_3D_annotation_boxes_for_visual_diagnostics_not_owner_or_eligibility_labels',
        LiDAR_status='pending' if front_only else 'decoded',human_verdict=None)
    dump(out/'manifest.json',summary);print(json.dumps(summary),flush=True)
    if point_cache is not None:
        import shutil
        for name in ('source_receipts.json','manifest.json'):shutil.copy2(cache_dir/name,out/('remote_points_'+name))
    return summary

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--scene',required=True);a.add_argument('--limit',type=int);a.add_argument('--sample',action='store_true');a.add_argument('--front-only',action='store_true');args=a.parse_args()
    build(args.scene,args.limit,args.sample,args.front_only)
