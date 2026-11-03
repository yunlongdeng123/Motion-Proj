"""CPU preparation of measured a6 evidence; no model and no invented color."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1')
INPUT = ROOT / 'inputs'
FULL = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026')
B = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_only_B10k_20261103_r1/formal10k')
BUNDLE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')
READER = Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101')
RAW = Path('/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files/training/segment-10094743350625019937_3420_000_3440_000_with_camera_labels.tfrecord')
OLD = Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137')
TRACK = '7MxwK9zTRZ9g5lc_ZC0z6A'
OCCLUDER = 'vBOasuXpjI7h7d-X_VEQdA'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, x):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp,p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for s in iter(lambda:f.read(8*1024**2), b''):
            h.update(s)
    return h.hexdigest()

def transform(x,T):
    return np.asarray(x) @ T[:3,:3].T + T[:3,3]

def savez_atomic(path, **arrays):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
        stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

def save_minimal():
    meta=read(INPUT/'a6_raw_diagnostic_metadata.json')
    src=np.load(INPUT/'actor_a6.npz')
    traj=meta['actors']['a6']['trajectory']
    box=np.asarray([x['dimensions'] for x in traj],float)
    assert len(traj)==199 and [x['raw'] for x in traj]==list(range(199))
    assert np.max(np.ptp(box,axis=0))<1e-6
    Ta=np.tile(np.eye(4),(199,1,1))
    for i,row in enumerate(traj):
        Ta[i,:3,:3]=np.asarray(row['orientation_world']).reshape(3,3)
        Ta[i,:3,3]=row['center_world']
    bundle=read(BUNDLE); wm=np.asarray(bundle['world_to_model'])
    tm=wm[None]@Ta
    xyz=src['actor_local_xyz'];rf=src['raw_frame'].astype(np.int16)
    world=np.einsum('nij,nj->ni',Ta[rf,:3,:3],xyz)+Ta[rf,:3,3]
    model=transform(world,wm)
    stamps=np.asarray(read(B/'actual_scene_binding.json')['timestamps'])
    time=np.interp(np.arange(199),[89,137],[stamps[0],stamps[-1]])
    time[:89]=stamps[0]+(np.arange(89)-89)*.02
    time[138:]=stamps[-1]+(np.arange(138,199)-137)*.02
    out=INPUT/'a6_geometry_minimal.npz'
    savez_atomic(out,canonical_xyz=xyz,raw_frame=rf,source_cloud_row=src['source_cloud_row'],
        world_xyz=world,model_xyz=model,box_lwh_m=box[0],box_lwh_per_raw=box,
        world_from_canonical=Ta,model_from_canonical=tm,model_from_world=wm,
        raw_ids=np.arange(199,dtype=np.int16),timestamp_micros=np.asarray([x['timestamp_micros'] for x in traj],np.int64),
        model_time=time,baseline_raw_ids=np.arange(89,138,dtype=np.int16),baseline_model_time=stamps,
        canonical_coordinate_axes=np.array(['x_forward_length','y_left_width','z_up_height']),
        rgb_valid=np.zeros(len(xyz),bool),rgb_unknown=np.ones(len(xyz),bool),
        omni_normalized_xyz=xyz[:,[0,2,1]]/(box[0].max()/2))
    # Geometry-only point conditioning. RGB is deliberately absent from this file.
    p=INPUT/'a6_omni_pointcloud.npz'
    savez_atomic(p,points=xyz[:,[0,2,1]]/(box[0].max()/2),native_canonical_xyz=xyz,
        full_box_lwh_m=box[0],normalization_half_max_box_m=box[0].max()/2,
        source_raw_frame=rf,source_cloud_row=src['source_cloud_row'])
    receipt=dict(status='READY_MEASURED_GEOMETRY_RGB_PENDING_VISIBILITY',point_count=len(xyz),
        full_box_lwh_m=box[0].tolist(),raw_frames=199,geometry_npz=str(out),geometry_sha256=sha(out),
        omni_npz=str(p),omni_sha256=sha(p),omni_axes='x_length,y_height,z_width; native[:,[0,2,1]]/(max(full_box_lwh)/2)',
        source='exact real measured rows from original raw TFRecord via CPU five LiDAR/two return decoder; per-frame annotated OBB actor-local coordinates',
        geometry_limit='OBB selection may contain ground or other object returns; not an exact semantic surface or completed mesh',
        rgb_policy='NO original diagnostic colors accepted: their front_color_valid has no visibility zbuffer',
        transforms='canonical->raw-world->B-model; authoritative native49 raw89..137 cameras will be audited against raw calibration',
        model_time_note='native49 exact; outside49 extrapolated at0.02 per raw frame for metadata only',
        GPU_used=False,training_updates=0,old_assets_modified=False)
    write(INPUT/'a6_geometry_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)

def box_pose(box, ego):
    c,s=np.cos(box['heading']),np.sin(box['heading'])
    T=np.eye(4);T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
    T[:3,3]=[box['center_x'],box['center_y'],box['center_z']]
    return ego@T

def project(x,K,W):
    cam=transform(x,W);h=cam@K.T
    return h[:,:2]/np.maximum(h[:,2:],1e-10)-.5,cam[:,2]

def depth_image(x,K,W,shape=(320,480)):
    uv,z=project(x,K,W);ij=np.floor(uv+.5).astype(int)
    ok=(z>.1)&(ij[:,0]>=0)&(ij[:,0]<shape[1])&(ij[:,1]>=0)&(ij[:,1]<shape[0])
    depth=np.full(shape,np.inf,np.float32)
    np.minimum.at(depth,(ij[ok,1],ij[ok,0]),z[ok])
    return depth,uv,z,ok

def box_image(box,T,K,W):
    import cv2
    signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    xyz=transform(signs*np.array([box['length'],box['width'],box['height']])/2,T)
    uv,z=project(xyz,K,W);mask=np.zeros((320,480),np.uint8)
    if (z>.1).all():
        hull=cv2.convexHull(np.round(uv).astype(np.int32));cv2.fillConvexPoly(mask,hull,1)
    return mask.astype(bool)

def ray_box_depth(box,T,K,W):
    # Native camera rays are parameterized by camera z in metres.
    yy,xx=np.indices((320,480));rays=np.c_[(xx.ravel()+.5-K[0,2])/K[0,0],(yy.ravel()+.5-K[1,2])/K[1,1],np.ones(320*480)]
    C=np.linalg.inv(W);origin=transform(C[None,:3,3],np.linalg.inv(T))[0]
    d=rays@C[:3,:3].T@T[:3,:3]
    half=np.array([box['length'],box['width'],box['height']])/2
    with np.errstate(divide='ignore',invalid='ignore'):
        lo=(-half-origin)/d;hi=(half-origin)/d
    enter=np.maximum(np.minimum(lo,hi).max(1),.1);leave=np.maximum(lo,hi).min(1)
    return np.where(leave>=enter,enter,np.inf).reshape(320,480).astype(np.float32)

def native49():
    import cv2
    from PIL import Image
    from scipy.ndimage import binary_erosion,binary_dilation,minimum_filter,distance_transform_edt
    out=INPUT/'a6_views';out.mkdir(exist_ok=True)
    geo=np.load(INPUT/'a6_geometry_minimal.npz');wm=geo['model_from_world'];scale=read(BUNDLE)['meter_to_model']
    records=read(B/'cameras.json');binding=read(B/'actual_scene_binding.json')
    rows=[];best=None;exact_transforms=[];camera_errors=[];all_points=[];all_frame=[];all_rgb=[];all_rgb_ok=[]
    for raw in range(89,138):
        fid=f'0008{raw:03d}';ann=read(OLD/'annotations'/f'{fid}.json')
        a6=next(x for x in ann['laser_labels'] if x['id']==TRACK)['box']
        a37=next(x for x in ann['laser_labels'] if x['id']==OCCLUDER)['box']
        ego=np.loadtxt(OLD/'pose'/f'{fid}.txt');Ta=box_pose(a6,ego);To=box_pose(a37,ego);Tm=wm@Ta
        cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (OLD/'calib'/f'{fid}.txt').read_text().splitlines() if ':' in s}
        W=np.eye(4);W[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);W=W@np.linalg.inv(ego)
        K=cal['P0'].reshape(3,4)[:,:3].copy();K[:2]/=4
        r=next(x for x in records if x['img_name']==fid);C=np.eye(4);C[:3,:3]=r['rotation'];C[:3,3]=r['position'];Wm=np.linalg.inv(C)
        Ccheck=wm@np.linalg.inv(W);Ccheck[:3,:3]/=scale
        error=float(np.abs(C-Ccheck).max());camera_errors.append(error)
        image_path=OLD/'image/image_0'/f'{fid}.png'
        image=cv2.resize(np.asarray(Image.open(image_path).convert('RGB'))/255.,(480,320)).astype(np.float32)
        dense=np.load('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023_dense_r7/frames/'+fid+'.npz')
        silhouette=dense['actor_labels']==4;occluder=dense['actor_labels']==5
        assert not (silhouette&occluder).any()
        old_body=np.asarray(Image.open(OLD/'dynamic_actor_assets/actor_3/body_masks'/f'{fid}.png').resize((480,320),Image.Resampling.NEAREST))>0
        assert np.array_equal(silhouette,old_body),'reviewed a6 label source changed'
        body=binary_erosion(silhouette,iterations=1)&~binary_dilation(occluder,iterations=1)
        cloud=np.fromfile(OLD/'velodyne'/f'{fid}.bin',np.float32).reshape(-1,6)
        world=transform(cloud[:,:3],ego)
        canonical=transform(world,np.linalg.inv(Ta))
        half=np.array([a6['length'],a6['width'],a6['height']])/2
        own=(np.abs(canonical)<=half+1e-6).all(1)
        # Exclude ground-adjacent lower slice from sparse photometric evidence, preserve it as geometry.
        measured=own&(canonical[:,2]>-half[2]+.10)
        full_depth,_,_,_=depth_image(world,K,W)
        dep=minimum_filter(full_depth,size=3,mode='constant',cval=np.inf)
        own_depth,uv,z,ok=depth_image(world[measured],K,W)
        local_xyz=canonical[measured];local_ids=np.flatnonzero(measured)
        ij=np.clip(np.floor(uv+.5).astype(int),[0,0],[479,319])
        depth_ok=ok&(z<=dep[ij[:,1],ij[:,0]]+.20)
        point_ok=depth_ok&body[ij[:,1],ij[:,0]]
        evidence_depth=np.full((320,480),np.inf,np.float32)
        np.minimum.at(evidence_depth,(ij[point_ok,1],ij[point_ok,0]),z[point_ok])
        evidence_depth=minimum_filter(evidence_depth,size=3,mode='constant',cval=np.inf)
        visible=body&np.isfinite(evidence_depth)&((evidence_depth<=dep+.20)|~np.isfinite(dep))
        envelope=box_image(a6,Ta,K,W)
        d6=ray_box_depth(a6,Ta,K,W);d37=ray_box_depth(a37,To,K,W)
        box_occluded=np.isfinite(d6)&np.isfinite(d37)&(d37+.1<d6)
        # Cuboid intersection is a possible-occlusion flag only. An observed a6 silhouette wins.
        hidden_a37=(box_occluded|occluder)&envelope&~body
        unknown=envelope&~visible
        rgb_point=np.zeros((len(local_xyz),3),np.float32);rgb_point[point_ok]=image[ij[point_ok,1],ij[point_ok,0]]
        uncertain_boundary=binary_dilation(silhouette,iterations=1)&~binary_erosion(silhouette,iterations=1)
        savez_atomic(out/f'{raw:03d}.npz',image_srgb=image,visible_evidence_mask=visible,
            actor_labels=dense['actor_labels'],visible_mask=silhouette,uncertain_boundary_mask=uncertain_boundary,
            known_visible_silhouette_mask=body,raw_reviewed_silhouette_mask=silhouette,
            alpha_positive_mask=visible,alpha_negative_mask=np.zeros_like(visible),unknown_mask=unknown,
            a37_occlusion_unknown_mask=hidden_a37,a37_visible_mask=occluder,a6_box_candidate_mask=envelope,
            measured_a6_depth_m=np.where(np.isfinite(evidence_depth),evidence_depth,0),
            camera_depth_m=np.where(np.isfinite(dep),dep,0),K=K,world_to_camera=W,model_to_camera=Wm,
            world_from_canonical=Ta,model_from_canonical=Tm,model_from_world=wm,
            box_lwh_m=np.array([a6['length'],a6['width'],a6['height']]),raw_frame=raw,
            model_time=binding['timestamps'][raw-89],point_canonical_xyz=local_xyz,
            point_source_cloud_row=local_ids,point_uv=uv.astype(np.float32),point_depth_m=z.astype(np.float32),
            point_rgb_srgb=rgb_point,point_rgb_valid=point_ok,
            mask_source=np.array('reviewed per-image a6 identity silhouette label4, not bbox; erode1px, a37 label5 dilate1; 5-LiDAR 2-return zbuffer min3x3 tolerance0.20m; actual a6 point support radius1px'),
            source_rgb=np.array(str(image_path)))
        row=dict(raw=raw,visible_evidence_pixels=int(visible.sum()),known_silhouette_pixels=int(body.sum()),
            a37_occlusion_unknown_pixels=int(hidden_a37.sum()),unknown_pixels=int(unknown.sum()),
            point_rgb_valid=int(point_ok.sum()),measured_obb_rows=int(own.sum()),camera_max_abs_error=error)
        rows.append(row);exact_transforms.append(Ta);all_points.append(local_xyz);all_frame.append(np.full(len(local_xyz),raw,np.int16));all_rgb.append(rgb_point);all_rgb_ok.append(point_ok)
        if best is None or visible.sum()>best[0]:best=(visible.sum(),raw,image,visible,body,envelope)
    # Native camera geometry is authoritative: this equality is independent of trained weights.
    if max(camera_errors)>1e-5:raise ValueError('B native camera mismatch '+str(max(camera_errors)))
    _,raw,image,visible,body,envelope=best
    yy,xx=np.where(envelope);x0=max(0,int(xx.min())-4);x1=min(480,int(xx.max())+5);y0=max(0,int(yy.min())-4);y1=min(320,int(yy.max())+5)
    crop=(np.clip(image[y0:y1,x0:x1]*255+.5,0,255)).astype(np.uint8)
    alpha=visible[y0:y1,x0:x1].astype(np.uint8)*255
    rgba=np.dstack([crop,alpha]);rgba[alpha==0,:3]=0
    cond=INPUT/'a6_conditioning_rgba.png';Image.fromarray(rgba).save(cond)
    # A second clearly labelled hint retains the independently reviewed visible silhouette.
    silhouette_rgba=np.dstack([crop,body[y0:y1,x0:x1].astype(np.uint8)*255]);silhouette_rgba[silhouette_rgba[...,3]==0,:3]=0
    Image.fromarray(silhouette_rgba).save(INPUT/'a6_conditioning_reviewed_silhouette_rgba.png')
    points=geo['canonical_xyz'].astype(np.float32);size=geo['box_lwh_m'].astype(np.float32)
    points_npz=INPUT/'a6_prior_inputs.npz';savez_atomic(points_npz,observed_points_actor_native_m=points,
        size_lwh_m=size,visible_points_confidence=np.ones(len(points),np.float32),
        source_raw_frame=geo['raw_frame'],source_cloud_row=geo['source_cloud_row'])
    contract=dict(status='READY_REAL_POINT_AND_SPARSE_VISIBLE_RGBA_CONDITIONING',points_npz=str(points_npz),
        points_sha256=sha(points_npz),conditioning_image=str(cond),conditioning_image_sha256=sha(cond),
        native_axis='length_width_height',omni_axis='length_height_width',normalization_half_span_m=float(geo['box_lwh_m'].max()/2),
        full_box_size_lwh_m=geo['box_lwh_m'].tolist(),raw_frame_count=199,true_visible_pixel_count=int(visible.sum()),
        conditioning_raw_frame=raw,conditioning_crop_xyxy_native480=[x0,y0,x1,y1],
        conditioning_alpha='Only measured depth-visible a6 pixels within eroded reviewed identity silhouette. All unknown/a37 pixels alpha=0, RGB=0.',
        image_provenance='Original same-frame Waymo FRONT image; native cv2 linear resize float64/255 to480x320; never rendered RGB.',
        conditioning_role='Prior hint only. Transparent pixels are UNKNOWN, never measured background or completed a6.',
        alternative_reviewed_silhouette_hint=str(INPUT/'a6_conditioning_reviewed_silhouette_rgba.png'),
        point_geometry_limit='Real per-frame OBB rows; may include some road/occluder returns. No learned completion or synthetic GT.',
        mask_source='Previously reviewed per-frame image identity silhouettes; not GT boxes. a6 label4 and a37 label5 are exclusive.',
        source_views=str(out),GPU_used=False,training_updates=0)
    write(INPUT/'a6_prior_contract.json',contract)
    receipt=dict(status='READY_NATIVE49_A6_OBSERVATIONS',raw_range=[89,137],views=str(out),rows=rows,
        total_visible_evidence_pixels=sum(x['visible_evidence_pixels'] for x in rows),
        native_camera_max_abs_error=max(camera_errors),native_camera_test='world_to_model @ inverse(raw pinhole W); rotation divided by meter_to_model; all49 match actual B cameras',
        mask_provenance='dense_r7 actor_labels==4 equals original reviewed actor3 body mask pixel-for-pixel at480x320; not bbox; a37==5',
        photometric_policy='erode silhouette1px, exclude dilated a37, require measured a6 full-cloud zbuffer support within1px and0.20m; all original RGB',
        alpha_policy='positive evidence only; alpha_negative_mask allFalse. Unknown must not become0-alpha/black RGB supervision.',
        a37_unknown_policy='a6 projected cuboid intersect near a37 cuboid or current a37 silhouette, minus observed a6 body; conservative possible-occlusion flag, not exact hidden silhouette',
        GPU_used=False,training_updates=0,old_assets_modified=False)
    write(INPUT/'a6_native49_receipt.json',receipt)
    savez_atomic(INPUT/'a6_native49_points.npz',canonical_xyz=np.concatenate(all_points),raw_frame=np.concatenate(all_frame),
        rgb_srgb=np.concatenate(all_rgb),rgb_valid=np.concatenate(all_rgb_ok),world_from_canonical=np.asarray(exact_transforms))
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}),flush=True)
    print(json.dumps(contract),flush=True)

def brown_project(world, camera_from_world, intrinsic):
    cam=transform(world,camera_from_world);z=cam[:,0]
    x=-cam[:,1]/np.maximum(z,1e-8);y=-cam[:,2]/np.maximum(z,1e-8)
    fx,fy,cx,cy,k1,k2,p1,p2,k3=intrinsic
    r2=x*x+y*y;radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    xd=x*radial+2*p1*x*y+p2*(r2+2*x*x)
    yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y
    uv=np.c_[fx*xd+cx,fy*yd+cy]
    # Clip in the undistorted pinhole frustum before accepting Brown-distorted pixels.
    pin=np.c_[fx*x+cx,fy*y+cy]
    valid=(z>.1)&(pin[:,0]>=0)&(pin[:,0]<1920)&(pin[:,1]>=0)&(pin[:,1]<1280)
    valid&=np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1280)
    return uv,z,valid

def colour_evidence():
    pts=[];colors=[];raws=[];valid=[];source=[];uv=[]
    for raw in range(89,138):
        with np.load(INPUT/'a6_views'/f'{raw:03d}.npz') as a:
            p=a['point_canonical_xyz'];pts.append(p);colors.append(a['point_rgb_srgb'])
            raws.append(np.full(len(p),raw,np.int16));valid.append(a['point_rgb_valid'])
            source.append(a['point_source_cloud_row']);uv.append(a['point_uv'])
    arrays=dict(canonical_xyz=np.concatenate(pts).astype(np.float32),rgb_srgb=np.concatenate(colors).astype(np.float32),
        raw_frame=np.concatenate(raws),valid_mask=np.concatenate(valid),source_cloud_row=np.concatenate(source),native_camera_uv=np.concatenate(uv))
    path=INPUT/'a6_colour_evidence.npz';savez_atomic(path,**arrays)
    receipt=dict(status='READY_MASKED_ACTUAL_LIDAR_COLOR_EVIDENCE',file=str(path),sha256=sha(path),
        rows=len(arrays['canonical_xyz']),valid_rows=int(arrays['valid_mask'].sum()),raw_range=[89,137],
        visibility='original RGB only; reviewed a6 silhouette erode1 and a37 label5 dilate1 exclusion; full5LiDAR2return zbuffer min3x3 tolerance0.20m',
        coordinates='a6 canonical xforward length,yleft width,zup height; metres',
        bottom_unobserved_band_m=[-.72,-.21044833958148956],
        limitation='bottom approximate0.51m has no real LiDAR observations; all completion/inferred color is prior, never measured GT',GPU_used=False,training_updates=0)
    write(INPUT/'a6_colour_evidence_receipt.json',receipt);print(json.dumps(receipt),flush=True)

def blocked_points(world, origin, boxes, exclude):
    blocked=np.zeros(len(world),bool);by_a37=np.zeros(len(world),bool)
    for track,box,T in boxes:
        if track==exclude:continue
        inv=np.linalg.inv(T);o=transform(origin[None],inv)[0]
        d=(world-origin)@T[:3,:3]
        half=np.array([box['length'],box['width'],box['height']])/2
        with np.errstate(divide='ignore',invalid='ignore'):
            lo=(-half-o)/d;hi=(half-o)/d
        enter=np.maximum(np.minimum(lo,hi).max(1),0);leave=np.maximum(lo,hi).min(1)
        # Stop the ray 0.10 m before the actual point, avoiding mere endpoint contact.
        hit=(leave>=enter)&(enter<1-.1/np.maximum(np.linalg.norm(world-origin,axis=1),.1))
        blocked|=hit
        if track==OCCLUDER:by_a37|=hit
    return blocked,by_a37

def full199():
    from PIL import Image
    from scipy.ndimage import minimum_filter,binary_erosion,binary_dilation
    sys.path.insert(0,str(READER))
    import waymo_numpy_cpu as decoder
    from fast_range_matrix import matrix
    decoder.matrix=matrix;pb=decoder.protobuf()
    source=np.load(INPUT/'actor_a6.npz');raws=source['raw_frame'].astype(int);ids=source['source_cloud_row'];n=len(raws)
    wm=np.asarray(read(BUNDLE)['world_to_model']);size=read(INPUT/'a6_geometry_receipt.json')['full_box_lwh_m']
    rgb=np.zeros((n,3),np.float32);uv_all=np.full((n,2),-1,np.float32);depth_all=np.zeros(n,np.float32)
    valid_geom=np.zeros(n,bool);valid_reviewed=np.zeros(n,bool);a37_block=np.zeros(n,bool);other_block=np.zeros(n,bool)
    points=np.zeros((n,3),np.float32);worlds=np.zeros((n,3),np.float64)
    transforms=[];camera_transforms=[];intrinsics=[];timestamps=[];frame_rows=[];exact_box_sizes=[]
    initial=RAW.stat();start=time.monotonic()
    roots=[OLD,Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/inputs/scene0008_077_125'),Path('/root/autodl-tmp/drivingeditor_repro/data/waymo_paper_scenes/scene0008_023_071')]
    for raw,payload,crc in decoder.records(RAW,198):
        frame=pb.Frame.FromString(payload);assert frame.context.name=='10094743350625019937_3420_000_3440_000'
        ego=np.asarray(frame.pose.transform).reshape(4,4);fid=f'0008{raw:03d}'
        label=next(x for x in frame.laser_labels if x.id==TRACK);b=label.box
        box={k:getattr(b,k) for k in ['center_x','center_y','center_z','length','width','height','heading']};Ta=box_pose(box,ego)
        assert np.max(np.abs(np.array([b.length,b.width,b.height])-size))<5e-6  # diagnostic JSON rounded to5 decimal places
        exact_box_sizes.append([b.length,b.width,b.height])
        full_rgb,front=decoder.front_rgb(frame)
        cal=next(x for x in frame.context.camera_calibrations if x.name==1)
        image_pose=np.asarray(front.pose.transform).reshape(4,4) if len(front.pose.transform)==16 else ego
        C=image_pose@np.asarray(cal.extrinsic.transform).reshape(4,4);W=np.linalg.inv(C);k=np.asarray(cal.intrinsic)
        # Re-decode every raw frame with the exact diagnostic decoder. Older prepared
        # clouds have identical row ordering but TOP float32 compensation may differ
        # by ~1mm, so they are not used for this exact provenance audit.
        cloud,_=decoder.lidar(frame,pb);reused=None
        world=transform(cloud[:,:3],ego);uv,z,ok=brown_project(world,W,k)
        safe_uv=np.where(ok[:,None],uv,0)
        ij=np.clip(np.floor(safe_uv+.5).astype(np.int32),[0,0],[1919,1279])
        # Actual all-sensor nearest-return zbuffer in original 1920x1280 image coordinates.
        depth=np.full((1280,1920),np.inf,np.float32)
        np.minimum.at(depth,(ij[ok,1],ij[ok,0]),z[ok]);depth=minimum_filter(depth,size=9,mode='constant',cval=np.inf)
        index=np.flatnonzero(raws==raw);source_id=ids[index];pworld=world[source_id]
        local=transform(pworld,np.linalg.inv(Ta)).astype(np.float32)
        err=float(np.abs(local-source['actor_local_xyz'][index]).max(initial=0))
        if err>5e-4:raise ValueError(f'source row/canonical mismatch raw{raw}: {err}')
        p_ij=ij[source_id];p_z=z[source_id]
        pass_depth=ok[source_id]&(p_z<=depth[p_ij[:,1],p_ij[:,0]]+.20)
        boxes=[]
        for lab in frame.laser_labels:
            bb=lab.box
            if min(bb.length,bb.width,bb.height)<=0:continue
            bx={name:getattr(bb,name) for name in ['center_x','center_y','center_z','length','width','height','heading']}
            boxes.append((lab.id,bx,box_pose(bx,ego)))
        blocked,blocked37=blocked_points(pworld,C[:3,3],boxes,TRACK)
        front_bbox=None
        for group in frame.projected_lidar_labels:
            if group.name!=1:continue
            for lab in group.labels:
                if lab.id.removesuffix('_FRONT')==TRACK:
                    bb=lab.box;front_bbox=np.array([bb.center_x-bb.length/2,bb.center_y-bb.width/2,bb.center_x+bb.length/2,bb.center_y+bb.width/2])
        if front_bbox is None:front_ok=np.zeros(len(index),bool)
        else:
            p_uv=uv[source_id];front_ok=(p_uv[:,0]>=front_bbox[0])&(p_uv[:,0]<=front_bbox[2])&(p_uv[:,1]>=front_bbox[1])&(p_uv[:,1]<=front_bbox[3])
        lower_guard=local[:,2]>-b.height/2+.10
        geometry_visible=pass_depth&~blocked&front_ok&lower_guard
        # Outside native49 these are sparse LiDAR/box-supported hints, not reviewed dense silhouette truth.
        reviewed=np.zeros(len(index),bool)
        if 89<=raw<=137:
            v=np.load(INPUT/'a6_views'/f'{raw:03d}.npz');small=np.clip(np.floor(uv[source_id]/4).astype(int),[0,0],[479,319])
            # A raw cuboid is only a possible-occluder envelope. It must not veto
            # a reviewed, measured-depth-visible a6 silhouette pixel.
            reviewed=pass_depth&lower_guard&v['known_visible_silhouette_mask'][small[:,1],small[:,0]]
        color=full_rgb[p_ij[:,1],p_ij[:,0]].astype(np.float32)/255.
        rgb[index]=np.where((geometry_visible|reviewed)[:,None],color,0)
        valid_geom[index]=geometry_visible;valid_reviewed[index]=reviewed;a37_block[index]=blocked37;other_block[index]=blocked
        points[index]=local;worlds[index]=pworld;uv_all[index]=uv[source_id];depth_all[index]=p_z
        transforms.append(Ta);camera_transforms.append(W);intrinsics.append(k);timestamps.append(int(frame.timestamp_micros))
        frame_rows.append(dict(raw=raw,point_count=len(index),canonical_source_max_abs_error_m=err,
            depth_visible=int(pass_depth.sum()),rgb_sparse_geometric_hint=int(geometry_visible.sum()),
            rgb_reviewed_native49=int(reviewed.sum()),a37_ray_blocked_points=int(blocked37.sum()),
            any_other_box_ray_blocked_points=int(blocked.sum()),projected_a6_front_bbox=front_bbox.tolist() if front_bbox is not None else None,
            reused_cloud=reused,crc=crc))
        if raw%20==0:
            progress=dict(status='A6_FULL199_CPU_EVIDENCE_RUNNING',completed=raw+1,elapsed_seconds=time.monotonic()-start)
            write(INPUT/'a6_full199_progress.json',progress);print(json.dumps(progress),flush=True)
    assert (initial.st_size,initial.st_mtime_ns)==(RAW.stat().st_size,RAW.stat().st_mtime_ns)
    Ts=np.asarray(transforms);timestamps=np.asarray(timestamps,np.int64)
    modeltime=np.interp(np.arange(199),[89,137],[-.48,.48]);modeltime[:89]=-.48+(timestamps[:89]-timestamps[89])/(timestamps[137]-timestamps[89])*.96;modeltime[138:]=.48+(timestamps[138:]-timestamps[137])/(timestamps[137]-timestamps[89])*.96
    modeltime[89:138]=np.asarray(read(B/'actual_scene_binding.json')['timestamps'])
    # Canonical 5cm voxels count unique raw frames, never treat 54k correlated rows as independent surfaces.
    voxel,inv=np.unique(np.floor(points/.05).astype(np.int32),axis=0,return_inverse=True)
    pairs=np.unique(np.c_[inv,raws],axis=0);frame_count=np.bincount(pairs[:,0],minlength=len(voxel)).astype(np.int16)
    centroids=np.column_stack([np.bincount(inv,weights=points[:,axis],minlength=len(voxel))/np.bincount(inv) for axis in range(3)])
    vox_rgb=np.zeros((len(voxel),3),np.float32);vox_count=np.zeros(len(voxel),np.int16);consensus=np.zeros(n,bool)
    color_accepted=valid_geom|valid_reviewed
    for i in np.unique(inv[color_accepted]):
        ii=np.flatnonzero((inv==i)&color_accepted)
        count=len(np.unique(raws[ii]));vox_count[i]=count
        med=np.median(rgb[ii],axis=0);vox_rgb[i]=med
        if count>=2:consensus[ii]=np.abs(rgb[ii]-med).mean(1)<=.10
    outfile=INPUT/'a6_full199_observations.npz'
    savez_atomic(outfile,canonical_xyz=points,world_xyz=worlds,model_xyz=transform(worlds,wm),raw_frame=raws.astype(np.int16),
        source_cloud_row=ids,rgb_srgb=rgb,rgb_valid=valid_reviewed,rgb_valid_reviewed_native49=valid_reviewed,
        rgb_valid_sparse_geometric_hint=valid_geom,rgb_two_frame_voxel_consensus=consensus,
        camera_uv_original=uv_all,camera_depth_m=depth_all,a37_ray_occluded=a37_block,other_box_ray_occluded=other_block,
        a37_raw_cuboid_ray_intersection_possible=a37_block,
        box_lwh_m=np.asarray(exact_box_sizes[0]),box_lwh_per_raw=np.asarray(exact_box_sizes),world_from_canonical=Ts,model_from_canonical=wm[None]@Ts,model_from_world=wm,
        image_pose_camera_from_world=np.asarray(camera_transforms),camera_intrinsics9=np.asarray(intrinsics),
        raw_ids=np.arange(199,dtype=np.int16),timestamp_micros=timestamps,model_time=modeltime,
        voxel_centroids_canonical=centroids.astype(np.float32),voxel_source_frame_count=frame_count,
        voxel_sparse_rgb_srgb=vox_rgb,voxel_sparse_rgb_frame_count=vox_count,point_voxel_id=inv.astype(np.int32))
    occupancy={}
    for name,axis in [('length_x',0),('width_y',1),('height_z',2)]:
        edges=np.linspace(-size[axis]/2,size[axis]/2,11);occupancy[name]=dict(edges_m=edges.tolist(),measured_rows=np.histogram(points[:,axis],edges)[0].tolist(),occupied5cm_voxels=np.histogram(centroids[:,axis],edges)[0].tolist())
    receipt=dict(status='PASS_REAL199_A6_GEOMETRY_AND_TIERED_RGB_EVIDENCE',output=str(outfile),sha256=sha(outfile),
        frames=199,point_count=n,voxel5cm_count=len(voxel),voxel_two_or_more_source_frames=int((frame_count>=2).sum()),
        geometry_point_range_m=[points.min(0).tolist(),points.max(0).tolist()],full_box_lwh_m=size,
        lower_geometry_unobserved_band_m=[-size[2]/2,float(points[:,2].min())],
        reviewed_native49_rgb_points=int(valid_reviewed.sum()),sparse_all199_geometric_rgb_hint_points=int(valid_geom.sum()),
        sparse_two_frame_color_consensus_points=int(consensus.sum()),a37_ray_occluded_point_count=int(a37_block.sum()),
        canonical_source_max_abs_error_m=max(x['canonical_source_max_abs_error_m'] for x in frame_rows),
        rgb_strict_policy='rgb_valid is ONLY native49 reviewed visible silhouette AND full measured zbuffer. Reviewed depth-visible a6 positive evidence overrides raw cuboid candidate intersections. Outside49 rgb_valid=False.',
        rgb_hint_policy='rgb_valid_sparse_geometric_hint uses same measured zbuffer, raw projected FRONT a6 bbox and all-other-OBB ray exclusion; NO outside49 silhouette. Hints are not dense GT.',
        RGB_projection='original1920x1280 RGB, Brown-Conrady9 intrinsic and image pose; finite undistorted frustum first; nearest full5-sensor2-return depth min9x9, tolerance0.20m',
        geometry_limit='measured OBB rows only, may include road/occluders; bottom/wheels and any space without points unknown; no learned completion enters observations',
        cuboid_warning='a37_ray_occluded is a conservative possible-occlusion CANDIDATE ONLY: raw a37 cuboid envelopes overlap many actually visible a6 points; not actual surface visibility or silhouette GT',
        rolling_shutter_limit='provided image pose only, no per-row rolling shutter or within-scan actor motion compensation',
        raw_TFRecord_CRC='all199 record length and payload CRC32C passed',raw_source_size_mtime_unchanged=True,
        occupancy_10bins=occupancy,frame_rows=frame_rows,wall_seconds=time.monotonic()-start,GPU_used=False,training_updates=0,old_assets_modified=False)
    write(INPUT/'a6_full199_receipt.json',receipt)
    write(INPUT/'a6_full199_progress.json',dict(status=receipt['status'],completed=199,wall_seconds=receipt['wall_seconds']))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['frame_rows','occupancy_10bins']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');parser.add_argument('--native49',action='store_true');parser.add_argument('--colour-evidence',action='store_true');args=parser.parse_args()
    if args.full:full199()
    elif args.native49:native49()
    elif args.colour_evidence:colour_evidence()
    else:save_minimal()
