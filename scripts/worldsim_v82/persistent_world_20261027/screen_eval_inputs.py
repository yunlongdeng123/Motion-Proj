"""候选池前置筛查：传感器、图像、轨迹、视差与逐操作适配。

metadata用data_cpu环境；图像/ORB用drivingeditor环境。不读取任何编辑效果。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,concurrent.futures as futures,ctypes,json,struct,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
OUT=ROOT/'evaluation_admission_v2'
DATA=ROOT/'evaluation_set_v1'


def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False));q.replace(p)


def sensor_worker(scene):
    from waymo_open_dataset import dataset_pb2
    path=OUT/'sensors'/(scene['context']+'.json')
    if path.exists():return read(path)
    raw=Path(scene['raw_path']);assert raw.stat().st_size==scene['raw_size_bytes'] and raw.stat().st_mtime_ns==scene['raw_mtime_ns']
    previous=ROOT/'evaluation_admission_v1/sensors'/(scene['context']+'.json')
    if previous.exists():
        data=read(previous);data['metadata_reused_from_unchanged_raw']=str(previous);write(path,data);return data
    selected=set(scene['selected_window']);rows=[]
    lib=ctypes.CDLL('/root/autodl-tmp/drivingeditor_repro/campaign/scene_search_mirror/libpreview_crc32c.so')
    crc=lib.preview_crc32c;crc.argtypes=[ctypes.c_char_p,ctypes.c_size_t];crc.restype=ctypes.c_uint32
    def check(b):
        v=crc(b,len(b));return (((v>>15)|(v<<17))+0xa282ead8)&0xffffffff
    with raw.open('rb') as stream:
        index=0
        while index<=max(selected):
            header=stream.read(12);size,checksum=struct.unpack('<QI',header);assert check(header[:8])==checksum
            if index not in selected:stream.seek(size+4,1);index+=1;continue
            payload=stream.read(size);assert check(payload)==struct.unpack('<I',stream.read(4))[0]
            frame=dataset_pb2.Frame.FromString(payload);assert frame.context.name==scene['context']
            camera=next(x for x in frame.context.camera_calibrations if x.name==1)
            pose=np.array(frame.pose.transform).reshape(4,4);extrinsic=np.array(camera.extrinsic.transform).reshape(4,4)
            optical=np.eye(4);optical[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]]
            c2w=pose@extrinsic@np.linalg.inv(optical)
            k=list(camera.intrinsic);K=np.array([[k[0]/4,0,k[2]/4],[0,k[1]/4,k[3]/4],[0,0,1.]])
            labels={x.id:x for x in frame.laser_labels};target=labels.get(scene['selected_track']);heading=None
            if target:
                axis=pose[:3,:3]@np.array([np.cos(target.box.heading),np.sin(target.box.heading),0.]);heading=float(np.arctan2(axis[1],axis[0]))
            rows.append(dict(raw=index,timestamp_micros=frame.timestamp_micros,pose=pose.tolist(),extrinsic=extrinsic.tolist(),
                             camera_world_from_optical=c2w.tolist(),K=K.tolist(),distortion=k[4:9],world_heading=heading,
                             image_width=camera.width,image_height=camera.height))
            index+=1
    result=dict(context=scene['context'],frames=rows,selected_record_crc_verified=True)
    write(path,result);return result


def longest_run(indices):
    longest=current=0;previous=None
    for x in sorted(indices):
        current=current+1 if previous is not None and x==previous+1 else 1
        longest=max(longest,current);previous=x
    return longest


def quality_worker(scene):
    import cv2
    from PIL import Image,ImageDraw
    key=scene['context'];destination=OUT/'scenes'/key
    if (destination/'screen.json').exists():return read(destination/'screen.json')
    policy=read(ROOT/'code/ADMISSION_POLICY.json');sensor=read(OUT/'sensors'/(key+'.json'))
    original=read(DATA/'scenes'/key/'scene.json');meta={r['raw']:r for r in original['raw_annotations']}
    sensor_rows=sensor['frames'];images=[];frames=[];target_rows=[];structural=[];flags=[]
    for row in sensor_rows:
        raw=row['raw'];image=np.array(Image.open(DATA/'scenes'/key/'gt_front'/f'{raw:03d}.png').convert('RGB'));gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        lap=float(cv2.Laplacian(gray,cv2.CV_64F).var());dark=float((gray<5).mean());bright=float((gray>250).mean())
        R=np.array(row['pose'])[:3,:3];E=np.array(row['extrinsic'])[:3,:3];K=np.array(row['K'])
        finite=all(np.isfinite(np.array(row[k])).all() for k in ['pose','extrinsic','K'])
        rotations=all(np.max(np.abs(a.T@a-np.eye(3)))<.001 and .999<np.linalg.det(a)<1.001 for a in [R,E])
        intrinsics=K[0,0]>0 and K[1,1]>0 and 0<K[0,2]<480 and 0<K[1,2]<320
        if not(finite and rotations and intrinsics):structural.append(dict(raw=raw,reason='invalid_calibration_or_pose_structure'))
        targets=[x for x in meta[raw]['front_vehicles'] if x['track']==scene['selected_track']]
        roi_lap=None;area=None;clipped=None
        if targets:
            target=targets[0];x,y,w,h=np.array(target['xywh'])/4;area=w*h
            left,right=max(0,int(x-w/2)),min(480,int(x+w/2));top,bottom=max(0,int(y-h/2)),min(320,int(y+h/2))
            clipped=(right-left)*(bottom-top)/max(area,1)
            if right>left+2 and bottom>top+2:roi_lap=float(cv2.Laplacian(gray[top:bottom,left:right],cv2.CV_64F).var())
            border=left<=2 or right>=478 or top<=2 or bottom>=318
            target_rows.append(dict(raw=raw,world=target.get('center_world'),speed=target.get('speed_mps'),area=area,clipped=clipped,border_touch=border,heading=row['world_heading']))
        frames.append(dict(raw=raw,laplacian_variance=lap,dark_fraction=dark,bright_fraction=bright,target_roi_laplacian=roi_lap,target_area_px=area))
        images.append(image)
    stamps=np.array([r['timestamp_micros'] for r in sensor_rows])/1e6;dt=np.diff(stamps)
    if not(dt>0).all():structural.append(dict(reason='nonincreasing_time'))
    positions=np.array([r['pose'] for r in sensor_rows])[:,:3,3];ego_speed=np.linalg.norm(np.diff(positions,axis=0),axis=1)/dt
    low=float(np.mean([r['laplacian_variance']<20 for r in frames]))
    if low>.25:flags.append('persistent_low_high_frequency_review_blur_vs_low_texture')
    if any(max(r['dark_fraction'],r['bright_fraction'])>.7 for r in frames):flags.append('exposure_or_valid_extreme_light_review')
    if dt.max()>.15:flags.append('timestamp_gap_review')
    if ego_speed.max()>60:flags.append('ego_pose_jump_review')
    indices=[r['raw'] for r in target_rows];gaps=max(np.diff(indices)-1,default=0);run=longest_run(indices)
    available=[r for r in target_rows if r['world'] is not None]
    target_speed=[];acceleration=[]
    if len(available)>2:
        times=np.array([meta[r['raw']]['timestamp_micros'] for r in available])/1e6
        velocity=np.diff(np.array([r['world'] for r in available]),axis=0)/np.diff(times)[:,None]
        target_speed=np.linalg.norm(velocity,axis=1).tolist()
        acceleration=(np.linalg.norm(np.diff(velocity,axis=0),axis=1)/np.diff(times)[1:]).tolist()
        if max(target_speed,default=0)>60:flags.append('target_world_track_jump_review')
        if max(acceleration,default=0)>20:flags.append('target_acceleration_or_annotation_jitter_review')
    heading_rows=[r for r in available if r['heading'] is not None]
    heading_rate=[]
    if len(heading_rows)>2:
        h=np.unwrap([r['heading'] for r in heading_rows]);t=np.array([meta[r['raw']]['timestamp_micros'] for r in heading_rows])/1e6
        heading_rate=(np.abs(np.diff(h)/np.diff(t))*180/np.pi).tolist()
        if max(heading_rate)>90:flags.append('target_heading_jump_review')
    # 背景ORB对应检验图像/标定/位姿一致性；低视差不等于坏输入。
    pairs=[];orb=cv2.ORB_create(nfeatures=1500);matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    for ia,ib in [(0,24),(24,48),(0,48)]:
        masks=[]
        for i in [ia,ib]:
            mask=np.full((320,480),255,np.uint8)
            for obj in meta[sensor_rows[i]['raw']]['front_vehicles']:
                x,y,w,h=np.array(obj['xywh'])/4
                cv2.rectangle(mask,(max(0,int(x-w/2)-3),max(0,int(y-h/2)-3)),(min(479,int(x+w/2)+3),min(319,int(y+h/2)+3)),0,-1)
            masks.append(mask)
        ka,da=orb.detectAndCompute(cv2.cvtColor(images[ia],cv2.COLOR_RGB2GRAY),masks[0]);kb,db=orb.detectAndCompute(cv2.cvtColor(images[ib],cv2.COLOR_RGB2GRAY),masks[1])
        pair=dict(raw=[sensor_rows[ia]['raw'],sensor_rows[ib]['raw']],status='insufficient_matches')
        if da is not None and db is not None:
            matches=[a for a,b in matcher.knnMatch(da,db,k=2) if a.distance<.75*b.distance]
            pair['ratio_test_matches']=len(matches)
            if len(matches)>=12:
                pa=np.array([ka[m.queryIdx].pt for m in matches]);pb=np.array([kb[m.trainIdx].pt for m in matches])
                _,ok=cv2.findFundamentalMat(pa,pb,cv2.FM_RANSAC,1.5,.99)
                if ok is not None:
                    ok=ok.ravel()>0;pa=pa[ok];pb=pb[ok];pair['image_ransac_inliers']=len(pa)
                    ca=sensor_rows[ia];cb=sensor_rows[ib];A=np.array(ca['camera_world_from_optical']);B=np.array(cb['camera_world_from_optical'])
                    baseline=np.linalg.norm(A[:3,3]-B[:3,3]);pair['translation_baseline_m']=float(baseline)
                    if len(pa)>=8 and baseline>.05:
                        na=cv2.undistortPoints(pa[:,None].astype(float),np.array(ca['K']),np.array(ca['distortion'])).reshape(-1,2)
                        nb=cv2.undistortPoints(pb[:,None].astype(float),np.array(cb['K']),np.array(cb['distortion'])).reshape(-1,2)
                        T=np.linalg.inv(B)@A;t=T[:3,3];skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
                        E=skew@T[:3,:3];ha=np.c_[na,np.ones(len(na))];hb=np.c_[nb,np.ones(len(nb))]
                        line_b=ha@E.T;line_a=hb@E;num=(hb*line_b).sum(1)**2;den=(line_a[:,:2]**2).sum(1)+(line_b[:,:2]**2).sum(1)
                        error=np.sqrt(num/np.maximum(den,1e-20))*float(np.array(ca['K'])[0,0])
                        ra=ha@A[:3,:3].T;rb=hb@B[:3,:3].T;ra/=np.linalg.norm(ra,axis=1)[:,None];rb/=np.linalg.norm(rb,axis=1)[:,None]
                        angles=np.arccos(np.clip((ra*rb).sum(1),-1,1))*180/np.pi
                        pair.update(status='measured',calibrated_epipolar_median_px=float(np.median(error)),calibrated_inlier_fraction=float((error<2).mean()),median_ray_angle_deg=float(np.median(angles)))
                        if (error<2).mean()<.5:flags.append('camera_pose_or_correspondence_consistency_review')
                    else:pair['status']='low_baseline_or_matches_not_calibration_failure'
        pairs.append(pair)
    med_area=float(np.median([r['area'] for r in target_rows])) if target_rows else 0
    med_speed=float(np.median(target_speed)) if target_speed else 0
    border_fraction=float(np.mean([r['border_touch'] for r in target_rows])) if target_rows else 1.
    motion_data=len(indices)>=40 and run>=35 and gaps<=2 and med_area>=300 and border_fraction<=.1 and not any('target_' in f for f in flags)
    delete_data=len(indices)>=15 and med_area>=100
    low_parallax=not any(p.get('median_ray_angle_deg',0)>=.5 and p.get('translation_baseline_m',0)>=.5 for p in pairs)
    difficulty=['occluded_background_status_pending','measured_road_support_pending']
    if low_parallax:difficulty.append('low_visual_parallax_keep_as_stratum')
    if indices and np.mean([r['clipped']<.9 for r in target_rows])>.1:difficulty.append('frequent_image_boundary_clipping')
    if border_fraction>.1:difficulty.append('frequent_border_truncation_keep_as_hard_asset_case')
    operations={}
    for op,eligible,dependencies in [
        ('no_edit',not structural,['native_reconstruction_QA']),
        ('delete_vehicle',delete_data,['Gaussian_instance_binding','reveal_domain_reference_or_unknown_label']),
        ('translate_vehicle',motion_data,['Gaussian_instance_binding','asset_view_coverage','destination_support_and_collision']),
        ('change_speed',motion_data and med_speed>=1,['Gaussian_instance_binding','temporal_asset_support','speed_trajectory_contract']),
        ('turn_vehicle',motion_data and len(heading_rows)>=20,['Gaussian_instance_binding','asset_orientation_coverage','turn_support_and_collision']),
        ('insert_asset',delete_data,['source_asset_QA','destination_support_and_collision']),
        ('camera_lateral',not structural,['observed_3D_camera_volume','disocclusion_unknown_domain']),
        ('camera_yaw',not structural,['observed_view_coverage','disocclusion_unknown_domain']),
        ('continuous_combination',motion_data,['individual_operation_QA','persistent_request_sequence','instance_conflict_checks']),
        ('repeated_reveal',delete_data,['world_region_identity','multi_request_comparison'])]:
        operations[op]=dict(raw_data_eligible=bool(eligible and not structural),status='prerequisites_pending' if eligible and not structural else 'raw_data_not_suitable_for_this_operation',pending=dependencies,formal_ready=False)
    result=dict(context=key,split=scene['split'],raw_input_status='structurally_invalid' if structural else ('review_required' if flags else 'raw_checks_passed'),
                structural_failures=structural,review_flags=sorted(set(flags)),image_metrics=frames,pose_max_speed_mps=float(ego_speed.max()),
                target=dict(track=scene['selected_track'],presence=len(indices),longest_contiguous=run,maximum_internal_gap_frames=int(gaps),median_area_at480=med_area,median_world_speed_mps=med_speed,max_acceleration_mps2=max(acceleration,default=0),heading_samples=len(heading_rows),border_touch_fraction=border_fraction),
                image_geometry_pairs=pairs,difficulty=difficulty,operations=operations,formal_admitted=False,
                does_not_use_edit_outputs=True,human_input_QA=None,geometry_support_QA='pending measured LiDAR visibility / native observed-region QA',
                caveats=['focus/exposure are review signals, not semantic proof of blur','epipolar mismatch can arise from rolling shutter or correspondences; not automatic calibration rejection','missing road support remains a hard case, not input corruption','physical-route overlap pending'])
    destination.mkdir(parents=True,exist_ok=True)
    sheet=Image.new('RGB',(3*480,350),'#132436');draw=ImageDraw.Draw(sheet)
    for j,i in enumerate([0,24,48]):
        draw.text((j*480+5,7),key+' / raw'+str(sensor_rows[i]['raw']),fill='white');sheet.paste(Image.fromarray(images[i]),(j*480,27))
    sheet.save(destination/'input_contact.jpg',quality=92)
    write(destination/'screen.json',result);print(key,result['raw_input_status'],len(flags),'flags',flush=True);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','sensors','quality'],default='all');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);manifest=read(DATA/'manifest.json')
    if args.stage=='all':
        subprocess.run(['/root/autodl-tmp/envs/dggt_data_cpu/bin/python',__file__,'--stage','sensors'],check=True)
        subprocess.run([sys.executable,__file__,'--stage','quality'],check=True);return
    worker=sensor_worker if args.stage=='sensors' else quality_worker
    results=[];errors=[]
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(worker,s):s['context'] for s in manifest['scenes']}
        for future in futures.as_completed(jobs):
            try:results.append(future.result())
            except Exception as e:errors.append(dict(context=jobs[future],error=repr(e)))
            write(OUT/'status.json',dict(phase=args.stage,completed=len(results),total=36,errors=errors))
    assert not errors,errors
    if args.stage=='quality':
        counts={status:sum(r['raw_input_status']==status for r in results) for status in sorted(set(r['raw_input_status'] for r in results))}
        operations={op:dict(raw_data_eligible=sum(r['operations'][op]['raw_data_eligible'] for r in results),formal_ready=0) for op in results[0]['operations']}
        write(OUT/'registry.json',dict(policy='ADMISSION_POLICY.json',counts=counts,operations=operations,scenes=sorted(results,key=lambda x:x['context']),formal_admitted=0,
                                     candidate_pool=36,method_outputs_used_for_selection=False,legacy_examples_preserved=True,quality_accepted=False,human_verdict=None))
        write(OUT/'status.json',dict(phase='input_and_operation_screen_complete',counts=counts,formal_admitted=0))


if __name__=='__main__':main()
