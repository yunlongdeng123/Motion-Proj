"""完整原片段车辆盘点；只读原始传感器/标注，不读取编辑参考或模型结果。

标注框和标注LiDAR计数是明确的注释辅助筛查，既非预测mask也非正式R4资格。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import gzip
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import time
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DECODER = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_debug/followup_20261013/red_preparation/waymo_numpy_cpu.py')
RULE = dict(version='annotation_assisted_preedit_queue_v1', minimum_front_frames=15,
            minimum_median_front_bbox_area_ratio=.001, minimum_median_front_lidar_box_count=10,
            maximum_median_bbox_truncation_fraction=.25,
            ranking='front_frames * sqrt(median_front_bbox_area_ratio) * min(median_front_lidar_count/50,1) * (1-median_box_overlap_proxy)',
            formal_eligibility=None, background_exposure_support='unknown_not_measured',
            body_mask_support='unknown_projected_boxes_are_not_body_masks',
            prediction_deployment_claim=False)

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def median(values):
    return float(np.median(values)) if len(values) else None

def longest(indices):
    best = run = 0; previous = -2
    for index in sorted(indices):
        run = run + 1 if index == previous + 1 else 1
        best = max(best, run); previous = index
    return best

def intersection(a, b):
    return max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))

def scan(raw, context, scene_number, output):
    began = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    (output/'previews').mkdir(); write(output/'screening_rules.json', RULE)
    spec=importlib.util.spec_from_file_location('waymo_cpu_decoder', DECODER)
    decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    pb=decoder.protobuf();crc=decoder.crc_function()
    tracks={};frames=[];class_counts=Counter();offsets={};front_jpegs={}
    with raw.open('rb') as stream, gzip.open(output/'all_vehicle_observations.jsonl.gz','wt',encoding='utf-8') as observations:
        index=0
        while True:
            offset=stream.tell();header=stream.read(12)
            if not header:break
            if len(header)!=12:raise ValueError('truncated record header')
            size,check=struct.unpack('<QI',header)
            if not 0<size<96*1024**2 or crc(header[:8])!=check:raise ValueError('length CRC/layout')
            data=stream.read(size);trailer=stream.read(4)
            if len(data)!=size or len(trailer)!=4 or crc(data)!=struct.unpack('<I',trailer)[0]:raise ValueError('payload CRC')
            frame=pb.Frame.FromString(data)
            if frame.context.name!=context:raise ValueError('context identity changed')
            pose=np.asarray(frame.pose.transform,dtype=float).reshape(4,4)
            ego_yaw=math.atan2(pose[1,0],pose[0,0])
            calibrations={int(c.name):c for c in frame.context.camera_calibrations}
            front_cal=calibrations[1]; width,height=int(front_cal.width),int(front_cal.height)
            projections={}; front_boxes={}
            for camera in frame.projected_lidar_labels:
                name=pb.CameraName.Name.Name(camera.name)
                for label in camera.labels:
                    suffix='_'+name
                    track_id=label.id[:-len(suffix)] if label.id.endswith(suffix) else label.id
                    b=label.box
                    # Waymo二维box: length是横向宽度，width是纵向高度。
                    box=[float(b.center_x-b.length/2),float(b.center_y-b.width/2),float(b.center_x+b.length/2),float(b.center_y+b.width/2)]
                    projections.setdefault(track_id,{})[int(camera.name)]=box
                    if camera.name==1 and label.type==1:front_boxes[track_id]=box
            frame_vehicle_count=0
            for label in frame.laser_labels:
                class_counts[int(label.type)]+=1
                if label.type!=1:continue
                frame_vehicle_count+=1; tid=label.id; b=label.box
                center=pose@np.array([b.center_x,b.center_y,b.center_z,1.])
                yaw=ego_yaw+float(b.heading)
                cameras=projections.get(tid,{})
                bins={};front_box=cameras.get(1); area_ratio=None;truncation=None;overlap=None
                for camera_name in cameras:
                    calibration=calibrations[camera_name]
                    extrinsic=np.asarray(calibration.extrinsic.transform,dtype=float).reshape(4,4)
                    camera_world=pose@extrinsic[:,3]
                    angle=(math.atan2(camera_world[1]-center[1],camera_world[0]-center[0])-yaw)%(2*math.pi)
                    bins[str(camera_name)]=int(angle/(2*math.pi)*12)%12
                if front_box:
                    area=max(0.,front_box[2]-front_box[0])*max(0.,front_box[3]-front_box[1])
                    clipped=intersection(front_box,[0,0,width,height])
                    area_ratio=clipped/(width*height)
                    truncation=1.-clipped/area if area else 1.
                    overlap=max([intersection(front_box,other)/area for other_tid,other in front_boxes.items() if other_tid!=tid]+[0.]) if area else 1.
                record=dict(frame_index=index,timestamp_micros=int(frame.timestamp_micros),track_id=tid,
                            center_world_m=center[:3].tolist(),heading_world_rad=yaw,
                            size_lwh_m=[float(b.length),float(b.width),float(b.height)],
                            annotation_speed_mps=float(math.hypot(label.metadata.speed_x,label.metadata.speed_y)),
                            annotation_lidar_points_in_box=int(label.num_lidar_points_in_box),
                            camera_names=sorted(cameras),view_12sector_bins_by_camera=bins,
                            front_bbox_xyxy=front_box,front_bbox_visible_area_ratio=area_ratio,
                            front_bbox_truncation_fraction=truncation,front_bbox_overlap_proxy=overlap)
                if not np.isfinite(center).all():raise ValueError('nonfinite legal source pose')
                tracks.setdefault(tid,[]).append(record)
                observations.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            offsets[index]=offset
            frames.append(dict(frame_index=index,timestamp_micros=int(frame.timestamp_micros),source_offset=offset,vehicle_labels=frame_vehicle_count,front_projected_vehicle_labels=len(front_boxes),source_wh=[width,height]))
            index+=1
    rows=[]
    for tid,records in tracks.items():
        front=[r for r in records if r['front_bbox_xyxy'] is not None]
        front_demo=[r for r in front if r['frame_index']<=89]
        areas=[r['front_bbox_visible_area_ratio'] for r in front]
        lidar=[r['annotation_lidar_points_in_box'] for r in front]
        truncation=[r['front_bbox_truncation_fraction'] for r in front]
        overlap=[r['front_bbox_overlap_proxy'] for r in front]
        reasons=[]
        if len(front)<RULE['minimum_front_frames']:reasons.append('few_front_projected_frames')
        if not areas or median(areas)<RULE['minimum_median_front_bbox_area_ratio']:reasons.append('small_front_projected_box')
        if not lidar or median(lidar)<RULE['minimum_median_front_lidar_box_count']:reasons.append('sparse_annotation_lidar_box_count')
        if not truncation or median(truncation)>RULE['maximum_median_bbox_truncation_fraction']:reasons.append('projected_box_truncation')
        score=(len(front)*math.sqrt(median(areas))*min(median(lidar)/50,1)*(1-median(overlap))) if front else 0.
        best=max(front,key=lambda r:r['front_bbox_visible_area_ratio']*(1-r['front_bbox_overlap_proxy'])*min(r['annotation_lidar_points_in_box']/50,1)) if front else None
        centers=np.array([r['center_world_m'] for r in records]);times=np.array([r['timestamp_micros']/1e6 for r in records])
        distance=float(np.linalg.norm(np.diff(centers[:,:2],axis=0),axis=1).sum())
        gaps=np.diff([r['frame_index'] for r in records])
        all_bins=sorted({v for r in front for v in r['view_12sector_bins_by_camera'].values()})
        row=dict(scene_number=scene_number,context_id=context,track_id=tid,actor_type='vehicle',
                 source_role='Waymo_annotation_assisted_preedit_screen_not_prediction',
                 labelled_frames=len(records),total_raw_frames=len(frames),first_frame=records[0]['frame_index'],last_frame=records[-1]['frame_index'],
                 front_projected_frames=len(front),demo_first90_front_frames=len(front_demo),longest_consecutive_front_run=longest([r['frame_index'] for r in front]),
                 median_front_bbox_area_ratio=median(areas),max_front_bbox_area_ratio=max(areas) if areas else None,
                 median_front_lidar_annotation_count=median(lidar),minimum_front_lidar_annotation_count=min(lidar) if lidar else None,
                 front_frames_with_nonzero_lidar=sum(x>0 for x in lidar),median_bbox_truncation_fraction=median(truncation),median_bbox_overlap_proxy=median(overlap),
                 median_annotation_speed_mps=median([r['annotation_speed_mps'] for r in records]),world_path_length_m=distance,
                 max_track_frame_gap=int(max(gaps)) if len(gaps) else None,front_and_covisible_camera_view_sectors=all_bins,
                 front_view_sectors=sorted({r['view_12sector_bins_by_camera']['1'] for r in front}),
                 view_support_role='12sector_30degree_annotation_heading_camera_proxy_not_hidden_surface_support',
                 best_preview_frame=best['frame_index'] if best else None,best_preview_box=best['front_bbox_xyxy'] if best else None,
                 preedit_queue='raw_observation_candidate_needs_mask_and_GS' if not reasons else 'retain_with_missing_or_weak_support',
                 reasons=reasons,screening_score=score,formal_eligibility=None,
                 body_mask_support=None,background_exposure_support=None,native_GS_actor_mapping=None,
                 immediately_native_editable=False,operation_eligibility={'delete':None,'shift':None,'rotate':None})
        rows.append(row)
    rows.sort(key=lambda r:(bool(r['reasons']),-r['screening_score'],r['track_id']))
    selected=[r for r in rows if not r['reasons']][:6]
    if not selected:selected=[r for r in rows if r['front_projected_frames']][:3]
    selected_indices=sorted({r['best_preview_frame'] for r in selected if r['best_preview_frame'] is not None}|{0,min(44,len(frames)-1),min(89,len(frames)-1)})
    with raw.open('rb') as stream:
        for idx in selected_indices:
            stream.seek(offsets[idx]);header=stream.read(12);size,check=struct.unpack('<QI',header)
            payload=stream.read(size);trailer=stream.read(4)
            if crc(header[:8])!=check or crc(payload)!=struct.unpack('<I',trailer)[0]:raise ValueError('preview record CRC')
            frame=pb.Frame.FromString(payload);front=next(im for im in frame.images if im.name==1)
            image=Image.open(io.BytesIO(front.image)).convert('RGB');image.thumbnail((960,640))
            image.save(output/'previews'/f'front_{idx:03d}.jpg',quality=90)
    for rank,row in enumerate(selected,1):
        row['preview_rank']=rank
        idx=row['best_preview_frame'];image=Image.open(output/'previews'/f'front_{idx:03d}.jpg').convert('RGB')
        draw=ImageDraw.Draw(image);sx=image.width/frames[idx]['source_wh'][0];sy=image.height/frames[idx]['source_wh'][1]
        box=row['best_preview_box'];rect=[box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy]
        draw.rectangle(rect,outline='#00ffbb',width=3);label=f"#{rank} {row['track_id']} | frame {idx}"
        draw.rectangle([0,0,image.width,24],fill='#10202d');draw.text((8,6),label,fill='white')
        relative=f'previews/actor_{rank:02d}_frame_{idx:03d}.jpg';image.save(output/relative,quality=92);row['preview_image']=relative
    write(output/'actors.json',dict(schema='waymo_preedit_all_vehicle_screen_v1',context_id=context,scene_number=scene_number,rules=RULE,rows=rows))
    with (output/'actors.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=[k for k in rows[0] if k not in ('preview_rank','preview_image')] if rows else []
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    write(output/'frame_inventory.json',frames)
    summary=dict(schema='waymo_preedit_screen_summary_v1',scene_number=scene_number,context_id=context,raw_path=str(raw),raw_bytes=raw.stat().st_size,
                 frames=len(frames),crc_checked_records=len(frames),context_identity_checked_records=len(frames),
                 all_vehicle_tracks=len(rows),front_visible_vehicle_tracks=sum(r['front_projected_frames']>0 for r in rows),
                 front15plus_vehicle_tracks=sum(r['front_projected_frames']>=15 for r in rows),raw_observation_candidates=sum(not r['reasons'] for r in rows),
                 retained_weak_or_missing_tracks=sum(bool(r['reasons']) for r in rows),
                 all_label_class_observation_counts=dict(class_counts),vehicle_observations=sum(r['labelled_frames'] for r in rows),
                 scan_role='complete_raw_annotation_assisted_vehicle_denominator_no_GS_or_edit_GT',
                 legal_source_refs=[str(raw),str(DECODER)],used_edit_reference=False,used_render_or_student_output=False,formal_eligibility=None,
                 immediately_native_editable=0,previews=len(selected),wall_seconds=time.monotonic()-began,gpu_calls=0,training_updates=0)
    write(output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--context',required=True)
    parser.add_argument('--scene-number',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    scan(args.raw,args.context,args.scene_number,args.output)
