"""在全部原片段中按合法输入选49帧准备窗口；不选择共享路由，不读取编辑结果。"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np

def build(root):
    all_rows=[];scenes=[]
    recipe=dict(version='preedit_49frame_preparation_queue_v1',window_frames=49,
                allowed_source='raw annotation-assisted track/camera/projection and box-LiDAR-count only',
                per_frame_support='sqrt(clipped bbox/image area) * min(box LiDAR count/50,1) * (1-projected-box-overlap proxy) * (1-box truncation)',
                window_selection='earliest maximum total support over49consecutive raw frames',
                actor_selection='top3 distinct vehicles per scene among existing preedit screen candidates; ranked by support in recommended window',
                route_selection=False,formal_eligibility=None,requires_before_new_native_edit=['target mask or explicitly registered alternative scope','frozen candidate row map and H0','native restore and actual shared none'],
                background_and_unseen_surface_support='unknown, not inferred from boxes')
    for folder in sorted(root.glob('scene_*')):
        if not (folder/'summary.json').exists():continue
        data=json.loads((folder/'actors.json').read_text());summary=json.loads((folder/'summary.json').read_text());n=summary['frames']
        if n<49:
            raise ValueError('49帧准备窗口要求完整原片段至少49帧，不能用卷积结果伪造不足帧数')
        byid={r['track_id']:[] for r in data['rows']}
        with gzip.open(folder/'all_vehicle_observations.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);byid[row['track_id']].append(row)
        ranked=[]
        for actor in data['rows']:
            score=np.zeros(n);visible=np.zeros(n,dtype=int);lidar=np.zeros(n)
            for row in byid[actor['track_id']]:
                idx=row['frame_index']
                if row['front_bbox_xyxy'] is None:continue
                visible[idx]=1;lidar[idx]=row['annotation_lidar_points_in_box']
                score[idx]=np.sqrt(row['front_bbox_visible_area_ratio'])*min(lidar[idx]/50,1)*(1-row['front_bbox_overlap_proxy'])*(1-row['front_bbox_truncation_fraction'])
            totals=np.convolve(score,np.ones(49),mode='valid');first=int(np.argmax(totals));last=first+48
            row=dict(scene_number=actor['scene_number'],context_id=actor['context_id'],track_id=actor['track_id'],
                     development_screen_candidate=not bool(actor['reasons']),screening_reasons=actor['reasons'],
                     total_raw_frames=n,total_front_projected_frames=actor['front_projected_frames'],
                     recommended_window_first=first,recommended_window_last=last,
                     recommended_window_front_frames=int(visible[first:last+1].sum()),
                     recommended_window_support_score=float(totals[first]/49),
                     recommended_window_lidar_positive_frames=int(((visible>0)&(lidar>0))[first:last+1].sum()),
                     median_front_bbox_area_ratio=actor['median_front_bbox_area_ratio'],
                     median_front_lidar_annotation_count=actor['median_front_lidar_annotation_count'],
                     front_view_sectors=actor['front_view_sectors'],
                     native_preparation_state='needs_GS_mask_and_frozen_target_candidate',formal_eligibility=None,
                     operation_eligibility={'delete':None,'shift':None,'rotate':None},
                     used_edit_result=False)
            ranked.append(row)
        ranked.sort(key=lambda r:(not r['development_screen_candidate'],-r['recommended_window_support_score'],r['track_id']))
        selected=[r for r in ranked if r['development_screen_candidate']][:3]
        for rank,row in enumerate(selected,1):row['preparation_priority_within_scene']=rank
        scenes.append(dict(scene_number=data['scene_number'],context_id=data['context_id'],all_vehicle_tracks=len(ranked),preedit_screen_candidates=sum(r['development_screen_candidate'] for r in ranked),selected_preparation_candidates=selected))
        all_rows.extend(ranked)
        (folder/'preparation_queue.json').write_text(json.dumps(dict(recipe=recipe,rows=ranked),ensure_ascii=False,indent=2)+'\n')
    result=dict(schema='horizonforge_preedit_preparation_queue_v1',recipe=recipe,scenes=scenes,registered_scenes=len(scenes),all_vehicle_tracks=len(all_rows),selected_preparation_candidates=sum(len(s['selected_preparation_candidates']) for s in scenes),formal_qualified_count=None,new_reconstruction_started=False)
    (root/'preedit_preparation_queue.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if all_rows:
        fields=sorted({k for r in all_rows for k in r})
        with (root/'all_actors_preparation_queue.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in all_rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    print(json.dumps(dict(scenes=len(scenes),actors=len(all_rows),selected=result['selected_preparation_candidates'])))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();build(args.root)
