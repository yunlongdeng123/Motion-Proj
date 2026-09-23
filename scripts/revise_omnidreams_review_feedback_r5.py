"""Apply the user's second-round case review without running any model."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box

from mine_lateral_review import V5, OLD as LEGACY, footprint, local_road, evaluate
from prepare_omnidreams_review_candidates import build_candidate, planned_pose, save, write_report
from prepare_omnidreams_cfbench import pose_at

BASE = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02')
PREVIOUS = BASE/'20260923-proposal-r4-driverq-review'
OUTPUT = BASE/'20260923-proposal-r5-human-feedback'
INDEX = Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/adapter_inputs/omnidreams/index.json')
DB = Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/driverq/processed-11-scenes.db')
MAP_INDEX = json.loads((Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/map_metadata_driverq11')/'scene-map-index.json').read_text())

# start is the first 10-Hz source frame; event is the first counterfactual frame.
PLANS = {
    'CFB-SPEED-ACTOR-02': dict(revision='R6', scene='756', start=50, event=110, key='5', camera=0,
        reason='按截图改框黑色汽车 #5。原片窗口从帧20–119后移3秒到帧50–149；黑车约5.5秒入画，6秒起才加速1.5×，避免在不可见时编辑。', reference=110),
    'CFB-LATERAL-EGO-02': dict(revision='R5', scene='382', start=60, event=65, camera=0, offset=3.5,
        reason='右移3.5米与施工区车辆/锥桶冲突；按截图改为向左3.5米。核对整车占地与标注车辆，另附俯视绿色目标车框。'),
    'CFB-LATERAL-ACTOR-03': dict(revision='R5', scene='350', start=0, event=5, key='9', camera=0, offset=3.5,
        reason='原骑行者右移会落入路外，整段换为 scene-0436 前视的行驶汽车 #9；向左3.5米，计划车框在可行驶区且未与标注目标相交。', reference=23),
    'CFB-REMOVE-02': dict(revision='R4', scene='204', start=40, event=45, key='49', camera=0,
        reason='换掉停车场车辆；选择前视经过自车行进走廊的动态汽车 #49，删除与通行判断有潜在关系。', reference=65),
    'CFB-REMOVE-03': dict(revision='R4', scene='350', start=0, event=5, key='9', camera=0,
        reason='换掉后视无关车辆；选择前视、距自车约12米的行驶汽车 #9。', reference=15),
    'CFB-REMOVE-04': dict(revision='R4', scene='350', start=90, event=95, key='0', camera=0,
        reason='换掉后视无关车辆；选择前视、距自车约11米的前车 #0。', reference=105),
    'CFB-REMOVE-05': dict(revision='R4', scene='296', start=60, event=65, key='21', camera=0,
        reason='换掉侧后方停车目标；选择日间前视、在自车前方约29米的摩托车 #21，供人工判断是否影响避让。', reference=85),
    'CFB-REMOVE-06': dict(revision='R4', scene='179', start=0, event=5, key='28', camera=0,
        reason='截图指出原施工车辆 #20 框错。更正为前方道路施工设备 #28，并在更清晰的第25帧画目标框；仍待人工核实规划关联。', reference=25),
    'CFB-INSERT-06': dict(revision='R4', scene='350', start=0, event=5, key='9', camera=0, offset_xyz=[8.0, 3.5, 0.0],
        reason='换掉自车静止片段；以 scene-0436 行驶汽车 #9 为供体，在其前方8米、左侧3.5米新增一车。自车10秒平均约5米/秒。', reference=5),
}


def scene_root(scene: str) -> Path:
    return (LEGACY if scene in {'179','191','204'} else V5)/scene


def amend_source(parent: str, source: dict, plan: dict) -> dict:
    source = copy.deepcopy(source)
    case = source['case']
    scene = plan['scene']
    root = scene_root(scene)
    scene_name = {'179':'scene-0230','204':'scene-0255','296':'scene-0379',
                  '350':'scene-0436','382':'scene-0471','756':'scene-0998'}[scene]
    case['dataset']['root'] = str(root.parent)
    case['dataset']['scene_id'] = scene
    case['anchor']['event_frame'] = plan['event']
    case['anchor']['pre_frames'] = plan['event']-plan['start']
    source.update(camera_index=plan['camera'], scene_name=scene_name,
                  location=MAP_INDEX[scene_name]['location'],
                  review_reference_frame=plan.get('reference'))
    if 'key' in plan:
        item = json.loads((root/'instances/instances_info.json').read_text())[plan['key']]
        target = case['target']
        target['class_name'] = item['class_name']
        if case['intervention']['family']=='actor_insertion':
            target['source_asset_actor_key'] = plan['key']
            target['source_asset_actor_id'] = item['id']
            target['entity_id'] = 'inserted:'+item['id']
            target['proposal_offset_actor_frame_m'] = plan['offset_xyz']
        else:
            target['actor_key'] = plan['key']
            target['entity_id'] = item['id']
        case['expected_outcome']['affected_entities'] = [target['entity_id']]
        source['target_key'] = plan['key']
    if 'offset' in plan:
        case['intervention']['counterfactual']['lateral_offset_m'] = plan['offset']
        case['intervention']['coordinate_convention'] = 'vehicle_forward_left_v2'
        case['expected_outcome']['direction'] = 'leftward displacement'
    return source


def ego_footprint(pose):
    # DriveStudio LiDAR: +X right, +Y forward. Conservative 4.5 x 2.0 m ego box.
    return Polygon([tuple((pose @ np.array([x, y, 0, 1]))[:2])
                    for x,y in [(-1,-2.25),(1,-2.25),(1,2.25),(-1,2.25)]])


def ego_gate(entry):
    case = entry['case']
    root = scene_root(case['dataset']['scene_id'])
    poses = {int(p.stem): np.loadtxt(p) for p in (root/'lidar_pose').glob('*.txt')}
    start = entry['source_frame_range'][0]
    xy = np.array([poses[f][:2,3] for f in range(start,start+100,5)])
    road = local_road(MAP_INDEX[entry['scene_name']]['location'],
                      box(xy[:,0].min()-15,xy[:,1].min()-15,xy[:,0].max()+15,xy[:,1].max()+15))
    info = json.loads((root/'instances/instances_info.json').read_text())
    road_count = 0
    collisions = []
    for frame in range(start,start+100,5):
        planned = planned_pose(case,poses,float(frame))
        area = ego_footprint(planned)
        road_count += road.covers(area)
        for key,item in info.items():
            ann = item['frame_annotations']
            if frame not in ann['frame_idx']:
                continue
            i = ann['frame_idx'].index(frame)
            other = footprint(np.asarray(ann['obj_to_world'][i]),ann['box_size'][i])
            if area.intersects(other):
                collisions.append([frame,key])
    return {'road_footprint_2hz':road_count,'sampled_frames':20,
            'annotated_actor_intersections_2hz':collisions,
            'unannotated_cones_checked_by_human_video':False}


def ego_bev(entry):
    case = entry['case']
    root = scene_root(case['dataset']['scene_id'])
    poses = {int(p.stem): np.loadtxt(p) for p in (root/'lidar_pose').glob('*.txt')}
    event = case['anchor']['event_frame']
    mark = event+18
    actual = pose_at(poses,float(mark))
    planned = planned_pose(case,poses,float(mark))
    center = (actual[:2,3]+planned[:2,3])/2
    right,forward = actual[:2,0],actual[:2,1]
    width,height,scale = 900,650,13
    def pixel(point):
        delta = np.asarray(point[:2])-center
        return (int(width/2+scale*float(delta@right)),
                int(height/2-scale*float(delta@forward)))
    region = box(center[0]-45,center[1]-45,center[0]+45,center[1]+45)
    road = local_road(MAP_INDEX[entry['scene_name']]['location'],region)
    image = Image.new('RGB',(width,height),'#f0f2f3')
    draw = ImageDraw.Draw(image)
    polygons = list(road.geoms) if hasattr(road,'geoms') else [road]
    for poly in polygons:
        if hasattr(poly,'exterior'):
            draw.polygon([pixel(p) for p in poly.exterior.coords],fill='#cad4d8',outline='#99aab1')
    # Show a short trajectory around the edit, not a static impossible camera-frame ego box.
    for label,color,fn in [('FACTUAL','#eab308',lambda f: pose_at(poses,float(f))),
                           ('PLANNED','#22c55e',lambda f: planned_pose(case,poses,float(f)))]:
        points = [pixel(fn(f)[:2,3]) for f in range(event,min(event+45,max(poses))+1,3) if fn(f) is not None]
        if len(points)>1:
            draw.line(points,fill=color,width=5)
    for pose,color in [(actual,'#facc15'),(planned,'#16a34a')]:
        area = ego_footprint(pose)
        draw.line([pixel(p) for p in area.exterior.coords],fill=color,width=8,joint='curve')
    draw.rectangle((15,15,480,86),fill='#19242a')
    draw.text((28,24),'YELLOW factual ego     GREEN planned ego',fill='white')
    draw.text((28,51),f'At t={(mark-entry["source_frame_range"][0])/10:.1f}s  |  + means vehicle left',fill='white')
    out = OUTPUT/'cases'/entry['case_id']/'target-reference.jpg'
    image.save(out,quality=92)
    entry['target_reference_s']=(mark-entry['source_frame_range'][0])/10
    entry['target_reference_kind']='ego_bev'
    entry['show_target_reference']=True


def add_evidence(entry, conn):
    case = entry['case']
    family = case['intervention']['family']
    if case['target']['role']=='ego':
        entry['input_check']['ego_full_footprint_and_actor_gate_2hz']=ego_gate(entry)
        gate=entry['input_check']['ego_full_footprint_and_actor_gate_2hz']
        if entry['parent_case_id']=='CFB-LATERAL-EGO-02':
            assert gate['road_footprint_2hz']==20 and not gate['annotated_actor_intersections_2hz']
        entry['selection_evidence']={'summary':f'计划车身在可行驶多边形内 {gate["road_footprint_2hz"]}/20 帧；与标注对象相交 {len(gate["annotated_actor_intersections_2hz"])} 次。锥桶仍按原片人工确认。'}
        return
    if family=='actor_lateral_relocation':
        root=scene_root(case['dataset']['scene_id'])
        result=evaluate(case['dataset']['scene_id'],case['target']['actor_key'],
                        MAP_INDEX[entry['scene_name']]['location'],entry['source_frame_range'][0],root.parent)
        gate=result[str(case['intervention']['counterfactual']['lateral_offset_m'])]
        assert gate['road_footprint_2hz']==20 and not gate['collisions_2hz'] and gate['visible_2hz']==20
        entry['input_check']['actor_full_footprint_and_collision_gate_2hz']=gate
        entry['selection_evidence']={'summary':'计划车身在可行驶区20/20帧、前视几何可见20/20帧、与标注对象相交0次（2Hz）；仍需人工核对无遮挡。'}
        return
    if family=='actor_insertion':
        entry['selection_evidence']={'summary':'DriverQ 轨迹筛出行驶场景；计划新车车身于2Hz采样19/19帧在可行驶区，和现有标注对象无交叠。需人工确认语义/规划价值。'}
        return
    if family=='actor_speed_change':
        entry['selection_evidence']={'summary':'黑车实例ID为 #5；从原始窗口删前3秒、末尾补3秒，黑车约5.5秒可见，6秒开始加速。'}
        return
    root=scene_root(case['dataset']['scene_id'])
    frame=case['anchor']['event_frame']
    lidar=np.loadtxt(root/f'lidar_pose/{frame:03d}.txt')
    ann=json.loads((root/'instances/instances_info.json').read_text())[case['target']['actor_key']]['frame_annotations']
    i=ann['frame_idx'].index(frame)
    point=np.asarray(ann['obj_to_world'][i])[:3,3]-lidar[:3,3]
    forward=float(point@lidar[:3,1]);left=float(point@(-lidar[:3,0]))
    ego_speeds=conn.execute('select ego_speed from ego_poses where scene_token=? and frame_idx between ? and ?',
         (MAP_INDEX[entry['scene_name']]['scene_token'],*entry['source_frame_range'])).fetchall()
    avg=float(np.mean([x[0] for x in ego_speeds]))
    entry['selection_evidence']={'front_gap_m_at_event':forward,'left_offset_m_at_event':left,
       'mean_ego_speed_mps':avg,
       'summary':f'目标在编辑起点位于自车前方{forward:.1f}米、横向{left:+.1f}米；原片自车平均{avg:.1f}米/秒。是否足以影响规划仍由你审核。'}


def copy_ego_reference(previous):
    olddir=PREVIOUS/'cases'/previous['case_id']
    newdir=OUTPUT/'cases'/previous['case_id']
    newdir.mkdir()
    for name in ['original-nuscenes.mp4','original-nuscenes.json']:
        os.link(olddir/name,newdir/name)
    entry=copy.deepcopy(previous)
    ego_bev(entry)
    save(newdir/'proposal.json',entry)
    return entry


def main():
    old=json.loads((PREVIOUS/'candidates.json').read_text())
    assert old['case_count']==24 and old['model_calls']==0
    assert all(not r['approval']['inference_allowed'] for r in old['cases'])
    inputs={r['case_id']:r for r in json.loads(INDEX.read_text())['cases']}
    assert not OUTPUT.exists(), 'Never overwrite an existing review run.'
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'cases').mkdir()
    conn=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
    updated=[];replacements=[]
    for previous in old['cases']:
        parent=previous['parent_case_id']
        if parent in {'CFB-LATERAL-EGO-01','CFB-LATERAL-EGO-03'}:
            updated.append(copy_ego_reference(previous))
            continue
        if parent not in PLANS:
            (OUTPUT/'cases'/previous['case_id']).symlink_to(PREVIOUS/'cases'/previous['case_id'],target_is_directory=True)
            updated.append(previous)
            continue
        plan=PLANS[parent]
        source=amend_source(parent,inputs[parent],plan)
        entry=build_candidate(source,inputs['CFB-SPEED-EGO-01'],OUTPUT,revision=plan['revision'])
        assert entry['source_frame_range']==[plan['start'],plan['start']+99],parent
        entry['supersedes_case_id']=previous['case_id']
        entry['revision_reason']=plan['reason']
        entry['prior_short_window_input_audit']={}
        entry['warnings']=[w for w in entry['warnings'] if '旧短窗检查' not in w]
        entry['show_target_reference']=True
        add_evidence(entry,conn)
        if case_family:=entry['case']['intervention']['family']:
            if case_family=='actor_lateral_relocation' and entry['case']['target']['role']=='ego':
                ego_bev(entry)
        save(OUTPUT/'cases'/entry['case_id']/'proposal.json',entry)
        updated.append(entry)
        replacements.append({'old':previous['case_id'],'new':entry['case_id'],'reason':plan['reason']})
    conn.close()
    result=copy.deepcopy(old)
    result.update(run_id=OUTPUT.name,created_utc=datetime.now(timezone.utc).isoformat(),
        previous_manifest=str(PREVIOUS/'candidates.json'),cases=updated,case_count=24,
        replacement=replacements,model_calls=0,new_ai_video_reviews=0,
        revision_summary='按第二轮截图反馈修订9例：黑车目标与时间窗、左移自车/行驶汽车、5个移除目标、动态插入场景。全部仍是待审提案，未跑推理；factual、counterfactual和评分保持空白。')
    shutil.copy2(PREVIOUS/'human-review-original-four.json',OUTPUT/'human-review-original-four.json')
    save(OUTPUT/'candidates.json',result)
    write_report(OUTPUT,result)
    print(json.dumps({'cases':24,'revised':len(replacements),'no_model_calls':True,
                      'run':str(OUTPUT),'ids':[r['new'] for r in replacements]}),flush=True)


if __name__=='__main__':
    main()
