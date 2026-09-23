"""将用户反馈和 DriverQ 11 场景查询结果写成新的24例待审版；零模型调用。"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
from shapely.geometry import Point, Polygon

from prepare_omnidreams_review_candidates import build_candidate, planned_pose, save, write_report
from prepare_omnidreams_cfbench import pose_at

BASE = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02')
OLD = BASE/'20260923-proposal-r3-replace-actor02'
NEW = BASE/'20260923-proposal-r4-driverq-review'
INPUT = Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/adapter_inputs/omnidreams/index.json')
DB = Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/driverq/processed-11-scenes.db')
V5 = '/root/autodl-tmp/data/worldsim_v5/drivestudio_processed_10Hz/trainval'
MAP = Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/map_metadata/boston-seaport.json')

# 与原 case 比较：新源由 DriverQ 查询的场景库筛出，原源保持处只修订干预方向文字/坐标。
CHANGES = {
    'CFB-SPEED-ACTOR-02': dict(revision='R5',scene='756',camera=0,event=25,key='11',
        reason='DriverQ 重新筛为 scene-0998 前视、单个突出汽车；替换先前与第3例重复的巴士窗口。0.5秒起加速1.5×。'),
    'CFB-SPEED-ACTOR-03': dict(revision='R4',scale=1.5,
        reason='采纳你的人工观察：同一段慢速巴士原片，反事实由0.5×减速改为1.5×加速。'),
    'CFB-LATERAL-EGO-01': dict(revision='R4',scene='425',camera=0,event=95,
        reason='发现旧版把自车雷达坐标+Y前进误当横向；换成 DriverQ 筛出的运动自车片段，+3.5米明确为向左。'),
    'CFB-LATERAL-EGO-02': dict(revision='R4',scene='382',camera=0,event=65,
        reason='按你的要求整段换为 scene-0471 行驶中的自车；−3.5米是沿行驶方向向右。'),
    'CFB-LATERAL-EGO-03': dict(revision='R4',
        reason='原片保留；修正自车横移轴：+3.5米是沿行驶方向向左，不是沿雷达局部Y前进。'),
    'CFB-LATERAL-ACTOR-01': dict(revision='R4',scene='425',camera=5,event=45,key='1',
        reason='按你的要求换掉刁钻的右前视；DriverQ 筛出 scene-0535 后视中持续可见的汽车，−3.5米是汽车自身向右。'),
    'CFB-LATERAL-ACTOR-02': dict(revision='R4',
        reason='原片与偏移保留；把+3.5米明确标成目标车沿行驶方向向左。'),
    'CFB-LATERAL-ACTOR-03': dict(revision='R4',
        reason='原片与偏移保留；把−3.5米明确标成目标骑行者沿行驶方向向右。'),
}


def driveable_polygons():
    m=json.loads(MAP.read_text())
    nodes={x['token']:(x['x'],x['y']) for x in m['node']}
    polygons={x['token']:x for x in m['polygon']}
    result=[]
    for token in (token for area in m['drivable_area'] for token in area['polygon_tokens']):
        p=polygons[token]
        try:
            poly=Polygon([nodes[n] for n in p['exterior_node_tokens']],
                         [[nodes[n] for n in h['node_tokens']] for h in p['holes']])
        except (KeyError,ValueError):
            continue
        if poly.is_valid and not poly.is_empty:
            result.append(poly)
    return result


def road_center_check(entry, polygons):
    case=entry['case']; root=Path(case['dataset']['root'])/case['dataset']['scene_id']
    key=case['target'].get('actor_key')
    if key is None:
        poses={int(p.stem):np.loadtxt(p) for p in (root/'lidar_pose').glob('*.txt')}
    else:
        ann=json.loads((root/'instances/instances_info.json').read_text())[key]['frame_annotations']
        poses={int(f):np.asarray(p) for f,p in zip(ann['frame_idx'],ann['obj_to_world'])}
    start=entry['source_frame_range'][0]
    values={name:[] for name in ['factual','counterfactual']}
    for f in range(start,start+100,5):
        for name,p in [('factual',pose_at(poses,float(f))),('counterfactual',planned_pose(case,poses,float(f)))]:
            if p is None:
                values[name].append(False)
                continue
            point=Point(float(p[0,3]),float(p[1,3]))
            values[name].append(any(poly.covers(point) for poly in polygons))
    return {name:{'in_drivable_centers':sum(v),'sampled_centers':len(v)} for name,v in values.items()}


def query_evidence(conn, entry):
    case=entry['case']; scene=entry['scene_name']; token=conn.execute(
        'SELECT scene_token FROM scenes WHERE scene_name=?',(scene,)).fetchone()[0]
    start,stop=entry['source_frame_range']
    if case['target']['role']=='ego':
        rows=conn.execute('SELECT ego_speed FROM ego_poses WHERE scene_token=? AND frame_idx BETWEEN ? AND ?',
                          (token,start,stop)).fetchall()
        assert len(rows)==100
        speeds=[r[0] for r in rows]
        return {'tool':'DriverQ SQLite pose query (DriveStudio 10Hz adapter)',
                'mean_speed_mps':float(np.mean(speeds)),
                'stopped_frames_below_0_5_mps':sum(v<.5 for v in speeds),
                'summary':f'DriverQ 自车运动查询：平均{np.mean(speeds):.1f}米/秒，10秒内低于0.5米/秒的帧数{sum(v<.5 for v in speeds)}/100。'}
    actor=case['target']['entity_id']; camera=entry['camera_name']
    rows=conn.execute('SELECT (bbox_x2-bbox_x1)*(bbox_y2-bbox_y1) area FROM visibility WHERE scene_token=? AND instance_token=? AND camera=? AND frame_idx BETWEEN ? AND ?',
                      (token,actor,camera,start,stop)).fetchall()
    assert rows
    return {'tool':'DriverQ SQLite projected visibility query (DriveStudio 10Hz adapter)',
            'visible_keyframes':len(rows),'total_sampled_keyframes':20,
            'median_projected_area_px2':float(np.median([r[0] for r in rows])),
            'summary':f'DriverQ 几何投影：目标在{len(rows)}/20个采样帧进入{camera}画面，框面积中位数{np.median([r[0] for r in rows]):.0f}px²；遮挡仍以原视频人工判断。'}


def main():
    assert DB.is_file()
    old=json.loads((OLD/'candidates.json').read_text())
    assert old['case_count']==24 and old['model_calls']==0
    assert all(row['approval']=={'status':'pending','reviewer':'user','inference_allowed':False} for row in old['cases'])
    inputs=json.loads(INPUT.read_text())['cases']
    by_parent={row['case_id']:row for row in inputs}
    conn=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
    polygons=driveable_polygons()
    assert len(polygons)>0
    NEW.mkdir(parents=True,exist_ok=True)
    assert not any(NEW.iterdir()), 'Refuse to overwrite a partially generated review run.'
    (NEW/'cases').mkdir()
    updated=[]
    replacement=[]
    for previous in old['cases']:
        parent=previous['parent_case_id']
        if parent not in CHANGES:
            (NEW/'cases'/previous['case_id']).symlink_to(OLD/'cases'/previous['case_id'],target_is_directory=True)
            updated.append(previous)
            continue
        plan=CHANGES[parent]
        source=copy.deepcopy(by_parent[parent])
        case=source['case']
        if 'scene' in plan:
            case['dataset']['root']=V5
            case['dataset']['scene_id']=plan['scene']
            case['anchor']['event_frame']=plan['event']
            source['camera_index']=plan['camera']
        if 'scale' in plan:
            case['intervention']['counterfactual']['speed_scale']=plan['scale']
            case['expected_outcome']['direction']='longer progress'
        if case['intervention']['family']=='actor_lateral_relocation':
            case['intervention']['coordinate_convention']='vehicle_forward_left_v2'
        if 'key' in plan:
            root=Path(case['dataset']['root'])/case['dataset']['scene_id']
            item=json.loads((root/'instances/instances_info.json').read_text())[plan['key']]
            case['target'].update(actor_key=plan['key'],entity_id=item['id'],class_name=item['class_name'])
            case['expected_outcome']['affected_entities']=[item['id']]
        if parent=='CFB-LATERAL-EGO-01':
            # +3.5 = 左，原版+Y实际上是前进；新场景的该方向中心在可行驶区。
            pass
        reuse=(OLD/'cases'/previous['case_id']) if 'scene' not in plan else None
        entry=build_candidate(source,inputs[0],NEW,revision=plan['revision'],reuse_original_from=reuse)
        entry['supersedes_case_id']=previous['case_id']
        entry['revision_reason']=plan['reason']
        entry['selection_evidence']=query_evidence(conn,entry)
        entry['show_target_reference']=case['target']['role']!='ego'
        if 'scene' in plan:
            entry['prior_short_window_input_audit']={}
            entry['warnings']=[w for w in entry['warnings'] if '旧短窗检查' not in w]
        if case['intervention']['family']=='actor_lateral_relocation':
            entry['input_check']['drivable_center_2hz']=road_center_check(entry,polygons)
            if parent in {'CFB-LATERAL-EGO-01','CFB-LATERAL-EGO-02','CFB-LATERAL-EGO-03','CFB-LATERAL-ACTOR-01'}:
                assert entry['input_check']['drivable_center_2hz']['counterfactual']['in_drivable_centers']==20, parent
        if parent in {'CFB-LATERAL-EGO-01','CFB-LATERAL-EGO-02'}:
            assert entry['selection_evidence']['stopped_frames_below_0_5_mps']==0
        if parent=='CFB-LATERAL-ACTOR-01':
            assert entry['selection_evidence']['visible_keyframes']==20
            assert entry['input_check']['target_projection_visibility']['counterfactual']['visible_frames']==100
        if parent=='CFB-SPEED-ACTOR-02':
            assert entry['selection_evidence']['visible_keyframes']==20
            assert entry['input_check']['target_projection_visibility']['counterfactual']['visible_frames']>=80
        if parent=='CFB-SPEED-ACTOR-03':
            assert entry['case']['intervention']['counterfactual']['speed_scale']==1.5
        save(NEW/'cases'/entry['case_id']/'proposal.json',entry)
        updated.append(entry)
        replacement.append({'old':previous['case_id'],'new':entry['case_id'],'reason':plan['reason']})
    conn.close()
    result=copy.deepcopy(old)
    result.update(run_id=NEW.name,created_utc=datetime.now(timezone.utc).isoformat(),
        previous_manifest=str(OLD/'candidates.json'),cases=updated,case_count=24,
        replacement=replacement,model_calls=0,new_ai_video_reviews=0,
        revision_summary='按你的反馈修订8例：巴士改加速，换掉静止自车/看不清的目标，用 DriverQ 重选两个速度/横移来源，修正全部横移的左右定义。此页只展示原始10秒视频和计划；推理结果与评分仍空。')
    shutil.copy2(OLD/'human-review-original-four.json',NEW/'human-review-original-four.json')
    save(NEW/'candidates.json',result)
    write_report(NEW,result)
    print(json.dumps({'run_id':NEW.name,'cases':24,'revised':len(replacement),
                      'model_calls':0,'approval':'pending','new_case_ids':[r['new'] for r in replacement]},ensure_ascii=False))


def finalize_existing():
    """给早于坐标契约字段写入的同一待审输出补齐显式契约；不改视频。"""
    manifest=json.loads((NEW/'candidates.json').read_text())
    assert manifest['model_calls']==0 and all(not r['approval']['inference_allowed'] for r in manifest['cases'])
    for entry in manifest['cases']:
        if entry['parent_case_id'] not in CHANGES or entry['case']['intervention']['family']!='actor_lateral_relocation':
            continue
        entry['case']['intervention']['coordinate_convention']='vehicle_forward_left_v2'
        save(NEW/'cases'/entry['case_id']/'proposal.json',entry)
    save(NEW/'candidates.json',manifest)
    write_report(NEW,manifest)
    print(json.dumps({'explicit_lateral_coordinate_convention':6,'model_calls':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--finalize-existing',action='store_true')
    args=parser.parse_args()
    finalize_existing() if args.finalize_existing else main()
