"""Replace three user-rejected insertion cases; no inference or video scoring."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from mine_insertion_review import V5, evaluate, iou
from motion_proj.cfbench.geometry import project_box, shifted_pose
from prepare_omnidreams_review_candidates import build_candidate, save, write_report
from revise_omnidreams_review_feedback_r5 import BASE, INDEX, MAP_INDEX

PREVIOUS = BASE/'20260923-proposal-r6-human-feedback'
OUTPUT = BASE/'20260923-proposal-r7-insertion-review'

PLANS = {
    'CFB-INSERT-02': dict(revision='R4', scene='425', scene_name='scene-0535', key='1', camera=5,
        start=90, event=95, offset=[-4.0,3.5,0.0], reference=110,
        reason='整段换为scene-0535后视行驶场景。黄框是清晰的行驶汽车 #1；绿框在其后方4米、左侧3.5米的相邻可行驶位置，原车保留。'),
    'CFB-INSERT-04': dict(revision='R4', scene='350', scene_name='scene-0436', key='0', camera=0,
        start=90, event=95, offset=[8.0,3.5,0.0], reference=140,
        reason='整段换为scene-0436前视行驶场景。黄框是前车 #0；绿框在其前方8米、左侧3.5米，框清楚落在道路上。'),
    'CFB-INSERT-05': dict(revision='R4', scene='756', scene_name='scene-0998', key='5', camera=0,
        start=90, event=95, offset=[8.0,0.0,0.0], reference=110,
        reason='整段换为scene-0998前视夜间交叉口。黄框是已核实的黑色汽车 #5；绿框在其沿轨迹前方8米，二者在第2秒参考帧均清楚可辨。夜间画质请人工审核。'),
}


def source_for(parent, original, plan):
    row=copy.deepcopy(original)
    case=row['case']
    root=V5/plan['scene']
    item=json.loads((root/'instances/instances_info.json').read_text())[plan['key']]
    case['dataset']['root']=str(V5)
    case['dataset']['scene_id']=plan['scene']
    case['anchor']['event_frame']=plan['event']
    case['anchor']['pre_frames']=plan['event']-plan['start']
    target=case['target']
    target.update(entity_id='inserted:'+item['id'],source_asset_actor_key=plan['key'],
                  source_asset_actor_id=item['id'],class_name=item['class_name'],
                  proposal_offset_actor_frame_m=plan['offset'])
    case['expected_outcome']['affected_entities']=[target['entity_id']]
    row.update(camera_index=plan['camera'],scene_name=plan['scene_name'],
               location=MAP_INDEX[plan['scene_name']]['location'],target_key=plan['key'],
               review_reference_frame=plan['reference'])
    return row


def geometry_gate(plan):
    option=(plan['scene'],plan['key'],plan['camera'],plan['start'],
            MAP_INDEX[plan['scene_name']]['location'],V5)
    candidates=evaluate(option)
    chosen=next((r for r in candidates if list(r['offset'])==plan['offset']),None)
    assert chosen is not None,plan
    assert chosen['road']==19 and chosen['visible']>=15
    assert chosen['annotated_actor_intersections']==0 and chosen['ego_footprint_intersections']==0
    root=V5/plan['scene']
    ann=json.loads((root/'instances/instances_info.json').read_text())[plan['key']]['frame_annotations']
    j=ann['frame_idx'].index(plan['reference'])
    pose=np.asarray(ann['obj_to_world'][j]);size=ann['box_size'][j]
    source=project_box(root,plan['reference'],plan['camera'],pose,size)
    planned=project_box(root,plan['reference'],plan['camera'],shifted_pose(pose,plan['offset']),size)
    assert source is not None and planned is not None
    rects=[source['bbox_xyxy'],planned['bbox_xyxy']]
    assert min(x['area_px2'] for x in [source,planned])>=15000
    assert all(r[2]-r[0]>=140 and r[3]-r[1]>=100 for r in rects)
    assert iou(*rects)<.15
    return {'source_projected_area_px2':round(source['area_px2']),
            'planned_projected_area_px2':round(planned['area_px2']),
            'reference_bbox_iou':round(iou(*rects),4),
            'road_footprint_2hz':chosen['road'],'planned_visible_2hz':chosen['visible'],
            'actor_intersections_2hz':0,'ego_intersections_2hz':0,
            'reference_frame':plan['reference'],'reference_second_in_clip':(plan['reference']-plan['start'])/10}


def main():
    old=json.loads((PREVIOUS/'candidates.json').read_text())
    assert old['case_count']==24 and old['model_calls']==0
    assert all(not r['approval']['inference_allowed'] for r in old['cases'])
    assert not OUTPUT.exists(),'Refuse to overwrite a prior review run.'
    original={r['case_id']:r for r in json.loads(INDEX.read_text())['cases']}
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'cases').mkdir()
    cases=[];changes=[]
    for previous in old['cases']:
        parent=previous['parent_case_id']
        if parent not in PLANS:
            (OUTPUT/'cases'/previous['case_id']).symlink_to(PREVIOUS/'cases'/previous['case_id'],target_is_directory=True)
            cases.append(previous)
            continue
        plan=PLANS[parent]
        gate=geometry_gate(plan)
        row=source_for(parent,original[parent],plan)
        reuse=PREVIOUS/'cases'/'CFB-REMOVE-04-R4-10S' if parent=='CFB-INSERT-04' else None
        entry=build_candidate(row,original['CFB-SPEED-EGO-01'],OUTPUT,
                              revision=plan['revision'],reuse_original_from=reuse)
        assert entry['source_frame_range']==[plan['start'],plan['start']+99]
        entry['supersedes_case_id']=previous['case_id']
        entry['revision_reason']=plan['reason']
        entry['input_check']['insertion_road_collision_and_reference_gate']=gate
        entry['selection_evidence']={'summary':
            f'原车/计划车参考框面积{gate["source_projected_area_px2"]}/{gate["planned_projected_area_px2"]}像素²；'
            f'计划车在19/19个0.5秒采样帧车身落在可行驶区域，19帧中至少15帧双框入画，'
            '与标注对象及自车车身相交0次。投影不代表无遮挡。'}
        entry['prior_short_window_input_audit']={}
        entry['warnings']=[w for w in entry['warnings'] if '旧短窗检查' not in w]
        entry['show_target_reference']=True
        save(OUTPUT/'cases'/entry['case_id']/'proposal.json',entry)
        cases.append(entry)
        changes.append({'old':previous['case_id'],'new':entry['case_id'],'reason':plan['reason']})
    assert len(cases)==24 and len(changes)==3
    manifest=copy.deepcopy(old)
    manifest.update(run_id=OUTPUT.name,created_utc=datetime.now(timezone.utc).isoformat(),
        previous_manifest=str(PREVIOUS/'candidates.json'),cases=cases,
        replacement=old['replacement']+changes,model_calls=0,new_ai_video_reviews=0,
        revision_summary='按你第三轮截图只替换INSERT-02、04、05的整个场景与供体；其余21例逐字段保留。新三例黄/绿框在参考帧清楚分离、计划车身道路与相交门控通过。全部仍待你确认，推理与评分留空。')
    shutil.copy2(PREVIOUS/'human-review-original-four.json',OUTPUT/'human-review-original-four.json')
    save(OUTPUT/'candidates.json',manifest)
    write_report(OUTPUT,manifest)
    print(json.dumps({'run':str(OUTPUT),'case_count':24,'revised':changes,'model_calls':0},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
