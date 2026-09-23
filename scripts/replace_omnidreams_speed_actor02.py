"""按用户要求仅替换多卡车加速候选；新版本，其余23例逐字段不变。"""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import av
from prepare_omnidreams_review_candidates import build_candidate, save, write_report

BASE=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02')
OLD=BASE/'20260923-proposal-r2-10s'
NEW=BASE/'20260923-proposal-r3-replace-actor02'
REPLACE='CFB-SPEED-ACTOR-02-R3-10S'
INPUT=Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/adapter_inputs/omnidreams/index.json')


def main():
    old=json.loads((OLD/'candidates.json').read_text())
    assert all(r['approval']['inference_allowed'] is False for r in old['cases'])
    inputs=json.loads(INPUT.read_text())['cases']
    row=copy.deepcopy(next(r for r in inputs if r['case_id']=='CFB-SPEED-ACTOR-02'))
    case=row['case']; root=Path(case['dataset']['root'])/'204'
    info=json.loads((root/'instances/instances_info.json').read_text())['3']
    assert info['class_name']=='vehicle.bus.rigid'
    case['dataset']['scene_id']='204'; case['anchor']['event_frame']=45
    case['target'].update(actor_key='3',class_name=info['class_name'],entity_id=info['id'])
    case['expected_outcome']['affected_entities']=[info['id']]
    row['camera_index']=5
    NEW.mkdir(parents=True,exist_ok=False)
    fresh=build_candidate(row,inputs[0],NEW,revision='R4')
    fresh.update(supersedes_case_id=REPLACE,
        revision_reason='按你的要求替换多辆卡车难辨认的片段：改为scene-0255后视画面中间的巴士。仍是10秒、0.5秒起1.5×加速。',
        target_description='画面中间的巴士 #3（不是自车）',show_target_reference=True,
        replacement_user_reason='这个case画面有好几辆卡车，不太好分辨，你换个case吧',
        source_overlap_note='与SPEED-ACTOR-03使用同一场景同一巴士，起点后移4秒；编辑分别为加速/减速，不作为独立场景。',
        prior_short_window_input_audit={})
    fresh['counterfactual_description']='从0.5秒开始，让画面中间的巴士 #3 沿原轨迹以1.5×速度推进（加速50%）；自车/相机和其他对象轨迹保持不变。'
    assert fresh['input_check']['trajectory_available_without_extrapolation']
    assert fresh['case']['intervention']==next(r for r in old['cases'] if r['case_id']==REPLACE)['case']['intervention']
    save(NEW/'cases'/fresh['case_id']/'proposal.json',fresh)
    result=copy.deepcopy(old); result.update(run_id=NEW.name,created_utc=datetime.now(timezone.utc).isoformat(),
        previous_manifest=str(OLD/'candidates.json'),replacement={'old':REPLACE,'new':fresh['case_id'],'reason':fresh['replacement_user_reason']})
    unchanged=0
    for i,r in enumerate(old['cases']):
        if r['case_id']==REPLACE:
            result['cases'][i]=fresh
        else:
            # 只引用已有不可变资源，不重复导出23段视频，也不修改旧case。
            (NEW/'cases'/r['case_id']).symlink_to(OLD/'cases'/r['case_id'],target_is_directory=True)
            assert result['cases'][i]==r
            unchanged+=1
    assert unchanged==23
    shutil.copy2(OLD/'human-review-original-four.json',NEW/'human-review-original-four.json')
    save(NEW/'candidates.json',result)
    write_report(NEW,result)
    with av.open(str(NEW/'cases'/fresh['case_id']/'original-nuscenes.mp4')) as c:
        s=c.streams.video[0]
        assert s.frames==100 and s.average_rate==10 and float(s.duration*s.time_base)==10
    print(json.dumps({'replacement':fresh['case_id'],'unchanged':unchanged,'input_check':fresh['input_check'],
                      'warnings':fresh['warnings'],'model_calls':0},ensure_ascii=False))


if __name__=='__main__':
    main()
