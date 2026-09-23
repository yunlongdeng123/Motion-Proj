"""Replace the short-visibility removal candidate; still no model inference."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3

from prepare_omnidreams_review_candidates import build_candidate, save, write_report
from revise_omnidreams_review_feedback_r5 import BASE, DB, INDEX, add_evidence, amend_source

PREVIOUS = BASE/'20260923-proposal-r5-human-feedback'
OUTPUT = BASE/'20260923-proposal-r6-human-feedback'
PARENT = 'CFB-REMOVE-05'
PLAN = dict(revision='R5',scene='756',start=40,event=45,key='1',camera=0,reference=55,
    reason='再次筛选：上版摩托车在10秒原片中不足半段可见，改为scene-0998前视的路口巴士 #1；目标在画内约8秒。自车先停后走与巴士离开同时出现，但是否有因果关系仍待人工审。')


def main():
    old=json.loads((PREVIOUS/'candidates.json').read_text())
    assert old['case_count']==24 and old['model_calls']==0
    assert all(not e['approval']['inference_allowed'] for e in old['cases'])
    assert not OUTPUT.exists()
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'cases').mkdir()
    inputs={e['case_id']:e for e in json.loads(INDEX.read_text())['cases']}
    result=[];change=None
    conn=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
    for previous in old['cases']:
        if previous['parent_case_id']!=PARENT:
            (OUTPUT/'cases'/previous['case_id']).symlink_to(PREVIOUS/'cases'/previous['case_id'],target_is_directory=True)
            result.append(previous)
            continue
        source=amend_source(PARENT,inputs[PARENT],PLAN)
        entry=build_candidate(source,inputs['CFB-SPEED-EGO-01'],OUTPUT,revision=PLAN['revision'])
        assert entry['source_frame_range']==[40,139]
        entry['supersedes_case_id']=previous['case_id']
        entry['revision_reason']=PLAN['reason']
        entry['prior_short_window_input_audit']={}
        entry['warnings']=[w for w in entry['warnings'] if '旧短窗检查' not in w]
        entry['show_target_reference']=True
        add_evidence(entry,conn)
        save(OUTPUT/'cases'/entry['case_id']/'proposal.json',entry)
        result.append(entry)
        change={'old':previous['case_id'],'new':entry['case_id'],'reason':PLAN['reason']}
    conn.close()
    assert change is not None and len(result)==24
    manifest=copy.deepcopy(old)
    manifest.update(run_id=OUTPUT.name,created_utc=datetime.now(timezone.utc).isoformat(),
        previous_manifest=str(PREVIOUS/'candidates.json'),cases=result,
        replacement=old['replacement']+[change],model_calls=0,new_ai_video_reviews=0,
        revision_summary='第二轮截图反馈已修订；追加一次CPU筛选，把仅短暂入画的摩托车移除例换成持续可见的路口巴士。全部24例仍待你确认，生成和评分留空。')
    shutil.copy2(PREVIOUS/'human-review-original-four.json',OUTPUT/'human-review-original-four.json')
    save(OUTPUT/'candidates.json',manifest)
    write_report(OUTPUT,manifest)
    print(json.dumps({'run':str(OUTPUT),'case_count':24,'replaced':change,
                      'model_calls':0},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
