"""CPU-only INSERT-04 lane correction and two explicit user approvals."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import mine_insertion_review as miner
from prepare_omnidreams_review_candidates import build_candidate, save, write_report
from revise_omnidreams_review_insertion_r7 import BASE, INDEX, PLANS, geometry_gate, source_for


PREVIOUS = BASE / '20260923-proposal-r7-insertion-review'
OUTPUT = BASE / '20260923-proposal-r8-insert04-lane-review'
PARENT = 'CFB-INSERT-04'
APPROVED = {'CFB-INSERT-02', 'CFB-INSERT-05'}


def main() -> None:
    old = json.loads((PREVIOUS / 'candidates.json').read_text())
    assert old['case_count'] == 24 and old['model_calls'] == 0
    assert not OUTPUT.exists(), 'Refuse to overwrite a previous review run'
    source = {row['case_id']: row for row in json.loads(INDEX.read_text())['cases']}
    plan = copy.deepcopy(PLANS[PARENT])
    plan.update(revision='R5', offset=[4.0, 2.0, 0.0],
                reason='按人工截图否决原 X +8米、Y +3.5米位置。仅缩短 X 会使框在画面更靠左，故同时收紧到 X +4米、Y +2米；新绿框需再次人工确认所在车道。')
    miner.OFFSETS = [tuple(plan['offset'])]
    gate = geometry_gate(plan)
    assert gate['road_footprint_2hz'] == 19 and gate['actor_intersections_2hz'] == 0

    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'cases').mkdir()
    cases = []
    for previous in old['cases']:
        parent = previous['parent_case_id']
        if parent == PARENT:
            row = source_for(parent, source[parent], plan)
            entry = build_candidate(row, source['CFB-SPEED-EGO-01'], OUTPUT,
                                    revision=plan['revision'],
                                    reuse_original_from=PREVIOUS / 'cases' / previous['case_id'])
            entry['supersedes_case_id'] = previous['case_id']
            entry['revision_reason'] = plan['reason']
            entry['input_check']['insertion_road_collision_and_reference_gate'] = gate
            entry['selection_evidence'] = {'summary':
                '新的局部偏移为X +4米、Y +2米；0.5秒采样的19/19帧车身在可行驶区域内，'
                '与已标注对象及自车车身相交0次。地图可行驶区域不等于车道合法性；仍待人工确认。'}
            entry['prior_short_window_input_audit'] = {}
            entry['warnings'] = [w for w in entry['warnings'] if '旧短窗检查' not in w]
            entry['show_target_reference'] = True
            entry['approval'].update(status='pending', reviewer='user', inference_allowed=False)
            save(OUTPUT / 'cases' / entry['case_id'] / 'proposal.json', entry)
        elif parent in APPROVED:
            shutil.copytree(PREVIOUS / 'cases' / previous['case_id'],
                            OUTPUT / 'cases' / previous['case_id'])
            entry = copy.deepcopy(previous)
            entry['approval'].update(status='approved', reviewer='user', inference_allowed=True,
                                     note='2026-09-23 用户明确表示其他两个新替换 case 审批通过；未启动推理。')
            entry['case']['qualification']['human_verdict'] = 'approved'
            save(OUTPUT / 'cases' / entry['case_id'] / 'proposal.json', entry)
        else:
            (OUTPUT / 'cases' / previous['case_id']).symlink_to(
                PREVIOUS / 'cases' / previous['case_id'], target_is_directory=True)
            entry = previous
        cases.append(entry)

    assert len(cases) == 24
    assert sum(row['approval']['status'] == 'approved' for row in cases) == 2
    manifest = copy.deepcopy(old)
    manifest.update(run_id=OUTPUT.name, created_utc=datetime.now(timezone.utc).isoformat(),
                    previous_manifest=str(PREVIOUS / 'candidates.json'), cases=cases,
                    replacement=old['replacement'] + [{
                        'old': 'CFB-INSERT-04-R4-10S', 'new': 'CFB-INSERT-04-R5-10S',
                        'reason': plan['reason']}], model_calls=0, new_ai_video_reviews=0,
                    revision_summary='INSERT-02与INSERT-05按用户反馈已批准；INSERT-04原绿框越出相邻车道，改为X +4米、Y +2米并继续待审。其余21例不变。无模型推理或评分。')
    shutil.copy2(PREVIOUS / 'human-review-original-four.json',
                 OUTPUT / 'human-review-original-four.json')
    save(OUTPUT / 'candidates.json', manifest)
    write_report(OUTPUT, manifest)
    print(json.dumps({'output': str(OUTPUT), 'new_case': 'CFB-INSERT-04-R5-10S',
                      'approved': sorted(APPROVED), 'pending_case': PARENT,
                      'model_calls': 0}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
