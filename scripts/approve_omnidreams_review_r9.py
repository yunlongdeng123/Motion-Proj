"""Freeze the user's approval of all 24 proposed cases; no inference or scoring."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from prepare_omnidreams_review_candidates import save, write_report
from revise_omnidreams_review_insertion_r7 import BASE


PREVIOUS = BASE / '20260923-proposal-r8-insert04-lane-review'
OUTPUT = BASE / '20260923-proposal-r9-all-approved'


def main() -> None:
    old = json.loads((PREVIOUS / 'candidates.json').read_text())
    assert old['case_count'] == 24 and len(old['cases']) == 24
    assert old['model_calls'] == 0 and old.get('new_ai_video_reviews', 0) == 0
    assert not OUTPUT.exists(), 'Refuse to overwrite an earlier approval snapshot'
    assert len({row['case_id'] for row in old['cases']}) == 24

    (OUTPUT / 'cases').mkdir(parents=True)
    cases = []
    for old_row in old['cases']:
        row = copy.deepcopy(old_row)
        cid = row['case_id']
        source = PREVIOUS / 'cases' / cid
        target = OUTPUT / 'cases' / cid
        assert source.is_dir() and (source / 'original-nuscenes.mp4').is_file()
        assert (source / 'original-nuscenes.json').is_file()
        assert not (source / 'factual.mp4').exists()
        assert not (source / 'counterfactual.mp4').exists()
        target.mkdir()
        for src in source.iterdir():
            if src.name == 'proposal.json':
                continue
            assert src.is_file(), src
            # All runs share the same volume: retain the reviewed media without duplication.
            os.link(src.resolve(), target / src.name)

        row['approval'].update(
            status='approved', reviewer='user', inference_allowed=True,
            note='用户于2026-09-23确认全部24个case及反事实提案通过；仅批准推理输入，未评价模型输出。',
        )
        row['case']['status'] = 'human_approved_no_inference'
        row['case']['qualification']['human_verdict'] = 'approved'
        save(target / 'proposal.json', row)
        cases.append(row)

    manifest = copy.deepcopy(old)
    manifest.update(
        run_id=OUTPUT.name,
        created_utc=datetime.now(timezone.utc).isoformat(),
        previous_manifest=str(PREVIOUS / 'candidates.json'),
        status='human_approved_no_inference',
        approval_scope='24 case selections and counterfactual proposals; not generated videos or scores',
        cases=cases,
        model_calls=0,
        new_ai_video_reviews=0,
        revision_summary='用户已批准24/24个10秒case及其反事实提案，包括修正后的INSERT-04。此批准仅覆盖推理输入，不代表生成视频质量或六维评分通过；GPU当前关闭，factual/counterfactual与评分仍为空。',
    )
    assert all(row['approval']['status'] == 'approved' and row['approval']['inference_allowed'] for row in cases)
    shutil.copy2(PREVIOUS / 'human-review-original-four.json',
                 OUTPUT / 'human-review-original-four.json')
    save(OUTPUT / 'candidates.json', manifest)
    write_report(OUTPUT, manifest)
    print(json.dumps({'output': str(OUTPUT), 'approved': len(cases),
                      'model_calls': 0}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
