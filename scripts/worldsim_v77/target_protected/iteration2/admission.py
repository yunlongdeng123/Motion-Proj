"""历史规则复现：已被同日用户五项训练输入标准取代，不得用于当前准入。

当前入口为merge_training_reviews.py及training_admission.json。本文件只保留
早先“不重审、低分淘汰”指令下的可复核计算，CLI必须显式选择历史复现。
"""
from pathlib import Path
import json,argparse,re
from collections import Counter

def decision(score,donor_instance,human_verdict=None,blocked_instances=()):
    if human_verdict=='reject_visible_contamination':return 'human_reject'
    if score<2:return 'ai_low_score_reject'
    if donor_instance in blocked_instances:return 'donor_blocked'
    return 'ai_2_candidate_not_qualified'

def main(root,manifest):
    reviews=json.loads((root/'ai_case_reviews.json').read_text())['reviews'];clips={c['case_id']:c for c in json.loads(manifest.read_text())['clips']}
    tok='08fba8bd56fd456c8364c93c6110af46';records=[]
    for r in reviews:
        c=clips[r['case_id']];human='reject_visible_contamination' if r['case_id']=='W029' else None
        records.append({'case_id':r['case_id'],'ai_score':r['score'],'donor_instance':c['donor_instance'],'donor_source_id':c['donor_source_id'],
            'human_verdict':human,'current_admission':decision(r['score'],c['donor_instance'],human,[tok]),'training_ready':False,
            'donor_reuse_allowed':c['donor_instance']!=tok,'old_independent_status':c['review']['synthetic_status'],
            're_review_scheduled':False})
    result={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r2','policy':'external AI<2 reject; AI=2 candidate only; explicit human rejection wins; no re-review of old data',
        'counts':dict(Counter(r['current_admission'] for r in records)),'blocked_donor_instances':[tok],
        'all_blocked_donor_descendants':sum(r['donor_instance']==tok for r in records),'training_ready':0,'cases':records}
    result['superseded']=True
    result['superseded_by']='training_admission.json'
    dest=root/'admission_before_training_contract_review.json'
    if dest.exists():raise FileExistsError('原历史快照已存在；不覆盖，也不作为当前准入')
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--reproduce-legacy',action='store_true');a=p.parse_args()
    if not a.reproduce_legacy:p.error('此规则已被取代。当前准入请运行merge_training_reviews.py')
    main(a.root,a.manifest)
