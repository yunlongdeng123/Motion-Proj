"""新训练判据优先于旧外观分；保留AI原分、独立新判断、工程证据与人工状态。"""
from pathlib import Path
from collections import Counter
import argparse,json

def read(p):return json.loads(Path(p).read_text())
def main(root,old):
    ai=read(root/'ai_case_reviews.json')['reviews'];visual=read(root/'training_usability_sol.json');tech=read(root/'training_input_audit.json')
    rv={r['case_id']:r for r in visual['reviews']};rt={r['case_id']:r for r in tech['cases']}
    expected={r['case_id'] for r in ai if r['score']==1};assert set(rv)==set(rt)==expected and len(expected)==37
    sources={c['case_id']:c for c in read(old/'synthetic_delivery/review_manifest.json')['clips']}
    cases=[]
    for a in ai:
        cid=a['case_id'];c=sources[cid];v=rv.get(cid);t=rt.get(cid);human='reject_spatial_placement' if cid=='D009' else None
        if cid=='W029':status='hold_user_reported_silhouette'
        elif human:status='reject'
        elif a['score']==0:status='reject'
        elif a['score']==2:status='ai_2_unverified'
        elif v['visual_status']=='reject' or not t['engineering_pass']:status='reject'
        elif v['visual_status']=='uncertain':status='uncertain'
        else:status='train_usable_pending_human'
        if v:
            assert v['reviewed_frames'] and all(0<=i<len(c['preview_frames']) for i in v['reviewed_frames'])
            assert v.get('human_verdict') is None
        cases.append({'case_id':cid,'original_ai_score':a['score'],'source_id':c['source_id'],'donor_source_id':c['donor_source_id'],
            'training_admission':status,'training_ready':False,'human_full_frame_approved':False,'human_verdict':human,
            'human_asset_feedback':('贴纸右下方多了一块，不干净，需要重新造；新训练判据下尚未单独认证' if cid=='W029' else None),
            'visual_reassessment':v,'engineering_audit':({k:t[k] for k in t if k not in ('frames','geometry_record')} if t else None),
            'original_reason':a['reason'],'original_failure_type':a['failure_type']})
    review_cases=[r for r in cases if r['original_ai_score']==1]
    summary={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r2','policy_revision':'training_input_quality_five_checks_20260930',
        'user_superseded_prior_no_re_review_and_low_score_reject_rule':True,'reassessed_cases':37,
        'reassessed_counts':dict(Counter(r['training_admission'] for r in review_cases)),'all_49_counts':dict(Counter(r['training_admission'] for r in cases)),
        'human_full_frame_review_received':False,'training_ready':0,
        'definition':'train_usable_pending_human = 五项训练输入检查下的技术可用候选，独立抽帧+全帧工程核验通过；不是人工全检完成，不启动训练。',
        'D009':'用户明确空间错误，保持reject，不因masked-X合同复活','cases':cases}
    (root/'training_admission.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    legacy=root/'admission.json'
    if legacy.exists():
        snapshot=root/'admission_before_training_contract_review.json'
        if not snapshot.exists():snapshot.write_bytes(legacy.read_bytes())
        legacy.write_text(json.dumps({'superseded':True,'current_admission':'training_admission.json','original_snapshot':snapshot.name,
            'reason':'用户最新要求按模型实际输入的五项标准重评1分例，旧外观低分拒绝政策已失效'},ensure_ascii=False,indent=2)+'\n')
    print({k:v for k,v in summary.items() if k!='cases'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args();main(a.root,a.old)
