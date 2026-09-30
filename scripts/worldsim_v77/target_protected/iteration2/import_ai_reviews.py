"""归档外部AI评分；与人工意见、旧独立QA分开，不提高训练准入。"""
import argparse, csv, json
from pathlib import Path
from collections import Counter

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p, d): Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def load_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def main(root, manifest):
    clips={c['case_id']:c for c in read(manifest)['clips']}
    batches=[('score_completed.csv','frame_scores_400.csv'),('score_completed (1).csv','frame_scores_410.csv')]
    cases=[]; frames=[]; errors=[]; summaries=[]
    for b,(cf,ff) in enumerate(batches,1):
        cr=load_csv(root/'raw'/cf); fr=load_csv(root/'raw'/ff)
        ci={r['case_id']:r for r in cr}; fi={}
        if len(ci)!=len(cr): errors.append(f'batch{b}:duplicate_case')
        for row in fr:
            cid=row['case_id']; n=int(row['frame']); score=int(row['score_0_2']); key=(cid,n)
            if key in fi: errors.append(f'batch{b}:duplicate_frame:{key}')
            fi[key]=row
            if cid not in clips or cid not in ci: errors.append(f'unknown_frame:{key}'); continue
            c=clips[cid]; f=c['preview_frames'][n]
            if int(row['timestamp_us'])!=f['timestamp_us']:errors.append(f'timestamp:{key}')
            if row.get('preview_source',row.get('input_image'))!=f['input']: errors.append(f'path:{key}')
            if score not in (0,1,2):errors.append(f'score:{key}')
            frames.append({'case_id':cid,'frame':n,'score':score,'timestamp_us':int(row['timestamp_us']),
                           'reviewer_type':'external_ai','batch':b,'raw_file':ff,'raw_row':row,'human_verdict':None})
        for row in cr:
            cid=row['case_id'];scores=json.loads(row['frame_scores_json']);evidence=json.loads(row['evidence_frames'])
            if cid not in clips: errors.append(f'unknown_case:{cid}');continue
            c=clips[cid];n=len(c['preview_frames'])
            if len(scores)!=n or any((cid,i) not in fi or int(fi[(cid,i)]['score_0_2'])!=s for i,s in enumerate(scores)):
                errors.append(f'frame_reconciliation:{cid}')
            if not all(0<=i<n for i in evidence):errors.append(f'evidence_frame:{cid}')
            cases.append({'case_id':cid,'source_id':c['source_id'],'scene':c['scene'],'donor_source_id':c['donor_source_id'],
                'reviewer_type':'external_ai','batch':b,'raw_file':cf,'score':int(row['score_0_2']),
                'frame_scores':scores,'evidence_frames':evidence,'input_difficulty':row['input_difficulty'],
                'failure_type':row['failure_type'],'reason':row['reason'],'human_verdict':None,'training_ready':False})
        summaries.append({'batch':b,'case_file':cf,'frame_file':ff,'cases':len(cr),'frames':len(fr),
                          'scores':dict(Counter(int(r['score_0_2']) for r in cr))})
    counts=Counter(r['case_id'] for r in cases)
    duplicates=[k for k,v in counts.items() if v>1]
    summary={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r2','stage':'CPU_revision',
        'batches':summaries,'case_rows':len(cases),'unique_cases':len(counts),'frame_rows':len(frames),
        'expected_cases':len(clips),'expected_frames':sum(len(c['preview_frames']) for c in clips.values()),
        'missing_cases':sorted(set(clips)-set(counts)),'duplicate_cases':duplicates,'errors':errors,
        'case_score_counts':dict(Counter(r['score'] for r in cases)),
        'frame_score_counts':dict(Counter(r['score'] for r in frames)),
        'all_frames_same_score_cases':sum(len(set(r['frame_scores']))==1 for r in cases),
        'human_full_review_received':False,'training_ready':0,
        'interpretation':'用户明确声明为AI review；文件中的逐帧评分记录不证明AI实际逐帧独立检查，不能替代人工全检。保留原理由及证据帧，不自动覆盖独立QA或人工结论。'}
    dump(root/'ai_case_reviews.json',{'reviews':cases});dump(root/'ai_frame_reviews.json',{'reviews':frames});dump(root/'ai_review_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    assert not errors and not duplicates, '评分对账失败，原始文件保持不变'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();main(a.root,a.manifest)
