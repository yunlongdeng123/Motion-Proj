"""将独立来源检查归档到预审页；不替换用户逐帧评分。"""
import argparse
import json
from collections import Counter
from pathlib import Path

def main(out,review,template):
    rows=json.loads(review.read_text(encoding='utf-8'))
    by_id={c['source_id']:c for c in rows['clips']}
    assert len(by_id)==len(rows['clips'])
    for row in by_id.values():
        for key in ['receiver_status','donor_status']:assert row[key] in ['pass','reject','uncertain']
        assert row['reviewed_frames'] and row['note'].strip()
    manifest=json.loads((out/'source_review_manifest.json').read_text(encoding='utf-8'))
    available={c['source_id'] for c in manifest['clips']}
    assert set(by_id)<=available,sorted(set(by_id)-available)
    for c in manifest['clips']:
        c['subagent_review']=by_id.get(c['source_id'])
        if c['subagent_review'] and c['source_status']=='geometry_reject':
            assert c['subagent_review']['receiver_status']!='pass' and c['subagent_review']['donor_status']!='pass'
        assert c['human_verdict'] is None
    manifest['subagent_review_summary']={
        'reviewed_sources':len(by_id),'pending':len(available-set(by_id)),
        'receiver':dict(Counter(c['receiver_status'] for c in by_id.values())),
        'donor':dict(Counter(c['donor_status'] for c in by_id.values())),
        'scope':'independent sampled-source-frame review; not synthesized pair approval or human verdict'}
    (out/'source_review_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (out/'subagent_source_reviews.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (out/'index.html').write_text(template.read_text(encoding='utf-8').replace('__DATA__',json.dumps(manifest,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8',newline='\n')
    print(json.dumps(manifest['subagent_review_summary'],ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args();main(a.out,a.review,Path(__file__).with_name('source_review.html'))
