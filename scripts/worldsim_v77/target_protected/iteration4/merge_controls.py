"""保留两次等预算CPU控制；旧通过预案不替换，只补此前无预案的scene。"""
import argparse,json
from collections import Counter
from pathlib import Path
def read(p):return json.loads(p.read_text())
def main(root):
    baseline=read(root/'pair_plan.json');diverse=read(root/'diverse_control/pair_plan.json')
    assert baseline['summary']['complete'] and diverse['summary']['complete']
    old={p['source_id']:p for p in baseline['plans']};new={p['source_id']:p for p in diverse['plans']}
    oldscenes={r['scene'] for r in read(root/'native10_factory/source_manifest.json')['clips'] if r['source_id'] in old}
    sources={r['source_id']:r for r in read(root/'native10_factory/source_manifest.json')['clips']}
    chosen=[p|{'planning_control':'original_top_six_windows'} for p in old.values()]
    additions=[]
    for sid,p in new.items():
        if sid in old or sources[sid]['scene'] in oldscenes:continue
        chosen.append(p|{'planning_control':'distinct_instance_top_six'});additions.append(sid);oldscenes.add(sources[sid]['scene'])
    s=baseline['summary']|{'planned_cases':len(chosen),'planned_scenes':len(oldscenes),'planned_types':dict(Counter(p['planned_type'] for p in chosen)),
         'baseline_planned_cases':len(old),'distinct_instance_planned_cases':len(new),'additional_receiver_windows':additions,
         'control_comparison':'same receiver pool, at most 6 donor windows and same 35 offsets per receiver per control; no threshold relaxation',
         'baseline_rejection_counts':baseline['summary']['rejection_counts'],'distinct_rejection_counts':diverse['summary']['rejection_counts']}
    (root/'selected_pair_plan.json').write_text(json.dumps({'summary':s,'plans':chosen},ensure_ascii=False,indent=2)+'\n')
    print('MERGED',s)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
