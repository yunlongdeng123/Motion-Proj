"""同一train试产的补充来源：补可放置空间与视角，不改变质量门槛。"""
import argparse,copy
from collections import Counter,defaultdict
from pathlib import Path
from prepare_source_pool import read,dump,resolve
from prepare_context_extract import context,extract

def main(root,meta,pub):
    extra=root/'source_extension';extra.mkdir(exist_ok=True)
    if not (extra/'source_selection.json').exists():
        old=read(root/'source_selection.json')['clips'];used={(c['scene'],c['camera'],c['start_keyframe']) for c in old}
        allc=read(root/'candidate_pool.json')['candidates'];rows=[]
        for c in allc:
            if c['location']!='boston-seaport' or not set(c['shards']).issubset({'03','07'}):continue
            if (c['scene'],c['camera'],c['start_keyframe']) in used:continue
            depth=c['actors'][0]['anchor_projection']['depth']
            if not 10<=depth<=32:continue
            if max(b-a for a,b in zip(c['keyframe_timestamps'],c['keyframe_timestamps'][1:]))>600000:continue
            rows.append(copy.deepcopy(c))
        # 一scene一个补充source。优先多真实actor、CAM_FRONT等此前稀少视角与中距；不看生成结果。
        grouped=defaultdict(list)
        for c in rows:grouped[c['scene']].append(c)
        selected=[min(arr,key=lambda c:(-len(c['actors']),c['camera']=='CAM_BACK',abs(c['actors'][0]['anchor_projection']['depth']-20),c['start_keyframe'])) for arr in grouped.values()]
        selected.sort(key=lambda c:(-len(c['actors']),c['camera']=='CAM_BACK',c['scene']))
        selected=selected[:40]
        for i,c in enumerate(selected):c['source_id']=f'T{i+1:03}'
        dump(extra/'source_selection.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','stage':'additional_train_sources',
              'reason':'Primary pool produced 6 feasible pairs and no dense cases; seek other windows/cameras before any gate relaxation',
              'archive_scope':['03','07'],'clips':selected,'human_verdict':None})
    sel=read(extra/'source_selection.json');print('EXTRA_SELECTED',len(sel['clips']),dict(Counter(c['camera'] for c in sel['clips'])),flush=True)
    if not (extra/'source_manifest.json').exists():resolve(meta,extra,sel)
    if not (extra/'required_files.json').exists():context(extra,meta)
    extract(extra,pub)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--meta',type=Path,required=True);p.add_argument('--pub',type=Path,required=True);a=p.parse_args();main(a.root,a.meta,a.pub)
