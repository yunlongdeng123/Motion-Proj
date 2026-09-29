"""合并互斥render分片并验证case完整，不以进程退出代替实际产物。"""
import argparse
from pathlib import Path
from geometry_factory import read,dump

def main(root,out,parts):
    expected=[r['case_id'] for r in read(root/'pair_candidates.json')['selected']]
    manifests=[read(out/f'synthetic_manifest.part{i}.json') for i in range(parts)]
    rows=[r for m in manifests for r in m['clips']]
    assert len({r['case_id'] for r in rows})==len(rows),'duplicate shard case'
    assert set(expected)=={r['case_id'] for r in rows},'missing or extra rendered cases'
    for r in rows:
        for p in [*r['videos'].values(),*r['contacts']]:
            assert (out/p).is_file(),p
    full={k:v for k,v in manifests[0].items() if k!='clips'}
    full['clips']=sorted(rows,key=lambda r:r['case_id'])
    dump(out/'synthetic_manifest.json',full)
    print('MERGED',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--parts',type=int,required=True);a=p.parse_args();main(a.root,a.out,a.parts)
