from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,read

def main():
    rows=[]
    for name in ['asset_candidates','asset_focus_candidates']:
        for p in (O/name).glob('N*.json'):rows+=read(p)['candidates']
    unique={(p['source_id'],p['asset'],p['offset_longitudinal_m'],p['offset_lateral_m']):p for p in rows};rows=list(unique.values())
    print('POOL',len(rows),'scenes',len({p['scene'] for p in rows}),'types',dict(Counter(p['type'] for p in rows)))
    for k in ['background','single_actor','dense_actors']:
        q=[p for p in rows if p['type']==k];print(k,len(q),'scenes',len({p['scene'] for p in q}),dict(Counter(p['scene'] for p in q)))
if __name__=='__main__':main()
