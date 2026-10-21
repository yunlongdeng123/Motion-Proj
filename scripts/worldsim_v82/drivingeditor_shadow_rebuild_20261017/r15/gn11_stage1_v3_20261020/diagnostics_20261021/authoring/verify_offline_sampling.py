"""从原CPU采样器提取独立行号参考，核对离线包完整性。"""
import argparse,json,sys,zlib,base64
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cloud_sampling import Sampler

def read(path):return json.loads(path.read_text().split(']=',1)[1].strip()[:-1])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--existing',type=Path);a=p.parse_args()
    index=read(a.root/'data/geometry_index.js');sampler=Sampler();reference=json.loads(a.existing.read_text()) if a.existing else {}
    for arm,meta in index['arms'].items():
        for branch in meta['branches'].values():
            assert sum(v['count'] for v in branch['chunks'])==branch['ordinary_total']
            for chunk in branch['chunks']:assert (a.root/chunk['path']).exists()
        assert len(meta['members'])==49
        for member in meta['members'].values():assert (a.root/member['path']).exists()
        if arm in reference:continue
        reference[arm]={}
        for actor in (5,37):
            q=sampler.query(dict(arm=arm,raw=102,static_rate=2,dynamic_rate=2,vehicle=actor,vehicle_rate=10,vehicle_scope='projection'))
            raw=zlib.decompress(base64.b64decode(q['data']));record={}
            for branch in ('static','dynamic'):
                desc=q['arrays'][branch+'_row']
                record[branch]=np.frombuffer(raw,dtype=desc['dtype'],count=desc['shape'][0],offset=desc['offset']).tolist()
            reference[arm][str(actor)]=record
        print(arm+' original CPU reference complete',flush=True)
    a.out.write_text(json.dumps(reference,separators=(',',':')))

if __name__=='__main__':main()
