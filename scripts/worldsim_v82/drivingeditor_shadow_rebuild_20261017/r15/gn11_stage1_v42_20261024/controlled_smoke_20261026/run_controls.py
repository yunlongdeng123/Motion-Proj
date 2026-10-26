import argparse,time,json
from pathlib import Path
from editor_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='controls_initial');p.add_argument('--frames',default='102,129,137');p.add_argument('--footprint',action='store_true');p.add_argument('--ray-depth',action='store_true');a=p.parse_args()
    dest=OUT/a.name;dest.mkdir(exist_ok=False);start=time.time()
    n=Native();gate,gate_meta=make_gate(n,a.footprint,a.ray_depth)
    roads={v:Road(FULL/f'{v}_initialization.npz') for v in ['window49','full199']}
    rows=[]
    with torch.no_grad():
        for raw in map(int,a.frames.split(',')):
            c=n.camera(raw);master=n.render(c,dynamic=True)['rgb'];save(dest/f'{raw}_master.png',master)
            ordinary=n.render(c);clean=n.render(c,suppress=gate)
            save(dest/f'{raw}_static.png',ordinary['rgb']);save(dest/f'{raw}_suppression.png',clean['rgb'])
            row=dict(raw=raw,variants={})
            for name,road in roads.items():
                for suffix,g in [('road_only',None),('road_suppression',gate)]:
                    r=n.render(c,road,g);save(dest/f'{raw}_{name}_{suffix}.png',r['rgb'])
                    row['variants'][name+'_'+suffix]=dict(critical={f'{x},{y}':dict(contribution=float(r['contribution'][0,y,x]),rgb=cpu(r['rgb'][:,y,x]).tolist()) for x,y in [(255,240),(256,219)]})
                    if raw==102:np.savez_compressed(dest/f'{raw}_{name}_{suffix}_float.npz',rgb=cpu(r['rgb']),contribution=cpu(r['contribution']))
                    del r
            repeat=n.render(c,dynamic=True)['rgb'];row['master_repeat_max_abs']=float((repeat-master).abs().max());assert row['master_repeat_max_abs']==0
            rows.append(row);print(json.dumps(row),flush=True)
    assert n.unchanged()
    result=dict(status='completed',rows=rows,gate=gate_meta,training_updates=0,elapsed=time.time()-start,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),master_unchanged=True,quality_accepted=False,human_verdict=None)
    (dest/'receipt.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
