"""只读紧凑状态，检查实际进程与产物时使用。"""
from nine_common import *
reg=read(ROOT/'registration.json');out={};plan=[]
for s in reg['scenes']:
    n=0
    for v in s['streams']:
        if not v['active']:continue
        rows=read(ROOT/s['name']/f"cam{v['camera']}/mask_stats.json")
        n+=sum(any(x['model']>0 for x in rows[start:min(start+10,30)]) for start in [0,9,18,27])
    plan.append(dict(scene=s['name'],nonempty_windows=n))
out['planned_generated_windows']=sum(r['nonempty_windows'] for r in plan)
for key in ['sequence','mask','drive','omega','finish']:
    p=ROOT/f'{key}_state.json'
    if p.exists():
        d=read(p);out[key]=dict(state=d['state'],current=d.get('current'),completed=len(d.get('completed',d.get('queried',[]))),error=d.get('error'))
        if key in ['drive','omega'] and d.get('completed'):
            last=d['completed'][-1];out[key]['last']={k:last[k] for k in ['scene','camera','start','frame','generated','seconds'] if k in last}
        if key=='drive':out[key]['generated_windows']=sum(x['generated'] for x in d.get('completed',[]))
out['remote_render_scenes']=[s['name'] for s in reg['scenes'] if (ROOT/s['name']/'actor_layers/render_summary.json').exists()]
import json
print(json.dumps(out,ensure_ascii=False))
