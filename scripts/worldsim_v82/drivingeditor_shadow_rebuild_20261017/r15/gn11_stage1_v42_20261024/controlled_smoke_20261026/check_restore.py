from pathlib import Path
import json,torch
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026')
a=torch.load(root/'observed_smoke/full199/checkpoint101_from0.pth',map_location='cpu',weights_only=False)
b=torch.load(root/'restore_check/full199/checkpoint101_from100.pth',map_location='cpu',weights_only=False)
rows=[]
def compare(x,y,path):
    if torch.is_tensor(x):
        rows.append(dict(path=path,equal=torch.equal(x,y),max_abs=float((x-y).abs().max()) if x.numel() and x.is_floating_point() else None))
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x:compare(x[k],y[k],path+'/'+str(k))
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y)
        for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
    else:
        import numpy as np
        equal=bool(np.array_equal(x,y)) if isinstance(x,np.ndarray) else x==y
        rows.append(dict(path=path,equal=equal))
for k in ['dc','opacity','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','sampler','step','frames','arm','master_files']:compare(a[k],b[k],k)
result=dict(all_equal=all(r['equal'] for r in rows),rows=rows,check='fresh process resume100 then step101 vs uninterrupted101; one additional engineering update',extra_updates=1)
(root/'restore_check/result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}));print([r for r in rows if not r['equal']])
