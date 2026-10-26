from pathlib import Path
import json,torch,numpy as np,argparse
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026')
parser=argparse.ArgumentParser();parser.add_argument('--original',default='edit_smoke_preserve');parser.add_argument('--resumed',default='edit_restore');args=parser.parse_args()
a=torch.load(root/args.original/'full199/checkpoint101_from0.pth',map_location='cpu',weights_only=False)
b=torch.load(root/args.resumed/'full199/checkpoint101_from100.pth',map_location='cpu',weights_only=False)
rows=[]
def compare(x,y,path):
    if torch.is_tensor(x):rows.append(dict(path=path,equal=torch.equal(x,y),max_abs=float((x-y).abs().max()) if x.numel() and x.is_floating_point() else None))
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x:compare(x[k],y[k],path+'/'+str(k))
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y)
        for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
    else:rows.append(dict(path=path,equal=bool(np.array_equal(x,y)) if isinstance(x,np.ndarray) else x==y))
for k in a:compare(a[k],b[k],k)
result=dict(all_equal=all(r['equal'] for r in rows),rows=rows,extra_updates=1,loaded_parameters_exact=json.loads((root/args.resumed/'full199/restore_loaded_exact.json').read_text()))
(root/args.resumed/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps([r for r in rows if not r['equal']],indent=2))
