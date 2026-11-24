from pathlib import Path
import json,numpy as np,torch
R=Path(__file__).resolve().parent/'training_v1/probe'
a=torch.load(R/'expected002616.pt',map_location='cpu',weights_only=False)['complete_state']
b=torch.load(R/'replay_actual_second.pt',map_location='cpu',weights_only=False)['complete_state'];rows=[]
def walk(a,b,path):
    if isinstance(a,dict):
        if a.keys()!=b.keys():rows.append(dict(path=path,keys_differ=True));return
        for k in a:walk(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):rows.append(dict(path=path,lengths=[len(a),len(b)]));return
        for i,(x,y)in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
    elif torch.is_tensor(a)or isinstance(a,np.ndarray):
        x=a.numpy()if torch.is_tensor(a)else a;y=b.numpy()if torch.is_tensor(b)else b
        if x.shape!=y.shape:rows.append(dict(path=path,shapes=[x.shape,y.shape]));return
        if not np.array_equal(x,y):
            diff=np.abs(x.astype(np.float64)-y.astype(np.float64));close=np.isclose(x,y,rtol=4*np.finfo(np.float32).eps,atol=4*np.finfo(np.float32).eps)
            rows.append(dict(path=path,dtype=str(x.dtype),changed=int(np.count_nonzero(diff)),max_abs=float(diff.max()),failed=int((~close).sum()),examples=np.flatnonzero(~close)[:8].tolist()))
    elif a!=b:rows.append(dict(path=path,expected=str(a),actual=str(b)))
walk(a,b,'');(R/'replay_state_diff.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
