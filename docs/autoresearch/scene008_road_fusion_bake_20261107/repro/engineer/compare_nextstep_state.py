"""CPU numerical disclosure of disposable next-step states; no tolerance pass."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from shared_fit_common import tree_hash,sha,write


def compare(expected,actual):
    leaves=[]
    def visit(a,b,path):
        if torch.is_tensor(a):
            assert torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype
            difference=(a.double()-b.double()).abs()
            leaves.append(dict(path=path,exact=torch.equal(a,b),maxabs=float(difference.max()) if a.numel() else 0.,
                changed_elements=int((a!=b).sum()),elements=a.numel()))
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a:visit(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
        elif isinstance(a,np.ndarray):
            leaves.append(dict(path=path,exact=np.array_equal(a,b),maxabs=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.,
                changed_elements=int(np.count_nonzero(a!=b)),elements=a.size))
        else:leaves.append(dict(path=path,exact=a==b,maxabs=0 if a==b else None))
    visit(expected,actual,'')
    return leaves


def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',type=Path,required=True);args=p.parse_args()
    expected_path=args.branch/'smoke/checkpoint000004_disposable_expected.pt';actual_path=args.branch/'restore/checkpoint000004.pt'
    expected=torch.load(expected_path,map_location='cpu',weights_only=False);actual=torch.load(actual_path,map_location='cpu',weights_only=False)
    entries=compare(expected,actual)
    result=dict(expected_sha256=sha(expected_path),actual_sha256=sha(actual_path),
        checkpoint_restore_before_next_steps= json.loads((args.branch/'restore/restore_checks.json').read_text()),
        new_nextstep_tree_exact=tree_hash(expected)==tree_hash(actual),leaves=entries,
        numerical_policy='report actual exactness and differences; no threshold relaxation or replacement of states',CPU_only=True)
    target=args.branch/'restore/nextstep_numeric_details.json';write(target,result)
    print(json.dumps(dict(path=str(target),all_exact=result['new_nextstep_tree_exact'],differences=[e for e in entries if not e['exact']])))


if __name__=='__main__':main()
