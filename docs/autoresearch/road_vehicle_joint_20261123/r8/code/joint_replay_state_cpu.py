"""Full checkpoint leaf evidence: exact restore, bounded CUDA floating replay."""
from pathlib import Path
import hashlib,json
import numpy as np


def compare_states(expected,actual,directory,strict=False,eps_multiplier=4):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    left,right,leaves,structures={},{},[],[]
    def walk(a,b,path):
        if isinstance(a,dict):
            assert isinstance(b,dict) and a.keys()==b.keys(),('state structure differs',path)
            structures.append(dict(path=path,kind='dict',keys=[repr(k) for k in a]))
            for k in a:walk(a[k],b[k],path+'/'+str(k))
            return
        if isinstance(a,(tuple,list)):
            assert type(a)==type(b) and len(a)==len(b),('state sequence differs',path)
            structures.append(dict(path=path,kind=type(a).__name__,length=len(a)))
            for k,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(k))
            return
        kind='array'
        if hasattr(a,'detach'):
            assert hasattr(b,'detach');a=a.detach().cpu().numpy();b=b.detach().cpu().numpy();kind='torch_tensor'
        elif a is None:
            assert b is None;structures.append(dict(path=path,kind='None'));return
        elif not isinstance(a,np.ndarray):
            assert type(a)==type(b);kind='python_'+type(a).__name__;a=np.asarray(a);b=np.asarray(b)
        assert a.dtype==b.dtype and a.shape==b.shape,(path,'dtype/shape differs')
        key=f'leaf{len(leaves):06d}';left[key]=a;right[key]=b
        float_leaf=a.dtype.kind in 'fc'
        counter_or_rng=(path.startswith('/sampler') or 'rng' in path.lower() or path.endswith('/logical_step') or path.endswith('/step'))
        exact=strict or counter_or_rng or not float_leaf
        if float_leaf:
            finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
            diff=abs(a.astype(np.float64)-b.astype(np.float64))
            bound=eps_multiplier*np.finfo(a.dtype).eps*(1+abs(a.astype(np.float64)))
            ok=finite and (np.array_equal(a,b) if exact else bool((diff<=bound).all()))
            maximum=float(diff.max()) if a.size else 0.;ratio=float((diff/np.maximum(bound,1e-300)).max()) if a.size else 0.
        else:finite=True;ok=bool(np.array_equal(a,b));maximum=None;ratio=None
        leaves.append(dict(key=key,path=path,kind=kind,dtype=str(a.dtype),shape=list(a.shape),elements=int(a.size),
            floating=float_leaf,required_exact=exact,counter_or_rng=counter_or_rng,finite=finite,pass_check=ok,
            max_abs=maximum,max_error_over_bound=ratio,expected_bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            replay_bytes_sha256=hashlib.sha256(b.tobytes()).hexdigest()))
    walk(expected,actual,'')
    np.savez_compressed(directory/'expected_state.npz',**left);np.savez_compressed(directory/'replay_state.npz',**right)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(schema='R8.complete_checkpoint_state_comparison.v2',complete=True,decision='PASS' if all(x['pass_check'] for x in leaves) else 'FAIL',
        strict_restore=strict,epsilon_multiplier=eps_multiplier,floating_rule='abs(replay-ref) <= 4*eps(dtype)*(1+abs(ref)) elementwise',
        counters_RNG_and_nonfloat_exact=True,structures=structures,leaves=leaves,
        expected_npz_sha256=digest(directory/'expected_state.npz'),replay_npz_sha256=digest(directory/'replay_state.npz'))
    (directory/'state_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result
