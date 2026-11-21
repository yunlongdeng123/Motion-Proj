"""Pinned SciPy1.13.1 reverse-communication L-BFGS-B with full workspace.

No Torch, native context, optimizer execution, or GPU import at module import.
Every FG/NEW_X is explicit; only a caller-supplied fixed objective is evaluated.
"""
from pathlib import Path
import copy,hashlib,json
import numpy as np


def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.loads(json.dumps(x,sort_keys=True,allow_nan=False))


class CheckpointedLBFGSB:
    SCHEMA='complete_reverse_communication_LBFGSB.v1'
    ARRAY_NAMES=('x','l','u','nbd','f','g','wa','iwa','task','csave','lsave','isave','dsave')

    def __init__(self,x0,evaluator,*,identity,m=10,maxls=20,factr=1e7,pgtol=1e-8,maxFG=2000):
        import scipy
        from scipy.optimize import _lbfgsb
        if scipy.__version__!='1.13.1'or not _lbfgsb.setulb.__doc__.startswith('setulb(m,x,l,u,nbd,f,g,'):
            raise RuntimeError('full-workspace wrapper requires qualified SciPy1.13.1 ABI; no silent x-only fallback')
        self.lib=_lbfgsb;self.evaluator=evaluator;self.original_shape=tuple(np.asarray(x0).shape)
        x=np.array(x0,np.float64,copy=True).reshape(-1)
        if not np.isfinite(x).all()or np.any((x<0)|(x>1)):raise ValueError('initial ci must be finite within[0,1]')
        self.n=len(x);self.recipe=dict(m=int(m),maxls=int(maxls),factr=float(factr),pgtol=float(pgtol),maxFG=int(maxFG))
        if m<1 or maxls<1 or factr<0 or pgtol<0 or maxFG<1:raise ValueError('invalid registered numerical recipe')
        self.identity=canonical(identity)
        self.abi=dict(scipy_version=scipy.__version__,binary=str(Path(_lbfgsb.__file__).resolve()),binary_sha256=file_sha(_lbfgsb.__file__),
                      signature=_lbfgsb.setulb.__doc__.splitlines()[0])
        integer=_lbfgsb.types.intvar.dtype;n=self.n
        self.x=x;self.l=np.zeros(n,np.float64);self.u=np.ones(n,np.float64);self.nbd=np.full(n,2,integer)
        self.f=np.array(0.,np.float64);self.g=np.zeros(n,np.float64)
        self.wa=np.zeros(2*m*n+5*n+11*m*m+8*m,np.float64);self.iwa=np.zeros(3*n,integer)
        self.task=np.zeros(1,'S60');self.task[:]=b'START';self.csave=np.zeros(1,'S60')
        self.lsave=np.zeros(4,integer);self.isave=np.zeros(44,integer);self.dsave=np.zeros(29,np.float64)
        self.fg_calls=0;self.accepted=0;self.trace=[];self.done=False;self.reason=None;self.at_boundary=True
        self.last_accepted_x=self.x.copy();self.last_accepted_f=None;self.last_accepted_g=None
        self.best_evaluated_x=None;self.best_evaluated_f=None;self.last_details={}

    @property
    def task_text(self):return self.task.tobytes().decode('ascii').rstrip('\x00 ').strip()

    def _FG(self):
        if not np.isfinite(self.x).all()or np.any((self.x<0)|(self.x>1)):raise RuntimeError('Fortran requested an infeasible color point')
        value,gradient,details=self.evaluator(self.x.reshape(self.original_shape).copy())
        gradient=np.asarray(gradient,np.float64).reshape(-1)
        if not np.isfinite(value)or gradient.shape!=(self.n,)or not np.isfinite(gradient).all():raise FloatingPointError('invalid native objective/VJP')
        self.f[...] = float(value);self.g[...] = gradient;self.fg_calls+=1;self.last_details=copy.deepcopy(details)
        if self.fg_calls==1:
            self.last_accepted_f=float(self.f);self.last_accepted_g=self.g.copy();self.last_accepted_x=self.x.copy()
        if self.best_evaluated_f is None or float(self.f)<self.best_evaluated_f:
            self.best_evaluated_f=float(self.f);self.best_evaluated_x=self.x.copy()

    @staticmethod
    def gap(x,g):return float(np.sum(np.asarray(g,np.float64)*(np.asarray(x,np.float64)-(np.asarray(g)<0)),dtype=np.float64))

    def next_accepted(self):
        if self.done:return None
        m=self.recipe['m'];self.at_boundary=False
        while True:
            self.lib.setulb(m,self.x,self.l,self.u,self.nbd,self.f,self.g,self.recipe['factr'],self.recipe['pgtol'],
                self.wa,self.iwa,self.task,-1,self.csave,self.lsave,self.isave,self.dsave,self.recipe['maxls'])
            task=self.task_text
            if task.startswith('FG'):
                if self.fg_calls>=self.recipe['maxFG']:
                    self.done=True;self.reason='registered_maxFG_cap_pending_line_search';self.at_boundary=True;return None
                self._FG()
            elif task.startswith('NEW_X'):
                self.accepted+=1;self.last_accepted_x=self.x.copy();self.last_accepted_f=float(self.f);self.last_accepted_g=self.g.copy()
                gap=self.gap(self.x,self.g)
                row=dict(accepted=self.accepted,FG=self.fg_calls,objective=float(self.f),measured_FW_gap=gap,
                    relative_FW_gap=gap/max(float(self.f),1e-3),task=task,strict_numerical_certificate=False)
                self.trace.append(row);self.at_boundary=True;return copy.deepcopy(row)
            else:
                self.done=True;self.reason=task;self.at_boundary=True;return None

    def stop_at_training_gate(self):
        if not self.at_boundary:raise RuntimeError('stop only at a complete reverse-communication boundary')
        self.done=True;self.reason='registered_measured_training_FW_gate'

    def snapshot(self):
        if not self.at_boundary:raise RuntimeError('cannot snapshot a partially evaluated callback')
        return dict(schema=self.SCHEMA,identity=copy.deepcopy(self.identity),recipe=copy.deepcopy(self.recipe),abi=copy.deepcopy(self.abi),
            original_shape=self.original_shape,n=self.n,arrays={k:getattr(self,k).copy()for k in self.ARRAY_NAMES},
            fg_calls=self.fg_calls,accepted=self.accepted,trace=copy.deepcopy(self.trace),done=self.done,reason=self.reason,
            last_accepted_x=self.last_accepted_x.copy(),last_accepted_f=self.last_accepted_f,
            last_accepted_g=None if self.last_accepted_g is None else self.last_accepted_g.copy(),
            best_evaluated_x=None if self.best_evaluated_x is None else self.best_evaluated_x.copy(),best_evaluated_f=self.best_evaluated_f,
            last_details=copy.deepcopy(self.last_details))

    def restore(self,saved):
        if saved.get('schema')!=self.SCHEMA or canonical(saved['identity'])!=self.identity or saved['recipe']!=self.recipe or saved['abi']!=self.abi:
            raise ValueError('solver source/recipe/private ABI differs; no x-only restart')
        if saved['n']!=self.n or tuple(saved['original_shape'])!=self.original_shape:raise ValueError('ci topology differs')
        if saved['fg_calls']<0 or saved['fg_calls']>self.recipe['maxFG']or saved['accepted']!=len(saved['trace']):raise ValueError('solver trace/FG budget invalid')
        for name in self.ARRAY_NAMES:
            a=saved['arrays'][name];current=getattr(self,name)
            if a.shape!=current.shape or a.dtype!=current.dtype:raise ValueError('workspace shape/dtype differs: '+name)
            if a.dtype.kind in 'f'and not np.isfinite(a).all():raise ValueError('nonfinite workspace: '+name)
        if not np.array_equal(saved['arrays']['l'],self.l)or not np.array_equal(saved['arrays']['u'],self.u)or not np.array_equal(saved['arrays']['nbd'],self.nbd):
            raise ValueError('global box changed')
        for name in self.ARRAY_NAMES:getattr(self,name)[...] = saved['arrays'][name]
        for name in ('fg_calls','accepted','trace','done','reason','last_accepted_x','last_accepted_f','last_accepted_g','best_evaluated_x','best_evaluated_f','last_details'):
            setattr(self,name,copy.deepcopy(saved[name]))
        self.at_boundary=True

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError('preserve prior full solver state')
        state=self.snapshot();arrays=state.pop('arrays')
        for name in ('last_accepted_x','last_accepted_g','best_evaluated_x'):
            value=state.pop(name);state[name+'_is_none']=value is None
            arrays['meta_'+name]=np.zeros(0,np.float64)if value is None else value
        np.savez(path,metadata=np.asarray(json.dumps(canonical(state),allow_nan=False)),**arrays)

    def load(self,path):
        with np.load(path,allow_pickle=False)as z:
            state=json.loads(str(z['metadata']));state['arrays']={k:z[k].copy()for k in self.ARRAY_NAMES}
            for name in ('last_accepted_x','last_accepted_g','best_evaluated_x'):
                state[name]=None if state.pop(name+'_is_none')else z['meta_'+name].copy()
        self.restore(state)
