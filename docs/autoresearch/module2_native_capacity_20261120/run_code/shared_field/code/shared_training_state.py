"""Passive full-batch state for fixed-GS shared-field RGB/DETAIL arms.

The caller owns native context, lease, calibrated common loss, execution and all
evaluation. This module never creates them and never starts a training loop.
"""
from pathlib import Path
import copy,hashlib,json,random
import numpy as np
import torch


def _cpu_copy(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:_cpu_copy(v)for k,v in x.items()}
    if isinstance(x,list):return [_cpu_copy(v)for v in x]
    if isinstance(x,tuple):return tuple(_cpu_copy(v)for v in x)
    return copy.deepcopy(x)


def _tensor_sha(x):
    a=x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()


def _identity(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _integer(x,name,maximum=None):
    if isinstance(x,bool)or not isinstance(x,(int,np.integer))or x<0 or(maximum is not None and x>maximum):
        raise ValueError('invalid '+name)
    return int(x)


class SharedTrainingState:
    """1200 logical full-batch steps: H-only 600, then L+H 600.

    Physical counters count actual calls, including replay. Loading an older
    checkpoint never decreases current counters. External hooks can aggregate
    several processes; a local checkpoint cannot count an unseen sibling run.
    """
    SCHEMA='native_capacity_shared_training_state.v1'
    CONFIG=dict(total_steps=1200,H_only_steps=600,low_lr=.01,high_lr=.003,
        betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,
        fullbatch='all injected frames, equal frame weight in external objective',private_regularizers=False)
    COUNTERS=('step_attempts','native_forward_calls','native_backward_calls','backward_calls','optimizer_step_calls',
        'parameter_updates','low_parameter_updates','high_parameter_updates','failed_step_attempts')

    def __init__(self,model,reader,context,objective,*,mode,source_identity,source_metadata,
                 purpose='formal',ledger_hook=None,stage_counts=None):
        if mode not in('rgb','detail'):raise ValueError('mode must be rgb/detail')
        if purpose not in('formal','smoke','profile','fixture'):raise ValueError('explicit checkpoint purpose required')
        self.config=copy.deepcopy(self.CONFIG)
        if stage_counts is not None:
            if purpose!='smoke'or len(stage_counts)!=2:raise ValueError('short stage counts are smoke-only')
            b,c=(_integer(x,'smoke stage count')for x in stage_counts)
            if b<1 or c<1 or b+c>2500:raise ValueError('smoke stage counts outside engineering cap')
            self.config.update(total_steps=b+c,H_only_steps=b)
        params=dict(model.named_parameters())
        if set(params)!= {'low','high'}or params['low']is not model.low or params['high']is not model.high:
            raise ValueError('only named low/high parameters may exist')
        if not all(bool(torch.isfinite(p).all())for p in params.values())or bool((model.high!=0).any()):
            raise ValueError('new state requires finite common L and exactly zero raw H, never profile hot-start')
        frames=tuple(map(int,context.frames))
        if len(frames)!=2 or set(frames)!={86,110}:raise ValueError('fixed fullbatch frames86/110 required')
        if not np.array_equal(reader.active_world_rows,context.active_world_rows):raise ValueError('reader/native topology differs')
        if not isinstance(source_identity,dict)or not isinstance(source_metadata,dict)or not source_identity or not source_metadata:
            raise ValueError('nonempty explicit source identity and original field metadata required')
        self.model=model;self.reader=reader;self.context=context;self.objective=objective;self.mode=mode
        self.frames=tuple(sorted(frames));self.purpose=purpose;self.source_identity=copy.deepcopy(source_identity)
        self.source_metadata=copy.deepcopy(source_metadata);self._source_sha=_identity([self.source_identity,self.source_metadata])
        self.ledger_hook=ledger_hook;self.cursor=0;self.physical={k:0 for k in self.COUNTERS};self.needs_restore=False
        self._initial_low=_cpu_copy(model.low)
        self._buffer_signature=self.buffer_signature()
        self._buffer_versions={k:(id(v),v._version)for k,v in model.named_buffers()}
        self.optimizer=torch.optim.Adam([
            dict(params=[model.low],name='low',lr=self.CONFIG['low_lr']),
            dict(params=[model.high],name='high',lr=self.CONFIG['high_lr'])],
            betas=self.CONFIG['betas'],eps=self.CONFIG['eps'],weight_decay=0.,foreach=False)
        self._set_stage()

    @property
    def stage(self):return 'B'if self.cursor<self.config['H_only_steps']else'C'

    def _set_stage(self):
        self.model.set_stage(self.stage)
        if self.model.low.requires_grad!=(self.stage=='C')or not self.model.high.requires_grad:
            raise RuntimeError('field stage did not freeze/enable the declared parameters')

    def buffer_signature(self):
        return {k:dict(shape=list(v.shape),dtype=str(v.dtype),SHA256=_tensor_sha(v))for k,v in self.model.named_buffers()}

    def assert_frozen(self,*,full=False):
        buffers=dict(self.model.named_buffers())
        if set(buffers)!=set(self._buffer_versions)or any((id(v),v._version)!=self._buffer_versions[k]for k,v in buffers.items()):
            raise RuntimeError('frozen field buffer identity/version changed')
        if full and self.buffer_signature()!=self._buffer_signature:raise RuntimeError('frozen field buffer bytes changed')

    def _event(self,key,amount=1):
        amount=_integer(amount,'physical counter increment');self.physical[key]+=amount
        if self.ledger_hook is not None:self.ledger_hook(dict(event=key,amount=amount,total=self.physical[key],cursor=self.cursor,purpose=self.purpose))

    def _native_backward(self,gradient):
        # One output hook per actual road-raster graph, not one per matrix read.
        self._event('native_backward_calls');return gradient

    def predictions(self,*,grad=True):
        """One shared Sigma read, then actual road raster for both full frames."""
        if not np.array_equal(self.reader.active_world_rows,self.context.active_world_rows):raise RuntimeError('native topology changed')
        manager=torch.enable_grad()if grad else torch.no_grad()
        with manager:
            colors=self.reader.colors(self.model);out={}
            for raw in self.frames:
                before=int(self.context.native_calls)
                try:
                    out[raw]=self.context.render_road(raw,colors)['rgb']
                    if grad and out[raw].requires_grad:out[raw].register_hook(self._native_backward)
                finally:
                    delta=int(self.context.native_calls)-before
                    if delta<0:raise RuntimeError('native call counter went backwards')
                    self._event('native_forward_calls',delta)
            return out

    def step(self):
        if self.needs_restore:raise RuntimeError('failed step requires explicit checkpoint restore')
        if self.cursor>=self.config['total_steps']:raise RuntimeError('registered step budget exhausted')
        self.assert_frozen();self._set_stage();self.optimizer.zero_grad(set_to_none=True)
        self._event('step_attempts');stage=self.stage;cursor=self.cursor
        frozen_low=_cpu_copy(self.model.low)if stage=='B'else None
        try:
            if self.objective.snapshot().get('normalizers')is None:raise RuntimeError('external common loss must be calibrated from same native ci0')
            predictions=self.predictions();loss,terms=self.objective.objective(predictions,self.mode)
            if loss.ndim or not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite/nonscalar common objective')
            self._event('backward_calls');loss.backward();grads={}
            for name in('low','high'):
                p=getattr(self.model,name)
                if not p.requires_grad:
                    if p.grad is not None:raise RuntimeError('inactive group gradient must be None')
                    grads[name]=None;continue
                if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('missing/nonfinite '+name+' gradient')
                grads[name]=dict(L2=float(p.grad.detach().double().norm()),max_abs=float(p.grad.detach().abs().max()))
            self._event('optimizer_step_calls');self.optimizer.step()
            self._event('parameter_updates');self._event('high_parameter_updates')
            if stage=='C':self._event('low_parameter_updates')
            if not all(bool(torch.isfinite(p).all())for p in(self.model.low,self.model.high)):raise FloatingPointError('nonfinite updated field')
            if frozen_low is not None and not torch.equal(frozen_low,self.model.low.detach().cpu()):raise RuntimeError('inactive low parameters changed')
            self.cursor+=1;self._set_stage();self.assert_frozen()
            return dict(cursor_before=cursor,cursor=self.cursor,fullbatch_cursor=self.cursor,stage=stage,
                next_stage='complete'if self.cursor==self.config['total_steps']else self.stage,mode=self.mode,loss=float(loss.detach()),
                gradients=grads,frames=list(self.frames),physical=dict(self.physical),terms=_cpu_copy(terms))
        except Exception:
            self.needs_restore=True;self._event('failed_step_attempts');raise

    def _rng(self):
        states=dict(python=random.getstate(),numpy=copy.deepcopy(np.random.get_state()),torch_cpu=torch.get_rng_state().clone())
        if self.model.high.device.type=='cuda':states['torch_cuda']=[x.clone()for x in torch.cuda.get_rng_state_all()]
        return states

    def snapshot(self):
        if self.needs_restore:raise RuntimeError('cannot checkpoint a failed partial step')
        self.assert_frozen(full=True)
        objective=self.objective.snapshot()
        if objective.get('normalizers')is None:raise RuntimeError('cannot checkpoint uncalibrated common objective')
        return dict(schema=self.SCHEMA,config=copy.deepcopy(self.config),purpose=self.purpose,mode=self.mode,
            source_identity=copy.deepcopy(self.source_identity),source_metadata=copy.deepcopy(self.source_metadata),source_SHA256=self._source_sha,
            frames=self.frames,cursor=self.cursor,fullbatch_cursor=self.cursor,stage=self.stage,
            params={k:_cpu_copy(getattr(self.model,k))for k in('low','high')},optimizer=_cpu_copy(self.optimizer.state_dict()),
            RNG=self._rng(),objective=_cpu_copy(objective),frozen_buffers=copy.deepcopy(self._buffer_signature),
            physical=dict(self.physical),checkpoint_phase='complete'if self.cursor==self.config['total_steps']else'training')

    def _validate_optimizer(self,saved,cursor):
        groups=saved['param_groups']
        if len(groups)!=2:raise ValueError('optimizer groups differ')
        for g,name,steps in zip(groups,('low','high'),(max(0,cursor-self.config['H_only_steps']),cursor)):
            if g.get('name')!=name or g['lr']!=self.CONFIG[name+'_lr']or tuple(g['betas'])!=self.CONFIG['betas']or g['eps']!=self.CONFIG['eps']or g['weight_decay']!=0 or g.get('foreach')is not False or len(g['params'])!=1:
                raise ValueError('optimizer group hyperparameters differ: '+name)
            s=saved['state'].get(g['params'][0],{})
            if steps==0:
                if s:raise ValueError('inactive/zero-step group has moments: '+name)
                continue
            if set(s)!= {'step','exp_avg','exp_avg_sq'}or float(s['step'])!=steps:raise ValueError('optimizer step differs: '+name)
            for k in('exp_avg','exp_avg_sq'):
                if s[k].shape!=getattr(self.model,name).shape or not bool(torch.isfinite(s[k]).all()):raise ValueError('invalid Adam moment: '+name)
            if bool((s['exp_avg_sq']<0).any()):raise ValueError('negative Adam variance')

    def restore(self,saved):
        self.assert_frozen(full=True)
        if saved['schema']!=self.SCHEMA or saved['config']!=self.config or saved['purpose']!=self.purpose or saved['mode']!=self.mode:
            raise ValueError('checkpoint method/phase/purpose differs')
        if tuple(saved['frames'])!=self.frames or saved['source_SHA256']!=self._source_sha or _identity([saved['source_identity'],saved['source_metadata']])!=self._source_sha:
            raise ValueError('checkpoint source identity differs')
        cursor=_integer(saved['cursor'],'cursor',self.config['total_steps'])
        if saved['fullbatch_cursor']!=cursor or saved['stage']!=('B'if cursor<self.config['H_only_steps']else'C')or saved['checkpoint_phase']!=('complete'if cursor==self.config['total_steps']else'training'):
            raise ValueError('checkpoint logical phase differs')
        if saved['frozen_buffers']!=self._buffer_signature:raise ValueError('checkpoint frozen buffers differ; never restore old gates')
        for name in('low','high'):
            x=saved['params'][name]
            if x.shape!=getattr(self.model,name).shape or not bool(torch.isfinite(x).all()):raise ValueError('invalid field parameter: '+name)
        if cursor<=self.config['H_only_steps']and not torch.equal(saved['params']['low'],self._initial_low):raise ValueError('low moved during H-only stage')
        if cursor==0 and bool((saved['params']['high']!=0).any()):raise ValueError('nonzero initial high')
        self._validate_optimizer(saved['optimizer'],cursor)
        counts={k:_integer(saved['physical'][k],'physical '+k)for k in self.COUNTERS}
        if counts['parameter_updates']<cursor or counts['high_parameter_updates']<cursor or counts['low_parameter_updates']<max(0,cursor-self.config['H_only_steps']):
            raise ValueError('physical update ledger is below logical history')
        rng=saved['RNG']
        if self.model.high.device.type=='cuda'and'torch_cuda'not in rng:raise ValueError('CUDA RNG missing for GPU resume')
        self.objective.restore(copy.deepcopy(saved['objective']))
        with torch.no_grad():
            for name in('low','high'):getattr(self.model,name).copy_(saved['params'][name].to(getattr(self.model,name)))
        self.optimizer.load_state_dict(copy.deepcopy(saved['optimizer']));self.optimizer.zero_grad(set_to_none=True)
        self.cursor=cursor;self._set_stage();self.needs_restore=False
        self.physical={k:max(self.physical[k],counts[k])for k in self.COUNTERS}
        random.setstate(rng['python']);np.random.set_state(copy.deepcopy(rng['numpy']));torch.set_rng_state(rng['torch_cpu'])
        if self.model.high.device.type=='cuda':
            torch.cuda.set_rng_state_all(rng['torch_cuda'])
        self.assert_frozen(full=True)
        return dict(cursor=self.cursor,stage=self.stage,physical=dict(self.physical),source_metadata=copy.deepcopy(self.source_metadata))

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError('preserve existing checkpoint: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True);torch.save(self.snapshot(),path);return path

    def load(self,path):return self.restore(torch.load(path,map_location='cpu',weights_only=False))
