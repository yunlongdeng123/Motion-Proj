"""匹配共享场监督/初值的独立RGB控制；模块导入和CPU fixture不加载native。"""
from pathlib import Path
import copy
import hashlib
import json
import random
import sys
import numpy as np
import torch

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
G=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v)for k,v in value.items()}
    if isinstance(value,list):return [cpu(v)for v in value]
    if isinstance(value,tuple):return tuple(cpu(v)for v in value)
    return copy.deepcopy(value)


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
    return a==b


def sampler_trace():
    value=[];generator=random.Random(101)
    for _ in range(400):
        pair=[86,110];generator.shuffle(pair);value.extend(pair)
    return value


class MatchedI0(torch.nn.Module):
    def __init__(self,c0,eligible):
        super().__init__();base=torch.as_tensor(c0,dtype=torch.float32).detach().clone()
        selected=torch.as_tensor(eligible,dtype=torch.bool)
        if base.ndim!=2 or base.shape[1]!=3 or selected.shape!=(len(base),):raise ValueError('ci0/scope形状错误')
        if not bool(torch.isfinite(base).all()) or not bool(((base>=0)&(base<=1)).all()):raise ValueError('ci0超出finite RGB box')
        self.register_buffer('c0',base);self.register_buffer('eligible',selected)
        self.register_buffer('ids',torch.nonzero(selected,as_tuple=False)[:,0])
        self.delta=torch.nn.Parameter(torch.zeros((int(selected.sum()),3),dtype=torch.float32))

    def free_colors(self):return (self.c0[self.ids]+self.delta).clamp(0,1)

    def forward(self):return self.c0.index_copy(0,self.ids,self.free_colors())

    def prior(self):return (self.free_colors()-self.c0[self.ids]).square().mean()

    def project(self):
        with torch.no_grad():self.delta.clamp_(-1,1)


class MatchedI0State:
    def __init__(self,c0,eligible,*,device='cpu',purpose='smoke',binding=None):
        if purpose not in ('smoke','formal'):raise ValueError('purpose必须显式smoke/formal')
        self.model=MatchedI0(c0,eligible).to(device)
        self.optimizer=torch.optim.Adam([self.model.delta],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.schedule=sampler_trace();self.cursor=0;self.trace=[];self.actual_updates=0
        self.purpose=purpose;self.binding=binding or {};self.arm='matched_I0'
        self.backward_calls=0

    def snapshot(self):
        value=dict(schema='matched_I0_full_state.v1',model=cpu(self.model.state_dict()),
            optimizer=cpu(self.optimizer.state_dict()),schedule=self.schedule.copy(),cursor=self.cursor,
            trace=self.trace.copy(),actual_updates=self.actual_updates,purpose=self.purpose,arm=self.arm,
            binding=copy.deepcopy(self.binding),RNG=dict(torch=torch.get_rng_state().clone(),numpy=np.random.get_state(),python=random.getstate()))
        if self.model.delta.device.type=='cuda':value['RNG']['CUDA']=[x.cpu().clone()for x in torch.cuda.get_rng_state_all()]
        return value

    def restore(self,saved):
        if saved['schema']!='matched_I0_full_state.v1' or saved['purpose']!=self.purpose or saved['arm']!=self.arm:
            raise ValueError('smoke/formal/arm checkpoint禁止混用')
        if saved['binding']!=self.binding or saved['schedule']!=self.schedule:raise ValueError('source/algorithm/sampler不同')
        cursor=saved['cursor']
        if not 0<=cursor<=800 or saved['trace']!=self.schedule[:cursor] or len(saved['trace'])!=cursor or saved['actual_updates']!=cursor:
            raise ValueError('cursor/trace/逻辑更新数不一致')
        for key in ['eligible','ids']:
            if not torch.equal(saved['model'][key].cpu(),getattr(self.model,key).cpu()):raise ValueError('world scope不同')
        c0_error=float((saved['model']['c0'].cpu()-self.model.c0.cpu()).abs().max())
        if c0_error>1e-6:raise ValueError('从同filteredA0重建的ci0与checkpoint不符')
        group=saved['optimizer']['param_groups']
        if len(group)!=1 or group[0]['lr']!=.01 or tuple(group[0]['betas'])!=(.9,.999) or group[0]['eps']!=1e-8 or group[0]['weight_decay']!=0:
            raise ValueError('Adam contract不同')
        adam_states=saved['optimizer']['state']
        if cursor==0 and adam_states:raise ValueError('fresh0步不应含旧Adam状态')
        if cursor>0:
            if set(adam_states)!={0}or not {'step','exp_avg','exp_avg_sq'}.issubset(adam_states[0]):
                raise ValueError('非0cursor缺少完整Adam状态')
            if float(adam_states[0]['step'])!=cursor:raise ValueError('Adam step与sampler cursor不同')
        for key,tensor in saved['model'].items():
            current=self.model.state_dict()[key]
            if tensor.shape!=current.shape or tensor.dtype!=current.dtype:raise ValueError('modelshape/dtype不符')
            if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all()):raise ValueError('model非finite')
        if not bool(((saved['model']['delta']>=-1)&(saved['model']['delta']<=1)).all()):raise ValueError('delta不在原界内')
        for index,values in saved['optimizer']['state'].items():
            if index!=0:raise ValueError('未知Adam参数')
            for name,value in values.items():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):raise ValueError('Adam非finite')
                if name in ('exp_avg','exp_avg_sq') and value.shape!=self.model.delta.shape:raise ValueError('Adam momentshape不符')
        self.model.load_state_dict(copy.deepcopy(saved['model']))
        # load_state_dict可共享CPU step tensor；必须deepcopy，防止污染待重放snapshot。
        self.optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        torch.set_rng_state(saved['RNG']['torch']);np.random.set_state(saved['RNG']['numpy']);random.setstate(saved['RNG']['python'])
        if 'CUDA'in saved['RNG']:torch.cuda.set_rng_state_all(saved['RNG']['CUDA'])
        self.cursor=cursor;self.trace=saved['trace'].copy();self.actual_updates=saved['actual_updates']
        return dict(rebuilt_c0_max_error_before_exact_saved_buffer_restore=c0_error,exact_snapshot_buffers_restored=True)

    def step_objective(self,objective):
        if self.cursor>=800:raise RuntimeError('800步硬预算，不自动延长')
        raw=self.schedule[self.cursor]
        self.optimizer.zero_grad(set_to_none=True);loss,terms=objective(raw,self.model())
        prior=self.model.prior();loss=loss+.001*prior;loss.backward();self.backward_calls+=1
        if not bool(torch.isfinite(loss)) or self.model.delta.grad is None or not bool(torch.isfinite(self.model.delta.grad).all()):
            raise FloatingPointError('loss/delta gradient非finite或断路')
        gradient_nonzero=int((self.model.delta.grad!=0).sum());gradient_max=float(self.model.delta.grad.detach().abs().max())
        self.optimizer.step();self.model.project();self.cursor+=1;self.actual_updates+=1;self.trace.append(raw)
        return dict(step=self.cursor,raw=raw,loss=float(loss.detach()),effective_delta_squared=float(prior.detach()),
            gradient_nonzero=gradient_nonzero,gradient_max=gradient_max,
            **{k:float(v.detach())for k,v in terms.items()})


class NativeMatchedI0State(MatchedI0State):
    def __init__(self,contract,*,lease_file,lease_owner,purpose):
        sys.path.insert(0,str(G/'code'))
        from road_capacity_native import assert_gpu_lease,load_context,load_npz
        assert_gpu_lease(lease_file,lease_owner)
        self.contract_path=Path(contract);spec=json.loads(self.contract_path.read_text())
        expected_parent=P.parent/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
        if spec.get('gpu_authorized')is not True or spec.get('formal_updates')!=800:
            raise ValueError('需要root显式登记gpu_authorized=true/唯一formal_updates800')
        if Path(spec['coverage_parent']).resolve()!=expected_parent.resolve():raise ValueError('固定N coverage parent不同')
        self.ctx=load_context(asset_dir=spec['coverage_parent'],lease_file=lease_file,lease_owner=lease_owner)
        torch.set_num_threads(1);torch.manual_seed(101);torch.cuda.manual_seed_all(101);np.random.seed(101);random.seed(101)
        scope=load_npz(P/'method_scope_r1/scope.npz');eligible=np.asarray(scope['eligible'],bool)
        if not np.array_equal(scope['active_world_rows'],self.ctx.active_world_rows)or int(eligible.sum())!=84266:
            raise ValueError('独立RGB必须同scientific84266 scope')
        # 同零修正权威field在同device重query；不从smoke或A200颜色warmstart。
        from world_material_field import WorldMaterialField
        field,metadata=WorldMaterialField.load(P/'cpu_exact_r1/field_initial.pt',device=self.ctx.base_rgb.device)
        if bool(field.low.detach().abs().max()!=0)or bool(field.high.detach().abs().max()!=0):raise ValueError('filteredA0初始field参数非0')
        ids=torch.as_tensor(np.flatnonzero(eligible),device=self.ctx.base_rgb.device)
        with torch.no_grad():
            init=self.ctx.base_rgb.detach().clone();init[ids]=field.query(indices=ids)[0]
        cache=load_npz(P/'matched_I0/input_r1/initialization.npz')
        if not np.array_equal(cache['active_world_rows'],self.ctx.active_world_rows)or not np.array_equal(cache['eligible'],eligible):raise ValueError('CPU init scope不符')
        self.initial_CPU_cache_error=float((init.cpu()-torch.as_tensor(cache['c0'])).abs().max())
        if self.initial_CPU_cache_error>1e-6:raise ValueError('CPU/GPU同filteredA0初始差>1e-6')
        self.field_metadata=metadata;self.targets={}
        for raw in (86,110):
            d=load_npz(P/f'stage0_native_r1/{raw:03d}_targets.npz')
            self.targets[raw]={k:torch.as_tensor(d[k],device=init.device)for k in ['A','C','S','fullS','fixed_alpha','target_A','target_C','target_S']}
        if sum(int(t['S'].sum())for t in self.targets.values())!=3131:raise ValueError('S3131监督被缩')
        binding_files=[self.contract_path,Path(__file__),P/'cpu_exact_r1/field_initial.pt',P/'method_scope_r1/scope.npz',
            P/'matched_I0/input_r1/initialization.npz',
            P/'cpu_exact_r1/A0.npz',P/'cpu_exact_r1/geometry_kernel.npz',P/'code/world_material_field.py',
            P/'code/road_chart_sampling.py',G/'code/road_capacity_native.py',
            expected_parent/'road.npz',expected_parent/'provenance.npz',
            expected_parent/'persistent/scene.json',expected_parent/'persistent/proof.json',
            *[P/f'stage0_native_r1/{r:03d}_targets.npz'for r in(86,110)]]
        for key in('method_contract','preparation'):
            if key in spec:binding_files.append(Path(spec[key]))
        binding={str(path):digest(path)for path in binding_files}
        super().__init__(init.detach().cpu().numpy(),eligible,device=init.device,purpose=purpose,binding=binding)
        del field

    def objective(self,raw,colors):
        out=self.ctx.render_road(raw,colors);t=self.targets[raw];alpha=t['fixed_alpha'][None]
        terms={k:(out['rgb']-alpha*t['target_'+k]).abs()[:,t[k]].sum()/(3*t[k].sum().clamp_min(1))for k in ['A','C','S']}
        return terms['A']+.5*terms['C']+terms['S'],terms

    def step(self):return self.step_objective(self.objective)


def cpu_fixture():
    torch.set_num_threads(1);torch.manual_seed(101);np.random.seed(101);random.seed(101)
    initial=np.full((7,3),.3,np.float32);eligible=np.array([1,0,1,1,0,1,0],bool)
    target=torch.full((7,3),.7)
    objective=lambda raw,colors:((colors-target).square().mean(),dict(photo=(colors-target).square().mean()))
    state=MatchedI0State(initial,eligible,purpose='smoke',binding={'fixture':True})
    assert torch.equal(state.model(),torch.as_tensor(initial))
    state.step_objective(objective);saved=state.snapshot();immutable=copy.deepcopy(saved)
    state.restore(saved);restore_exact=equal(saved,state.snapshot());assert restore_exact
    state.step_objective(objective);next_a=state.snapshot()
    assert equal(saved,immutable),'restore+nextstep污染snapshot，包括Adam CPU step'
    replay=MatchedI0State(initial,eligible,purpose='smoke',binding={'fixture':True});replay.restore(saved)
    replay.step_objective(objective);next_b=replay.snapshot();assert equal(next_a,next_b)
    assert torch.equal(state.model()[~state.model.eligible],state.model.c0[~state.model.eligible])
    with torch.no_grad():state.model.delta.fill_(2)
    state.model.project();assert float(state.model.delta.max())==1 and float(state.model().max())<=1
    rejected=False
    try:MatchedI0State(initial,eligible,purpose='formal',binding={'fixture':True}).restore(saved)
    except ValueError:rejected=True
    assert rejected
    return dict(schema='matched_I0_CPU_fixture.v1',passed=True,GPU_executed=False,native_loaded=False,
        exact_initial_c0=True,scope_outside_colors_exact=True,delta_box_and_output_clip=True,
        effective_delta_regularizer=.001,complete_restore_exact=restore_exact,nextstep_fullstate_exact=True,
        saved_snapshot_not_mutated_by_restore_nextstep=True,CPU_Adam_step_alias_regression_pass=True,
        smoke_to_formal_rejected=True,trace=sampler_trace(),sampler_seed101=True,formal_budget=800)
