"""共享世界材质原生训练状态与阶段入口；formal必须另获root锁定许可。"""
from pathlib import Path
import argparse,copy,hashlib,json,random,sys,time
import numpy as np
import torch
from world_material_field import WorldMaterialField,blur

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
G=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)

def rng():
    return dict(torch=torch.get_rng_state(),CUDA=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())

def restore_rng(value):
    torch.set_rng_state(value['torch']);torch.cuda.set_rng_state_all(value['CUDA'])
    np.random.set_state(value['numpy']);random.setstate(value['python'])

def rgb_l1(pred,target,mask):
    return (pred-target).abs()[:,mask].sum()/(3*mask.sum().clamp_min(1))

def configure_parameter_masks(model,eligible):
    """共享参数仅开放实际合法查询的节点；highpass halo仍受source1原始mask约束。"""
    from world_material_field import sample_field
    low=torch.zeros_like(model.low_parameter_mask);high=torch.zeros_like(model.high_parameter_mask)
    ids=torch.as_tensor(np.flatnonzero(eligible),device=model.center_uv.device)
    with torch.no_grad():
        for begin in range(0,len(ids),1024):
            i=ids[begin:begin+1024]
            uv=model.sample_uv[i].double() if hasattr(model,'sample_uv') else model.center_uv[i,None]+torch.einsum('nij,qj->nqi',model.chol_uv[i],model.nodes)
            good=sample_field(model.valid,uv)[...,0]>1e-12
            points=uv[good]
            for target,scale in [(high,1),(low,model.low_stride)]:
                v=points/scale;lo=torch.floor(v).long()
                for dy,dx in [(0,0),(0,1),(1,0),(1,1)]:
                    x,y=lo[:,0]+dx,lo[:,1]+dy
                    ok=(x>=0)&(x<target.shape[1])&(y>=0)&(y<target.shape[0])
                    target[y[ok],x[ok]]=1
        direct=high.clone()
        radius=int(np.ceil(3*model.highpass_sigma_cells))
        halo=torch.nn.functional.max_pool2d(high.permute(2,0,1)[None],2*radius+1,stride=1,padding=radius)[0].permute(1,2,0)
        model.high_parameter_mask.copy_(halo*(model.mH>0));model.high_output_mask.copy_(direct*(model.mH>0))
        model.low_parameter_mask.copy_(low)
    return dict(low_trainable_scalars=int(low.sum())*3,high_trainable_scalars=int(model.high_parameter_mask.sum())*3,
        high_regularizer_support_nodes=int(model.high_output_mask.sum()),source_mask_outside_highpass_latent_frozen=True)

class TrainingState:
    def __init__(self,field,scope,targets,contract,lease_owner,purpose='smoke'):
        sys.path.insert(0,str(G/'code'));sys.path.insert(0,str(REPO/'scripts/worldsim_v82'))
        from road_capacity_native import assert_gpu_lease,load_context,load_npz
        assert_gpu_lease(P/'gpu_lease.json',lease_owner)
        if purpose not in ['smoke','formal']:raise ValueError('训练状态角色无效')
        self.purpose=purpose;self.contract=json.loads(Path(contract).read_text());self.field_path=Path(field)
        self.scope_path=Path(scope);self.target_path=Path(targets)
        self.ctx=load_context(asset_dir=self.contract['coverage_parent'],lease_file=P/'gpu_lease.json',lease_owner=lease_owner)
        torch.set_num_threads(1);torch.manual_seed(101);torch.cuda.manual_seed_all(101);np.random.seed(101);random.seed(101)
        self.model,self.field_metadata=WorldMaterialField.load(field,device='cpu')
        if len(self.model.weights)!=225 or not hasattr(self.model,'sample_uv'):
            raise ValueError('必须先通过真实chart GH15 query资格，不能用旧global-affine草稿启动')
        s=load_npz(scope)
        if not np.array_equal(s['active_world_rows'],self.ctx.active_world_rows):raise ValueError('field scope与当前coverage active顺序不符')
        self.eligible=np.asarray(s['eligible'],bool)
        self.mask_statistics=configure_parameter_masks(self.model,self.eligible)
        self.model=self.model.to(self.ctx.base_rgb.device)
        self.ids=torch.as_tensor(np.flatnonzero(self.eligible),device=self.ctx.base_rgb.device)
        self.targets={}
        for raw in (86,110):
            d=load_npz(Path(targets)/f'{raw:03d}_targets.npz')
            self.targets[raw]={key:torch.as_tensor(d[key],device=self.ctx.base_rgb.device) for key in ['A','C','S','target_A','target_C','target_S','fixed_alpha','fullS']}
        if sum(int(t['S'].sum()) for t in self.targets.values())!=3131:raise ValueError('S3131完整监督分母改变')
        self.optimizer=torch.optim.Adam([dict(params=[self.model.low],lr=.01),dict(params=[self.model.high],lr=.003)],betas=(.9,.999),eps=1e-8)
        self.schedule=[];sampler=random.Random(101)
        for _ in range(400):
            pair=[86,110];sampler.shuffle(pair);self.schedule.extend(pair)
        self.cursor=0;self.trace=[];self.stage='A';self.arm='common';self.actual_updates=0
        files=[Path(field),Path(scope),Path(contract),Path(__file__),Path(__file__).with_name('world_material_field.py'),
            Path(__file__).with_name('road_chart_sampling.py'),P/'cpu_numeric_r1/receipt.json',P/'cpu_exact_r1/A0.npz',
            P/'cpu_exact_r1/geometry_kernel.npz',P/'method_scope_r1/receipt.json',P/'stage0_native_r1/receipt.json',
            *[Path(targets)/f'{raw:03d}_targets.npz' for raw in (86,110)]]
        self.binding={str(p):digest(p) for p in files}
        self.frozen_versions={k:v._version for k,v in self.model.named_buffers()}

    def colors(self,center_only=False):
        value,mass=self.model.query(indices=self.ids,center_only=center_only)
        if not bool(torch.isfinite(value).all()) or bool((mass<=1e-12).any()):raise RuntimeError('合法field查询出现空核/非finite')
        color=self.ctx.base_rgb.clone();color[self.ids]=value
        return color

    def regularizers(self):
        low,_=self.model.fields();mask=self.model.low_parameter_mask[...,0]>0
        mx=mask[:,1:]&mask[:,:-1];my=mask[1:]&mask[:-1]
        energy=((low[:,1:]-low[:,:-1]).square()[mx].sum()+(low[1:]-low[:-1]).square()[my].sum())/(3*(mx.sum()+my.sum()).clamp_min(1))
        h=self.model.effective_high_statistics(self.model.high_output_mask)
        return dict(low_adjacent_delta_squared=energy,effective_high_energy=h['effective_high_energy'],
            effective_high_lowfrequency_leakage=h['effective_high_lowfrequency_energy'],
            effective_high_weighted_DC=h['effective_high_DC'].square().mean())

    def objective(self,raw):
        colors=self.colors();out=self.ctx.render_road(raw,colors);t=self.targets[raw];alpha=t['fixed_alpha'][None]
        terms={key:rgb_l1(out['rgb'],alpha*t['target_'+key],t[key]) for key in ['A','C','S']}
        reg=self.regularizers();weights=self.contract['optimizer']['regularizers']
        loss=terms['A']+.5*terms['C']+terms['S']+sum(weights[k]*v for k,v in reg.items())
        return loss,dict(**terms,**reg)

    def step(self,stage=None):
        if stage:self.stage=stage
        self.model.set_stage(self.stage);raw=self.schedule[self.cursor]
        self.optimizer.zero_grad(set_to_none=True);loss,terms=self.objective(raw);loss.backward()
        if not torch.isfinite(loss):raise RuntimeError('loss nonfinite')
        for p in [self.model.low,self.model.high]:
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()):raise RuntimeError('共享field gradient nonfinite')
        self.optimizer.step();self.cursor+=1;self.actual_updates+=1;self.trace.append(raw)
        return dict(cursor=self.cursor,stage=self.stage,raw=raw,loss=float(loss),**{k:float(v.detach()) for k,v in terms.items()})

    def snapshot(self):
        return dict(schema='multiscale_complete_training_state.v1',params=cpu(dict(low=self.model.low,high=self.model.high)),
            optimizer=cpu(self.optimizer.state_dict()),RNG=cpu(rng()),schedule=self.schedule.copy(),cursor=self.cursor,
            trace=self.trace.copy(),stage=self.stage,arm=self.arm,binding=self.binding.copy(),actual_updates=self.actual_updates,purpose=self.purpose)

    def restore(self,saved,allow_common_fork=False):
        if saved['binding']!=self.binding:raise ValueError('checkpoint source/algorithm binding不同')
        if saved.get('purpose')!=self.purpose:raise ValueError('smoke/formal checkpoint禁止混用')
        if saved['arm']!=self.arm and not (allow_common_fork and saved['arm']=='common'):raise ValueError('checkpoint分支角色不同')
        if not 0<=saved['cursor']<=800 or saved['schedule']!=self.schedule:raise ValueError('checkpoint sampler范围/顺序不同')
        if len(saved['trace'])!=saved['cursor'] or saved['trace']!=saved['schedule'][:saved['cursor']] or saved['actual_updates']!=saved['cursor']:
            raise ValueError('checkpoint sampler trace/逻辑更新数不同')
        groups=saved['optimizer']['param_groups']
        if len(groups)!=2 or groups[0]['lr']!=.01 or groups[1]['lr']!=.003:raise ValueError('Adam group/LR不同')
        for index,adam in saved['optimizer']['state'].items():
            shape=self.model.low.shape if index==0 else self.model.high.shape
            if index not in (0,1):raise ValueError('Adam未知参数')
            for key,value in adam.items():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):raise ValueError('Adam非finite')
                if key in ['exp_avg','exp_avg_sq'] and value.shape!=shape:raise ValueError('Adam moment shape不同')
        with torch.no_grad():
            for key in ['low','high']:
                x=saved['params'][key];p=getattr(self.model,key)
                if x.shape!=p.shape or not bool(torch.isfinite(x).all()):raise ValueError('checkpoint field shape/finite错误')
                p.copy_(x.to(p.device))
        # Adam可保留CPU step tensor别名；必须deepcopy，不能让下一步修改调用方snapshot。
        self.optimizer.load_state_dict(copy.deepcopy(saved['optimizer']));restore_rng(saved['RNG'])
        self.cursor=saved['cursor'];self.trace=saved['trace'].copy();self.stage=saved['stage'];self.arm=saved['arm'];self.actual_updates=saved['actual_updates']
        self.model.set_stage(self.stage)

    def save_checkpoint(self,path):torch.save(self.snapshot(),Path(path))

    def assert_frozen(self):
        for key,value in self.model.named_buffers():
            if value._version!=self.frozen_versions[key]:raise RuntimeError('frozen field/kernel/mask改变: '+key)
        self.ctx.assert_frozen()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args()
    if not args.inspect_only:raise SystemExit('状态实现已就绪；生命周期/正式controller由独立入口显式调度，禁止默认训练')
    print(json.dumps(dict(executable_state_ready=True,formal_default_enabled=False,independent_ci_parameters=False)))
