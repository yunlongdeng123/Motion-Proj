"""Module2阶段1：冻结道路位置/法向/材质，优化opacity与预声明有界面内sigma。

评价域与有梯度的监督域分别保存；无几何和未知材质孔洞仍进入评价分母。
旧版道路 alpha 仅用于范围外保留正则，不被称为独立真值。
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import copy
import random
import numpy as np


@dataclass(frozen=True)
class CoverageContract:
    formal_steps: int = 600
    opacity_lr: float = .05
    outside_alpha_weight: float = 1.
    retained_alpha_weight: float = 4.
    seed: int = 101
    train_tangent_scales: bool = False
    tangent_scale_max_factor: float = 1.15
    tangent_scale_lr: float = .005
    loss: str = 'road_only_alpha_mse'

    def __post_init__(self):
        if self.formal_steps != 600 or type(self.seed) is not int:
            raise ValueError('阶段1合同固定600正式步，seed必须为整数')
        if any(not np.isfinite(v) or v < 0 for v in [self.opacity_lr,self.outside_alpha_weight,self.retained_alpha_weight]) or self.opacity_lr == 0:
            raise ValueError('学习率与损失权重必须有限且非负')
        if (not np.isfinite(self.tangent_scale_max_factor) or not 1<=self.tangent_scale_max_factor<=1.15
                or not np.isfinite(self.tangent_scale_lr) or self.tangent_scale_lr<=0):
            raise ValueError('面内scale上界必须<=1.15且学习率有限正数')
        if self.loss != 'road_only_alpha_mse':
            raise ValueError('未登记的监督合同')


def opacity_scope(road, provenance):
    """生成完整world-row固定训练集；可信几何的opacity可训练，几何本身冻结。"""
    active = np.asarray(road['active'], bool)
    n = len(active)
    def field(name):
        a = np.asarray(provenance[name])
        if a.shape != (n,):
            raise ValueError('道路来源拓扑不一致: '+name)
        return a.astype(bool)
    guarded = field('guarded_world_rows') if 'guarded_world_rows' in provenance else np.zeros(n,bool)
    minimum = np.asarray(road.get('minimum_model_depth', np.zeros(n)))
    if minimum.shape != (n,) or not np.isfinite(minimum).all():
        raise ValueError('持久深度保护必须与完整道路行一致')
    request_field='coverage_requested' if 'coverage_requested' in provenance else 'requested'
    gates = dict(active=active, coverage_requested=field(request_field), legal=field('legal_write'),
        supported=field('surface_supported'), material_available=~field('unknown'),
        retained_world_unlocked=~guarded, legacy_near_unlocked=minimum<=0)
    eligible = np.logical_and.reduce(list(gates.values()))
    count = n;sequential=[];current=np.ones(n,bool)
    for name,mask in gates.items():
        after=current&mask
        sequential.append(dict(gate=name,before=int(current.sum()),excluded=int((current&~mask).sum()),after=int(after.sum())))
        current=after
    return eligible, dict(schema='road_opacity_scope.v1',complete_world_rows=count,
        trainable_world_rows=int(eligible.sum()),independent_gate_counts={k:int(v.sum()) for k,v in gates.items()},
        coverage_request_field=request_field,generated_colour_request_unchanged=True,
        sequential_exclusions=sequential,guarded_world_rows=int(guarded.sum()),
        trained_parameters=['road_opacity_logits'],frozen=['road_xyz','road_z','road_normal','road_tangent_sigma','road_normal_sigma',
            'road_reference_srgb','native_static','native_dynamic_vehicles','environment','cameras','ownership','existing_static_gates'],
        trusted_geometry_opacity_allowed=True)


def tangent_scale_scope(road, provenance, opacity_eligible, max_factor=1.15, cell_m=.02):
    """只扩张非可信补全面内sigma；每行最终3σ留在合法表面/材料域内部。"""
    from scipy.spatial import cKDTree
    if not 1<=max_factor<=1.15:raise ValueError('scale上界超出预登记15%')
    n=len(road['active']);keys=np.asarray(provenance['world_keys'])
    if keys.shape!=(n,2) or keys.dtype.kind not in 'iu':raise ValueError('需要完整固定world-grid keys')
    ij=keys-keys.min(0);width,height=ij.max(0)+1
    if n!=width*height or len(np.unique(ij,axis=0))!=n:raise ValueError('world-grid必须完整互异')
    # center训练授权来自opacity_eligible的coverage request；footprint边界来自
    # 物理可信路面与可用材质，不能再把生成/请求图格边界当作道路存在边界。
    legal=(np.asarray(provenance['legal_write'],bool)
        &np.asarray(provenance['surface_supported'],bool)&~np.asarray(provenance['unknown'],bool))
    guarded=np.asarray(provenance.get('guarded_world_rows',np.zeros(n,bool)),bool)
    legal &= ~guarded & (np.asarray(road.get('minimum_model_depth',np.zeros(n)))<=0)
    # 使用真实世界XY距离而非将参考grid间距假定为精确米制距离。
    world=np.asarray(provenance['world_xyz'],np.float64)
    grid=np.empty((height,width),np.int64);grid[ij[:,1],ij[:,0]]=np.arange(n)
    xy=world[grid,:2]
    max_dx=float(np.linalg.norm(np.diff(xy,axis=1),axis=2).max())
    max_dy=float(np.linalg.norm(np.diff(xy,axis=0),axis=2).max())
    half_diagonal=.5*np.hypot(max_dx,max_dy)
    edge=(ij[:,0]==0)|(ij[:,1]==0)|(ij[:,0]==width-1)|(ij[:,1]==height-1)
    blocked=~legal|edge
    candidates=np.flatnonzero(np.asarray(opacity_eligible,bool))
    distance=np.zeros(n,np.float64)
    nearest,_=cKDTree(world[blocked,:2]).query(world[candidates,:2])
    distance[candidates]=np.maximum(nearest-half_diagonal,0)
    sigma=np.asarray(road['gaussian_sigma_m']);largest=np.max(sigma[:,:2],axis=1)
    upper=np.minimum(max_factor,np.maximum(1.,(distance-1e-4)/(3*largest)))
    trusted_field='coverage_trusted' if 'coverage_trusted' in provenance else 'trusted_geometry'
    trusted=np.asarray(provenance[trusted_field],bool)
    eligible=np.asarray(opacity_eligible,bool)&~trusted&(upper>1.001)
    upper[~eligible]=1
    if np.any(3*largest[eligible]*upper[eligible]>=distance[eligible]):raise RuntimeError('最终3σ越过合法表面边界')
    return eligible,upper,distance,dict(schema='bounded_road_tangent_scale_scope.v1',trainable_world_rows=int(eligible.sum()),
        max_factor=float(max_factor),minimum_boundary_margin_m=1e-4,grid_cell_m=cell_m,
        distance='actual world XY nearest illegal/boundary node minus maximum cell half-diagonal',
        maximum_grid_half_diagonal_m=float(half_diagonal),
        trusted_mask_field=trusted_field,
        center_scope='opacity eligible coverage request',footprint_scope='physical legal supported material-available road excluding retained guards',
        trusted_geometry_sigma_frozen=True,normal_sigma_frozen=True,normal_and_xyz_frozen=True,
        upper_bound_uses_final_sigma=True,minimum_factor=1.,only_expansion=True)


def coverage_domains(view, labels, provenance, *, target_actor_id=10, eligible_world=None):
    """旧冻结评价域不缩小；监督域只取合法有表面、有材质的世界请求行。"""
    index = np.asarray(view['index'])
    labels = np.asarray(labels)
    if index.shape != labels.shape or index.dtype.kind not in 'iu':
        raise ValueError('道路像素索引与实例标签形状不一致')
    n = len(provenance['requested'])
    if (index < 0).any() or (index >= n).any():
        raise ValueError('world-row索引越界')
    k = (labels > 0)&(labels != target_actor_id)
    evaluation = (np.asarray(view['target'],bool)&np.asarray(view['inside'],bool)
        &~np.asarray(view['retained_reveal_guard'],bool)&~np.asarray(view['retained_box_guard'],bool)&~k)
    coverage_requested=provenance.get('coverage_requested',provenance['requested'])
    world = (np.asarray(coverage_requested,bool)&np.asarray(provenance['legal_write'],bool)
        &np.asarray(provenance['surface_supported'],bool)&~np.asarray(provenance['unknown'],bool))
    guarded = np.asarray(provenance.get('guarded_world_rows',np.zeros(n,bool)),bool)
    legal_supervision = evaluation & world[index] & ~guarded[index]
    if eligible_world is None:
        eligible_world=world&~guarded
    eligible_world=np.asarray(eligible_world)
    if eligible_world.shape!=(n,) or eligible_world.dtype!=np.bool_:raise ValueError('训练行授权拓扑无效')
    fit = legal_supervision & eligible_world[index]
    return dict(evaluation=evaluation, fit=fit, retained=k, outside=~fit&~k,
        legal_but_frozen=legal_supervision&~eligible_world[index],
        geometry_unsupported=evaluation&~np.asarray(provenance['surface_supported'],bool)[index],
        material_unknown=evaluation&np.asarray(provenance['unknown'],bool)[index],
        frozen_world_guard=evaluation&guarded[index])


def alpha_coverage_metrics(alpha, domains, *, maximum_alpha=None):
    mask = np.asarray(domains['evaluation'],bool)
    a = np.asarray(alpha)
    if a.shape == (1,)+mask.shape:a=a[0]
    if a.shape != mask.shape or not np.isfinite(a).all():
        raise ValueError('alpha评价输入无效')
    total=int(mask.sum());gap=mask&(a<.05);thin=mask&(a>=.05)&(a<=.9);supported=mask&(a>.9)
    result=dict(evaluation_pixels=total,gap_pixels=int(gap.sum()),thin_pixels=int(thin.sum()),supported_pixels=int(supported.sum()),
        alpha_mean=float(a[mask].mean()) if total else None,
        geometry_unsupported_pixels=int(domains['geometry_unsupported'].sum()),
        material_unknown_pixels=int(domains['material_unknown'].sum()),
        fit_pixels=int(domains['fit'].sum()),guarded_evaluation_pixels=int(domains['frozen_world_guard'].sum()))
    if 'surface_intersection_failed' in domains:result['surface_intersection_failed_pixels']=int(domains['surface_intersection_failed'].sum())
    if 'legal_but_frozen' in domains:result['legal_but_frozen_pixels']=int(domains['legal_but_frozen'].sum())
    if maximum_alpha is not None:
        m=np.asarray(maximum_alpha)
        if m.shape == (1,)+mask.shape:m=m[0]
        result.update(opacity_maximum_gap_pixels=int((mask&(m<.05)).sum()),
            opacity_maximum_thin_pixels=int((mask&(m>=.05)&(m<=.9)).sum()),
            opacity_can_close_gap_pixels=int((gap&(m>=.05)).sum()),
            opacity_can_reach_supported_pixels=int((mask&(a<=.9)&(m>.9)).sum()),
            opacity_uncontrollable_gap_pixels=int((gap&(m<.05)).sum()),
            opacity_uncontrollable_gap_fraction=float((gap&(m<.05)).sum()/max(int(gap.sum()),1)))
    return result


def _tensor_difference(a,b,path=''):
    import torch
    if torch.is_tensor(a):
        return [] if torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu()) else [path]
    if isinstance(a,np.ndarray):
        return [] if isinstance(b,np.ndarray) and np.array_equal(a,b) else [path]
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():return [path]
        return [p for k in a for p in _tensor_difference(a[k],b[k],path+'/'+str(k))]
    if isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):return [path]
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in _tensor_difference(x,y,path+'/'+str(i))]
    return [] if a==b else [path]


class CoverageTrainer:
    """通过注入的真实road-only renderer训练，持有独立logits而不改源模型。"""
    schema='road_coverage_training.v1'

    def __init__(self, base_opacity, active_train_ids, frame_keys, render_alpha,
                 domains, baseline_alpha, binding, contract=None, *,base_scales=None,active_scale_ids=None,scale_upper=None,accounting_role='formal'):
        import torch
        self.torch=torch;self.contract=contract or CoverageContract()
        if accounting_role not in ('phase0','smoke','formal'):raise ValueError('未知更新记账角色')
        self.accounting_role=accounting_role
        self.base_opacity=base_opacity.detach().clone()
        if (self.base_opacity.ndim!=1 or not self.base_opacity.is_floating_point() or not torch.isfinite(self.base_opacity).all()
                or (self.base_opacity<0).any() or (self.base_opacity>1).any()):raise ValueError('基准opacity必须是有限[0,1]向量')
        ids=torch.as_tensor(active_train_ids,device=base_opacity.device,dtype=torch.long)
        if ids.ndim!=1 or not len(ids) or len(torch.unique(ids))!=len(ids) or (ids<0).any() or (ids>=len(base_opacity)).any():
            raise ValueError('需要非空、互异的活跃道路训练行索引')
        self.ids=ids;self.keys=tuple(frame_keys)
        if not self.keys or len(set(self.keys))!=len(self.keys):raise ValueError('训练帧必须非空互异')
        torch.manual_seed(self.contract.seed);np.random.seed(self.contract.seed);random.seed(self.contract.seed)
        self.logits=torch.nn.Parameter(torch.logit(self.base_opacity[ids].clamp(1e-5,1-1e-5)))
        self.base_scales=None;self.scale_ids=None;self.scale_upper=None;self.scale_factor=None
        groups=[dict(params=[self.logits],lr=self.contract.opacity_lr)]
        if self.contract.train_tangent_scales:
            self.base_scales=base_scales.detach().clone()
            self.scale_ids=torch.as_tensor(active_scale_ids,device=base_opacity.device,dtype=torch.long)
            self.scale_upper=torch.as_tensor(scale_upper,device=base_opacity.device,dtype=base_opacity.dtype).clone()
            if (self.base_scales.shape!=(len(base_opacity),3) or self.scale_ids.ndim!=1 or not len(self.scale_ids)
                    or len(torch.unique(self.scale_ids))!=len(self.scale_ids) or (self.scale_ids<0).any() or (self.scale_ids>=len(base_opacity)).any()
                    or self.scale_upper.shape!=self.scale_ids.shape or not torch.isfinite(self.scale_upper).all()
                    or (self.scale_upper<1).any() or (self.scale_upper>self.contract.tangent_scale_max_factor+1e-6).any()):
                raise ValueError('有界scale参数或授权无效')
            if (not torch.isin(self.scale_ids,self.ids).all() or not torch.isfinite(self.base_scales).all()
                    or (self.base_scales<=0).any()):raise ValueError('scale必须属于同一合法opacity行且sigma有限正数')
            self.scale_factor=torch.nn.Parameter(torch.ones_like(self.scale_upper))
            groups.append(dict(params=[self.scale_factor],lr=self.contract.tangent_scale_lr))
        self.optimizer=torch.optim.Adam(groups,eps=1e-15)
        self.sampler=random.Random(self.contract.seed);self.remaining=[];self.step=0
        self.render_alpha=render_alpha;self.domains=domains;self.baseline_alpha=baseline_alpha
        self.binding=copy.deepcopy(binding)

    def opacity(self, logits=None):
        t=self.torch;value=self.logits if logits is None else logits
        return self.base_opacity.index_copy(0,self.ids,value.sigmoid())

    def scales(self,factor=None):
        if self.scale_factor is None:return None
        t=self.torch;factor=self.scale_factor if factor is None else factor
        base=self.base_scales[self.scale_ids]
        changed=t.cat([base[:,:2]*factor[:,None],base[:,2:3]],1)
        return self.base_scales.index_copy(0,self.scale_ids,changed)

    def loss_for(self,key):
        t=self.torch
        alpha=self.render_alpha(key,self.opacity(),self.scales()) if self.scale_factor is not None else self.render_alpha(key,self.opacity())
        dom=self.domains[key];base=self.baseline_alpha[key]
        def mean(value,mask):
            return value[mask].mean() if mask.any() else self.logits.sum()*0
        coverage=mean((alpha-1).square(),dom['fit'])
        outside=mean((alpha-base).square(),dom['outside'])
        retained=mean((alpha-base).square(),dom['retained'])
        loss=coverage+self.contract.outside_alpha_weight*outside+self.contract.retained_alpha_weight*retained
        return loss,dict(coverage=float(coverage.detach()),outside_alpha_regularizer=float(outside.detach()),retained_alpha_regularizer=float(retained.detach()))

    def update(self):
        t=self.torch
        if self.step>=self.contract.formal_steps:raise RuntimeError('正式更新预算已达到600步')
        if not self.remaining:
            self.remaining=list(self.keys);self.sampler.shuffle(self.remaining)
        key=self.remaining.pop();self.optimizer.zero_grad(set_to_none=True)
        loss,parts=self.loss_for(key)
        if not bool(t.isfinite(loss)):raise RuntimeError('nonfinite coverage loss')
        loss.backward()
        if self.logits.grad is None or not bool(t.isfinite(self.logits.grad).all()):raise RuntimeError('nonfinite opacity gradient')
        grad=self.logits.grad.detach()
        gradient=dict(nonzero_gradient_parameters=int((grad.abs()>1e-12).sum()),gradient_l1=float(grad.abs().sum()),gradient_max=float(grad.abs().max()))
        if self.scale_factor is not None:
            sg=self.scale_factor.grad
            if sg is None or not bool(t.isfinite(sg).all()):raise RuntimeError('nonfinite tangent scale gradient')
            gradient.update(nonzero_scale_gradient_parameters=int((sg.abs()>1e-12).sum()),scale_gradient_max=float(sg.abs().max()))
        self.optimizer.step()
        with t.no_grad():
            low=t.logit(self.logits.new_tensor(1e-5));high=t.logit(self.logits.new_tensor(1-1e-5))
            self.logits.clamp_(low,high)
            if self.scale_factor is not None:self.scale_factor.copy_(t.minimum(self.scale_factor.clamp_min(1),self.scale_upper))
        self.step+=1
        return dict(step=self.step,key=key,loss=float(loss.detach()),**parts,**gradient)

    def state(self):
        t=self.torch
        return dict(schema=self.schema,step=self.step,logits=self.logits.detach().clone(),ids=self.ids.detach().clone(),
            accounting_role=self.accounting_role,
            scale_ids=None if self.scale_ids is None else self.scale_ids.detach().clone(),
            scale_factor=None if self.scale_factor is None else self.scale_factor.detach().clone(),
            scale_upper=None if self.scale_upper is None else self.scale_upper.detach().clone(),
            optimizer=copy.deepcopy(self.optimizer.state_dict()),sampler=self.sampler.getstate(),remaining=self.remaining.copy(),
            binding=copy.deepcopy(self.binding),contract=asdict(self.contract),keys=self.keys,
            torch_rng=t.get_rng_state(),cuda_rng=t.cuda.get_rng_state_all() if self.logits.is_cuda else [],
            numpy_rng=np.random.get_state(),python_rng=random.getstate())

    def restore(self,state):
        t=self.torch
        if state.get('schema')!=self.schema or state.get('binding')!=self.binding or state.get('contract')!=asdict(self.contract) or tuple(state.get('keys',()))!=self.keys:
            raise ValueError('checkpoint改变了输入绑定、帧或训练合同')
        if state.get('accounting_role')!=self.accounting_role:raise ValueError('checkpoint改变了smoke/正式更新记账角色')
        if not t.equal(state['ids'].to(self.ids.device),self.ids):raise ValueError('checkpoint训练行索引改变')
        if type(state.get('step')) is not int or not 0<=state['step']<=self.contract.formal_steps:
            raise ValueError('checkpoint正式步数越界')
        if (not isinstance(state.get('remaining'),list) or len(set(state['remaining']))!=len(state['remaining'])
                or not set(state['remaining'])<=set(self.keys)):
            raise ValueError('checkpoint采样队列无效')
        if state['logits'].shape!=self.logits.shape or not bool(t.isfinite(state['logits']).all()):
            raise ValueError('checkpoint opacity参数无效')
        if self.scale_factor is not None:
            if (not t.equal(state['scale_ids'].to(self.scale_ids.device),self.scale_ids)
                    or not t.equal(state['scale_upper'].to(self.scale_upper.device),self.scale_upper)
                    or state['scale_factor'].shape!=self.scale_factor.shape or not bool(t.isfinite(state['scale_factor']).all())
                    or (state['scale_factor']<1).any() or (state['scale_factor'].to(self.scale_upper.device)>self.scale_upper).any()):
                raise ValueError('checkpoint scale授权或参数无效')
        with t.no_grad():self.logits.copy_(state['logits'])
        if self.scale_factor is not None:
            with t.no_grad():self.scale_factor.copy_(state['scale_factor'])
        self.optimizer.load_state_dict(state['optimizer']);self.step=state['step'];self.sampler.setstate(state['sampler']);self.remaining=state['remaining'].copy()
        t.set_rng_state(state['torch_rng'].cpu())
        if self.logits.is_cuda:t.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        differences=_tensor_difference(state,self.state())
        if differences:raise RuntimeError('full-state restore mismatch: '+str(differences))
        return dict(full_state_restored=True,step=self.step,differences=[])
