"""Fixed-road bounded RGB capacity diagnostic; import does not load a native scene.

The primary operator is an injected actual native raster context.  An optional
CSR read represents selected native pixel rows only; it is never exported as a
replacement scene image.  Geometry and alpha are not optimization variables.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ColorLayout:
    active_world_rows: np.ndarray
    eligible_active: np.ndarray

    def __post_init__(self):
        ids=np.asarray(self.active_world_rows)
        gate=np.asarray(self.eligible_active)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(np.diff(ids.astype(np.int64))<=0):
            raise ValueError('active_world_rows must be unique, strictly increasing global world IDs')
        if gate.dtype!=bool or gate.shape!=ids.shape:raise ValueError('eligible_active must be a boolean active-local mask')
        ids=ids.astype(np.int64,copy=True);gate=gate.copy();ids.setflags(write=False);gate.setflags(write=False)
        object.__setattr__(self,'active_world_rows',ids);object.__setattr__(self,'eligible_active',gate)

    @property
    def signature(self):
        return hashlib.sha256(self.active_world_rows[self.eligible_active].astype('<i8').tobytes()).hexdigest()

    @property
    def eligible_world_rows(self):return self.active_world_rows[self.eligible_active]


class BoundedGSColors(nn.Module):
    """Only eligible per-Gaussian RGB is trainable; readonly rows remain bit-exact.

    This is a conditional in-sample capacity diagnostic, not a reusable material
    function or a claim about novel views.  Caller must project after any update.
    """
    SCHEMA='fixed_road_bounded_GS_RGB.v1'

    def __init__(self,c0,layout):
        super().__init__();c0=torch.as_tensor(c0)
        if not c0.is_floating_point()or c0.shape!=(len(layout.active_world_rows),3):raise ValueError('c0 must be active-local Nx3 float RGB')
        if not bool(torch.isfinite(c0).all())or bool(((c0<0)|(c0>1)).any()):raise ValueError('c0 must be finite native shader sRGB in [0,1]')
        self.layout=layout
        self.register_buffer('c0',c0.detach().clone())
        self.register_buffer('eligible_positions',torch.as_tensor(np.flatnonzero(layout.eligible_active),device=c0.device))
        self.rgb=nn.Parameter(self.c0[self.eligible_positions].clone())

    def forward(self):
        if not bool(torch.isfinite(self.rgb).all())or bool(((self.rgb<0)|(self.rgb>1)).any()):
            raise ValueError('capacity RGB left its box; project before rendering')
        return self.c0.index_copy(0,self.eligible_positions,self.rgb)

    @torch.no_grad()
    def project_(self):
        if not bool(torch.isfinite(self.rgb).all()):raise ValueError('nonfinite RGB cannot be projected')
        self.rgb.clamp_(0,1)

    def payload(self):
        return dict(schema=self.SCHEMA,c0=self.c0.detach().cpu().clone(),rgb=self.rgb.detach().cpu().clone(),
                    active_world_rows=self.layout.active_world_rows.copy(),eligible_active=self.layout.eligible_active.copy(),
                    scope_sha256=self.layout.signature,shader='native_encoded_sRGB',geometry_alpha_frozen=True,
                    role='in_sample_conditional_capacity_diagnostic_not_production_material')

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError('preserve the previous color oracle checkpoint')
        self();torch.save(self.payload(),path)

    @classmethod
    def load(cls,path,*,device='cpu'):
        return cls.from_payload(torch.load(path,map_location='cpu',weights_only=False),device=device)

    @classmethod
    def from_payload(cls,payload,*,device='cpu'):
        if payload.get('schema')!=cls.SCHEMA:raise ValueError('unknown color capacity schema')
        layout=ColorLayout(payload['active_world_rows'],payload['eligible_active'])
        if payload.get('scope_sha256')!=layout.signature:raise ValueError('world identity mismatch')
        model=cls(payload['c0'].to(device),layout)
        rgb=payload['rgb'].to(device=device,dtype=model.rgb.dtype)
        if rgb.shape!=model.rgb.shape:raise ValueError('RGB topology mismatch')
        with torch.no_grad():model.rgb.copy_(rgb)
        model();return model


class InjectedNativeGSOperator:
    """Passive adapter. Context creation and GPU authorization belong to root.

    Existing scene._bindings().raster receives each individual c_i.  This does
    not use aggregate eligible_weight*pixel_material as the color model.
    """
    def __init__(self,ctx,layout,*,image_hw=(1280,1920)):
        self.ctx=ctx;self.layout=layout;self.image_hw=tuple(image_hw)
        if not np.array_equal(ctx.active_world_rows,layout.active_world_rows):raise ValueError('native active world identity differs')
        if ctx.base_rgb.shape!=(len(layout.active_world_rows),3):raise ValueError('native color packet topology differs')
        self.fixed_alpha={}
        for raw in ctx.frames:
            camera=ctx.camera(raw)
            if (camera.image_height,camera.image_width)!=self.image_hw:raise ValueError('native camera is not the registered full resolution')
            packet=ctx.base_packet(raw)
            for key in ('means3D','opacities','scales','rotations'):
                value=packet.get(key)
                if torch.is_tensor(value)and value.requires_grad:raise ValueError('geometry/coverage must be detached')

    def road(self,raw,colors):
        value=self.ctx.render_road(int(raw),colors)
        rgb=value['rgb'];alpha=value['alpha'].reshape(self.image_hw)
        if rgb.shape!=(3,*self.image_hw):raise ValueError('native returned an unexpected RGB shape')
        # A custom raster autograd Function may mark every output differentiable
        # when only colors require grad. Detach the fixed RHS; qualification must
        # separately show zero opacity/scale trainables and invariant alpha.
        native_alpha_requires_grad=alpha.requires_grad
        alpha=alpha.detach()
        if raw in self.fixed_alpha:
            if not torch.equal(alpha,self.fixed_alpha[raw]):raise ValueError('fixed native alpha changed')
        else:self.fixed_alpha[raw]=alpha.detach().clone()
        return dict(rgb=rgb,alpha=alpha,source='actual_native_perGaussian_raster',
                    native_alpha_output_requires_grad=native_alpha_requires_grad)

    def assert_frozen(self):self.ctx.assert_frozen()


class NativeSparseColorRead(nn.Module):
    """Optional selected-pixel loss operator P=P0+W(c-c0).

    W must contain one column per eligible Gaussian and actual native alpha*T
    entries.  Aggregate eligible weight is insufficient.  No native CSR exporter
    is claimed by this module; real receipts remain pending until GPU qualified.
    """
    def __init__(self,W,p0,c0_eligible,pixel_ids,layout,*,receipt):
        super().__init__()
        if not W.is_sparse:raise ValueError('W must be a native contribution COO sparse tensor')
        W=W.coalesce();ids=np.asarray(pixel_ids)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids):raise ValueError('pixel IDs must be explicit unique full-image indices')
        if W.shape!=(len(ids),int(layout.eligible_active.sum())):raise ValueError('operator columns must be individual eligible Gaussians')
        if bool((W.values()<0).any())or not bool(torch.isfinite(W.values()).all()):raise ValueError('native weights must be finite nonnegative alpha*T')
        if p0.shape!=(len(ids),3)or c0_eligible.shape!=(W.shape[1],3):raise ValueError('operator color shape mismatch')
        if receipt.get('scope_sha256')!=layout.signature:raise ValueError('operator scope binding differs')
        role=receipt.get('role')
        if role not in ('synthetic_unit_test','actual_native_perGaussian_alphaT'):raise ValueError('unqualified aggregate/pixel-overlay read is forbidden')
        if role!='synthetic_unit_test':
            if receipt.get('native_qualified')is not True or receipt.get('forward_qualified')is not True or receipt.get('adjoint_qualified')is not True:
                raise ValueError('actual native forward+adjoint qualification is required')
        self.layout=layout;self.pixel_ids=ids.astype(np.int64,copy=True);self.receipt=copy.deepcopy(receipt)
        self.register_buffer('W',W);self.register_buffer('p0',p0.detach().clone());self.register_buffer('c0_eligible',c0_eligible.detach().clone())

    def forward(self,eligible_rgb):return self.p0+torch.sparse.mm(self.W,eligible_rgb-self.c0_eligible)


def gaussian_masked_hf(image,context,sigma):
    """Encoded-sRGB screen HF, same source-normalized constant-boundary Gaussian.

    Native units are pixels at the registered camera resolution.  Context/core
    are frozen input masks; caller keeps excluded/full counts independently.
    """
    if image.ndim!=3 or image.shape[0]!=3:raise ValueError('image must be CHW')
    context=torch.as_tensor(context,device=image.device,dtype=torch.bool)
    if context.shape!=image.shape[-2:]or sigma<=0:raise ValueError('invalid Gaussian context/sigma')
    radius=int(3*float(sigma)+.5)
    x=torch.arange(-radius,radius+1,dtype=image.dtype,device=image.device)
    k=torch.exp(-.5*(x/float(sigma)).square());k=k/k.sum()
    def blur(v):
        channels=v.shape[1]
        v=F.conv2d(v,k.reshape(1,1,1,-1).expand(channels,1,1,-1),padding=(0,radius),groups=channels)
        return F.conv2d(v,k.reshape(1,1,-1,1).expand(channels,1,-1,1),padding=(radius,0),groups=channels)
    mass=blur(context.to(image.dtype)[None,None])[0,0]
    clean=torch.where(context[None],image,torch.zeros_like(image))
    low=blur(clean[None])[0]/mass.clamp_min(torch.finfo(image.dtype).tiny)[None]
    return torch.where(context[None],clean-low,torch.zeros_like(image)),mass


def source_rgb_mse(pred,alpha,teacher,full_source_mask):
    """Premultiplied RGB objective; fixed alpha on RHS, complete source denominator.

    It intentionally leaves unsupported/read-only pixels in the scalar loss.
    Their lack of color gradient is reported separately, never removed as a fit.
    """
    mask=torch.as_tensor(full_source_mask,dtype=torch.bool,device=pred.device)
    if alpha.requires_grad:raise ValueError('coverage cannot receive photo gradients')
    if mask.shape!=pred.shape[-2:]or teacher.shape!=pred.shape:raise ValueError('source target shape mismatch')
    if not bool(mask.any()):raise ValueError('empty full source has no normalized objective')
    return (pred-alpha[None]*teacher).square()[:,mask].mean()


def source_stable_hf_mse(pred,alpha,teacher,context,core,sigma,*,alpha_min=.95):
    """Optional detail objective; no default weight or pass criterion is selected."""
    if alpha.requires_grad:raise ValueError('coverage cannot receive HF gradients')
    context=torch.as_tensor(context,dtype=torch.bool,device=pred.device)
    core=torch.as_tensor(core,dtype=torch.bool,device=pred.device)
    if bool((context&(alpha<alpha_min)).any())or bool((core&~context).any()):raise ValueError('HF context must be common frozen stable-alpha support')
    if not bool(core.any()):raise ValueError('empty strict core is N/A, not a zero-error success')
    straight=torch.where(context[None],pred/alpha.clamp_min(alpha_min)[None],torch.zeros_like(pred))
    hp,_=gaussian_masked_hf(straight,context,sigma);ht,_=gaussian_masked_hf(teacher,context,sigma)
    return (hp-ht).square()[:,core].mean()


def capture_full_state(model,optimizer,*,step,trace,sampler_state,identity,role):
    """Full restart state; no optimizer update and no CUDA call on CPU."""
    if len(trace)!=step:raise ValueError('step/trace mismatch')
    result=dict(schema='fixed_road_RGB_oracle_checkpoint.v1',model=model.payload(),optimizer=copy.deepcopy(optimizer.state_dict()),
        step=int(step),trace=copy.deepcopy(trace),sampler_state=copy.deepcopy(sampler_state),identity=copy.deepcopy(identity),role=role,
        torch_rng=torch.get_rng_state().clone(),numpy_rng=copy.deepcopy(np.random.get_state()),python_rng=copy.deepcopy(random.getstate()))
    if model.rgb.is_cuda:result['cuda_rng']=[x.clone()for x in torch.cuda.get_rng_state_all()]
    return result


def restore_full_state(saved,model,optimizer,*,identity,role):
    if saved.get('schema')!='fixed_road_RGB_oracle_checkpoint.v1'or saved['identity']!=identity or saved['role']!=role:raise ValueError('checkpoint identity/role differs')
    if saved['step']<0 or len(saved['trace'])!=saved['step']:raise ValueError('checkpoint cursor/trace invalid')
    payload=saved['model']
    restored=BoundedGSColors.from_payload(payload,device=model.rgb.device)
    if (restored.layout.signature!=model.layout.signature or
        not np.array_equal(restored.layout.active_world_rows,model.layout.active_world_rows)or
        not np.array_equal(restored.layout.eligible_active,model.layout.eligible_active)or
        not torch.equal(restored.c0,model.c0)):raise ValueError('checkpoint common parent/scope differs')
    with torch.no_grad():model.rgb.copy_(restored.rgb)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    for state in optimizer.state.values():
        for key,value in state.items():
            if torch.is_tensor(value)and not bool(torch.isfinite(value).all()):raise ValueError('optimizer state is nonfinite: '+key)
            if key in ('exp_avg','exp_avg_sq','max_exp_avg_sq')and value.shape!=model.rgb.shape:raise ValueError('optimizer moment topology differs')
    torch.set_rng_state(saved['torch_rng'].clone());np.random.set_state(copy.deepcopy(saved['numpy_rng']));random.setstate(copy.deepcopy(saved['python_rng']))
    if model.rgb.is_cuda:
        if 'cuda_rng'not in saved:raise ValueError('CUDA checkpoint lacks full device RNG')
        torch.cuda.set_rng_state_all([x.clone()for x in saved['cuda_rng']])
    return dict(step=saved['step'],trace=copy.deepcopy(saved['trace']),sampler_state=copy.deepcopy(saved['sampler_state']))
