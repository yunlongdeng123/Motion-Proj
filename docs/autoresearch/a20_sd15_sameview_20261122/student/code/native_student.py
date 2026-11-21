"""Real object-centric Gaussian distillation; default is a CPU-only readiness check.

Every CUDA entry needs a root approval JSON bound to this contract and asset SHA.
Geometry (40) and material (80) are separate invocations with an explicit grey gate.
There is no full-scene training loss and no automatic promotion from numeric scores.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import copy
import importlib.util
import math
import random
import sys
import time
import numpy as np
from common import BUILD, TRACK, bank, check_approval, cpu_environment, read, ref, sha, synthetic_specs, validate_contract, validate_packet, write


def preflight(args):
    c=validate_contract(read(args.contract));a=read(args.assets/'assets_receipt.json')
    for name in ('student_init','geometry_target'):
        p=args.assets/(name+'.npz')
        if a[name]['sha256']!=sha(p):raise ValueError('Asset changed: '+name)
    if a['contract']['sha256']!=sha(args.contract):raise ValueError('Assets bind a different contract')
    n=validate_packet(bank(args.assets/'student_init.npz'))
    teacher=args.assets/'teacher_actor.npz'
    ready=bool(a.get('training_ready')) and teacher.exists()
    if ready:
        if a['teacher']['output']['sha256']!=sha(teacher):raise ValueError('Teacher changed')
        validate_packet(bank(teacher),allow_opacity_boundary=True)
    return c,a,dict(schema='a20.object_centric_distillation.cpu_preflight.v1',passed=True,
        training_ready=ready,contract=ref(args.contract),assets_receipt=ref(args.assets/'assets_receipt.json'),
        student_rows=n,teacher_rows=410080 if ready else None,gpu_used=False,training_updates=0,
        teacher_ready=teacher.exists() and bool(a.get('teacher')),
        heldout_opened=[],missing_dependencies=[] if ready else ['approved opaque anchors and exact same-SLAT activated Gaussian teacher'])


def legacy_module(args):
    spec=importlib.util.spec_from_file_location('_a20_native_student_binding',args.legacy)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    if (m.TARGET,m.DELETE,m.TRACK)!=(21,10,TRACK):raise ValueError('Native identity mismatch')
    m.ROOT=args.output;m.FRAMES=BUILD;m.TRAIN=BUILD
    return m


class Runtime:
    def __init__(self,args,c,assets):
        self.args=args;self.c=c;self.assets=assets;self.m=legacy_module(args)
        m=self.m;torch=m.torch;self.torch=torch;self.F=m.F
        if not torch.cuda.is_available():raise RuntimeError('Authorized real smoke requires CUDA')
        torch.manual_seed(c['optimizer']['seed']);torch.cuda.manual_seed_all(c['optimizer']['seed'])
        np.random.seed(c['optimizer']['seed']);random.seed(c['optimizer']['seed'])
        self.scene,self.manifest,self.loader=m.load_persistent_handoff(args.road_source/'persistent/scene.json',args.road_source/'persistent/proof.json')
        for model in (self.scene.static,self.scene.dynamic,self.scene.environment):
            if hasattr(model,'parameters'):
                for p in model.parameters():p.requires_grad_(False)
        self.raster=self.scene._bindings().raster;self.pipe=self.scene.pipe
        self.synthetic_pipe=copy.deepcopy(self.pipe)
        # The new positive-fov projection must use its own matching covariance camera settings.
        self.synthetic_pipe.neg_fov=False
        self.background=torch.zeros(3,device='cuda');self.calls=0;self.backward_calls=0
        self.student=self.actor_type()(bank(args.assets/'student_init.npz')).cuda()
        self.teacher=self.actor_type()(bank(args.assets/'teacher_actor.npz'),exact_teacher_geometry=True).cuda()
        for p in self.teacher.parameters():p.requires_grad_(False)
        self.target=m.ten(bank(args.assets/'geometry_target.npz')['xyz'])
        self.target_missing=m.ten(bank(args.assets/'geometry_target.npz')['missing'],torch.bool)
        self.cameras={};self.poses={};self.depth_units={};self.real={}
        for spec in synthetic_specs(c):
            key=spec['key'];W=m.ten(spec['camera_from_actor']).T;P=m.ten(spec['projection_matrix']).T
            self.cameras[key]=SimpleNamespace(image_width=512,image_height=512,FoVx=math.radians(spec['fov_degrees']),
                FoVy=math.radians(spec['fov_degrees']),world_view_transform=W,full_proj_transform=W@P,
                camera_center=m.ten(spec['camera_center_actor_m']))
            self.poses[key]=torch.eye(4,device='cuda');self.depth_units[key]=1.
        transform=np.asarray(self.manifest.world_to_model)
        for raw in BUILD:
            key=f'build_{raw:03d}';self.cameras[key]=self.scene.camera_provider(raw)
            if (self.cameras[key].image_width,self.cameras[key].image_height)!=(1152,768):
                raise ValueError('BUILD native camera must be 1152x768')
            obs=m.read(m.BANK/'observations'/f'{raw:03d}.json');box=next(x for x in obs['boxes'] if x['track']==TRACK)
            self.poses[key]=m.ten(transform@np.asarray(box['world_from_actor']).reshape(4,4))
            self.depth_units[key]=float(np.linalg.norm(transform[:3,:3],axis=0).mean())
            original=m.Image.open(m.BANK/obs['image_path']).convert('RGB').resize((1152,768),m.Image.Resampling.BOX)
            self.real[key]={'rgb':m.ten(np.asarray(original).astype(np.float32).transpose(2,0,1)/255.)}
            folder=args.conditioning/'train'/f'{raw:03d}';mapping=read(folder/'mapping.json')
            if (mapping['raw'],mapping['native_instance_label'],mapping['track'])!=(raw,21,TRACK):raise ValueError('Real input mismatch')
            w,h=mapping['image_native_size'];x0,y0,x1,y1=mapping['crop_native_xyxy_half_open']
            for short,name in [('V','visible_mask'),('U','unknown_potential_mask'),('B','known_background_mask')]:
                full=np.ones((h,w),np.uint8)*255 if short=='B' else np.zeros((h,w),np.uint8)
                full[y0:y1,x0:x1]=np.asarray(m.Image.open(folder/(name+'.png')).convert('L'))
                self.real[key][short]=m.ten(np.asarray(m.Image.fromarray(full).resize((1152,768),m.Image.Resampling.NEAREST))>0,torch.bool)
        self.initial_protected={k:v.detach().clone() for k,v in self.student.values('eval').items()}

    def actor_type(self):
        m=self.m;torch=m.torch;F=m.F;s=self.c['student']
        class Actor(torch.nn.Module):
            def __init__(self,a,exact_teacher_geometry=False):
                super().__init__();self.metadata=copy.deepcopy(a);self.exact_teacher_geometry=exact_teacher_geometry
                for key in ('xyz','scales','rotation','coefficients','opacity'):
                    self.register_buffer('base_'+key,m.ten(a[key]))
                self.register_buffer('missing',m.ten(a['missing'],torch.bool))
                self.register_buffer('protected',m.ten(a['observed_protected'],torch.bool))
                self.register_buffer('appearance_trainable',m.ten(a['observed_protected'],torch.bool))
                self.xyz_delta=torch.nn.Parameter(torch.zeros_like(self.base_xyz))
                self.scale_delta=torch.nn.Parameter(torch.zeros_like(self.base_scales))
                self.rotation_delta=torch.nn.Parameter(torch.zeros_like(self.base_rotation))
                self.opacity_delta=torch.nn.Parameter(torch.zeros_like(self.base_opacity))
                self.dc=torch.nn.Parameter(self.base_coefficients[:,0].clone())
                self.configure('geometry')
            def configure(self,stage):
                self.stage=stage
                for name,p in self.named_parameters():p.requires_grad_(name=='dc' if stage=='material' else name!='dc')
            def values(self,role):
                missing=self.missing.float();xyz=self.base_xyz+missing[:,None]*s['center_delta_bound_m']*self.xyz_delta.tanh()
                lo,hi=s['scale_ratio_bounds']
                delta=self.scale_delta.tanh()
                # Asymmetric bounds around ratio=1 preserve zero-delta initialization.
                ratio=delta*torch.where(delta>=0,delta.new_tensor(math.log(hi)),delta.new_tensor(-math.log(lo)))
                scales=self.base_scales*(missing[:,None]*ratio).exp()
                q=F.normalize(self.base_rotation+missing[:,None]*s['rotation_delta_bound']*self.rotation_delta.tanh(),dim=1)
                opacity=(self.base_opacity+missing*s['opacity_delta_bound']*self.opacity_delta.tanh()).clamp(1e-5,1-1e-5)
                if self.exact_teacher_geometry:
                    xyz=self.base_xyz;scales=self.base_scales;q=self.base_rotation;opacity=self.base_opacity
                dc=self.dc
                if role=='teacher_loss':
                    # Native raster sees the full object; protected rows retain values but not generated gradients.
                    select=self.missing
                    xyz=torch.where(select[:,None],xyz,xyz.detach());scales=torch.where(select[:,None],scales,scales.detach())
                    q=torch.where(select[:,None],q,q.detach());opacity=torch.where(select,opacity,opacity.detach())
                    dc=torch.where(select[:,None],dc,dc.detach())
                elif role=='real_RGB':
                    xyz=xyz.detach();scales=scales.detach();q=q.detach();opacity=opacity.detach()
                    dc=torch.where(self.appearance_trainable[:,None],dc,dc.detach())
                coeff=torch.cat([dc[:,None],self.base_coefficients[:,1:]],1)
                return dict(xyz=xyz,scales=scales,rotation=q,opacity=opacity,coefficients=coeff)
            def rows(self,pose,cam,role):
                v=self.values(role);R=pose[:3,:3]/pose[:3,:3].norm(dim=0).mean();scale=pose[:3,:3].norm(dim=0).mean()
                xyz=v['xyz']@pose[:3,:3].T+pose[:3,3]
                local=F.normalize(F.normalize(xyz-cam.camera_center,dim=1)@R,dim=1)
                color=(torch.einsum('ni,nic->nc',m.basis3(local),v['coefficients'])+.5).clamp_min(0)
                qp=m.ten(m.quat_from_matrix(R.detach().cpu().numpy()[None]))[0]
                rotation=m.qmul(qp.expand_as(v['rotation']),v['rotation'])
                depth=xyz@cam.world_view_transform[:3,2]+cam.world_view_transform[3,2]
                feature=xyz.new_zeros((len(xyz),24));feature[:,0]=depth;feature[:,1]=1
                feature[:,2]=self.missing;feature[:,3]=self.protected
                return dict(means3D=xyz,means2D=torch.zeros_like(xyz),colors_precomp=color,shs=None,
                    features=feature,opacities=v['opacity'][:,None],scales=v['scales']*scale,rotations=rotation,
                    cov3D_precomp=None,mask=depth>.00377)
            def export(self):
                result=copy.deepcopy(self.metadata)
                result.update({k:m.cpu(v).numpy() for k,v in self.values('eval').items()})
                result['observed_protected']=m.cpu(self.protected).numpy();result['missing']=m.cpu(self.missing).numpy()
                appearance=m.cpu(self.appearance_trainable).numpy()
                result['observed_appearance_prior_geometry']=appearance&~result.get('is_opaque_anchor',np.zeros(len(appearance),bool))
                result['observed_geometry_measured_opaque']=result.get('is_opaque_anchor',np.zeros(len(appearance),bool))
                result['protected_depth_ambiguous']=m.cpu(self.protected).numpy()&~appearance&~result['observed_geometry_measured_opaque']
                return result
        return Actor

    def optimizer(self,stage):
        self.student.configure(stage);o=self.c['optimizer'];groups=[]
        for name,p in self.student.named_parameters():
            if p.requires_grad:groups.append(dict(params=[p],lr=o['dc_lr' if name=='dc' else name.replace('_delta','')+'_delta_lr'],name=name))
        return self.torch.optim.Adam(groups,betas=tuple(o['betas']),eps=o['eps'])

    def render(self,actor,key,role,grey=False):
        rows=actor.rows(self.poses[key],self.cameras[key],role)
        if grey:rows=dict(rows,colors_precomp=self.torch.full_like(rows['colors_precomp'],.5))
        out=self.raster(self.cameras[key],self.synthetic_pipe if key.startswith('novel') else self.pipe,0,rows,self.background)
        self.calls+=1;alpha=out['alpha'].squeeze(0)
        neutral=self.c['loss']['material_neutral_background_srgb']
        return dict(rgb=out['rgb']+(1-alpha[None])*neutral,
            rgb_unpremult_diagnostic=out['rgb']/alpha.detach().clamp_min(.95),rgb_premult=out['rgb'],alpha=alpha,
            depth=out['depth_sum'].squeeze(0)/alpha.clamp_min(.90)/self.depth_units[key],
            missing_alpha=out['extra'][0],observed_alpha=out['extra'][1])

    def cache(self):
        t=self.torch;folder=self.args.cache
        if folder.exists():raise ValueError('Teacher cache is immutable; use a fresh directory')
        folder.mkdir(parents=True,exist_ok=False);entries=[]
        with t.inference_mode():
            initial={key:self.render(self.teacher,key,'eval') for key in self.real}
            support=self.qualify_native_appearance(initial)
            for key in self.cameras:
                out=self.render(self.teacher,key,'eval');path=folder/(key+'.npz')
                np.savez_compressed(path,**{k:self.m.cpu(v).numpy() for k,v in out.items()})
                entries.append(dict(key=key,shape=list(out['alpha'].shape),packet=ref(path)))
        np.savez_compressed(folder/'native_support.npz',**support)
        write(folder/'receipt.json',dict(schema='a20.frozen_teacher_native_cache.v1',contract=ref(self.args.contract),
            teacher=ref(self.args.assets/'teacher_actor.npz'),entries=entries,actual_native_raster=True,
            assets_receipt_sha256=sha(self.args.assets/'assets_receipt.json'),
            native_support=ref(folder/'native_support.npz'),
            observed_geometry_policy='Only root-approved opaque measured anchors; never infer measured geometry from RGB V.',
            observed_appearance_policy='Actual native teacher depth/alpha consistency in a BUILD V pixel; CPU mesh-first-surface candidates additionally stay conservatively protected. Ambiguous candidates receive no generated or real-RGB updates. Mesh/GS decoder supports are never assumed identical.',
            generated_views_sampled=False,downsampling=False,heldout_opened=[],training_updates=0))

    def qualify_native_appearance(self,targets):
        t=self.torch;result={}
        def front_support(actor_xyz):
            support=t.zeros(len(actor_xyz),device='cuda',dtype=t.bool)
            for raw in BUILD:
                key=f'build_{raw:03d}';pose=self.poses[key];cam=self.cameras[key];target=targets[key]
                xyz=actor_xyz@pose[:3,:3].T+pose[:3,3];uv,depth=self.m.project(xyz,cam)
                pixel=uv.round().long();h,w=target['alpha'].shape
                inside=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)&(depth>0)
                ids=t.nonzero(inside).reshape(-1);xx=pixel[ids,0];yy=pixel[ids,1]
                native_visible=(target['alpha'][yy,xx]>=self.c['loss']['teacher_depth_alpha_minimum'])&self.real[key]['V'][yy,xx]
                consistent=(depth[ids]/self.depth_units[key]-target['depth'][yy,xx]).abs()<=self.c['loss']['native_appearance_depth_consistency_m']
                support[ids]|=native_visible&consistent
            return support
        with t.inference_mode():
            for label,actor in [('student',self.student),('teacher',self.teacher)]:
                support=front_support(actor.base_xyz)
                actor.protected.logical_or_(support);actor.missing.copy_(~actor.protected)
                actor.appearance_trainable.copy_(support)
                result[label+'_appearance_trainable']=self.m.cpu(support).numpy()
                result[label+'_protected_depth_ambiguous']=self.m.cpu(actor.protected&~support).numpy()
                result[label+'_protected']=self.m.cpu(actor.protected).numpy();result[label+'_missing']=self.m.cpu(actor.missing).numpy()
            self.target_missing.logical_and_(~front_support(self.target))
            result['target_missing']=self.m.cpu(self.target_missing).numpy()
        return result

    def load_cache(self):
        a=read(self.args.cache/'receipt.json')
        if a['contract']['sha256']!=sha(self.args.contract) or a['teacher']['sha256']!=sha(self.args.assets/'teacher_actor.npz') or a['assets_receipt_sha256']!=sha(self.args.assets/'assets_receipt.json'):
            raise ValueError('Teacher cache binding changed')
        result={}
        for item in a['entries']:
            p=self.args.cache/(item['key']+'.npz')
            if item['packet']['sha256']!=sha(p):raise ValueError('Teacher cached target changed')
            result[item['key']]={k:self.m.ten(v) for k,v in bank(p).items()}
        support=self.args.cache/'native_support.npz'
        if a['native_support']['sha256']!=sha(support):raise ValueError('Native appearance support changed')
        s=bank(support)
        for label,actor in [('student',self.student),('teacher',self.teacher)]:
            actor.appearance_trainable.copy_(self.m.ten(s[label+'_appearance_trainable'],self.torch.bool))
            actor.protected.copy_(self.m.ten(s[label+'_protected'],self.torch.bool));actor.missing.copy_(self.m.ten(s[label+'_missing'],self.torch.bool))
        self.target_missing.copy_(self.m.ten(s['target_missing'],self.torch.bool))
        return result

    def mean(self,x,mask):
        return (x*mask).sum()/mask.sum().clamp_min(1)

    def material_mask(self,target,key):
        cfg=self.c['loss'];mask=(target['alpha']>=cfg['teacher_material_alpha_minimum'])&\
            (target['missing_alpha']>cfg['teacher_positive_alpha_threshold'])&\
            (target['observed_alpha']<cfg['teacher_observed_alpha_maximum'])
        return mask&self.real[key]['U'] if key.startswith('build') else mask

    def highpass(self,rgb):
        k=rgb.new_tensor([.054488685,.244201342,.402619947,.244201342,.054488685])
        kernel=(k[:,None]*k[None,:])[None,None].expand(3,1,5,5)
        return rgb-self.F.conv2d(rgb[None],kernel,padding=2,groups=3)[0]

    def appearance_loss(self,rgb,target,mask):
        rgbL1=self.mean((rgb-target).abs().mean(0),mask)
        edge=self.m.edge(rgb,target,mask)
        # Require the entire fixed 5x5 support to have high-confidence source alpha.
        inner=self.F.avg_pool2d(mask.float()[None,None],5,1,2)[0,0]>=1-1e-6
        hf=self.mean((self.highpass(rgb)-self.highpass(target)).abs().mean(0),inner)
        return rgbL1+.1*edge+.1*hf

    def material_coverage(self,targets,tag='coverage'):
        threshold=self.c['loss']['student_material_alpha_minimum'];records=[]
        required=self.c['loss']['material_fixed_domain_required_coverage_fraction']
        with self.torch.inference_mode():
            for key,target in targets.items():
                out=self.render(self.student,key,'eval')
                domains={'teacher_fixed_high_confidence':self.material_mask(target,key)}
                if key.startswith('build'):domains['real_fixed_V']=self.real[key]['V']
                for name,mask in domains.items():
                    n=int(mask.sum());covered=(out['alpha'][mask]>=threshold);fraction=float(covered.float().mean()) if n else None
                    values=self.m.cpu(out['alpha'][mask]).numpy();holes=mask&(out['alpha']<threshold)
                    p=self.args.output/'coverage'/tag/(key+'_'+name+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(p,fixed_domain=self.m.cpu(mask).numpy(),hole_mask=self.m.cpu(holes).numpy(),
                        hole_yx=self.m.cpu(self.torch.nonzero(holes)).numpy())
                    records.append(dict(key=key,domain=name,pixels=n,student_alpha_minimum=float(values.min()) if n else None,
                        student_alpha_p01=float(np.quantile(values,.01)) if n else None,student_alpha_p05=float(np.quantile(values,.05)) if n else None,
                        fraction_student_alpha_at_least_0p95=fraction,hole_pixels=int(holes.sum()),hole_distribution=ref(p),
                        passed=fraction>=required if n else True))
        return dict(teacher_mask_alpha_minimum=.95,student_alpha_minimum=threshold,
            fixed_domain_required_coverage_fraction=required,student_mask_may_shrink_fixed_domain=False,
            passed=all(r['passed'] for r in records),records=records)

    def teacher_losses(self,out,target,key,stage):
        cfg=self.c['loss'];pos=(target['missing_alpha']>cfg['teacher_positive_alpha_threshold'])&(target['observed_alpha']<cfg['teacher_observed_alpha_maximum'])
        neg=(target['alpha']<cfg['teacher_background_alpha_maximum'])&(out['observed_alpha'].detach()<.01)
        if key.startswith('build'):
            pos=pos&self.real[key]['U'];neg=neg&self.real[key]['U']
        if stage=='material':
            return self.appearance_loss(out['rgb'],target['rgb'],self.material_mask(target,key))
        alpha=self.mean((out['alpha']-target['alpha']).abs(),pos|neg)
        depth_mask=pos&(target['alpha']>=cfg['teacher_depth_alpha_minimum'])
        depth=self.mean(self.F.huber_loss(out['depth'],target['depth'],reduction='none',delta=cfg['depth_huber_delta_m']),depth_mask)
        return alpha+depth

    def chamfer(self):
        t=self.torch;x=self.student.values('teacher_loss')['xyz'][self.student.missing]
        stride=max(1,math.ceil(len(x)/self.c['loss']['geometry_student_subsample_rows']));x=x[::stride]
        y=self.target[self.target_missing];chunk=self.c['loss']['geometry_knn_chunk_rows']
        xy=t.cat([t.cdist(a[None],y[None])[0].min(1).values for a in x.split(chunk)])
        yx=t.cat([t.cdist(a[None],x[None])[0].min(1).values for a in y.split(chunk)])
        return (xy.mean()+yx.mean())/2

    def loss(self,stage,step,targets,audit=False):
        t=self.torch;s=self.student;novel=f'novel_{step%6:02d}';build=f'build_{BUILD[step%2]:03d}'
        a=self.render(s,novel,'teacher_loss');b=self.render(s,build,'teacher_loss')
        generated=self.teacher_losses(a,targets[novel],novel,stage)+self.teacher_losses(b,targets[build],build,stage)
        logs={};isolations={}
        if stage=='geometry':
            B=self.real[build]['B'];V=self.real[build]['V']
            realV=self.mean((b['alpha']-1).square(),V);realB=self.mean(b['alpha'].square(),B)
            # V is a preservation constraint. Missing support cannot be rewarded for covering real V.
            fixedV=realV.detach();reg=sum(p.square().mean() for name,p in s.named_parameters() if name!='dc')
            loss=generated+.2*self.chamfer()+.2*fixedV+.2*realB+.001*reg
            if audit:
                gs=t.autograd.grad(generated,[s.xyz_delta,s.scale_delta],retain_graph=True,allow_unused=True)
                for name,g in zip(('native_raster_xyz_gradient','native_raster_scale_gradient'),gs):
                    maximum=0. if g is None else float(g.abs().max());isolations[name]=maximum
                    if not math.isfinite(maximum) or maximum<=0:raise RuntimeError('Native geometry raster gradient disconnected: '+name)
                for name,p in s.named_parameters():
                    if p.requires_grad:
                        g=t.autograd.grad(generated,p,retain_graph=True,allow_unused=True)[0]
                        mx=0. if g is None else float(g[s.protected].abs().max())
                        if mx!=0:raise RuntimeError('Generated geometry gradient reached observed support')
                isolations['generated_to_observed_max']=0.
            logs.update(real_V_alpha=float(realV.detach()),real_B_alpha=float(realB.detach()))
        else:
            real=self.render(s,build,'real_RGB');V=self.real[build]['V']
            realRGB=self.appearance_loss(real['rgb'],self.real[build]['rgb'],V);loss=generated+realRGB
            if audit:
                gg=t.autograd.grad(generated,s.dc,retain_graph=True)[0]
                rg=t.autograd.grad(realRGB,s.dc,retain_graph=True)[0]
                if float(gg[s.protected].abs().max())!=0 or float(rg[~s.appearance_trainable].abs().max())!=0:
                    raise RuntimeError('Generated/real DC gradient domains overlap')
                isolations=dict(generated_to_observed_max=0.,real_RGB_to_missing_max=0.)
            logs['real_RGB']=float(realRGB.detach())
        if not bool(t.isfinite(loss)):raise FloatingPointError('Nonfinite actual raster loss')
        logs.update(loss=float(loss.detach()),generated=float(generated.detach()),novel=novel,build=build,**isolations)
        return loss,logs

    def step(self,stage,step,opt,targets,audit=False):
        self.student.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
        loss,logs=self.loss(stage,step,targets,audit);loss.backward();self.backward_calls+=1
        for name,p in self.student.named_parameters():
            if p.requires_grad and (p.grad is None or not bool(self.torch.isfinite(p.grad).all())):
                raise FloatingPointError('Missing/nonfinite active gradient '+name)
        opt.step();self.prove_protected()
        return logs

    def prove_protected(self):
        p=self.student.protected;now=self.student.values('eval')
        for k in ('xyz','scales','rotation','opacity'):
            if not self.torch.equal(now[k][p],self.initial_protected[k][p]):
                raise RuntimeError('Protected observation geometry changed: '+k)

    def capture(self,opt,stage,completed,role):
        t=self.torch
        return self.m.cpu(dict(schema='a20.complete_distill_checkpoint.v1',contract_sha256=sha(self.args.contract),
            assets_sha256=sha(self.args.assets/'assets_receipt.json'),model=self.student.state_dict(),optimizer=opt.state_dict(),
            stage=stage,completed=completed,role=role,
            geometry_checkpoint_sha256=sha(self.args.geometry_checkpoint) if self.args.geometry_checkpoint else None,
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=t.get_rng_state(),cuda=t.cuda.get_rng_state_all())))

    def restore(self,state,opt,stage):
        if state['contract_sha256']!=sha(self.args.contract) or state['assets_sha256']!=sha(self.args.assets/'assets_receipt.json'):
            raise ValueError('Checkpoint contract/asset mismatch')
        if state['stage']!=stage:raise ValueError('Optimizer stage mismatch')
        if stage=='material' and state.get('geometry_checkpoint_sha256')!=sha(self.args.geometry_checkpoint):
            raise ValueError('Material checkpoint belongs to another approved geometry')
        self.student.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
        self.torch.set_rng_state(state['rng']['torch']);self.torch.cuda.set_rng_state_all(state['rng']['cuda'])
        self.prove_protected()

    def evaluate(self,targets):
        result=[]
        with self.torch.inference_mode():
            for key in self.cameras:
                out=self.render(self.student,key,'eval');tar=targets[key]
                pos=(tar['missing_alpha']>.05)&(tar['alpha']>=.90)
                result.append(dict(key=key,alpha_L1=float((out['alpha']-tar['alpha']).abs().mean()),
                    missing_depth_huber_m=float(self.mean(self.F.huber_loss(out['depth'],tar['depth'],reduction='none',delta=.05),pos))))
                mask=self.material_mask(tar,key);item=result[-1]
                coverage=float((out['alpha'][mask]>=.95).float().mean())>=.99 if mask.any() else False
                item['fixed_high_confidence_pixels']=int(mask.sum());item['fixed_high_confidence_student_coverage_passed']=coverage
                inner=self.F.avg_pool2d(mask.float()[None,None],5,1,2)[0,0]>=1-1e-6
                item['masked_edge_L1']=float(self.m.edge(out['rgb'],tar['rgb'],mask))
                item['high_frequency']=self.m.hf(out['rgb'],tar['rgb'],inner)
                item['detail_color_definition']='premultiplied over fixed neutral 0.5; entire fixed source mask including holes'
                diagnostic=mask&(out['alpha']>=.95)
                item['unpremult_color_diagnostic_only']=dict(pixels=int(diagnostic.sum()),
                    L1=float(self.mean((out['rgb_unpremult_diagnostic']-tar['rgb_unpremult_diagnostic']).abs().mean(0),diagnostic)))
                student=out['alpha']>.5;teacher=tar['alpha']>.5
                item['silhouette_IoU']=float((student&teacher).sum()/(student|teacher).sum().clamp_min(1))
                def dilate(a,radius):return self.F.max_pool2d(a.float()[None,None],2*radius+1,1,radius)[0,0]>0
                sb=student&dilate(~student,1);tb=teacher&dilate(~teacher,1)
                precision=float((sb&dilate(tb,2)).sum()/sb.sum().clamp_min(1));recall=float((tb&dilate(sb,2)).sum()/tb.sum().clamp_min(1))
                item['boundary_2px']=dict(precision=precision,recall=recall,F1=2*precision*recall/max(precision+recall,1e-12))
                folder=self.args.output/'eval'/key
                self.m.png(folder/'student_rgb_premult.png',out['rgb_premult']);self.m.png(folder/'teacher_rgb_premult.png',tar['rgb_premult'])
                self.m.png(folder/'student_alpha.png',out['alpha']);self.m.png(folder/'teacher_alpha.png',tar['alpha'])
                grey=self.render(self.student,key,'eval',grey=True);self.m.png(folder/'student_grey.png',grey['rgb_premult'])
        return result


def tree_maxdiff(a,b,t):
    if t.is_tensor(a):
        if a.dtype==t.bool:return 0. if t.equal(a,b) else float('inf')
        return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):return max([tree_maxdiff(a[k],b[k],t) for k in a] or [0.])
    if isinstance(a,(tuple,list)):return max([tree_maxdiff(x,y,t) for x,y in zip(a,b)] or [0.])
    return 0. if a==b else float('inf')


def run(args,c,assets):
    for key in ('legacy','road_source','conditioning','cache'):
        if getattr(args,key) is None:raise ValueError('Native entry requires --'+key.replace('_','-'))
    if not args.authorize_gpu or not args.approval:raise ValueError('Explicit root-bound GPU approval is required')
    bindings=dict(assets_sha256=sha(args.assets/'assets_receipt.json'),trainer_sha256=sha(__file__))
    if args.phase=='material':
        if not args.geometry_checkpoint:raise ValueError('Material entry requires the approved 40-step geometry checkpoint')
        bindings['geometry_checkpoint_sha256']=sha(args.geometry_checkpoint)
    approval=check_approval(args.approval,args.contract,args.phase,bindings)
    if approval.get('opaque_anchors_approved') is not True and args.phase!='cache':
        raise ValueError('Root has not approved the selected opaque anchors bound to this asset receipt')
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'entry.json',dict(contract=ref(args.contract),approval=ref(args.approval),phase=args.phase,
        heldout_opened=[],training_updates=0))
    runtime=Runtime(args,c,assets);t=runtime.torch;t.cuda.reset_peak_memory_stats();started=time.monotonic()
    if args.phase=='cache':runtime.cache();return
    targets=runtime.load_cache();stage='material' if args.phase=='material' else 'geometry'
    opt=runtime.optimizer(stage);start=0;role='smoke_only' if args.phase=='smoke' else 'formal'
    restore_path=args.resume or (args.geometry_checkpoint if stage=='material' else None)
    if restore_path:
        state=t.load(restore_path,map_location='cpu',weights_only=False)
        if state['role']!='formal':raise ValueError('Smoke artifacts can never initialize formal training')
        if state['contract_sha256']!=sha(args.contract) or state['assets_sha256']!=sha(args.assets/'assets_receipt.json'):
            raise ValueError('Cross-stage checkpoint contract/asset mismatch')
        if stage=='material' and state['stage']=='geometry':
            if state['completed']!=40:raise ValueError('Material entry needs completed 40-step geometry')
            runtime.student.load_state_dict(state['model'],strict=True)
            random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
            t.set_rng_state(state['rng']['torch']);t.cuda.set_rng_state_all(state['rng']['cuda'])
        else:runtime.restore(state,opt,stage);start=state['completed']
    if stage=='material':
        coverage=runtime.material_coverage(targets,tag='material_entry');write(args.output/'material_entry_coverage.json',coverage)
        if not coverage['passed']:raise RuntimeError('Fixed teacher/real-V domain has insufficient student coverage; color entry is blocked')
    count=20 if args.phase=='smoke' else c['stages'][stage]['formal_steps'];records=[];restore_proof=None
    for step in range(start,count):
        logs=runtime.step(stage,step,opt,targets,audit=step==start);records.append(dict(step=step+1,**logs))
        if args.phase=='smoke' and step+1==10:
            saved=runtime.capture(opt,stage,10,role);path=args.output/'checkpoint_step010.pt';runtime.m.save(path,saved)
            random_probe=lambda:[random.random(),float(np.random.rand()),float(t.rand(())),float(t.rand((),device='cuda'))]
            first_rng_probe=random_probe()
            first=runtime.step(stage,10,opt,targets,audit=True);after=runtime.capture(opt,stage,11,role)
            runtime.restore(t.load(path,map_location='cpu',weights_only=False),opt,stage)
            replay_rng_probe=random_probe()
            replay=runtime.step(stage,10,opt,targets,audit=True);replayed=runtime.capture(opt,stage,11,role)
            difference=max(tree_maxdiff(after['model'],replayed['model'],t),tree_maxdiff(after['optimizer'],replayed['optimizer'],t))
            if first_rng_probe!=replay_rng_probe or difference>1e-5 or abs(first['loss']-replay['loss'])>1e-5:
                raise RuntimeError('Complete checkpoint replay diverged')
            restore_proof=dict(checkpoint=ref(path),full_model_optimizer_RNG_loaded=True,replayed_next_step=11,
                maximum_model_optimizer_difference=difference,loss_difference=abs(first['loss']-replay['loss']),
                python_numpy_torch_CPU_CUDA_rng_probes_exact=True,replay_extra_optimizer_steps=2)
            # Resume the original budget at completed=10; both replay probes are separate smoke overhead.
            runtime.restore(saved,opt,stage)
        if (step+1)%10==0:
            runtime.m.save(args.output/f'checkpoint_step{step+1:03d}.pt',runtime.capture(opt,stage,step+1,role))
            write(args.output/'progress.json',dict(phase=args.phase,completed=step+1,budget=count,records=records))
    final=runtime.capture(opt,stage,count,role);checkpoint=args.output/'checkpoint_final.pt';runtime.m.save(checkpoint,final)
    probe=None
    if args.phase=='smoke':
        # A disposable same-support asset probes the next-stage interface at genuinely high coverage.
        # It is never the formal independently resampled student and supplies no distillation-quality evidence.
        actual_student=runtime.student;actual_protected=runtime.initial_protected
        original_V={k:v['V'] for k,v in runtime.real.items()}
        runtime.student=copy.deepcopy(runtime.teacher)
        runtime.student.dc.data.add_(.01)
        runtime.initial_protected={k:v.detach().clone() for k,v in runtime.student.values('eval').items()}
        for key in runtime.real:
            runtime.real[key]['V']=original_V[key]&(targets[key]['alpha']>=.95)
        probe_opt=runtime.optimizer('material');probe_coverage=runtime.material_coverage(targets,tag='temporary_smoke_probe')
        if not probe_coverage['passed']:raise RuntimeError('Temporary smoke material probe did not reach fixed-domain coverage')
        probe=runtime.step('material',0,probe_opt,targets,audit=True)
        # The probe validates next-stage wiring only; its state is destroyed before export.
        runtime.student=actual_student;runtime.initial_protected=actual_protected
        for key in runtime.real:runtime.real[key]['V']=original_V[key]
        runtime.student.configure('geometry');runtime.restore(final,opt,'geometry')
        probe=dict(temporary_material_VJP_and_step_passed=True,discarded=True,formal_material_gate_created=False,
            geometry_support='temporary teacher-support copy for interface testing only',formal_student_used=False,
            quality_improvement_claim=False,coverage=probe_coverage,metrics=probe)
    exported=runtime.student.export();asset=args.output/'actor.npz';np.savez_compressed(asset,**exported)
    reloaded=bank(asset);validate_packet(reloaded)
    if any(not np.array_equal(reloaded[k],exported[k]) for k in ('xyz','scales','rotation','coefficients','opacity')):
        raise RuntimeError('Actor export reload mismatch')
    metrics=runtime.evaluate(targets);coverage=runtime.material_coverage(targets,tag='final');t.cuda.synchronize()
    receipt=dict(schema='a20.object_centric_distillation.execution.v1',phase=args.phase,role=role,completed=count,
        exact_formal_stage_budget=c['stages'][stage]['formal_steps'],contract=ref(args.contract),checkpoint=ref(checkpoint),
        actor=ref(asset),records=records,metrics=metrics,material_fixed_domain_coverage=coverage,restore_proof=restore_proof,next_stage_probe=probe,
        native_renderer_calls=runtime.calls,backward_calls=runtime.backward_calls,
        heldout_opened=[],full_scene_loss=False,frozen_road_and_context=True,protected_support_geometry_exact=True,
        observed_geometry_means_only_approved_opaque_anchors=True,
        geometry_gate_passed=False,material_automatically_allowed=False,
        state='awaiting_explicit_geometry_grey_review' if stage=='geometry' and role=='formal' else 'completed_unpromoted',
        wall_seconds=time.monotonic()-started,cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=t.cuda.max_memory_reserved())
    write(args.output/'receipt.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['cpu-check','cache','smoke','geometry','material'],default='cpu-check')
    for k in ('contract','assets','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('legacy','road-source','conditioning','cache','approval','resume','geometry-checkpoint'):p.add_argument('--'+k,type=Path)
    p.add_argument('--authorize-gpu',action='store_true');args=p.parse_args()
    if args.phase=='cpu-check':cpu_environment()
    c,assets,report=preflight(args)
    if args.phase=='cpu-check':write(args.output/'cpu_preflight.json',report);print(__import__('json').dumps(report,indent=2))
    else:
        if not report['training_ready'] and not (args.phase=='cache' and report['teacher_ready']):
            raise ValueError('CPU assets are not yet training-ready')
        run(args,c,assets)
