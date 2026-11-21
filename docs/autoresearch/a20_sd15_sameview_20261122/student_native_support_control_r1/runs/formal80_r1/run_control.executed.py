"""Native support + real appearance control. Formal80 needs a separate root gate."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import random
import sys
import time
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def bootstrap(args):
    c=read(args.contract);b=Path(c['baseline']['root'])
    for name,path in [('contract',b/'contract.json'),('assets_receipt',b/'assets_object_clean_color_r1/assets_receipt.json'),
                      ('runtime',b/'code/native_student.py'),('teacher_packet',b/'assets_object_clean_color_r1/teacher_actor.npz')]:
        if sha(path)!=c['baseline'][name+'_sha256']:raise ValueError('Sealed baseline changed: '+name)
    sys.path.insert(0,str(b/'code'));from native_student import Runtime
    a=copy.copy(args);a.assets=b/'assets_object_clean_color_r1';a.conditioning=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1/conditioning')
    a.legacy=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1/code/frozen_formal_r2/run_joint.py')
    a.road_source=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/appearance_fit/asset')
    a.geometry_checkpoint=None;a.cache=args.assets/'teacher_cache'
    r=Runtime(a,read(b/'contract.json'),read(b/'assets_object_clean_color_r1/assets_receipt.json'))
    return c,b,r


def control_actor(r,a):
    t=r.torch;m=r.m;Base=r.actor_type()
    class Actor(Base):
        def __init__(self,a):
            super().__init__(a,exact_teacher_geometry=True)
            for name in ('dc','xyz_delta','scale_delta','rotation_delta','opacity_delta'):delattr(self,name)
            ids=np.flatnonzero(a['observed_appearance_prior_geometry'])
            self.register_buffer('observed_ids',m.ten(ids,t.long));self.register_buffer('source_coefficients',r.teacher.base_coefficients.detach().clone())
            self.observed_dc=t.nn.Parameter(self.base_coefficients[self.observed_ids,0].detach().clone())
            self.appearance_trainable.copy_(m.ten(a['observed_appearance_prior_geometry'],t.bool))
        def configure(self,stage):
            for name,p in self.named_parameters():p.requires_grad_(name=='observed_dc')
        def values(self,role):
            dc=self.base_coefficients[:,0].index_copy(0,self.observed_ids,self.observed_dc)
            coeff=t.cat([dc[:,None],self.base_coefficients[:,1:]],1)
            return dict(xyz=self.base_xyz,scales=self.base_scales,rotation=self.base_rotation,opacity=self.base_opacity,coefficients=coeff)
        def export(self):
            out=copy.deepcopy(self.metadata);out.update({k:m.cpu(v).numpy() for k,v in self.values('eval').items()});return out
    return Actor(a).cuda()


def prove_frozen(r):
    s=r.student;t=r.torch;source=r.teacher
    for key in ('xyz','scales','rotation','opacity'):
        if not t.equal(getattr(s,'base_'+key),getattr(source,'base_'+key)):raise RuntimeError('Native geometry changed: '+key)
    unknown=~s.appearance_trainable
    coeff=s.values('eval')['coefficients']
    if not t.equal(coeff[unknown],source.base_coefficients[unknown]):raise RuntimeError('Unknown appearance changed')
    if not t.equal(coeff[:,1:],source.base_coefficients[:,1:]):raise RuntimeError('Higher SH changed')


def prepare(args,c,b,r):
    t=r.torch;m=r.m;args.assets.mkdir(parents=True,exist_ok=False)
    source=bank(b/'assets_object_clean_color_r1/teacher_actor.npz');n=len(source['xyz'])
    color=t.zeros((n,3),device='cuda');count=t.zeros(n,device='cuda');pixels=[];valids=[];rgbs=[];rgb_refs=[]
    for raw in (86,110):
        key=f'build_{raw:03d}';target={k:m.ten(v) for k,v in bank(b/'teacher_cache_object_r1'/(key+'.npz')).items()}
        pose=r.poses[key];cam=r.cameras[key];xyz=r.teacher.base_xyz@pose[:3,:3].T+pose[:3,3]
        uv,depth=m.project(xyz,cam);p=uv.round().long();h,w=target['alpha'].shape
        inside=(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)&(depth>0)
        ids=t.nonzero(inside).reshape(-1);xx=p[ids,0];yy=p[ids,1]
        good=(target['alpha'][yy,xx]>=.90)&r.real[key]['V'][yy,xx]&((depth[ids]/r.depth_units[key]-target['depth'][yy,xx]).abs()<=.075)
        selected=ids[good];rgb=t.zeros((n,3),device='cuda');rgb[selected]=r.real[key]['rgb'][:,p[selected,1],p[selected,0]].T
        valid=t.zeros(n,device='cuda',dtype=t.bool);valid[selected]=True;color+=rgb;count+=valid
        pixels.append(m.cpu(p).numpy());valids.append(m.cpu(valid).numpy());rgbs.append(m.cpu(rgb).numpy())
        obs=m.read(m.BANK/'observations'/f'{raw:03d}.json');rgb_refs.append(dict(raw=raw,observation=ref(m.BANK/'observations'/f'{raw:03d}.json'),
            original_RGB=ref(m.BANK/obs['image_path']),V=ref(r.args.conditioning/'train'/f'{raw:03d}'/'visible_mask.png')))
    observed=count>0;ids=t.nonzero(observed).reshape(-1);source['coefficients'][m.cpu(observed).numpy(),0]=m.cpu((color[observed]/count[observed,None]-.5)/.28209479177387814).numpy()
    source['observed_appearance_prior_geometry']=m.cpu(observed).numpy();source['observed_protected']=source['observed_appearance_prior_geometry'].copy()
    source['missing']=~source['observed_protected'];source['is_opaque_anchor']=np.zeros(n,bool)
    np.savez_compressed(args.assets/'control_init.npz',**source)
    oi=m.cpu(ids).numpy();np.savez_compressed(args.assets/'real_RGB_source_trace.npz',observed_row_ids=oi,
        source_raw=np.array([86,110]),native_pixel_xy=np.stack(pixels,1)[oi],valid_by_view=np.stack(valids,1)[oi],
        original_RGB_at_pixel=np.stack(rgbs,1)[oi])
    r.student=control_actor(r,source);r.teacher.protected.copy_(observed);r.teacher.missing.copy_(~observed);r.teacher.appearance_trainable.copy_(observed)
    cache=args.assets/'teacher_cache';cache.mkdir(exist_ok=False);entries=[]
    with t.inference_mode():
        for key in r.cameras:
            out=r.render(r.teacher,key,'eval');p=cache/(key+'.npz');np.savez_compressed(p,**{k:m.cpu(v).numpy() for k,v in out.items()});entries.append(dict(key=key,packet=ref(p)))
    prove_frozen(r);equality=geometry_equality(r)
    (args.assets/'run_control.prepared.py').write_bytes(Path(__file__).read_bytes())
    receipt=dict(schema='a20.native_support_control.assets.v1',contract=ref(args.contract),source_teacher=ref(b/'assets_object_clean_color_r1/teacher_actor.npz'),
        control_init=ref(args.assets/'control_init.npz'),source_trace=ref(args.assets/'real_RGB_source_trace.npz'),source_RGB=rgb_refs,
        native_rows=n,observed_DC_rows=len(oi),unknown_DC_rows=n-len(oi),extra_measurement_gaussians=0,
        geometry_bitwise_preserved=True,unknown_DC_bitwise_preserved=True,independent_compression_or_full_distillation=False,
        geometry_render_equality=equality,teacher_cache=entries,training_updates=0,heldout_opened=[],
        runtime_source=ref(args.assets/'run_control.prepared.py'))
    write(args.assets/'assets_receipt.json',receipt)


def targets(args,r):
    receipt=read(args.assets/'assets_receipt.json')
    if receipt['contract']['sha256']!=sha(args.contract):raise ValueError('Control asset contract mismatch')
    if receipt['control_init']['sha256']!=sha(args.assets/'control_init.npz'):raise ValueError('Control init changed')
    result={}
    for entry in receipt['teacher_cache']:
        p=args.assets/'teacher_cache'/(entry['key']+'.npz')
        if sha(p)!=entry['packet']['sha256']:raise ValueError('Control teacher cache changed')
        result[entry['key']]={k:r.m.ten(v) for k,v in bank(p).items()}
    r.student=control_actor(r,bank(args.assets/'control_init.npz'))
    r.teacher.protected.copy_(r.student.appearance_trainable);r.teacher.missing.copy_(~r.teacher.protected)
    return result


def geometry_equality(r):
    result=[]
    with r.torch.inference_mode():
        for key in r.cameras:
            a=r.render(r.student,key,'eval',grey=True);b=r.render(r.teacher,key,'eval',grey=True)
            result.append(dict(key=key,alpha_bitwise_equal=bool(r.torch.equal(a['alpha'],b['alpha'])),grey_bitwise_equal=bool(r.torch.equal(a['rgb_premult'],b['rgb_premult'])),
                alpha_max_difference=float((a['alpha']-b['alpha']).abs().max()),grey_max_difference=float((a['rgb_premult']-b['rgb_premult']).abs().max())))
    if not all(x['alpha_bitwise_equal'] and x['grey_bitwise_equal'] for x in result):raise RuntimeError('Native-support alpha/grey changed')
    return result


def capture(args,r,opt,completed,role):
    t=r.torch
    return r.m.cpu(dict(schema='a20.native_control.complete_mutable_state.v1',contract_sha256=sha(args.contract),assets_sha256=sha(args.assets/'assets_receipt.json'),
        immutable_full_geometry_and_unknown_DC_sha256=sha(args.assets/'control_init.npz'),observed_dc=r.student.observed_dc,
        optimizer=opt.state_dict(),completed=completed,role=role,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=t.get_rng_state(),cuda=t.cuda.get_rng_state_all())))


def restore(args,r,opt,state):
    if state['contract_sha256']!=sha(args.contract) or state['assets_sha256']!=sha(args.assets/'assets_receipt.json') or state['immutable_full_geometry_and_unknown_DC_sha256']!=sha(args.assets/'control_init.npz'):
        raise ValueError('Full state immutable source binding mismatch')
    r.student.observed_dc.data.copy_(state['observed_dc'].to('cuda'));opt.load_state_dict(state['optimizer'])
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);r.torch.set_rng_state(state['rng']['torch']);r.torch.cuda.set_rng_state_all(state['rng']['cuda']);prove_frozen(r)


def step(r,opt,k):
    key=f'build_{(86,110)[k%2]:03d}';opt.zero_grad(set_to_none=True);out=r.render(r.student,key,'real_RGB');V=r.real[key]['V']
    loss=r.appearance_loss(out['rgb'],r.real[key]['rgb'],V)
    if not bool(r.torch.isfinite(loss)):raise FloatingPointError('Nonfinite native-control loss')
    loss.backward();g=r.student.observed_dc.grad
    if g is None or not bool(r.torch.isfinite(g).all()) or float(g.abs().max())<=0:raise RuntimeError('Observed DC gradient disconnected/nonfinite')
    maximum=float(g.abs().max());opt.step();prove_frozen(r)
    return dict(step=k+1,build=key,loss=float(loss.detach()),observed_DC_gradient_maximum=maximum)


def evaluate(r,tar):
    prior=r.evaluate(tar);real=[]
    with r.torch.inference_mode():
        for key in r.real:
            out=r.render(r.student,key,'eval');V=r.real[key]['V'];gt=r.real[key]['rgb'];inner=r.F.avg_pool2d(V.float()[None,None],5,1,2)[0,0]>=1-1e-6
            real.append(dict(key=key,RGB_L1=float(r.mean((out['rgb']-gt).abs().mean(0),V)),edge_L1=float(r.m.edge(out['rgb'],gt,V)),HF=r.m.hf(out['rgb'],gt,inner)))
    return dict(prior_unknown_monitor_only=prior,real_V=real)


def train(args,c,r,tar):
    t=r.torch;from native_student import tree_maxdiff
    formal=args.phase=='formal';role='formal80' if formal else 'smoke5_only';budget=80 if formal else 5
    coverage=r.material_coverage(tar,tag='entry');write(args.output/'entry_coverage.json',coverage)
    if formal and not coverage['passed']:raise RuntimeError('Control source high-coverage gate failed; formal80 closed')
    opt=t.optim.Adam([r.student.observed_dc],lr=.01,betas=(.9,.999),eps=1e-8)
    records=[];proof=None;start=0;started=time.monotonic();t.cuda.reset_peak_memory_stats()
    if args.resume:
        if not formal:raise ValueError('Smoke is always from the clean control initialization')
        saved=t.load(args.resume,map_location='cpu',weights_only=False)
        if saved['role']!='formal80':raise ValueError('Formal control must never resume a smoke state')
        restore(args,r,opt,saved);start=saved['completed']
    for k in range(start,budget):
        records.append(step(r,opt,k))
        if not formal and k==1:
            saved=capture(args,r,opt,2,role);p=args.output/'checkpoint_replay_source_step002.pt';r.m.save(p,saved)
            probe=lambda:[random.random(),float(np.random.rand()),float(t.rand(())),float(t.rand((),device='cuda'))]
            ra=probe();a=step(r,opt,2);sa=capture(args,r,opt,3,role)
            restore(args,r,opt,t.load(p,map_location='cpu',weights_only=False));rb=probe();b=step(r,opt,2);sb=capture(args,r,opt,3,role)
            difference=max(tree_maxdiff(sa['observed_dc'],sb['observed_dc'],t),tree_maxdiff(sa['optimizer'],sb['optimizer'],t))
            if ra!=rb or difference>1e-5 or abs(a['loss']-b['loss'])>1e-5:raise RuntimeError('Full control state replay failed')
            proof=dict(checkpoint=ref(p),all_mutable_model_optimizer_and_four_RNG_restored=True,immutable_full_model_hash_verified=True,
                RNG_probe_exact=True,maximum_model_optimizer_difference=difference,loss_difference=abs(a['loss']-b['loss']),extra_optimizer_steps=2)
            restore(args,r,opt,saved)
        if formal and (k+1)%20==0:r.m.save(args.output/f'checkpoint_step{k+1:03d}.pt',capture(args,r,opt,k+1,role))
    r.m.save(args.output/'checkpoint_final.pt',capture(args,r,opt,budget,role));prove_frozen(r)
    np.savez_compressed(args.output/'actor.npz',**r.student.export());equality=geometry_equality(r);metrics=evaluate(r,tar);final_coverage=r.material_coverage(tar,tag='final');t.cuda.synchronize()
    receipt=dict(schema='a20.native_support_appearance_control.execution.v1',phase=args.phase,role=role,steps=budget,records=records,
        contract=ref(args.contract),runtime=ref(__file__),assets=ref(args.assets/'assets_receipt.json'),actor=ref(args.output/'actor.npz'),checkpoint=ref(args.output/'checkpoint_final.pt'),
        geometry_bitwise_preserved=True,unknown_DC_bitwise_preserved=True,geometry_render_equality=equality,restore_proof=proof,
        metrics=metrics,coverage=final_coverage,compression_or_full_distillation=False,teacher_unknown_is_loss=False,
        formal80_auto_allowed=False,heldout_opened=[],native_calls=r.calls,optimizer_steps_including_replay=budget+(2 if not formal else 0),
        wall_seconds_excluding_load=time.monotonic()-started,cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=t.cuda.max_memory_reserved())
    (args.output/'run_control.executed.py').write_bytes(Path(__file__).read_bytes());write(args.output/'receipt.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['prepare','smoke','formal'],required=True)
    for k in ('contract','assets','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--approval',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--authorize-control-smoke',action='store_true');args=p.parse_args()
    if args.phase!='formal' and not args.authorize_control_smoke:raise ValueError('Root authorization is required for prepare/smoke')
    if args.phase=='formal':
        approval=read(args.approval)
        if approval.get('approved') is not True or approval.get('contract_sha256')!=sha(args.contract) or approval.get('runtime_sha256')!=sha(__file__) or approval.get('assets_sha256')!=sha(args.assets/'assets_receipt.json'):
            raise ValueError('Root formal80 approval binding missing')
        smoke=args.contract.parent/'runs/smoke_r1/receipt.json';s=read(smoke)
        if approval.get('smoke_receipt_sha256')!=sha(smoke) or s['steps']!=5 or not s['restore_proof']['all_mutable_model_optimizer_and_four_RNG_restored'] or not s['geometry_bitwise_preserved'] or not s['unknown_DC_bitwise_preserved']:
            raise ValueError('Root formal80 approval must bind the completed five-step smoke')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();c,b,r=bootstrap(args)
    if args.phase=='prepare':prepare(args,c,b,r)
    else:tar=targets(args,r);train(args,c,r,tar)
