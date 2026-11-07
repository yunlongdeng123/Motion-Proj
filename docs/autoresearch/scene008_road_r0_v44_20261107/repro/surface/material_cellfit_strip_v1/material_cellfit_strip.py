"""R0 fixed-world-strip per-cell material control; matched-field initialization.

The source scene and all geometry/opacity are detached. Only eligible persistent
new-road SH0/sRGB cells change. This is observed appearance fitting, not hidden
texture ground truth or learned-road-prior success.
"""
import os
for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[_k]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=CAM/'scene008_road_r0_v44_20261107_r1'
V2_SHA='6112be03108b5a9b1a6deb2f605ef850dcd8ba5018346fa822993177815807a2'
INIT_SHA='c13108b3fa58d945aed8a228616e48beba45cf00615b7f34f7fd071bd78f4c7f'
HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(x for x in range(89,138) if x not in HELD)
EVAL=(90,91,92,105,113,130,137)
SEED=20261107
LR=.01
ANCHOR=.01
TEXTURE=.001
SOURCE_DELTA_BOUND=.15
INIT_NORMALIZED_EPS=1e-7


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(4194304),b''):h.update(x)
    return h.hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def tree_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if hasattr(x,'detach'):
            visit(x.detach().cpu().contiguous().numpy())
        elif isinstance(x,np.ndarray):
            h.update(str((x.shape,x.dtype.str)).encode());h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=repr):visit(k);visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def cpu_tree(x):
    if hasattr(x,'detach'):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    return copy.deepcopy(x)


class Sampler:
    def __init__(self):
        self.generator=np.random.default_rng(SEED);self.order=self.generator.permutation(BUILD)
        self.cursor=0;self.epoch=0
    def next(self):
        if self.cursor==40:
            self.order=self.generator.permutation(BUILD);self.cursor=0;self.epoch+=1
        raw=int(self.order[self.cursor]);self.cursor+=1;return raw
    def state(self):
        return dict(generator=copy.deepcopy(self.generator.bit_generator.state),order=self.order.copy(),cursor=self.cursor,epoch=self.epoch)
    def restore(self,s):
        self.generator.bit_generator.state=s['generator'];self.order=s['order'].copy();self.cursor=s['cursor'];self.epoch=s['epoch']
    def peek(self,n=2):
        s=self.state();values=[self.next() for _ in range(n)];self.restore(s);return values


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'contract.json').exists():raise FileExistsError('Preserve existing cellfit contract')
    assert sha(args.asset)==V2_SHA and sha(args.initial_material)==INIT_SHA
    a=load(args.asset);init=load(args.initial_material);ids=init['extension_ids']
    assert np.array_equal(ids,np.flatnonzero(a['active'].astype(bool)&(a['local_xyz'][:,0]>19.75)))
    c=init['colours_srgb'];source=a['reference_srgb'][ids]
    assert np.isfinite(c).all() and c.min()>0 and c.max()<1
    assert np.abs(c.astype(float)-source).max()<=SOURCE_DELTA_BOUND+1e-7
    low=np.maximum(0,source.astype(float)-SOURCE_DELTA_BOUND);high=np.minimum(1,source.astype(float)+SOURCE_DELTA_BOUND)
    normalized=np.clip((c-low)/(high-low),INIT_NORMALIZED_EPS,1-INIT_NORMALIZED_EPS)
    initial_interval_difference=float(np.max(np.abs((low+(high-low)*normalized).astype(np.float32)-c)))
    assert initial_interval_difference<=1e-7
    support=init['support_active'].astype(bool)
    assert support.sum()==10098 and (~support).sum()==4563
    assert np.array_equal(c[~support],source[~support])
    keys=a['keys'][ids];lookup={tuple(keys[i]):i for i in np.flatnonzero(support)};edges=[]
    for i,k in enumerate(keys):
        if not support[i]:continue
        for d in ((1,0),(0,1)):
            j=lookup.get((int(k[0]+d[0]),int(k[1]+d[1])))
            if j is not None:edges.append((i,j))
    edges=np.asarray(edges,np.int64).reshape(-1,2)
    np.savez_compressed(args.output/'prepared.npz',extension_ids=ids,initial_field_colours_srgb=c,
        source_colours_srgb=source,interval_low=low,interval_high=high,keys=keys,local_xyz=a['local_xyz'][ids],edges=edges,support_active=support,
        source_valid_count199=a['source_valid_count'][ids],source_agree_count199=a['agree_count'][ids],
        source_valid_count49=a['source_valid'][89:138,ids].sum(0).astype(np.int16),
        source_photometric_confidence=a['photometric_confidence'][ids])
    shutil.copy2(args.field_directory/'supervision_masks.npz',args.output/'supervision_masks.npz')
    snapshot=args.output/'input_snapshot';snapshot.mkdir(exist_ok=True)
    shutil.copy2(RUN/'source/r0_runtime.py',snapshot/'r0_runtime.py')
    shutil.copy2(RUN/'source/corridor_runtime.py',snapshot/'corridor_runtime.py')
    field=read(args.field_directory/'field_contract.json')
    immutable={str(args.asset):V2_SHA,str(args.initial_material):INIT_SHA,
        str(Path(__file__).resolve()):sha(__file__),str(snapshot/'r0_runtime.py'):sha(snapshot/'r0_runtime.py'),
        str(snapshot/'corridor_runtime.py'):sha(snapshot/'corridor_runtime.py')}
    for raw in set(BUILD+EVAL):
        p=CAM/f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz';immutable[str(p)]=sha(p)
    contract=dict(schema='R0_PERSISTENT_STRIP_NEW_ROAD_CELL_MATERIAL_CONTROL_V1',asset=str(args.asset),asset_sha256=V2_SHA,
        initial_material=str(args.initial_material),initial_material_sha256=INIT_SHA,
        extension_rows=len(ids),effective_active_rows=10098,inactive_rows=4563,
        support='Frozen CorridorRuntime world strip; outside rows must have zero marker gradient and exact source RGB',
        neighbour_edges=len(edges),neighbour_policy='both endpoints in active strip only',BUILD=list(BUILD),HELD=list(HELD),states=['keep','delete'],
        supervision='Exact frozen field-supervision masks: original BUILD40 road excluding visible/reviewed/two-car box_candidate and other_vehicle_guard; no road_edit_rgb/hidden pixels',
        supervision_masks_sha256=sha(args.output/'supervision_masks.npz'),prepared_sha256=sha(args.output/'prepared.npz'),
        original_field_contract_sha256=sha(args.field_directory/'field_contract.json'),
        eligibility='Before fit, sum actual renderer gradient of sum RGB on each fixed BUILD/state road mask wrt per-new-road RGB. Strict positive summed mass eligible; exact-zero columns frozen at initial field. No epsilon/visibility sweep.',
        eligibility_preforwards=80,eligibility_prebackwards=80,
        parameterization='low+(high-low)*sigmoid(logit((field-low)/(high-low), float64)+delta_float32), then float32; low=max(0,source-.15),high=min(1,source+.15). Initial normalized coordinate only clipped to[1e-7,1-1e-7] for finite logit; initial forward difference measured<=1e-7. Only eligible Parameters; unqualified rows exact field bytes',
        source_delta_bound=SOURCE_DELTA_BOUND,numerical_colour_tolerance=1e-7,
        prepared_initial_interval_colour_maxabs_difference=initial_interval_difference,
        optimizer=dict(name='Adam',lr=LR,betas=[.9,.999],eps=1e-8,asset_step='one BUILD frame, equal keep/delete losses, one optimizer.step',formal_asset_steps=1000),
        losses=dict(data='.5*(mean_RGB_MSE_keep+mean_RGB_MSE_delete) on same fixed true road mask',
            anchor='0.01*mean((new_sRGB-initial_field_sRGB)^2) on eligible cells',
            texture='0.001*mean(((new-init)[i]-(new-init)[j])^2) over physical 8cm four-neighbour edges; preserves initial texture rather than smoothing absolute RGB'),
        lifecycle=dict(smoke_steps=32,smoke_eval_raws=list(EVAL),restore='new process: params+Adam+Python/NumPy/Torch CPU/CUDA RNG+sampler bitwise check',
            disposable_next_steps=2,formal='fresh field initialization, fresh Adam/RNG/sampler; exactly1000 separate steps',
            formal_start_guard='if real smoke median step*1000>=3600s require >=600s real smoke before formal'),
        budgets=dict(prefit_forward_backward=80,smoke_forward_backward=64,smoke_evaluation_forward=14,
            disposable_forward_backward=4,formal_forward_backward=2000,formal_steps=1000),
        frozen='all V2 geometry, active, normals, scales, opacity, old/native/C2/road/gates/env/camera; detached rows except last-N new RGB',
        first_ETA_minutes=[10,25],ETA_basis='prior actual400 feature forwards=41.279s, backward+scene overhead estimated; update with32 actual paired steps',
        torch_allocator_fraction=.30,minimum_gpu_free_MiB=20*1024,allowed_other_gpu_pids=[13030,15730],
        hidden_GT_used=False,source_proxy_seen_all199=True,HELD_is_not_blinded=True,prior_success_claim=False,
        immutable_input_sha256=immutable)
    write(args.output/'contract.json',contract);print(json.dumps(dict(contract=str(args.output/'contract.json'),rows=len(ids),edges=len(edges))),flush=True)


def frozen(args):
    c=read(args.output/'contract.json')
    for path,digest in c['immutable_input_sha256'].items():assert sha(path)==digest,path
    assert sha(args.output/'prepared.npz')==c['prepared_sha256']
    assert sha(args.output/'supervision_masks.npz')==c['supervision_masks_sha256']
    return c,load(args.output/'prepared.npz'),load(args.output/'supervision_masks.npz')


def cpu_check(args):
    import torch
    generator=np.random.default_rng(2)
    source=generator.uniform(.2,.8,(100,3)).astype(np.float32)
    initial=(source+generator.uniform(-.15,.15,(100,3))).astype(np.float32);initial[0]=source[0]+np.float32(.15)
    low=torch.tensor(np.maximum(0,source.astype(float)-.15));high=torch.tensor(np.minimum(1,source.astype(float)+.15))
    v=torch.tensor(initial,dtype=torch.float64);normal=((v-low)/(high-low)).clamp(INIT_NORMALIZED_EPS,1-INIT_NORMALIZED_EPS)
    roundtrip=(low+(high-low)*torch.sigmoid(torch.logit(normal))).float().numpy()
    assert np.max(np.abs(roundtrip-initial))<=1e-7
    s=Sampler();[s.next() for _ in range(32)];state=s.state();future=s.peek(2);hash0=tree_hash(state)
    restored=Sampler();restored.restore(state)
    assert tree_hash(restored.state())==hash0 and [restored.next() for _ in range(2)]==future
    eligible=np.arange(0,100,2);delta=torch.zeros((50,3),requires_grad=True)
    original=torch.tensor(initial);train=(low[eligible]+(high[eligible]-low[eligible])*torch.sigmoid(torch.logit(normal[eligible])+delta.double())).float()
    colours=original.index_copy(0,torch.tensor(eligible),train);(colours[0].sum()).backward()
    assert delta.grad[0].abs().sum()>0 and delta.grad[1:].abs().sum()==0
    assert np.array_equal(colours.detach().numpy()[1::2],initial[1::2])
    args.output.mkdir(parents=True,exist_ok=True)
    obj=dict(status='PASS_CELL_MATERIAL_CPU_CHECKS',initial_sRGB_roundtrip_maxabs=float(abs(roundtrip-initial).max()),
        source_delta_bound=.15,interval_parameterization=True,sampler_bitwise_restore=True,
        next_two_sampler_frames=future,unqualified_rows_frozen_exact=True,only_eligible_colour_gradient=True,GPU_used=False)
    write(args.output/'cpu_checks.json',obj);print(json.dumps(obj),flush=True)


def setup(args):
    import torch
    c,p,m=frozen(args)
    query=lambda *x:subprocess_output(['nvidia-smi',*x,'--format=csv,noheader,nounits'])
    gpu=query('--query-gpu=memory.free,memory.used,utilization.gpu');apps=query('--query-compute-apps=pid,process_name,used_memory')
    unknown=[x for x in apps.splitlines() if x and int(x.split(',')[0]) not in c['allowed_other_gpu_pids']+[os.getpid()]]
    if int(gpu.split(',')[0])<c['minimum_gpu_free_MiB'] or unknown:raise RuntimeError(str((gpu,apps)))
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.30)
    sys.path.insert(0,str(args.output/'input_snapshot'))
    spec=importlib.util.spec_from_file_location('_cellfit_frozen_corridor',args.output/'input_snapshot/corridor_runtime.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    model=module.CorridorRuntime(args.asset,material=args.initial_material)
    assert np.array_equal(model.extension.ids,p['extension_ids'])
    assert np.array_equal(model.extension.active.cpu().numpy(),p['support_active'])
    return torch,c,p,m,model,dict(gpu=gpu,processes=apps,own_pid=os.getpid())


def subprocess_output(command):
    import subprocess
    return subprocess.check_output(command,text=True).strip()


def rendered(model,raw,state,colours):
    scene=model.scene(raw,delete=(state=='delete'),extension=True,retire=False,material=True)
    # Explicitly detach every original input; only the supplied new RGB leaf/graph
    # is reachable by autograd. No private material mutation or parameter write.
    rows={k:None if v is None else v.detach() for k,v in scene['rows'].items()}
    import torch
    n=len(colours);rows['colors_precomp']=torch.cat((rows['colors_precomp'][:-n],colours),0)
    out=model.rt.raster(scene['cam'],model.rt.pipe,0,rows,scene['bg'].detach())
    rgb=out['rgb']+(1-out['alpha'])*scene['env'].detach()
    return rgb,out


def qualify(args):
    torch,c,p,m,model,gpu=setup(args);dest=args.output/'eligibility';dest.mkdir(exist_ok=False)
    initial=torch.as_tensor(p['initial_field_colours_srgb'],device='cuda')
    masses=np.zeros((2,len(initial)),np.float64);frame_counts=np.zeros((2,len(initial)),np.int16);records=[];start=time.time()
    for raw in BUILD:
        domain=torch.as_tensor(m[f'raw{raw:03d}'],device='cuda')
        for si,state in enumerate(('keep','delete')):
            probe=initial.detach().clone().requires_grad_(True)
            rgb,out=rendered(model,raw,state,probe)
            gradient=torch.autograd.grad(rgb[:,domain].sum(),probe)[0].detach().cpu().numpy()
            assert np.isfinite(gradient).all() and gradient.min()>=0
            mass=gradient.mean(1);masses[si]+=mass;frame_counts[si]+=(mass>0)
            wsum=float(out['extra'][16][domain].sum().detach().cpu())
            conservation=abs(float(mass.sum())-wsum)
            assert conservation<.001+1e-5*max(wsum,1)
            records.append(dict(raw=raw,state=state,positive_rows=int((mass>0).sum()),marker_mass=float(mass.sum()),
                extension_alphaT_sum=wsum,marker_conservation_abs=conservation))
            del probe,rgb,out,gradient
        model.rt._data.clear();torch.cuda.empty_cache()
    eligible=masses.sum(0)>0
    assert eligible.any()
    assert not eligible[~p['support_active']].any() and np.array_equal(masses[:,~p['support_active']],np.zeros_like(masses[:,~p['support_active']]))
    np.savez_compressed(dest/'qualified_rows.npz',eligible=eligible,eligible_local_indices=np.flatnonzero(eligible),
        observed_mass_by_state=masses,observed_frame_count_by_state=frame_counts,extension_ids=p['extension_ids'])
    receipt=dict(status='PASS_TRUE_RENDERER_PREFIT_COLUMN_ELIGIBILITY',forwards=80,backwards=80,
        eligible_rows=int(eligible.sum()),frozen_exact_zero_rows=int((~eligible).sum()),records=records,
        eligibility_sha256=sha(dest/'qualified_rows.npz'),outside_strip_exact_zero_gradient=True,
        runtime=model.receipt(),seconds=time.time()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),preflight_gpu=gpu,optimizer_steps=0,original_updates=0)
    write(dest/'receipt.json',receipt);frozen(args);print(json.dumps({k:receipt[k] for k in ('status','eligible_rows','frozen_exact_zero_rows','seconds')}),flush=True)


def rng_state(torch):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(torch,r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch_cpu']);torch.cuda.set_rng_state_all(r['torch_cuda'])


class Material:
    def __init__(self,torch,p,q):
        self.torch=torch;self.initial=torch.as_tensor(p['initial_field_colours_srgb'],device='cuda')
        self.ids=torch.as_tensor(q['eligible_local_indices'],device='cuda',dtype=torch.long)
        self.low=torch.as_tensor(p['interval_low'],device='cuda',dtype=torch.float64)
        self.high=torch.as_tensor(p['interval_high'],device='cuda',dtype=torch.float64)
        normalized=((self.initial[self.ids].double()-self.low[self.ids])/(self.high[self.ids]-self.low[self.ids])).clamp(INIT_NORMALIZED_EPS,1-INIT_NORMALIZED_EPS)
        self.logit=torch.logit(normalized).detach()
        self.delta=torch.nn.Parameter(torch.zeros((len(self.ids),3),device='cuda'))
        self.edges=torch.as_tensor(p['edges'],device='cuda',dtype=torch.long)
        self.optimizer=torch.optim.Adam([self.delta],lr=LR)
    def colours(self):
        train=(self.low[self.ids]+(self.high[self.ids]-self.low[self.ids])*self.torch.sigmoid(self.logit+self.delta.double())).float()
        return self.initial.index_copy(0,self.ids,train)
    def regularization(self):
        residual=self.colours()-self.initial
        anchor=residual[self.ids].square().mean()
        texture=(residual[self.edges[:,0]]-residual[self.edges[:,1]]).square().mean()
        return ANCHOR*anchor+TEXTURE*texture,anchor,texture


def checkpoint(args,folder,step,material,sampler,torch):
    state=dict(step=step,delta=material.delta.detach().cpu().clone(),optimizer=cpu_tree(material.optimizer.state_dict()),
        sampler=sampler.state(),rng=cpu_tree(rng_state(torch)),next_two_frames=sampler.peek(2),
        contract_sha256=sha(args.output/'contract.json'),qualification_sha256=sha(args.output/'eligibility/qualified_rows.npz'))
    target=folder/f'checkpoint{step:06d}.pt'
    if target.exists():raise FileExistsError(target)
    torch.save(state,target)
    loaded=torch.load(target,map_location='cpu',weights_only=False)
    assert tree_hash(loaded)==tree_hash(state)
    return target,tree_hash(state)


def evaluate(args,folder,material,model,torch):
    from PIL import Image
    rows=[]
    with torch.no_grad():
        colours=material.colours()
        for raw in EVAL:
            obs=model.rt.raw(raw)['obs'];gt=np.asarray(obs['rgb'],np.float32).transpose(2,0,1)
            if gt.max()>1:gt/=255
            domains=dict(observed_road=obs['road'].astype(bool),a6=obs['visible'][0].astype(bool),a37=obs['visible'][1].astype(bool),
                protected=obs['protect'].astype(bool)&~obs['visible'].astype(bool).any(0))
            for state in ('keep','delete'):
                rgb,out=rendered(model,raw,state,colours);v=rgb.detach().cpu().numpy();error=abs(v-gt).mean(0)
                Image.fromarray(np.uint8(np.clip(v.transpose(1,2,0),0,1)*255+.5)).save(folder/f'eval_{raw:03d}_{state}.png')
                rows.append(dict(raw=raw,state=state,metrics={k:dict(pixels=int(mask.sum()),MAE=float(error[mask].mean()))
                    for k,mask in domains.items() if mask.any() and not(k=='a37' and state=='delete')}))
            model.rt._data.clear()
    write(folder/'evaluation.json',dict(forwards=14,raws=list(EVAL),source_role='internal visible-only evaluation; no hidden GT',rows=rows))


def train(args):
    if args.phase=='formal':
        smoke=read(args.output/'smoke/receipt.json');assert read(args.output/'restore/receipt.json')['steps_executed']==2
        if smoke['formal_ETA_seconds']>=3600 and smoke['seconds']<600:
            raise RuntimeError('TEN_MINUTE_REAL_SMOKE_REQUIRED_BEFORE_FORMAL')
    torch,c,p,m,model,gpu=setup(args);q=load(args.output/'eligibility/qualified_rows.npz')
    assert sha(args.output/'eligibility/qualified_rows.npz')==read(args.output/'eligibility/receipt.json')['eligibility_sha256']
    folder=args.output/args.phase;folder.mkdir(exist_ok=False)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    material=Material(torch,p,q);sampler=Sampler();step=0;restored=None
    assert not q['eligible'][~p['support_active']].any()
    initial_difference=float((material.colours().detach()-material.initial).abs().max().cpu())
    assert initial_difference<=1e-7
    if args.phase=='restore':
        state=torch.load(args.output/'smoke/checkpoint000032.pt',map_location='cpu',weights_only=False)
        assert state['contract_sha256']==sha(args.output/'contract.json')
        assert state['qualification_sha256']==sha(args.output/'eligibility/qualified_rows.npz')
        with torch.no_grad():material.delta.copy_(state['delta'].cuda())
        material.optimizer.load_state_dict(state['optimizer']);sampler.restore(state['sampler']);restore_rng(torch,state['rng']);step=state['step']
        restored=dict(delta=tree_hash(material.delta)==tree_hash(state['delta']),optimizer=tree_hash(material.optimizer.state_dict())==tree_hash(state['optimizer']),
            RNG=tree_hash(rng_state(torch))==tree_hash(state['rng']),sampler=tree_hash(sampler.state())==tree_hash(state['sampler']),
            next_two_frames=sampler.peek(2)==state['next_two_frames'])
        assert all(restored.values());write(folder/'restore_checks.json',restored)
    budget={'smoke':32,'restore':2,'formal':1000}[args.phase];initial_step=step;records=[];step_times=[];start=time.time()
    for offset in range(budget):
        began=time.time();raw=sampler.next();obs=model.rt.raw(raw)['obs']
        target=np.asarray(obs['rgb'],np.float32).transpose(2,0,1)
        if target.max()>1:target/=255
        gt=torch.as_tensor(target,device='cuda');domain=torch.as_tensor(m[f'raw{raw:03d}'],device='cuda')
        material.optimizer.zero_grad(set_to_none=True);data_loss=0.
        for state_name in ('keep','delete'):
            rgb,out=rendered(model,raw,state_name,material.colours())
            loss=.5*(rgb[:,domain]-gt[:,domain]).square().mean()
            assert torch.isfinite(loss);loss.backward();data_loss+=float(loss.detach().cpu());del rgb,out,loss
        regular,anchor,texture=material.regularization();assert torch.isfinite(regular);regular.backward()
        assert material.delta.grad is not None and torch.isfinite(material.delta.grad).all()
        gradient_norm=float(material.delta.grad.norm().detach().cpu());material.optimizer.step();step+=1
        colours=material.colours().detach();assert torch.isfinite(colours).all() and colours.min()>=0 and colours.max()<=1
        assert float((colours-torch.as_tensor(p['source_colours_srgb'],device='cuda')).abs().max().cpu())<=SOURCE_DELTA_BOUND+1e-7
        assert torch.equal(colours[~torch.as_tensor(q['eligible'],device='cuda')],material.initial[~torch.as_tensor(q['eligible'],device='cuda')])
        model.rt.assert_unchanged();assert model.extension.unchanged()
        step_times.append(time.time()-began)
        records.append(dict(step=step,raw=raw,data_MSE=data_loss,anchor=float(anchor.detach().cpu()),texture=float(texture.detach().cpu()),gradient_norm=gradient_norm,seconds=step_times[-1]))
        if step%100==0 or offset==budget-1:checkpoint(args,folder,step,material,sampler,torch)
        if (offset+1)%16==0 or offset==budget-1:print(json.dumps(records[-1]),flush=True)
        model.rt._data.clear()
    if args.phase=='smoke':evaluate(args,folder,material,model,torch)
    runtime=model.receipt();formal_eta=float(np.median(step_times)*1000)
    receipt=dict(status='PASS_CELL_MATERIAL_'+args.phase.upper(),steps_executed=budget,initial_step=initial_step,final_step=step,
        formal_asset_steps=1000 if args.phase=='formal' else 0,disposable_steps=2 if args.phase=='restore' else 0,
        forwards=2*budget+(14 if args.phase=='smoke' else 0),backwards=2*budget,regularizer_backward_calls=budget,
        backward_definition='two native CUDA RGB backwards per paired asset step; regularizer backward calls listed separately',
        eligible_cells=int(q['eligible'].sum()),frozen_exact_zero_cells=int((~q['eligible']).sum()),trainable_scalars=int(q['eligible'].sum()*3),
        optimizer='Adam',learning_rate=LR,records=records,restored=restored,runtime_frozen=runtime,
        original_parameter_updates=0,geometry_updates=0,opacity_updates=0,torch_allocator_fraction=.30,
        initial_forward_colour_maxabs_difference=initial_difference,source_delta_bound=SOURCE_DELTA_BOUND,
        final_source_delta_maxabs=float((colours-torch.as_tensor(p['source_colours_srgb'],device='cuda')).abs().max().cpu()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        seconds=time.time()-start,median_paired_step_seconds=float(np.median(step_times)),formal_ETA_seconds=formal_eta,
        preflight_gpu=gpu,shutdown_performed=False)
    write(folder/'receipt.json',receipt)
    if args.phase=='formal':
        value=material.colours().detach().cpu().numpy();assert np.array_equal(value[~q['eligible']],p['initial_field_colours_srgb'][~q['eligible']])
        assert np.array_equal(value[~p['support_active']],p['source_colours_srgb'][~p['support_active']])
        metadata=dict(role='fitted BUILD40 visible-road SH0/sRGB material; not true hidden texture',source_asset_sha256=V2_SHA,
            initial_field_sha256=INIT_SHA,formal_steps=1000,lr=LR,anchor=ANCHOR,texture=TEXTURE,source_delta_bound=SOURCE_DELTA_BOUND,
            qualification_sha256=sha(args.output/'eligibility/qualified_rows.npz'),original_updates=0,hidden_GT_used=False)
        np.savez_compressed(args.output/'material_override.npz',asset_sha256=np.array(V2_SHA),extension_ids=p['extension_ids'],colours_srgb=value,
            initial_field_colours_srgb=p['initial_field_colours_srgb'],source_colours_srgb=p['source_colours_srgb'],
            eligible=q['eligible'],support_active=p['support_active'],observed_mass_by_state=q['observed_mass_by_state'],observed_frame_count_by_state=q['observed_frame_count_by_state'],
            source_valid_count199=p['source_valid_count199'],source_agree_count199=p['source_agree_count199'],source_valid_count49=p['source_valid_count49'],
            eligible_local_indices=q['eligible_local_indices'],raw_logit_delta=material.delta.detach().cpu().numpy(),metadata_json=np.array(json.dumps(metadata,sort_keys=True)))
        write(args.output/'export_receipt.json',dict(material=str(args.output/'material_override.npz'),sha256=sha(args.output/'material_override.npz'),metadata=metadata))
    frozen(args);print(json.dumps({k:receipt[k] for k in ('status','steps_executed','seconds','formal_ETA_seconds')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','cpu-check','qualify','smoke','restore','formal'))
    p.add_argument('--asset',type=Path,default=RUN/'surface/frontfill_v2/road_reference_199.npz')
    p.add_argument('--initial-material',type=Path,default=RUN/'surface/material_worldfield11x9_strip_v1/material_override.npz')
    p.add_argument('--field-directory',type=Path,default=RUN/'surface/material_worldfield11x9_strip_v1')
    p.add_argument('--output',type=Path,default=RUN/'surface/material_cellfit_strip_v1')
    args=p.parse_args()
    if args.phase in ('prepare','cpu-check'):os.environ['CUDA_VISIBLE_DEVICES']=''
    {'prepare':prepare,'cpu-check':cpu_check,'qualify':qualify,'smoke':train,'restore':train,'formal':train}[args.phase](args)


if __name__=='__main__':main()
