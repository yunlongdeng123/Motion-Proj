"""New-teacher fixed-budget queue. Smoke gates formal work; no sweeps or power actions."""
from pathlib import Path
import argparse,copy,gc,json,os,random,subprocess,sys,time,traceback
import numpy as np
from runtime1152 import *

paths()
import torch
from PIL import Image
from world_texture_runtime import WorldTextureRuntime
from fixed_material_read_operator import FixedReadOperator
from material_readers import GaussianAverageReader,PixelWorldReader
from field_training_state import balanced_schedule,cpu_clone,max_tree_difference,rng_state,restore_rng
from build_training_inputs import common_low_mask
from native_capacity_loss import NativeCapacityLoss
sys.path.insert(0,str(B/'numerical_completion/code'))
from checkpointed_lbfgsb import CheckpointedLBFGSB
from fp64_reduction_loss import FP64ReductionLoss

DEVICE='cuda'
OPT=dict(low_lr=.01,high_lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0.)
REG=dict(low_adjacent_delta_squared=.01,effective_high_energy=.1,
         effective_high_lowfrequency_leakage=.1,effective_high_weighted_DC=.1)
COUNTERS={};MANIFEST=[];CTX=None;INPUTS=None;NATIVE_PRIOR=0

def read(p):return json.loads(Path(p).read_text())
def save_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
def tensor(a,dtype=None):return torch.as_tensor(np.array(a,copy=True),device=DEVICE,dtype=dtype)
def seeded():random.seed(101);np.random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101)
def record(kind,n=1,role='formal'):
    key=role+':'+kind;COUNTERS[key]=COUNTERS.get(key,0)+n
def persist_counters():
    write(ROOT/'compute_ledger.json',dict(counters=COUNTERS,native_process_total=NATIVE_PRIOR+(CTX.native_calls if CTX is not None else 0),
        native_preparation=read(ROOT/'native/receipt.json'),updated_at=time.time()))
def loss_state(v):
    v=copy.deepcopy(v);v['spec']['scales']=tuple(v['spec']['scales'])
    v['spec']['source_weights']=tuple(tuple(x) for x in v['spec']['source_weights']);return v
def workspace_difference(a,b,path='root'):
    """Numeric workspace tolerance; Fortran task/csave bytes must be exact."""
    if isinstance(a,np.ndarray) and a.dtype.kind in 'SUO':
        if not isinstance(b,np.ndarray) or a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a,b):
            raise ValueError('Fortran string workspace differs: '+path)
        return 0.
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():raise ValueError('workspace keys differ: '+path)
        return max((workspace_difference(a[k],b[k],path+'.'+str(k)) for k in a),default=0.)
    if isinstance(a,(tuple,list)):
        if type(a) is not type(b) or len(a)!=len(b):raise ValueError('workspace sequence differs: '+path)
        return max((workspace_difference(x,y,path+'.'+str(i)) for i,(x,y) in enumerate(zip(a,b))),default=0.)
    if isinstance(a,(float,np.floating)):
        if not np.isfinite(a) or not np.isfinite(b):raise ValueError('nonfinite workspace scalar: '+path)
        return abs(float(a)-float(b))
    return max_tree_difference(a,b,path)

def qualify_basis(eligible):
    dest=ROOT/'native/qualified_basis';receipt=dest/'receipt.json'
    if receipt.exists():
        assert read(receipt)['scope_SHA256']==sha(ROOT/'cpu_inputs/manifest.json');return dest
    sys.path.insert(0,str(ROOT/'code'))
    from native_basis1152_evidence import export_native_basis,scope_signature
    auth=dict(schema='pixel_native_basis_smoke.v1',phase='native_smoke',native_allowed=True,gpu_authorized=True,
        formal_allowed=False,frames=list(FRAMES),scope_world_rows_sha256=scope_signature(CTX.active_world_rows,eligible),
        max_native_forward_calls=200,delete_actor_ids=[10],optimizer_updates=0)
    rows=[]
    with torch.no_grad():
        for raw in FRAMES:
            directory=dest/f'{raw:03d}'
            meta=read(directory/'basis.json') if (directory/'basis.json').exists() else export_native_basis(CTX,raw,eligible,directory,authorization=auth)
            assert meta['joint']['K_eligible_weight_max']==0,'no screen-K repair or silent scope shrink'
            rows.append(dict(raw=raw,K_eligible_weight_max=0,eligible_rows=int(eligible.sum())))
    write(receipt,dict(complete=True,rows=rows,scope_SHA256=sha(ROOT/'cpu_inputs/manifest.json'),native_forward_calls=10))
    return dest

class CSRRead:
    """Same fixed FP64 coefficients, CSR CUDA forward and adjoint."""
    def __init__(self,root):
        self.op=FixedReadOperator.load(root)
        self.row_ids=self.op.row_ids;self.mass=self.op.mass;self.KH=self.op.KH;self.cache={}
    def torch_query(self,low,high,gate):
        if gate not in self.cache:
            def csr(m):
                m=m.tocsr();return torch.sparse_csr_tensor(tensor(m.indptr,torch.int64),tensor(m.indices,torch.int64),tensor(m.data,torch.float64),size=m.shape)
            self.cache[gate]=(tensor(self.op.base,torch.float64),csr(self.op.KL),csr(self.op.KH[gate]))
        base,L,H=self.cache[gate]
        return (base+torch.sparse.mm(L,low.reshape(-1,3).double())+torch.sparse.mm(H,high.reshape(-1,3).double())).clamp(0,1).float()
    def clear(self):self.cache.clear();self.op._torch_cache.clear()

class FilteredTexture(WorldTextureRuntime):
    def __init__(self,*args,sigma_uv,**kwargs):
        super().__init__(*args,**kwargs);self.sigma_uv=tuple(sigma_uv);self.kernels={}
    def fields(self):
        L,H=super().fields();key=(str(H.device),H.dtype)
        if key not in self.kernels:
            values=[]
            for sigma in (self.sigma_uv[1],self.sigma_uv[0]):
                radius=int(3*sigma+.5);x=torch.arange(-radius,radius+1,device=H.device,dtype=torch.float64)
                k=torch.exp(-.5*(x/sigma)**2);values.append((radius,(k/k.sum()).to(H.dtype)))
            self.kernels[key]=values
        import torch.nn.functional as F
        (rv,kv),(ru,ku)=self.kernels[key];v=H.permute(2,0,1)[None]
        v=F.conv2d(F.pad(v,(0,0,rv,rv),mode='replicate'),kv[None,None,:,None].expand(3,1,-1,1),groups=3)
        v=F.conv2d(F.pad(v,(ru,ru,0,0),mode='replicate'),ku[None,None,None,:].expand(3,1,1,-1),groups=3)
        return L,v[0].permute(1,2,0)
    def save(self,path,metadata):
        torch.save(dict(schema='common1152_source_filtered_texture.v1',state=self.state_dict(),metadata=metadata,
            sigma_uv=self.sigma_uv,settings=dict(low_stride=self.low_stride,low_bound=self.low_bound,high_bound=self.high_bound)),path)

class Adapter:
    parent_rgb=None
    def __init__(self,ctx):self.ctx=ctx;self.parent_rgb=ctx.base_rgb.detach()
    def require_execution(self,role):
        self.ctx.require_lease()
        if role=='formal' and not read(ROOT/'smoke/receipt.json').get('verified'):raise RuntimeError('actual smoke gate missing')
    def render_average(self,raw,colors,joint=False):
        out=self.ctx.render_joint(raw,colors) if joint else self.ctx.render_road(raw,colors)
        return dict(rgb=out['rgb'],alpha=out['alpha'],K=self.ctx.K_masks[raw])
    def assert_frozen(self):self.ctx.assert_frozen()

class Inputs:
    def __init__(self):
        self.manifest=read(ROOT/'cpu_inputs/manifest.json');assert self.manifest['complete']
        self.ledger=load(self.manifest['permission_ledger']);self.r1=load(ROOT/'r1_inputs/initial.npz')
        self.eligible=self.ledger['common_write_active'].astype(bool)
        assert np.array_equal(self.ledger['active_world_rows'],CTX.active_world_rows)
        assert not np.any(self.eligible & ~load(ROOT/'native/basis/086/row_scope.npz')['eligible_active'])
        self.targets={};self.regions={}
        for raw in (86,110):
            d=load(Path(self.manifest['targets_root'])/f'{raw:03d}_targets.npz')
            assert d['fixed_alpha'].shape==(768,1152)
            self.targets[raw]={k:tensor(d[k],torch.bool if k in ['A','C','S','fullS'] else torch.float32) for k in ['A','C','S','fullS','fixed_alpha','target_A','target_C','target_S']}
            contexts={k:np.asarray(d.get(k+'_core95',d[k]&(d['fixed_alpha']>=.95)),bool) for k in ['A','C','S']}
            self.regions[raw]=dict(alpha=d['fixed_alpha'],targets={k:d['target_'+k] for k in ['A','C','S']},contexts=contexts,scores=contexts)
            for k in ['A','C','S']:
                assert self.targets[raw][k].any()
        basis_root=qualify_basis(self.eligible)
        self.basis={raw:{mode:load(basis_root/f'{raw:03d}'/(mode+'_basis.npz')) for mode in ['road','joint']} for raw in FRAMES}
        self.r1op=CSRRead(ROOT/'r1_inputs/operator');self.avg=CSRRead(self.manifest['average_operator'])
        self.pixel={raw:CSRRead(Path(self.manifest['pixel_operator_root'])/f'{raw:03d}') for raw in FRAMES}
        sigma_path=self.manifest.get('source_footprint_json',str(ROOT/'cpu_inputs/source_footprint.json'))
        sd=read(sigma_path);self.sigma=sd.get('sigma_uv',sd.get('sigma_uv_u_v'))
        if self.sigma is None:raise ValueError('M12 predeclared physical footprint sigma missing')
        self.adapter=Adapter(CTX)
        files=[ROOT/'contract.json',ROOT/'teacher/manifest.json',ROOT/'camera_manifest.json',ROOT/'cpu_inputs/manifest.json',
            Path(self.manifest['permission_ledger']),ROOT/'r1_inputs/receipt.json',Path(sigma_path),Path(__file__),Path(__file__).with_name('runtime1152.py')]
        files += [Path(self.manifest['targets_root'])/f'{r:03d}_targets.npz' for r in (86,110)]
        self.binding={str(p):sha(p) for p in files}
        write(ROOT/'source_binding.json',self.binding)
    def world(self,group,gate,filtered=False):
        d=self.r1 if group=='1' else self.ledger
        kwargs=dict(low_mask=d['low_mask'] if group=='1' else common_low_mask(d['common_latent_mask']),low_stride=6,low_bound=.3,high_bound=.04 if group=='1' else .34)
        cls=FilteredTexture if filtered else WorldTextureRuntime
        if filtered:kwargs['sigma_uv']=self.sigma
        return cls(d['common_base_srgb'],d['common_material_valid'],d['observed_H_prior'] if gate=='observed' else d['mH_union'],
            d['common_latent_mask'],d['common_reg_support'],**kwargs).to(DEVICE)
    def reader(self,group,kind,gate):
        if kind=='pixel':
            # New operators use explicit observed/teacher keys; old reader accepts
            # arbitrary gate names and never infers permission from their spelling.
            if not all(gate in op.KH for op in self.pixel.values()):raise ValueError('new pixel permission key missing: '+gate)
            return PixelWorldReader(self.pixel,gate,self.basis,DEVICE,self.adapter,{})
        eligible=self.r1['eligible'] if group=='1' else self.eligible
        op=self.r1op if group=='1' else self.avg
        return GaussianAverageReader(op,gate,CTX.active_world_rows,eligible,CTX.base_rgb,self.adapter,{})
    def clear(self):
        for op in [self.r1op,self.avg,*self.pixel.values()]:op.clear()
        gc.collect();torch.cuda.empty_cache()

class Task:
    def __init__(self,group,arm,kind='avg',gate='observed',loss='photo',role='formal',total=800,filtered=False):
        seeded();self.group=group;self.arm=arm;self.kind=kind;self.gate=gate;self.loss_mode=loss;self.role=role;self.total=total
        self.model=INPUTS.world(group,gate,filtered);self.reader=INPUTS.reader(group,kind,gate)
        self.eligible=INPUTS.r1['eligible'] if group=='1' else INPUTS.eligible
        self.filtered=filtered;self.cursor=0;self.trace=[];self.schedule=balanced_schedule(total,101);self.stage='A';self.loss=None
        self.independent=arm=='matched_I0'
        if self.independent:
            self.initial=self.reader.colors(self.model).detach();del self.model
            sys.path.insert(0,str(C/'road_multiscale_material_20261118_r1/code'))
            from matched_i0_training import MatchedI0
            self.model=MatchedI0(self.initial.cpu(),self.eligible).to(DEVICE)
            self.optimizer=torch.optim.Adam([self.model.delta],lr=.01,betas=(.9,.999),eps=1e-8)
        else:self.optimizer=torch.optim.Adam([dict(params=[self.model.low],lr=.01),dict(params=[self.model.high],lr=.003)],betas=(.9,.999),eps=1e-8)
        self.identity=dict(group=group,arm=arm,reader=kind,gate=gate,loss=loss,role=role,total=total,source_binding=INPUTS.binding,filtered=filtered)
    @property
    def folder(self):return ROOT/('formal' if self.role=='formal' else 'smoke/work')/('r'+self.group)/self.arm
    def colors(self):return self.model() if self.independent else self.reader.colors(self.model)
    def stage_for_cursor(self):
        if self.independent:return 'I'
        if self.group=='3':return 'B' if self.cursor<600 else 'C'
        if self.arm=='L800':return 'A'
        return 'A' if self.cursor<200 else 'B' if self.cursor<500 else 'C'
    def prediction(self,raw,joint=False):
        if self.independent:
            colors=self.colors();rd=CTX.render_road(raw,colors)
            packet=dict(road_rgb=rd['rgb'],road_alpha=rd['alpha'].squeeze(),K=CTX.K_masks[raw])
            if joint:
                j=CTX.render_joint(raw,colors);packet.update(joint_rgb=j['rgb'],joint_alpha=j['alpha'].squeeze())
            return packet
        return self.reader.render(self.model,raw,joint=joint)
    def prepare_loss(self):
        if self.loss is not None:return
        self.loss=NativeCapacityLoss(INPUTS.regions,device=DEVICE)
        path=ROOT/('formal' if self.role=='formal' else 'smoke')/'r3_normalizers.json'
        if path.exists():self.loss.restore(loss_state(read(path)))
        else:
            with torch.no_grad():pred={r:self.prediction(r)['road_rgb'] for r in (86,110)}
            record('nativeF',2,self.role);state=self.loss.calibrate(pred)
            state['teacher_label_holdout_policy']='all qualified new1152 teacher fitting; no held selection'
            write(path,state)
    def objective(self):
        if self.group=='3':
            self.prepare_loss();pred={r:self.prediction(r)['road_rgb'] for r in (86,110)}
            record('nativeF',2,self.role);value,terms=self.loss.objective(pred,self.loss_mode)
            return value,dict(total=float(value.detach())),2
        raw=self.schedule[self.cursor];pred=self.prediction(raw)['road_rgb'];target=INPUTS.targets[raw];alpha=target['fixed_alpha'][None]
        terms={k:(pred-alpha*target['target_'+k]).abs()[:,target[k]].mean() for k in ['A','C','S']}
        value=terms['A']+.5*terms['C']+terms['S']
        if self.independent:value=value+.001*self.model.prior()
        else:
            reg=self.model.regularizers();value=value+sum(REG[k]*reg[k] for k in REG)
        record('pixelF' if self.kind=='pixel' else 'nativeF',1,self.role)
        return value,{k:float(v.detach()) for k,v in terms.items()},0 if self.kind=='pixel' else 1
    def step(self,stage=None):
        self.reader.require_execution(self.role);self.stage=stage or self.stage_for_cursor()
        if not self.independent:self.model.set_stage(self.stage)
        self.optimizer.zero_grad(set_to_none=True);t=time.monotonic();value,terms,backs=self.objective()
        if not torch.isfinite(value):raise FloatingPointError('nonfinite training objective')
        value.backward()
        grads={k:float(p.grad.abs().max()) for k,p in self.model.named_parameters() if p.grad is not None}
        if not grads or not all(np.isfinite(list(grads.values()))):raise FloatingPointError('nonfinite/disconnected field gradients')
        self.optimizer.step()
        if self.independent:self.model.project()
        self.trace.append(self.schedule[self.cursor]);self.cursor+=1
        record('updates',1,self.role);record('nativeB',backs,self.role);record('pixelB',1 if self.kind=='pixel' else 0,self.role)
        return dict(cursor=self.cursor,loss=float(value.detach()),stage=self.stage,gradients=grads,seconds=time.monotonic()-t,**terms)
    def snapshot(self):
        params={k:p for k,p in self.model.named_parameters()}
        return dict(schema='common1152_complete_training_state.v1',identity=self.identity,params=cpu_clone(params),
            optimizer=cpu_clone(self.optimizer.state_dict()),RNG=cpu_clone(rng_state(DEVICE)),cursor=self.cursor,trace=self.trace.copy(),
            schedule=self.schedule.copy(),stage=self.stage,loss_normalizers=self.loss.snapshot() if self.loss is not None else None)
    def restore(self,s,common=False):
        if s['schema']!='common1152_complete_training_state.v1':raise ValueError('wrong checkpoint schema')
        if common:
            assert s['cursor']==(200 if self.role=='formal' else 2) and s['identity']['group']==self.group and s['identity']['reader']==self.kind
            assert s['identity']['role']==self.role and s['identity']['source_binding']==self.identity['source_binding']
            assert not bool(s['params']['high'].any()),'common A must have H exactly zero'
        elif s['identity']!=self.identity:raise ValueError('checkpoint scientific identity mismatch')
        assert s['schedule']==self.schedule and s['trace']==s['schedule'][:s['cursor']]
        with torch.no_grad():
            for k,p in self.model.named_parameters():p.copy_(s['params'][k].to(p.device))
        self.optimizer.load_state_dict(copy.deepcopy(s['optimizer']));restore_rng(s['RNG'],DEVICE)
        self.cursor=s['cursor'];self.trace=s['trace'].copy();self.stage=s['stage']
        if s.get('loss_normalizers') is not None:
            self.loss=NativeCapacityLoss(INPUTS.regions,device=DEVICE);self.loss.restore(s['loss_normalizers'])
    def checkpoint(self):
        self.folder.mkdir(parents=True,exist_ok=True);s=self.snapshot();save_torch(self.folder/'checkpoint.pt',s)
        if self.cursor%100==0:save_torch(self.folder/f'checkpoint_{self.cursor:04d}.pt',s)
    def authority(self):
        p=self.folder/'authority.pt';self.folder.mkdir(parents=True,exist_ok=True)
        metadata=dict(identity=self.identity,cursor=self.cursor,source_binding=INPUTS.binding,source_sigma_uv=INPUTS.sigma if self.filtered else None)
        if self.independent:save_torch(p,dict(schema='common1152_independent_colors.v1',colors=self.colors().detach().cpu(),eligible=self.eligible,metadata=metadata))
        else:self.model.save(p,metadata)
        return p

def export(task):
    authority=task.authority();rows=[];dest=ROOT/'exports'/('r'+task.group)/task.arm;dest.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for raw in FRAMES:
            pkt=task.prediction(raw,joint=True);record('export_nativeF',0 if task.kind=='pixel' else 2,task.role)
            arrays={k:(v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k,v in pkt.items() if k in ['road_rgb','road_alpha','joint_rgb','joint_alpha','K','reader_source_mass','fallback_mask','eligible_weight_road','eligible_weight_joint']}
            base=INPUTS.basis[raw];mask=base['joint']['K']
            alphaerr=max(float(np.max(np.abs(arrays[k+'_alpha']-base[k]['alpha']))) for k in ['road','joint'])
            kerr=float(np.max(np.abs(arrays['joint_rgb']-base['joint']['baseline_rgb'])[:,mask]))
            if alphaerr!=0 or kerr!=0:raise RuntimeError('fixed geometry/keep protection failed '+str((task.arm,raw,alphaerr,kerr)))
            npz=dest/f'{raw:03d}.npz';np.savez_compressed(npz,**arrays)
            paths={}
            for view in ['road','joint']:
                image=dest/f'{raw:03d}_{view}.png';rgb=np.rint(np.clip(arrays[view+'_rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)
                assert rgb.shape==(768,1152,3);Image.fromarray(rgb).save(image);paths[view+'_png']=str(image.relative_to(ROOT))
            rows.append(dict(group=task.group,arm=task.arm,name=task.arm,budget=str(task.cursor)+' accepted updates',raw=raw,packet=str(npz.relative_to(ROOT)),**paths,
                authority=str(authority.relative_to(ROOT)),authority_SHA256=sha(authority),resolution=[1152,768],
                renderer='actual1152_native_eligible_weights_pixel_world' if task.kind=='pixel' else 'actual1152_ordinary_GS',
                data_role='all_qualified_newteacher_fit',K_RGB_max=kerr,alpha_max=alphaerr))
    # The learned field is the authority; ordinary road cache is an opt-in derived asset.
    if task.kind!='pixel':
        world=np.zeros(len(CTX.road['active']),bool);world[CTX.active_world_rows]=task.eligible
        asset=task.folder/'asset'
        if asset.exists():
            with np.load(asset/'road.npz',allow_pickle=False) as z:stored=z['reference_srgb'][CTX.active_world_rows]
            assert float(np.max(np.abs(stored-task.colors().detach().cpu().numpy())))<=1e-6,'replayed derived asset differs'
        else:CTX.save_asset(task.colors().detach(),world,asset,training_record=dict(role='new1152_teacher_fit',authority=str(authority),authority_SHA256=sha(authority),optimizer_steps=task.cursor))
    write(dest/'receipt.json',dict(complete=True,rows=rows,authority=str(authority),authority_SHA256=sha(authority),cursor=task.cursor))
    keep=[r for r in MANIFEST if not(r['group']==task.group and r['arm']==task.arm)]
    MANIFEST[:]=keep+rows;write(ROOT/'export_manifest.json',dict(schema='common1152_exports.v1',rows=MANIFEST,complete=False))
    persist_counters();return rows

def run_task(task,end,prefix=None,export_end=True):
    ck=task.folder/'checkpoint.pt'
    if ck.exists():task.restore(torch.load(ck,map_location='cpu',weights_only=False))
    elif prefix is not None:task.restore(prefix,common=True)
    while task.cursor<end:
        r=task.step()
        if task.cursor%50==0:
            task.checkpoint();persist_counters();status('training',arm=task.arm,group=task.group,cursor=task.cursor,budget=end,PID=os.getpid())
            print(json.dumps(dict(arm=task.arm,**r)),flush=True)
    task.checkpoint()
    if export_end:export(task)
    return task.snapshot()

def new_r3(arm,role='formal',total=1200):
    t=Task('3',arm,gate='teacher',loss='rgb' if arm=='G_RGB' else 'detail',role=role,total=total)
    prefix=ROOT/'shared/r2_avg_A200.pt'
    if role=='formal':s=torch.load(prefix,map_location='cpu',weights_only=False)
    else:s=torch.load(ROOT/'smoke/r3_seed_A.pt',map_location='cpu',weights_only=False)
    with torch.no_grad():t.model.low.copy_(s['params']['low'].to(DEVICE));t.model.high.zero_()
    return t

def numerical(role='formal',maxFG=2000,smoke_seed=None):
    folder=ROOT/('formal/r3/G_NUMERIC' if role=='formal' else 'smoke/numeric');folder.mkdir(parents=True,exist_ok=True)
    if role=='formal':seedfile=ROOT/'formal/r3/G_DETAIL/authority.pt';model,_=WorldTextureRuntime.load(seedfile,device=DEVICE);reader=INPUTS.reader('3','avg','teacher')
    else:model,reader=smoke_seed.model,smoke_seed.reader
    with torch.no_grad():seed=reader.colors(model).detach()
    positions=tensor(np.flatnonzero(INPUTS.eligible),torch.long)
    def evaluator(x):
        free=tensor(x,torch.float32).requires_grad_(True);colors=seed.index_copy(0,positions,free.reshape(-1,3));pred={}
        for raw in (86,110):pred[raw]=CTX.render_road(raw,colors)['rgb']
        value,_=precise.objective(pred,'detail');gradient=torch.autograd.grad(value,free)[0]
        record('FG',1,role);record('nativeF',2,role);record('nativeB',2,role)
        return float(value.detach()),gradient.detach().cpu().numpy().astype(np.float64),{}
    precise=FP64ReductionLoss(INPUTS.regions,device=DEVICE)
    norm=ROOT/('formal/r3_normalizers.json' if role=='formal' else 'smoke/r3_normalizers.json');precise.restore(loss_state(read(norm)))
    # Qualify the unchanged FP32 prediction/derivative path before FP64 reductions.
    original=NativeCapacityLoss(INPUTS.regions,device=DEVICE);original.restore(loss_state(read(norm)))
    free=seed[positions].clone().requires_grad_(True);colors=seed.index_copy(0,positions,free);pred={r:CTX.render_road(r,colors)['rgb'] for r in (86,110)}
    f32,_=original.objective(pred,'detail');f64,_=precise.objective(pred,'detail')
    g32=torch.autograd.grad(f32,free,retain_graph=True)[0];g64=torch.autograd.grad(f64,free)[0]
    fr=abs(float(f32)-float(f64))/max(abs(float(f64)),1e-12)
    gr=float((g32.double()-g64.double()).norm())/max(float(g64.double().norm()),1e-12)
    qualification=dict(verified=fr<=1e-6 and gr<=1e-5,function_relative=fr,gradient_L2_relative=gr,same_native_prediction=True,extra_qualification_compute=True)
    record('nativeF',2,role+'_qualification');record('nativeB',4,role+'_qualification');write(folder/'precision_qualification.json',qualification)
    if not qualification['verified']:raise RuntimeError('fixed-point reduction precision qualification failed')
    del original,free,colors,pred,f32,f64,g32,g64
    solver=CheckpointedLBFGSB(seed[positions].cpu().numpy().astype(np.float64),evaluator,identity=dict(role=role,binding=INPUTS.binding),maxFG=maxFG)
    pointer=folder/'solver_latest.json'
    if pointer.exists():
        meta=read(pointer);solver.load(meta['workspace']);restore_rng(torch.load(meta['rng'],map_location='cpu',weights_only=False),DEVICE)
    replay=None
    while not solver.done:
        row=solver.next_accepted()
        if row is None:break
        if role=='smoke' or solver.accepted%100==0:
            path=folder/f'solver_{solver.fg_calls:06d}_{solver.accepted:06d}.npz'
            if not path.exists():solver.save(path)
            rng=path.with_suffix('.rng.pt');save_torch(rng,rng_state(DEVICE));write(pointer,dict(workspace=str(path),rng=str(rng)))
            persist_counters();status('numeric_smoke' if role=='smoke' else 'training_numeric',arm='G_NUMERIC',FG=solver.fg_calls,budget=maxFG,PID=os.getpid())
        if role=='smoke' and solver.accepted==1:
            before=solver.snapshot();expected=solver.next_accepted();expected_state=solver.snapshot()
            solver.load(path);restore_rng(torch.load(rng,map_location='cpu',weights_only=False),DEVICE);actual=solver.next_accepted()
            diff=workspace_difference(expected_state,solver.snapshot());assert diff<1e-5
            replay=dict(verified=True,complete_Fortran_workspace=True,next_accepted_max=diff,string_workspace_bitexact=True)
            break
        if role=='formal' and row['relative_FW_gap']<=.01:solver.stop_at_training_gate()
    if role=='smoke':write(folder/'receipt.json',dict(verified=replay is not None,replay=replay,FG=solver.fg_calls));return
    colors=seed.index_copy(0,positions,tensor(solver.last_accepted_x.reshape(-1,3),torch.float32)).detach()
    class NumericTask:
        group='3';arm='G_NUMERIC';kind='avg';eligible=INPUTS.eligible;cursor=solver.accepted;folder=ROOT/'formal/r3/G_NUMERIC';role='formal'
        def authority(self):
            p=self.folder/'authority.pt';save_torch(p,dict(schema='common1152_bounded_GS_colors.v1',colors=colors.cpu(),eligible=self.eligible,solver_reason=solver.reason,source_binding=INPUTS.binding));return p
        def colors(self):return colors
        def prediction(self,raw,joint=False):
            rd=CTX.render_road(raw,colors);v=dict(road_rgb=rd['rgb'],road_alpha=rd['alpha'].squeeze(),K=CTX.K_masks[raw])
            if joint:
                j=CTX.render_joint(raw,colors);v.update(joint_rgb=j['rgb'],joint_alpha=j['alpha'].squeeze())
            return v
    export(NumericTask())
    write(folder/'receipt.json',dict(complete=True,FG=solver.fg_calls,accepted=solver.accepted,objective=solver.last_accepted_f,
        relative_FW_gap=solver.gap(solver.last_accepted_x,solver.last_accepted_g)/max(solver.last_accepted_f,1e-3),termination=solver.reason,strict_certificate=False))

def smoke():
    dest=ROOT/'smoke';dest.mkdir(exist_ok=True);start=time.monotonic();checks=[];timings={}
    variants=[('1','ABC800','avg','observed',False),('1','matched_I0','avg','observed',False),
        ('2','M11','pixel','teacher',False),('2','M12','pixel','teacher',True)]
    r3seed=Task('2','common_avg',role='smoke',total=800)
    for _ in range(2):r3seed.step('A')
    prefix=r3seed.snapshot();save_torch(dest/'r3_seed_A.pt',prefix)
    fork=Task('2','M10',gate='teacher',role='smoke');fork.restore(prefix,common=True)
    assert torch.equal(fork.model.H_gate,tensor(INPUTS.ledger['mH_union'][...,None],torch.float64))
    fork.step('B');fork.step('C');checks.append(dict(arm='common_A_to_teacher_branch',gate_preserved=True,next_stage_BC=True))
    del r3seed,fork
    for group,arm,kind,gate,filtered in variants:
        task=Task(group,arm,kind,gate,role='smoke',filtered=filtered)
        t=time.monotonic()
        for stage in (['I','I'] if task.independent else ['A','A','B','B','C','C']):task.step(stage)
        path=dest/(arm+'_complete.pt');save_torch(path,task.snapshot());before=task.snapshot()
        result=task.step('I' if task.independent else 'C');expected=task.snapshot()
        fresh=Task(group,arm,kind,gate,role='smoke',filtered=filtered);fresh.restore(torch.load(path,map_location='cpu',weights_only=False));fresh.step('I' if task.independent else 'C')
        diff=max_tree_difference(expected,fresh.snapshot());assert diff<2e-5,(arm,diff)
        # Persistence through the actual authority representation and frozen reader.
        authority=task.authority();save_torch(task.folder/'smoke_checkpoint.pt',before)
        with torch.no_grad():packet=fresh.prediction(110,joint=True)
        assert tuple(packet['road_rgb'].shape)==(3,768,1152)
        with torch.no_grad():
            if task.independent:
                saved=torch.load(authority,map_location='cpu',weights_only=False);reloaded=CTX.render_road(110,saved['colors'].to(DEVICE))['rgb']
            else:
                saved=torch.load(authority,map_location='cpu',weights_only=False)
                if filtered:
                    d=saved['state'];m=FilteredTexture(d['base'].numpy(),d['valid'][...,0].numpy(),d['H_gate'][...,0].numpy(),
                        d['latent_mask'][...,0].numpy(),d['reg_support'][...,0].numpy(),low_mask=d['low_mask'][...,0].numpy(),sigma_uv=saved['sigma_uv'],**saved['settings']).to(DEVICE);m.load_state_dict(d)
                else:m,_=WorldTextureRuntime.load(authority,device=DEVICE)
                reloaded=fresh.reader.render(m,110)['road_rgb'];del m
            reload_error=float((reloaded-packet['road_rgb']).abs().max());assert reload_error<2e-5
            base=INPUTS.basis[110];K=tensor(base['joint']['K'],torch.bool)
            assert float((packet['joint_rgb']-tensor(base['joint']['baseline_rgb'])).abs()[:,K].max())==0
            assert torch.equal(packet['road_alpha'],tensor(base['road']['alpha']))
        assert all(np.isfinite(list(result['gradients'].values()))) and max(result['gradients'].values())>0
        timings[arm]=result['seconds'];checks.append(dict(arm=arm,full_state_next_step_max=diff,authority_reload_RGB_max=reload_error,finite_nonzero_gradient=True,save=True,real_resolution=[1152,768],seconds=time.monotonic()-t))
        write(dest/(arm+'_lifecycle_receipt.json'),checks[-1])
        del task,fresh;INPUTS.clear()
    task=new_r3('G_DETAIL',role='smoke')
    task.step('B');task.step('C');task.checkpoint();path=task.folder/'checkpoint.pt';task.step('C');expected=task.snapshot()
    fresh=new_r3('G_DETAIL',role='smoke');fresh.restore(torch.load(path,map_location='cpu',weights_only=False));r=fresh.step('C')
    diff=max_tree_difference(expected,fresh.snapshot());assert diff<2e-5
    checks.append(dict(arm='G_DETAIL',RGB_HF_Sobel_finite=True,full_state_next_step_max=diff,next_stage_C=True));timings['G_DETAIL']=r['seconds']
    numerical(role='smoke',maxFG=20,smoke_seed=fresh)
    estimate=2200*timings['ABC800']+3400*max(timings['M11'],timings['M12'])+2400*timings['G_DETAIL']+2000*timings['G_DETAIL']
    long_required=estimate>=3600
    # Real lifecycle extension is isolated and discarded, never a formal warm start.
    cycle=0
    while long_required and time.monotonic()-start<600:
        if fresh.cursor>=1190:fresh=new_r3('G_DETAIL',role='smoke')
        fresh.step('C');cycle+=1
        if cycle%50==0:fresh.checkpoint();persist_counters();status('long_smoke',elapsed_seconds=time.monotonic()-start,estimated_formal_seconds=estimate,PID=os.getpid())
    CTX.assert_frozen();persist_counters()
    receipt=dict(schema='common1152_actual_smoke.v1',verified=True,native_GPU_executed=True,checks=checks,
        numeric_workspace=read(dest/'numeric/receipt.json'),estimated_formal_seconds=estimate,
        hour_scale_ten_minute_required=long_required,actual_smoke_seconds=time.monotonic()-start,
        engineering_updates_separate=True,formal_warm_start=False,all_outputs_resolution=[1152,768])
    write(dest/'receipt.json',receipt);lease=read(ROOT/'gpu_lease.json');lease['formal_allowed']=True;write(ROOT/'gpu_lease.json',lease)
    print('SMOKE_PASSED',json.dumps(receipt),flush=True);del task,fresh;INPUTS.clear()

def formal():
    assert read(ROOT/'smoke/receipt.json')['verified']
    shared=ROOT/'shared';shared.mkdir(exist_ok=True)
    r1=Task('1','common_r1');export(r1)
    A=run_task(r1,200,export_end=False);save_torch(shared/'r1_A200.pt',A);del r1
    for arm in ['L800','ABC800']:
        task=Task('1',arm);run_task(task,800,prefix=A);del task;INPUTS.clear()
    task=Task('1','matched_I0');run_task(task,800);del task;INPUTS.clear()
    for kind,arms in [('avg',[('M00','observed',False),('M10','teacher',False)]),('pixel',[('M01','observed',False),('M11','teacher',False),('M12','teacher',True)])]:
        common=Task('2','common_'+kind,kind);export(common)
        A=run_task(common,200,export_end=False);save_torch(shared/('r2_'+kind+'_A200.pt'),A);del common
        for arm,gate,filtered in arms:
            task=Task('2',arm,kind,gate,filtered=filtered);run_task(task,800,prefix=A);del task;INPUTS.clear()
    initial=new_r3('F0');export(initial);del initial;INPUTS.clear()
    for arm in ['G_RGB','G_DETAIL']:
        task=new_r3(arm);run_task(task,1200);del task;INPUTS.clear()
    numerical()

def main():
    global CTX,INPUTS,COUNTERS,MANIFEST,NATIVE_PRIOR
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);start=time.monotonic();status('waiting_cpu_inputs',PID=os.getpid(),complete=False)
    while True:
        a=ROOT/'cpu_inputs/manifest.json'
        if a.exists() and read(a).get('complete') and (ROOT/'r1_inputs/receipt.json').exists():break
        failure=ROOT/'cpu_inputs/failed.json'
        if failure.exists():raise RuntimeError('CPU input preparation failed: '+failure.read_text())
        time.sleep(5)
    try:
        lease=read(ROOT/'gpu_lease.json');lease.update(active=True,owner=OWNER,gpu_authorized=True);write(ROOT/'gpu_lease.json',lease)
        if (ROOT/'compute_ledger.json').exists():
            old=read(ROOT/'compute_ledger.json');COUNTERS=old['counters'];NATIVE_PRIOR=old.get('native_process_total',0)
        CTX=context();INPUTS=Inputs()
        if (ROOT/'export_manifest.json').exists():MANIFEST=read(ROOT/'export_manifest.json')['rows']
        if not (ROOT/'smoke/receipt.json').exists() or not read(ROOT/'smoke/receipt.json').get('verified'):
            status('actual_smoke',PID=os.getpid());smoke()
        status('training',PID=os.getpid(),actual_smoke_passed=True);formal()
        CTX.assert_frozen();persist_counters();CTX.close();CTX=None
        status('CPU_bench',PID=os.getpid());manifest=read(ROOT/'export_manifest.json');manifest['complete']=True;write(ROOT/'export_manifest.json',manifest)
        cmd=[sys.executable,str(ROOT/'bench/evaluate_and_package.py'),'--manifest',str(ROOT/'export_manifest.json')]
        subprocess.run(cmd,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        bundle=ROOT/'final_bundle.tar.gz'
        if not bundle.is_file():raise FileNotFoundError('CPU packager did not produce final_bundle.tar.gz')
        receipt=dict(schema='common1152_queue_complete.v1',complete=True,endpoints=11,export_rows=len(MANIFEST),
            seconds=time.monotonic()-start,counters=COUNTERS,GPU_keep_on=True,production_promoted=False,
            human_verdict=None,final_bundle_SHA256=sha(bundle),bundle_sha256=sha(bundle),scientific_scope='newteacher all-qualified fitting, not held validation')
        write(ROOT/'receipt.json',receipt);write(ROOT/'status.json',dict(status='complete',state='complete',phase='done',PID=os.getpid(),**receipt))
    except Exception:
        error=traceback.format_exc();persist_counters();write(ROOT/'failure.json',dict(error=error,counters=COUNTERS,PID=os.getpid(),time=time.time()))
        write(ROOT/'status.json',dict(status='failed',state='failed',phase='failed',complete=False,error=error,PID=os.getpid()));raise
    finally:
        if CTX is not None:CTX.close()

if __name__=='__main__':main()
