"""Fixed same-seen native objective callbacks for the independent numerical phase."""
from pathlib import Path
import copy,hashlib,json,random,sys,time
import numpy as np

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=C/'road_teacher_detail_cpu_20261119_r1'
B=Q/'gpu_20261119/native_capacity_20261120_r1';ROOT=B/'numerical_completion'
FRAMES=(86,110)

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class NumericalRuntime:
    def __init__(self,output,*,role,owner,ledger_file,maxFG):
        import torch
        sys.path[:0]=[str(B/'code'),str(B/'gs_oracle/code'),str(ROOT/'code')]
        from native_capacity_inputs import make_context,real_frames
        from native_capacity_loss import NativeCapacityLoss
        from fp64_reduction_loss import FP64ReductionLoss
        from run_registered_experiments import canonical_loss_snapshot
        self.torch=torch;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True);self.role=role
        self.ledger_file=Path(ledger_file);self.maxFG=int(maxFG)
        self.ledger=read(self.ledger_file)if self.ledger_file.exists()else dict(FG=0,nativeF=0,nativeB=0,torchBackward=0,accepted=0,rows=[])
        self.lease=self.output/'gpu_lease.json'
        write(self.output/'authorization.json',dict(GPU_authorized=True,phase='numerical_'+role,root_authorization='Root independently authorized unchanged boxed seen objective numerical completion and complete workspace lifecycle',
            role=role,maxFG=maxFG,main4arms_unchanged=True,keepGPUon=True))
        write(self.lease,dict(gpu_authorized=True,active=True,owner=owner,authorization=str(self.output/'authorization.json')))
        self.ctx=make_context(self.lease,owner);torch.set_num_threads(1)
        frames,frame_binding=real_frames('seen')
        self.frame_alpha={r:torch.as_tensor(frames[r]['alpha'],device=self.ctx.base_rgb.device)for r in FRAMES}
        normpath=B/'formal_r1/common_loss_snapshot.json';normalizers=canonical_loss_snapshot(read(normpath))
        self.original=NativeCapacityLoss(frames,device='cuda');self.original.restore(normalizers)
        self.precise=FP64ReductionLoss(frames,device='cuda');self.precise.restore(normalizers)
        if self.original.snapshot()!=self.precise.snapshot():raise RuntimeError('normalizers/inputs changed under reduction-only precision')
        seed_asset=B/'formal_r1/G_DETAIL/asset'
        with np.load(seed_asset/'road.npz',allow_pickle=False)as z:
            for key,value in self.ctx.road.items():
                if key!='reference_srgb'and not np.array_equal(z[key],value):raise ValueError('seed changed fixed geometry/coverage:'+key)
            seed=z['reference_srgb'][self.ctx.active_world_rows].copy()
        with np.load(B/'formal_r1/common_initial_ci.npz',allow_pickle=False)as z:
            self.eligible=z['eligible_active'].copy()
            if not np.array_equal(z['active_world_rows'],self.ctx.active_world_rows):raise ValueError('world row identity differs')
        if self.eligible.dtype!=bool or int(self.eligible.sum())!=89887:raise ValueError('scope differs')
        self.pos=torch.as_tensor(np.flatnonzero(self.eligible),device=self.ctx.base_rgb.device)
        self.seed=torch.as_tensor(seed,device=self.ctx.base_rgb.device)
        if not torch.equal(self.seed[~torch.as_tensor(self.eligible,device=self.seed.device)],self.ctx.base_rgb[~torch.as_tensor(self.eligible,device=self.seed.device)]):raise ValueError('readonly seed changed')
        self.x0=seed[self.eligible].astype(np.float64)
        paths=[Path(__file__),ROOT/'code/checkpointed_lbfgsb.py',ROOT/'code/fp64_reduction_loss.py',
            ROOT/'code/run_numerical_completion.py',ROOT/'protocol.json',
            B/'code/native_capacity_inputs.py',B/'code/native_capacity_loss.py',normpath,seed_asset/'road.npz',
            seed_asset/'provenance.npz',seed_asset/'persistent/scene.json',seed_asset/'persistent/proof.json',B/'formal_r1/common_initial_ci.npz']
        self.binding={str(p):sha(p)for p in paths};self.binding.update(frame_binding['files'])
        original_binding=read(B/'formal_r1/source_freeze.json')['binding']
        if any(sha(p)!=h for p,h in original_binding.items()):raise RuntimeError('original registered inputs/native code changed')
        self.binding.update(original_binding)
        self.identity=dict(role=role,precision='originalE_Gaussian_Sobel_FP32; square_mean_scalar_FP64',binding=self.binding,
            scope_sha256=hashlib.sha256(self.ctx.active_world_rows[self.eligible].astype('<i8').tobytes()).hexdigest())
        self.cache={};write(self.output/'source_binding.json',self.identity)

    def _record(self):write(self.ledger_file,self.ledger)

    def _predictions(self,x):
        torch=self.torch;p=torch.as_tensor(x,device=self.seed.device,dtype=self.seed.dtype).detach().requires_grad_(True)
        colors=self.seed.index_copy(0,self.pos,p.reshape(-1,3));pred={};alpha={}
        for raw in FRAMES:
            self.ctx.require_lease();value=self.ctx.render_road(raw,colors);self.ledger['nativeF']+=1
            if not torch.equal(value['alpha'].detach().squeeze(),self.frame_alpha[raw]):raise RuntimeError('native alpha changed')
            pred[raw]=value['rgb'];alpha[raw]=value['alpha'].detach()
        return p,colors,pred,alpha

    def qualify_precision(self):
        torch=self.torch
        if self.ledger['FG']+2>self.maxFG:raise RuntimeError('smoke FG cap before precision qualification')
        p,colors,pred,alpha=self._predictions(self.x0)
        f32,_=self.original.objective(pred,'detail');f64,_=self.precise.objective(pred,'detail')
        g32=torch.autograd.grad(f32,p,retain_graph=True)[0];g64=torch.autograd.grad(f64,p)[0]
        self.ledger['FG']+=2;self.ledger['nativeB']+=4;self.ledger['torchBackward']+=2
        value_error=abs(float(f32)-float(f64))/max(abs(float(f64)),1e-12)
        gradient_error=float((g32.double()-g64.double()).norm())/max(float(g64.double().norm()),1e-12)
        result=dict(point='fixed registered G_DETAIL final ci, no held read',function32=float(f32),function64=float(f64),
            function_relative=value_error,gradient_L2_relative=gradient_error,gradient_max_abs=float((g32-g64).abs().max()),
            same_prediction_reused=True,original_F0_normalizers_unchanged=True,passed=value_error<=1e-6 and gradient_error<=1e-5)
        self._record();write(self.output/'precision_qualification.json',result)
        if not result['passed']:raise RuntimeError('reduction precision qualification unsafe; no method selection/restart')
        return result

    def evaluate(self,x):
        torch=self.torch
        if self.ledger['FG']>=self.maxFG:raise RuntimeError('registered total FG cap exhausted')
        torch.cuda.synchronize();t=time.perf_counter();p,colors,pred,alpha=self._predictions(x)
        f,terms=self.precise.objective(pred,'detail');g=torch.autograd.grad(f,p)[0];torch.cuda.synchronize()
        self.ledger['FG']+=1;self.ledger['nativeB']+=2;self.ledger['torchBackward']+=1
        value=float(f.detach());gradient=g.detach().cpu().numpy().astype(np.float64)
        self.ledger['rows'].append(dict(FG=self.ledger['FG'],objective=value,seconds=time.perf_counter()-t))
        self.cache=dict(x=np.asarray(x).copy(),colors=colors.detach().cpu().numpy(),pred={r:P.detach().cpu().numpy()for r,P in pred.items()},
            alpha={r:a.cpu().numpy()for r,a in alpha.items()})
        self._record();return value,gradient,dict(role='same_seen_DETAIL_numerical_completion')

    def rng(self):
        t=self.torch
        return dict(python=random.getstate(),numpy=copy.deepcopy(np.random.get_state()),torch=t.get_rng_state().clone(),cuda=[s.clone()for s in t.cuda.get_rng_state_all()])

    def restore_rng(self,s):
        t=self.torch;random.setstate(s['python']);np.random.set_state(copy.deepcopy(s['numpy']));t.set_rng_state(s['torch'].clone());t.cuda.set_rng_state_all([x.clone()for x in s['cuda']])

    def close(self):
        try:
            self.ctx.assert_frozen()
            if any(sha(p)!=h for p,h in self.binding.items()):raise RuntimeError('numerical phase input/source changed')
        finally:
            self.ctx.close();state=read(self.lease);state.update(active=False,ended_at=time.time());write(self.lease,state)
