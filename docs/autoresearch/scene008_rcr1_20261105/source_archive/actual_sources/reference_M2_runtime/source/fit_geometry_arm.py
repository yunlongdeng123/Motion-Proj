"""M2/M3 shared SH0 appearance fit: only the geometry input differs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time,hashlib,math
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz,save_npz,vstack
from scipy.ndimage import binary_erosion
from deterministic_csr import Matrix
from sh_asset import SHAsset,C0
from scene_runtime import Runtime,raster,tagged
from linear_weights import design
import scene_runtime as _runtime_module
assert Path(_runtime_module.__file__).resolve().parent==Path(__file__).resolve().parent,'Wrong runtime source binding'

RUN=Path(__file__).resolve().parent.parent;PARENT=RUN.parent/'scene008_a6_complete_asset_20261105'
HELD=(89,95,101,105,113,121,125,130,137);BUILD=tuple(r for r in range(89,138) if r not in HELD)
STEPS=2000;LAMBDA=.001
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def contract(arm,geometry):
    return dict(arm=arm,geometry=str(geometry),geometry_sha256=sha(geometry),BUILD=BUILD,internal_HELD=HELD,
      target='Exact same27429 original SH0 fit pixels; no new predicted-coverage selection',
      material='SH0 RGB only; all higher SH exactlyzero; shared across all frames and states',
      formal_fullbatch_steps=STEPS,ridge_lambda=LAMBDA,optimizer='Deterministic projected FISTA, per-point diagonal bound',
      objective='mean_pixel squaredRGB + .001 mean_point squaredRGB deviation from fixed prior',
      c0='M2: same BUILD reliable native-colour rule; M3: exact saved M2 per-row c0 and flags reused',
      initialization='Fixed c0 for bothM2 andM3; M3 does not start from M2 fitted material',
      forbidden=['Per-part support rules','New opacity fit','Geometry changes during colour fitting','A37 or road RGB supervision'],
      observation_scope='Internal same-scene mechanism control, not independent geometry/generalization')

@torch.no_grad()
def collect(arm,geometry):
    dest=RUN/arm;dest.mkdir(exist_ok=True);rt=Runtime();asset=SHAsset.from_geometry(geometry).cuda();N=len(asset.xyz)
    matrices=[];targets=[];mass=np.zeros(N);views=np.zeros(N,np.int64)
    prior_sum=np.zeros((N,3));prior_mass=np.zeros(N);samples=[];records=[]
    for raw in BUILD:
        d=rt.raw(raw);mask=binary_erosion(d['a6_visible'].cpu().numpy(),iterations=1)
        lin=design(d['cam'],rt.pipe,asset.rows(d['pose'],d['cam'],rt.pipe,1,precomputed=True),torch.as_tensor(mask,device='cuda'))
        W=lin['matrix'];A=W.multiply(1/np.maximum(lin['alpha'][:,None],1e-5)).tocsr()
        pixel=lin['pixels'];target=d['rgb'].flatten(1)[:,pixel].T.cpu().numpy();native=rt.native_reference(raw)
        na=native['alpha'][0].flatten()[pixel].cpu().numpy()
        nc=native['rgb_unpremultiplied'].flatten(1)[:,pixel].T.cpu().numpy()
        reliable=(na>=.9)&(np.abs(nc-target).mean(1)<=.05)
        prior_sum+=W[reliable].T@nc[reliable];prior_mass+=np.asarray(W[reliable].sum(0)).ravel()
        fm=np.asarray(W.sum(0)).ravel();mass+=fm;views+=fm>=.05
        matrices.append(A);targets.append(target);samples.append(target[::max(1,len(target)//1000)])
        records.append(dict(raw=raw,fit=len(pixel),alpha_min=float(lin['alpha'].min()),low_alpha_under02=int((lin['alpha']<.2).sum()),
          colour_maxabs=lin['colour_maxabs'],alpha_maxabs=lin['alpha_maxabs'],nnz=A.nnz))
        print(json.dumps(dict(stage='collect',arm=arm,**records[-1])),flush=True)
        del d,lin,native;rt._data.clear();torch.cuda.empty_cache()
    A=vstack(matrices,format='csr');target=np.concatenate(targets).astype(np.float32)
    with np.load(PARENT/'appearance/appearance_inputs.npz') as old:assert np.array_equal(target,old['target'].astype(np.float32))
    median=np.median(np.concatenate(samples),axis=0);c0=np.tile(median,(N,1));support=prior_mass>=.05
    c0[support]=prior_sum[support]/prior_mass[support,None];c0=np.clip(c0,0,1).astype(np.float32)
    inferred=(views<2)|(mass<1)
    if arm=='m3':
        with np.load(RUN/'m2/inputs.npz') as fixed:
            assert fixed['prior'].shape==(N,3)
            c0=fixed['prior'].copy();inferred=fixed['inferred'].copy()
    save_npz(dest/'design.npz',A)
    np.savez_compressed(dest/'inputs.npz',target=target,prior=c0,inferred=inferred,mass=mass,views=views,
          geometry_path=np.asarray(str(geometry)),geometry_hash=np.asarray(asset.geometry_hash))
    assert rt.modelunchanged() and asset.assert_geometry_unchanged()
    write(dest/'collection.json',dict(frames=records,rows=A.shape[0],columns=N,nnz=A.nnz,
       geometry_hash=asset.geometry_hash,design_sha256=sha(dest/'design.npz'),inputs_sha256=sha(dest/'inputs.npz'),
       exactly_same_original_target=True,source_raw_frames=BUILD,prior_reused_from_M2=arm=='m3'))

class Solver:
    def __init__(self,arm):
        dest=RUN/arm;A=load_npz(dest/'design.npz').astype(np.float32);self.N=A.shape[1];self.M=A.shape[0]
        upper=np.asarray(abs(A).T@np.asarray(abs(A).sum(1)).ravel()).ravel()
        self.D=torch.tensor(2*(upper/self.M+LAMBDA/self.N),device='cuda')[:,None]
        self.A=Matrix(A);self.AT=Matrix(A.T)
        with np.load(dest/'inputs.npz') as x:
            self.target=torch.tensor(x['target'],device='cuda');self.prior=torch.tensor(x['prior'],device='cuda')
            self.inferred=x['inferred'].copy();path=str(x['geometry_path'])
        self.asset=SHAsset.from_geometry(path).cuda()
    def prediction(self,x):return self.A(x)
    def objective(self,x):return float((self.A(x)-self.target).double().square().sum()/self.M+
        LAMBDA*(x-self.prior).double().square().sum()/self.N)
    def gradient(self,x):return 2*self.AT(self.A(x)-self.target)/self.M+2*LAMBDA/self.N*(x-self.prior)
    def begin(self):
        x=self.prior.clone();return dict(x=x,y=x.clone(),t=1.,loss=self.objective(x),step=0,restarts=0)
    def step(self,state):
        x,y,t=state['x'],state['y'],state['t'];z=(y-self.gradient(y)/self.D).clamp(0,1)
        loss=self.objective(z);restarts=state['restarts']
        if loss>state['loss']+1e-9:t=1.;z=(x-self.gradient(x)/self.D).clamp(0,1);loss=self.objective(z);restarts+=1
        assert math.isfinite(loss) and bool(torch.isfinite(z).all())
        tn=(1+math.sqrt(1+4*t*t))/2
        return dict(x=z,y=z+(t-1)/tn*(z-x),t=tn,loss=loss,step=state['step']+1,restarts=restarts)
    def export(self,state,path):
        coefficients=self.asset.coefficients.new_zeros((self.N,16,3));coefficients[:,0]=(state['x']-.5)/C0
        self.asset.set_coefficients(coefficients,self.inferred);self.asset.metadata.update(fit_material='SH0_only',higher_SH_frozen_zero=True,steps=state['step'])
        return self.asset.export(path)

@torch.no_grad()
def smoke(arm):
    dest=RUN/arm/'smoke';dest.mkdir(exist_ok=True);start=time.time();s=Solver(arm);state=s.begin()
    for _ in range(32):state=s.step(state)
    torch.save(state,dest/'full_state.pt');restored=torch.load(dest/'full_state.pt',map_location='cuda',weights_only=False)
    next1=s.step(state);next2=s.step(restored);assert all(torch.equal(next1[k],next2[k]) for k in ('x','y'))
    result=s.export(state,dest/'asset.npz');fresh=SHAsset.from_geometry(dest/'asset.npz').cuda();rt=Runtime()
    for raw in (105,113,130):
        d=rt.raw(raw);k=rt.render_scene(raw,fresh);z=rt.render_scene(raw,fresh,delete=True)
        assert k['meta']['native_a6_rows_input']==0 and z['meta']['native_a37_packed_rows']==0
        rt._data.clear()
    assert fresh.assert_geometry_unchanged() and rt.modelunchanged()
    seconds=time.time()-start
    write(dest/'receipt.json',dict(status='PASS',seconds=seconds,updates_extra=32,formal_updates=0,
      exact_full_state_resume=True,next_stage_scene_entry=True,exported=result,projected_upper_formal_seconds=seconds/32*STEPS))
    if seconds/32*STEPS>=3600:raise RuntimeError('Long run: real10min smoke required before formal')
    print(json.dumps(dict(arm=arm,stage='smoke',status='PASS')),flush=True)

@torch.no_grad()
def fit(arm):
    dest=RUN/arm;assert json.loads((dest/'smoke/receipt.json').read_text())['status']=='PASS'
    start=time.time();s=Solver(arm);state=s.begin();initial=state['loss'];trace=[]
    for _ in range(STEPS):
        state=s.step(state)
        if state['step']%100==0:
            row=dict(step=state['step'],objective=state['loss'],seconds=time.time()-start,restarts=state['restarts']);trace.append(row);print(json.dumps(row),flush=True)
        if state['step'] in (1000,STEPS):torch.save(state,dest/f'checkpoint{state["step"]:06d}.pt')
    exported=s.export(state,dest/'complete_a6.npz')
    write(dest/'fit_receipt.json',dict(status='PASS',formal_steps=STEPS,smoke_steps_extra=32,geometry_updates_during_fit=0,
       objective_initial=initial,objective_final=state['loss'],seconds=time.time()-start,exported=exported,
       BUILD_normalized_RGB_MAE=float((s.A(state['x'])-s.target).abs().double().mean()),
       projected_update_maxabs=float(((state['x']-s.gradient(state['x'])/s.D).clamp(0,1)-state['x']).abs().max()),trace=trace))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','smoke','fit']);p.add_argument('arm',choices=['m2','m3']);p.add_argument('--geometry');args=p.parse_args()
    dest=RUN/args.arm;dest.mkdir(exist_ok=True)
    if args.stage=='collect':
        geometry=Path(args.geometry).resolve();write(dest/'contract.json',contract(args.arm,geometry));collect(args.arm,geometry)
    elif args.stage=='smoke':smoke(args.arm)
    else:fit(args.arm)

if __name__=='__main__':main()
