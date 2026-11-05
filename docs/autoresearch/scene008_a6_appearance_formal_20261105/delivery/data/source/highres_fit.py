"""Observed-only full-resolution material control. PVD/model weights frozen."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import argparse, copy, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.sparse import save_npz, load_npz, vstack
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD=CAM/'scene008_a6_asset_controls_20261105'
RUN=CAM/'scene008_a6_appearance_formal_20261105'
RAW=CAM/'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137'
sys.path.insert(0,str(OLD/'source'))
from scene_runtime import Runtime,raster,tagged
from sh_asset import SHAsset,C0
from linear_weights import design
from deterministic_csr import Matrix
HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(r for r in range(89,138) if r not in HELD)
STEPS=1000
RIDGE=.001

def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def camera_hr(cam):
    c=copy.copy(cam)
    c.image_width=1920;c.image_height=1280
    for key in ('fx','fy','cx','cy'): setattr(c,key,getattr(cam,key)*4)
    # Full-image FoV, projection, camera center and time are unchanged.
    return c

@torch.no_grad()
def collect():
    dest=RUN/'highres';dest.mkdir(exist_ok=False);rt=Runtime()
    asset=SHAsset.from_geometry(OLD/'m2/complete_a6.npz').cuda()
    mats=[];targets=[];raws=[];records=[];start=time.time()
    for raw in BUILD:
        d=rt.raw(raw);c=camera_hr(d['cam'])
        # Exactly the old extra-eroded visible internal domain, lifted 4x.
        # The old high-resolution body mask is itself nearest-neighbor lifted.
        low=binary_erosion(d['a6_visible'].cpu().numpy(),iterations=1)
        mask=np.repeat(np.repeat(low,4,axis=0),4,axis=1)
        img=np.asarray(Image.open(RAW/'image/image_0'/f'0008{raw:03d}.png').convert('RGB'))
        assert img.shape==(1280,1920,3)
        lin=design(c,rt.pipe,asset.rows(d['pose'],c,rt.pipe,1,precomputed=True),torch.as_tensor(mask,device='cuda'))
        use=lin['alpha']>.9
        A=lin['matrix'][use].multiply(1/lin['alpha'][use,None]).tocsr();A.eliminate_zeros()
        pixels=lin['pixels'][use];target=img.reshape(-1,3)[pixels].astype(np.float32)/255
        mats.append(A);targets.append(target);raws.append(np.full(len(target),raw,np.int16))
        records.append(dict(raw=raw,rows=len(target),nnz=A.nnz,RGB_sha256=sha(RAW/'image/image_0'/f'0008{raw:03d}.png'),alpha_min=float(lin['alpha'][use].min())))
        print(json.dumps(dict(stage='HR_collect',raw=raw,rows=len(target))),flush=True)
        rt._data.clear();torch.cuda.empty_cache()
    A=vstack(mats,format='csr');save_npz(dest/'design.npz',A)
    np.savez_compressed(dest/'inputs.npz',target=np.concatenate(targets),raw=np.concatenate(raws),unknown=A.getnnz(axis=0)==0)
    write(dest/'collection.json',dict(status='PASS',frames=records,rows=A.shape[0],columns=A.shape[1],nnz=A.nnz,
        source_resolution=[1920,1280],old_resolution=[480,320],source_mask='Same old eroded visible interiors lifted4x; no new boundary claims',
        no_HELD_RGB=True,geometry_hash=asset.geometry_hash,seconds=time.time()-start))
    assert rt.modelunchanged() and asset.assert_geometry_unchanged()

class Solver:
    def __init__(self):
        A=load_npz(RUN/'highres/design.npz').astype(np.float32);self.N=A.shape[1];self.M=A.shape[0]
        self.asset=SHAsset.from_geometry(OLD/'m2/complete_a6.npz').cuda()
        z=np.load(RUN/'highres/inputs.npz');self.unknown=z['unknown'].copy()
        self.target=torch.as_tensor(z['target'],device='cuda')
        self.prior=self.asset.rgb.detach().clone()
        upper=np.asarray(A.T@np.asarray(A.sum(1)).ravel()).ravel()
        self.D=torch.tensor(2*(upper/self.M+RIDGE/self.N),device='cuda',dtype=torch.float32)[:,None]
        self.A=Matrix(A);self.AT=Matrix(A.T)
    def objective(self,x):return float((self.A(x)-self.target).double().square().sum()/self.M+RIDGE*(x-self.prior).double().square().sum()/self.N)
    def gradient(self,x):return 2*self.AT(self.A(x)-self.target)/self.M+2*RIDGE/self.N*(x-self.prior)
    def begin(self):
        x=self.prior.clone();return dict(x=x,y=x.clone(),step=0,t=1.,loss=self.objective(x),restarts=0)
    def step(self,s):
        x,y,t=s['x'],s['y'],s['t'];z=(y-self.gradient(y)/self.D).clamp(0,1);loss=self.objective(z);restarts=s['restarts']
        if loss>s['loss']+1e-9:
            t=1.;z=(x-self.gradient(x)/self.D).clamp(0,1);loss=self.objective(z);restarts+=1
        assert math.isfinite(loss) and torch.isfinite(z).all()
        assert torch.equal(z[torch.as_tensor(self.unknown,device='cuda')],self.prior[torch.as_tensor(self.unknown,device='cuda')])
        tn=(1+math.sqrt(1+4*t*t))/2
        return dict(x=z,y=z+(t-1)/tn*(z-x),step=s['step']+1,t=tn,loss=loss,restarts=restarts)
    def export(self,s,p):
        coeff=self.asset.coefficients.clone();coeff[:,0]=(s['x']-.5)/C0
        self.asset.set_coefficients(coeff)
        self.asset.metadata['highres_observation_control']=dict(resolution=[1920,1280],steps=s['step'],PVD_weight_updates=0,geometry_updates=0,source_domain='Same BUILD40 conservative visible interiors',inferred_colours_not_truth=True)
        return self.asset.export(p)

@torch.no_grad()
def fit(smoke=False):
    dest=RUN/'highres'/('smoke' if smoke else 'formal');dest.mkdir(exist_ok=False)
    start=time.time();s=Solver();state=s.begin();initial=state['loss'];load=time.time()-start
    count=32 if smoke else STEPS
    if not smoke:
        receipt=json.loads((RUN/'highres/smoke/receipt.json').read_text());assert receipt['status']=='PASS'
        assert receipt['source_sha256']==sha(__file__)
    for _ in range(count):
        state=s.step(state)
        if state['step']%100==0:print(json.dumps(dict(stage='HR_fit',step=state['step'],loss=state['loss'],seconds=time.time()-start)),flush=True)
        if state['step'] in (500,1000):torch.save(state,dest/f'checkpoint{state["step"]:06d}.pt')
    torch.save(state,dest/'full_state.pt')
    restored=torch.load(dest/'full_state.pt',map_location='cuda',weights_only=False)
    n1=s.step(state);n2=s.step(restored);assert all(torch.equal(n1[k],n2[k]) for k in ('x','y'))
    result=s.export(state,dest/'complete_a6.npz');fresh=SHAsset.from_geometry(dest/'complete_a6.npz').cuda()
    rt=Runtime();d=rt.raw(106);out=rt.render_scene(106,fresh,delete=True);assert out['meta']['native_a6_rows_input']==0
    assert rt.modelunchanged() and fresh.assert_geometry_unchanged()
    elapsed=time.time()-start;projected=max(0,elapsed-load)/count*STEPS+load+120
    status='PASS' if not smoke or projected<3600 else 'REQUIRES_REAL_10MIN_SMOKE'
    write(dest/'receipt.json',dict(status=status,colour_fit_steps=count,neural_training_steps=0,PVD_weight_updates=0,geometry_updates=0,
        seconds=elapsed,projected_formal_seconds=projected,full_state_exact_resume=True,next_stage_scene=True,source_sha256=sha(__file__),
        loss_initial=initial,loss_final=state['loss'],BUILD_RGB_MAE=float((s.A(state['x'])-s.target).abs().double().mean()),
        unknown=int(s.unknown.sum()),peak_vram=torch.cuda.max_memory_allocated(),asset=result))
    print(json.dumps(dict(stage='HR_smoke' if smoke else 'HR_fit_done',status=status,seconds=elapsed,projected_formal_seconds=projected)),flush=True)
    if status!='PASS':raise RuntimeError(status)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','smoke','fit']);a=p.parse_args()
    if a.stage=='collect':collect()
    else:fit(a.stage=='smoke')
