"""M1: fixed full PVD geometry, constrained canonical SH3 material only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time,math
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix,load_npz,save_npz,vstack
from deterministic_csr import Matrix
from sh_asset import SHAsset,basis3
from scene_runtime import Runtime,raster,tagged

RUN=Path(__file__).resolve().parent.parent
PARENT=RUN.parent/'scene008_a6_complete_asset_20261105'
HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(r for r in range(89,138) if r not in HELD)
BOUND=math.sqrt(15/(4*math.pi));LAMBDA=.001;STEPS=2000;C0=.28209479177387814

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def contract():
    return dict(arm='M1_fixed_geometry_SH3',parent=str(PARENT),geometry='Exactly original complete2048 PVD asset',
      build=BUILD,internal_held=HELD,formal_fullbatch_steps=STEPS,geometry_updates=0,
      target='Identical27429 BUILD eroded visible a6 RGB pixels and normalized alpha*T rows as SH0',
      material='Canonical-local SH3:2048x16x3 coefficients; rotate to world for native raster',
      objective='sumRGB_squared/M + .001/N * sum((DC_RGB-c0)^2 + high_SH_squared/(4pi))',
      optimizer='Deterministic diagonally preconditioned projected FISTA; per-point scalar metric; monotone restart',
      constraint='sqrt(15/(4pi))*L2(high_SH_channel)<=min(DC_RGB,1-DC_RGB), all directions RGB in[0,1]',
      initial='Previous fitted SH0 asset; same native-conditioned c0; all high SH zero',
      inferred='Original source flags unchanged; extra directional coefficients are fitted only with BUILD data',
      forbidden=['Geometry adjustment','Opacity adjustment','New fit masks','HELD appearance fitting','Native a6 renderer rows'],
      evaluation='Identical all49 fixed RGB/depth/silhouette and paired keep/delete preservation metrics',
      limitations='Internal appearance control, full-scene geometry context; no independent hidden truth or generalization')

def project(theta):
    """Exact Euclidean projection per point/channel onto the SH colour cone.

    By the addition theorem sum(Y_l^2,l=1..3)=15/(4pi). Feasible(d,||h||)
    is the triangle(0,0),(.5,.5/BOUND),(1,0); rotate the norm back to h.
    """
    d=theta[:,0];h=theta[:,1:];r=torch.linalg.vector_norm(h,dim=1)
    feasible=(d>=0)&(d<=1)&(BOUND*r<=torch.minimum(d,1-d))
    # Nearest points on the three triangle edges, including their endpoints.
    candidates=[]
    candidates.append((d.clamp(0,1),torch.zeros_like(r)))
    bx=.5;by=.5/BOUND;den=bx*bx+by*by
    t=((d*bx+r*by)/den).clamp(0,1);candidates.append((t*bx,t*by))
    t=(((d-1)*(-bx)+r*by)/den).clamp(0,1);candidates.append((1-t*bx,t*by))
    ds=torch.stack([p[0] for p in candidates]);rs=torch.stack([p[1] for p in candidates])
    which=((ds-d).square()+(rs-r).square()).argmin(0,keepdim=True)
    dp=torch.gather(ds,0,which)[0];rp=torch.gather(rs,0,which)[0]
    dp=torch.where(feasible,d,dp);rp=torch.where(feasible,r,rp)
    output=theta.clone();output[:,0]=dp;output[:,1:]=h*(rp/r.clamp_min(1e-30))[:,None]
    return output

def coeffs(theta):
    out=theta.clone();out[:,0]=(out[:,0]-.5)/C0;return out

@torch.no_grad()
def build_design():
    dest=RUN/'m1';dest.mkdir(exist_ok=True);rt=Runtime();asset=SHAsset.from_geometry(PARENT/'appearance/complete_a6_fitted.npz').cuda()
    A0=load_npz(PARENT/'appearance/appearance_design.npz');records=json.loads((PARENT/'appearance/collection.json').read_text())['frames']
    matrices=[];offset=0;direction_rows=[]
    for record in records:
        raw=record['raw'];assert raw in BUILD;d=rt.raw(raw);pose=d['pose'];R=pose[:3,:3]
        xyz=asset.xyz@R.T+pose[:3,3]
        directions=torch.nn.functional.normalize((xyz-d['cam'].camera_center)@R,dim=1)
        b=basis3(directions).cpu().numpy();b[:,0]=1
        count=record['fit'];a=A0[offset:offset+count].tocoo();offset+=count
        rows=np.repeat(a.row,16);cols=(a.col[:,None]*16+np.arange(16)).ravel()
        values=(a.data[:,None]*b[a.col]).ravel()
        matrices.append(coo_matrix((values,(rows,cols)),shape=(count,2048*16)).tocsr())
        direction_rows.append(dict(raw=raw,fit=count));rt._data.clear()
    assert offset==A0.shape[0] and rt.modelunchanged()
    A=vstack(matrices,format='csr');A.eliminate_zeros();save_npz(dest/'design_SH3.npz',A)
    write(dest/'design.json',dict(shape=A.shape,nnz=A.nnz,parent_design_sha256=sha(PARENT/'appearance/appearance_design.npz'),
        design_sha256=sha(dest/'design_SH3.npz'),frames=direction_rows,geometry_hash=asset.geometry_hash,
        held_used_for_fit=False))
    print(json.dumps(dict(stage='design',shape=A.shape,nnz=A.nnz)),flush=True)

class Solver:
    def __init__(self):
        A=load_npz(RUN/'m1/design_SH3.npz').astype(np.float32);N=A.shape[1]//16;self.N=N;self.M=A.shape[0]
        abs_A=abs(A);row_sum=np.asarray(abs_A.sum(1)).ravel()
        upper=np.asarray(abs_A.T@row_sum).reshape(N,16).max(1)
        self.D=torch.tensor(2*(upper/self.M+LAMBDA/N),device='cuda')[:,None,None]
        self.A=Matrix(A);self.AT=Matrix(A.T)
        with np.load(PARENT/'appearance/appearance_inputs.npz') as saved:
            self.target=torch.tensor(saved['target'],dtype=torch.float32,device='cuda')
            self.prior=torch.zeros((N,16,3),device='cuda');self.prior[:,0]=torch.tensor(saved['prior'],dtype=torch.float32,device='cuda')
        self.weight=torch.full((1,16,1),1/(4*math.pi),device='cuda');self.weight[:,0]=1
        source=SHAsset.from_geometry(PARENT/'appearance/complete_a6_fitted.npz')
        self.initial=self.prior.clone();self.initial[:,0]=source.rgb.cuda()
        self.asset=source.cuda();self.best_loss=None
    def prediction(self,x):return self.A(x.reshape(-1,3))
    def objective(self,x):
        residual=self.prediction(x)-self.target
        return float(residual.double().square().sum()/self.M+LAMBDA*(self.weight*(x-self.prior).square()).double().sum()/self.N)
    def gradient(self,x):
        residual=self.prediction(x)-self.target
        return 2*self.AT(residual).reshape(self.N,16,3)/self.M+2*LAMBDA/self.N*self.weight*(x-self.prior)
    def begin(self):
        x=project(self.initial);return dict(x=x,y=x.clone(),t=1.,step=0,loss=self.objective(x),restarts=0)
    def step(self,state):
        x,y,t=state['x'],state['y'],state['t']
        candidate=project(y-self.gradient(y)/self.D);loss=self.objective(candidate);restarts=state['restarts']
        if loss>state['loss']+1e-9:
            t=1.;candidate=project(x-self.gradient(x)/self.D);loss=self.objective(candidate);restarts+=1
        assert math.isfinite(loss) and bool(torch.isfinite(candidate).all())
        tnext=(1+math.sqrt(1+4*t*t))/2
        return dict(x=candidate,y=candidate+(t-1)/tnext*(candidate-x),t=tnext,step=state['step']+1,
                    loss=loss,restarts=restarts)
    def export(self,state,path):
        self.asset.set_coefficients(coeffs(state['x']));self.asset.metadata.update(contract=contract(),solver_step=state['step'])
        return self.asset.export(path)

@torch.no_grad()
def smoke():
    dest=RUN/'m1/smoke';dest.mkdir(exist_ok=True);start=time.time();s=Solver();state=s.begin()
    for _ in range(32):state=s.step(state)
    torch.cuda.synchronize();steps_seconds=time.time()-start
    torch.save(state,dest/'full_state.pt');restored=torch.load(dest/'full_state.pt',map_location='cuda',weights_only=False)
    assert all(torch.equal(state[k],restored[k]) for k in ('x','y'))
    original_next=s.step(state);restored_next=s.step(restored)
    assert all(torch.equal(original_next[k],restored_next[k]) for k in ('x','y'))
    exported=s.export(state,dest/'asset_SH3.npz');fresh=SHAsset.from_geometry(dest/'asset_SH3.npz').cuda();rt=Runtime();records=[]
    for raw in (105,113,130):
        data=rt.raw(raw);rows=fresh.rows(data['pose'],data['cam'],rt.pipe,16,precomputed=False)
        black=raster(data['cam'],rt.pipe,3,tagged(rows,'asset'),rt.n.bg.new_zeros(3))
        pre=fresh.rows(data['pose'],data['cam'],rt.pipe,1,precomputed=True)
        explicit=raster(data['cam'],rt.pipe,0,tagged(pre,'asset'),rt.n.bg.new_zeros(3))
        error=float((black['rgb']-explicit['rgb']).abs().max());assert error<5e-6,error
        keep=rt.render_scene(raw,fresh);delete=rt.render_scene(raw,fresh,delete=True)
        records.append(dict(raw=raw,SH_rotation_precomputed_maxabs=error,
            native_a6_rows=keep['meta']['native_a6_rows_input'],asset_rows=keep['meta']['asset_rows']))
        assert records[-1]['native_a6_rows']==0 and delete['meta']['native_a37_packed_rows']==0
        rt._data.clear()
    assert rt.modelunchanged() and fresh.assert_geometry_unchanged()
    projected_seconds=steps_seconds/32*STEPS
    write(dest/'receipt.json',dict(status='PASS',seconds=time.time()-start,solver32_seconds_including_load=steps_seconds,
      projected_formal_seconds=projected_seconds,exact_full_state_resume=True,standalone_export=exported,frames=records,
      appearance_updates=32,formal_updates=0,geometry_updates=0,peak_vram_bytes=torch.cuda.max_memory_allocated()))
    print(json.dumps(dict(stage='smoke',status='PASS',projected_formal_seconds=projected_seconds)),flush=True)
    if projected_seconds>=3600:raise RuntimeError('Projected >=1hour: extend real smoke to10min before formal')

@torch.no_grad()
def fit():
    dest=RUN/'m1';assert json.loads((dest/'smoke/receipt.json').read_text())['status']=='PASS'
    start=time.time();s=Solver();state=s.begin();initial_loss=state['loss'];rows=[]
    for _ in range(STEPS):
        state=s.step(state)
        if state['step'] in (1,50) or state['step']%100==0:
            row=dict(step=state['step'],objective=state['loss'],restarts=state['restarts'],seconds=time.time()-start)
            rows.append(row);print(json.dumps(row),flush=True)
        if state['step'] in (1000,STEPS):torch.save(state,dest/f'checkpoint{state["step"]:06d}.pt')
    exported=s.export(state,dest/'complete_a6_SH3.npz')
    prediction=s.prediction(state['x']);projected=project(state['x']-s.gradient(state['x'])/s.D)
    write(dest/'fit_receipt.json',dict(status='PASS',formal_steps=STEPS,smoke_steps_extra=32,geometry_updates=0,
        objective_initial=initial_loss,objective_final=state['loss'],seconds=time.time()-start,exported=exported,
        BUILD_normalized_RGB_MAE=float((prediction-s.target).abs().double().mean()),
        projected_update_maxabs=float((projected-state['x']).abs().max()),
        coefficient_norm_max=float(torch.linalg.vector_norm(state['x'][:,1:],dim=1).max()),
        all_direction_colour_bound_max_violation=float((BOUND*torch.linalg.vector_norm(state['x'][:,1:],dim=1)-
             torch.minimum(state['x'][:,0],1-state['x'][:,0])).max()),trace=rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['design','smoke','fit']);args=p.parse_args()
    (RUN/'m1').mkdir(exist_ok=True);path=RUN/'m1/contract.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(contract()))
    else:write(path,contract())
    {'design':build_design,'smoke':smoke,'fit':fit}[args.stage]()

if __name__=='__main__':main()
