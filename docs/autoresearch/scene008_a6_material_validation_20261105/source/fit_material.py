"""Common visible-material fit followed by strictly data-preserving extension."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time,hashlib,math,argparse
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign');OLD=CAM/'scene008_a6_asset_controls_20261105'
RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(OLD/'source'))
from deterministic_csr import Matrix
from sh_asset import SHAsset,C0
from scene_runtime import Runtime
from surface_material import build_surface_graph,build_parent_surface_graph,extend_material,verify_train_invariance
STEPS=2000;LAMBDA=.001;CASES=('left','right','top');METHODS=('median','child_nearest','child_graph','parent_nearest','parent_graph')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def graph():
    g=dict(np.load(OLD/'geometry_dense.npz'))
    return dict(child=build_surface_graph(g['xyz'],g['normal'],g['normal_valid']),parent=build_parent_surface_graph(g))

class Solver:
    def __init__(self,case):
        folder=RUN/'cases'/case;A=load_npz(folder/'train_design.npz').astype(np.float32)
        self.sparse=A;self.N=A.shape[1];self.M=A.shape[0]
        upper=np.asarray(abs(A).T@np.asarray(abs(A).sum(1)).ravel()).ravel()
        self.D=torch.tensor(2*(upper/self.M+LAMBDA/self.N),device='cuda')[:,None]
        self.A=Matrix(A);self.AT=Matrix(A.T)
        with np.load(folder/'train_inputs.npz') as x:
            self.target=torch.tensor(x['target'],device='cuda');self.prior=torch.tensor(x['prior'],device='cuda')
            self.unknown=x['unknown'].copy();self.global_rgb=x['global_rgb'].copy()
        assert np.array_equal(A.getnnz(axis=0)==0,self.unknown)
    def objective(self,x):return float((self.A(x)-self.target).double().square().sum()/self.M+LAMBDA*(x-self.prior).double().square().sum()/self.N)
    def gradient(self,x):return 2*self.AT(self.A(x)-self.target)/self.M+2*LAMBDA/self.N*(x-self.prior)
    def begin(self):
        x=self.prior.clone();return dict(x=x,y=x.clone(),t=1.,loss=self.objective(x),step=0,restarts=0)
    def step(self,state):
        x,y,t=state['x'],state['y'],state['t'];z=(y-self.gradient(y)/self.D).clamp(0,1);loss=self.objective(z);restarts=state['restarts']
        if loss>state['loss']+1e-9:t=1.;z=(x-self.gradient(x)/self.D).clamp(0,1);loss=self.objective(z);restarts+=1
        assert math.isfinite(loss) and bool(torch.isfinite(z).all())
        assert torch.equal(z[torch.as_tensor(self.unknown,device='cuda')],self.prior[torch.as_tensor(self.unknown,device='cuda')])
        tn=(1+math.sqrt(1+4*t*t))/2
        return dict(x=z,y=z+(t-1)/tn*(z-x),t=tn,loss=loss,step=state['step']+1,restarts=restarts)

def export(path,rgb,unknown,metadata):
    asset=SHAsset.from_geometry(OLD/'geometry_dense.npz')
    coeff=np.zeros((len(rgb),16,3),np.float32);coeff[:,0]=(rgb-.5)/C0
    # Conservative: fit support is separately recorded, not a true material label.
    asset.set_coefficients(coeff,np.ones(len(rgb),bool));asset.metadata['material_validation']=metadata
    result=asset.export(path);assert asset.assert_geometry_unchanged();return result

def extend_all(g,known_fit,unknown,global_rgb,dest,Atrain,tag):
    dest.mkdir(exist_ok=True);reports=[];decoded_baseline=None
    for method in METHODS:
        # Missing rows are omitted, not passed as fitted/hidden colour hints.
        topology='parent' if method.startswith('parent_') else 'child'
        rule=method.split('_')[-1] if method!='median' else 'median'
        result=extend_material(g[topology],known_fit[~unknown].copy(),unknown,global_rgb,method=rule)
        rgb=np.asarray(result.rgb,np.float32)
        assert np.array_equal(rgb[~unknown],known_fit[~unknown]) and np.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1
        inv=verify_train_invariance(Atrain,known_fit,rgb)
        np.savez_compressed(dest/f'{method}_colours.npz',rgb=rgb,unknown=unknown,propagated=result.propagated,unreachable=result.unreachable)
        asset=export(dest/f'{method}_asset.npz',rgb,unknown,dict(case=tag,method=method,neural_training=False,
            unknown_stays_inferred=True,topology=topology,changed_only_zero_RGB_columns=True,not_hidden_texture_truth=True))
        decoded=SHAsset.from_geometry(dest/f'{method}_asset.npz').rgb.numpy()
        if decoded_baseline is None:decoded_baseline=decoded.copy()
        decoded_inv=verify_train_invariance(Atrain,decoded_baseline,decoded)
        reports.append(dict(method=method,export=asset,propagated=int(np.asarray(result.propagated).sum()),unreachable=int(np.asarray(result.unreachable).sum()),
            nonzero_columns_RGB_bitwise_unchanged=True,train_invariance=inv,export_reload_train_invariance=decoded_inv,extension_audit=result.audit))
    return reports

@torch.no_grad()
def smoke():
    root=RUN/'smoke';root.mkdir(exist_ok=False);start=time.time();g=graph();reports=[];projected_fit=0.
    for case in CASES:
        s=Solver(case);state=s.begin();folder=root/case;folder.mkdir()
        begin_fit=time.time()
        for _ in range(32):state=s.step(state)
        fit_seconds=time.time()-begin_fit
        torch.save(state,folder/'full_state.pt');reloaded=torch.load(folder/'full_state.pt',map_location='cuda')
        next1=s.step(state);next2=s.step(reloaded);assert all(torch.equal(next1[k],next2[k]) for k in ('x','y'))
        cols=state['x'].cpu().numpy();begin_extension=time.time();methods=extend_all(g,cols,s.unknown,s.global_rgb,folder/'extensions',s.sparse,case)
        extension_seconds=time.time()-begin_extension;projected_fit+=fit_seconds/32*STEPS+extension_seconds
        rt=Runtime()
        for method in METHODS:
            asset=SHAsset.from_geometry(folder/'extensions'/f'{method}_asset.npz').cuda()
            for raw in (105,113,130):
                for delete in (False,True):
                    out=rt.render_scene(raw,asset,delete=delete);assert out['meta']['native_a6_rows_input']==0
                rt._data.clear()
        assert rt.modelunchanged();reports.append(dict(case=case,finite=True,exact_full_state_nextstep=True,next_stage_extensions_and_joint_scene=True,
            measured_32step_fit_seconds=fit_seconds,extension_seconds=extension_seconds,methods=methods))
        del s,state,rt;torch.cuda.empty_cache()
    seconds=time.time()-start;projected=projected_fit+600.
    status='REQUIRES_REAL_10MIN_SMOKE' if projected>=3600 else 'PASS'
    write(root/'receipt.json',dict(status=status,seconds=seconds,projected_total_formal_seconds=projected,
        projection='Measured optimizer rate x formal budget + measured extension time +600sec reserved full49 rendering/CPU evaluation',
        source_sha256=sha(__file__),module_sha256=sha(Path(__file__).with_name('surface_material.py')),
        contract_sha256=sha(RUN/'cases/contract.json'),colour_steps_extra=32*3,nextstep_replays_extra=2*3,cases=reports))
    if status!='PASS':raise RuntimeError('Real10min smoke required for projected hourlong run')

@torch.no_grad()
def fit():
    receipt=json.loads((RUN/'smoke/receipt.json').read_text());assert receipt['status']=='PASS'
    assert receipt['source_sha256']==sha(__file__) and receipt['module_sha256']==sha(Path(__file__).with_name('surface_material.py'))
    assert receipt['contract_sha256']==sha(RUN/'cases/contract.json')
    dest=RUN/'formal';dest.mkdir(exist_ok=False);g=graph();reports=[]
    for case in CASES:
        folder=dest/case;folder.mkdir();start=time.time();s=Solver(case);state=s.begin();initial=state['loss']
        for _ in range(STEPS):
            state=s.step(state)
            if state['step'] in (1000,STEPS):torch.save(state,folder/f'checkpoint{state["step"]:06d}.pt')
        rgb=state['x'].cpu().numpy();np.savez_compressed(folder/'common_fit.npz',rgb=rgb,unknown=s.unknown,global_rgb=s.global_rgb)
        methods=extend_all(g,rgb,s.unknown,s.global_rgb,folder/'extensions',s.sparse,case)
        row=dict(case=case,steps=STEPS,objective_initial=initial,objective_final=state['loss'],seconds=time.time()-start,
            train_RGB_MAE=float((s.A(state['x'])-s.target).abs().double().mean()),
            projected_update_maxabs=float(((state['x']-s.gradient(state['x'])/s.D).clamp(0,1)-state['x']).abs().max()),methods=methods)
        reports.append(row);print(json.dumps(row),flush=True)
        del s,state;torch.cuda.empty_cache()
    write(dest/'receipt.json',dict(status='PASS_COLOUR_FIT_AND_EXTENSION',formal_colour_steps=6000,neural_model_training_steps=0,cases=reports))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','fit']);args=p.parse_args();globals()[args.stage]()
