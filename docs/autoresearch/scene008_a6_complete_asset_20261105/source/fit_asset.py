"""Complete fixed-PVD appearance control. One shared asset, BUILD RGB only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse, hashlib, json, time
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.sparse import vstack, eye, save_npz, load_npz
from scipy.optimize import lsq_linear
from full_pvd import FixedAsset, RUN
from scene_runtime import Runtime
from linear_weights import design

HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(r for r in range(89,138) if r not in HELD)
LAMBDA=1e-3

def write(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))

def contract():
    return dict(name='Complete a6 asset: fixed full PVD / BUILD-only appearance',
      build=BUILD,internal_held=HELD,geometry_source='Original full 2048 PVD output; 200 conditioning + 1848 generated',
      geometry_frozen=['centres','normals','tangent scales','normal thickness','opacity','trajectory'],
      appearance='2048 shared object-local SH0 RGB triplets, bounded sparse ridge least squares',
      target='Original raw RGB on 1-pixel eroded confirmed-visible a6, proposal alpha >= .2',
      normalized_fit=True,ridge_lambda=LAMBDA,prior='Reliable native a6 conditional colours; unseen global BUILD RGB median',
      reliable_native='Only real BUILD a6 visible pixels; native isolated alpha>=.9; RGB L1 mean discrepancy<=.05',
      objective='mean_pixels ||(W/alpha)c-I||^2 + .001 mean_points ||c-c0||^2',
      rgb_bounds=[0,1],geometry_updates=0,neural_training_steps=0,
      scene='Native10K + frozen road1K + frozen previously saved 11x9 world road colour field',
      replacement='Remove ALL native a6 rows; insert same complete asset in keep and delete; delete only a37',
      forbidden=['a37 or road RGB supervision','unobserved native colour transfer','lower/side-specific clipping','per-frame colours','geometry fitting'],
      evaluation='All49 fixed full observed mask and LiDAR denominator; no proposal-alpha denominator shrink',
      limitation='Internal appearance holdout only: native/PVD geometry uses full-scene context; 14 conditioning rows are HELD frames; no generalization claim')

@torch.no_grad()
def collect(dest):
    start=time.time();runtime=Runtime();asset=FixedAsset.from_geometry(RUN/'geometry.npz').cuda()
    matrices=[];targets=[];mass=np.zeros(2048);views=np.zeros(2048,np.int64)
    prior_sum=np.zeros((2048,3));prior_mass=np.zeros(2048);raw_samples=[];records=[]
    for raw in BUILD:
        d=runtime.raw(raw);visible=d['a6_visible'].cpu().numpy().astype(bool)
        interior=binary_erosion(visible,iterations=1)
        rows=asset.rows(d['pose'],d['cam'],runtime.pipe,1,precomputed=True)
        lin=design(d['cam'],runtime.pipe,rows,torch.as_tensor(interior,device='cuda'))
        use=lin['alpha']>=.2;W=lin['matrix'][use]
        A=W.multiply(1/np.maximum(lin['alpha'][use,None],1e-8)).tocsr()
        pixel=lin['pixels'][use];raw_rgb=d['rgb'].flatten(1)[:,pixel].T.cpu().numpy()
        native=runtime.native_reference(raw)
        native_alpha=native['alpha'][0].flatten()[pixel].cpu().numpy()
        native_rgb=native['rgb_unpremultiplied'].flatten(1)[:,pixel].T.cpu().numpy()
        reliable=(native_alpha>=.9)&(np.abs(native_rgb-raw_rgb).mean(1)<=.05)
        reliable_W=W[reliable]
        prior_mass+=np.asarray(reliable_W.sum(0)).ravel()
        prior_sum+=reliable_W.T@native_rgb[reliable]
        frame_mass=np.asarray(W.sum(0)).ravel();mass+=frame_mass;views+=frame_mass>=.05
        matrices.append(A);targets.append(raw_rgb);raw_samples.append(raw_rgb[::max(1,len(raw_rgb)//1000)])
        records.append(dict(raw=raw,visible=int(visible.sum()),interior=int(interior.sum()),fit=len(pixel),
            omitted_low_alpha=int((~use).sum()),reliable_native=int(reliable.sum()),
            alpha_maxabs=lin['alpha_maxabs'],colour_maxabs=lin['colour_maxabs'],nnz=A.nnz))
        print(json.dumps(dict(stage='collect',**records[-1])),flush=True)
        del d,rows,lin,native;runtime._data.clear();torch.cuda.empty_cache()
    A=vstack(matrices,format='csr');target=np.concatenate(targets).astype(np.float64)
    median=np.median(np.concatenate(raw_samples),axis=0)
    c0=np.tile(median,(2048,1));supported=prior_mass>=.05
    c0[supported]=prior_sum[supported]/prior_mass[supported,None]
    c0=np.clip(c0,0,1)
    save_npz(dest/'appearance_design.npz',A)
    np.savez_compressed(dest/'appearance_inputs.npz',target=target,prior=c0,mass=mass,views=views,prior_mass=prior_mass,
         global_unknown_rgb=median,geometry_hash=np.asarray(asset.geometry_hash))
    assert runtime.modelunchanged() and asset.assert_geometry_unchanged()
    write(dest/'collection.json',dict(frames=records,rows=A.shape[0],columns=A.shape[1],nnz=A.nnz,
       geometry_hash=asset.geometry_hash,seconds=time.time()-start,runtime=runtime.meta,
       appearance_sources='BUILD real visible a6 only; native reliable appearance only within same mask',
       input_sha256=hashlib.sha256((dest/'appearance_inputs.npz').read_bytes()).hexdigest()))

def solve(dest):
    start=time.time();A=load_npz(dest/'appearance_design.npz')
    with np.load(dest/'appearance_inputs.npz') as x:
        target=x['target'];prior=x['prior'];mass=x['mass'];views=x['views'];expected=str(x['geometry_hash'])
    asset=FixedAsset.from_geometry(RUN/'geometry.npz');assert asset.geometry_hash==expected
    # Multiplying mean-loss objective by M gives this exact ridge coefficient.
    strength=np.sqrt(LAMBDA*A.shape[0]/2048)
    augmented=vstack((A,strength*eye(2048)),format='csr');colours=[];logs=[]
    for channel in range(3):
        result=lsq_linear(augmented,np.r_[target[:,channel],strength*prior[:,channel]],bounds=(0,1),
             tol=1e-7,lsmr_tol=1e-7,max_iter=100,lsmr_maxiter=500,verbose=0)
        if not result.success:raise RuntimeError(str(result.message))
        colours.append(result.x);logs.append(dict(channel=channel,status=int(result.status),iterations=int(result.nit),
            cost=float(result.cost),optimality=float(result.optimality),message=str(result.message)))
        print(json.dumps(dict(stage='solve',**logs[-1])),flush=True)
    colour=np.stack(colours,1).astype(np.float32);inferred=(views<2)|(mass<1)
    asset.set_colours(prior.astype(np.float32),inferred);asset.metadata.update(contract=contract(),material='native-conditioned prior before raw RGB fit')
    prior_export=asset.export(dest/'complete_a6_prior.npz')
    asset.set_colours(colour,inferred);asset.metadata.update(contract=contract(),material='BUILD raw RGB fitted SH0')
    export=asset.export(dest/'complete_a6_fitted.npz')
    reloaded=FixedAsset.from_geometry(dest/'complete_a6_fitted.npz')
    assert np.array_equal(reloaded.rgb.numpy(),colour) and reloaded.geometry_hash==asset.geometry_hash
    write(dest/'fit_receipt.json',dict(status='PASS',export=export,prior_export=prior_export,solvers=logs,
       geometry_updates=0,neural_training_steps=0,shared_RGB_parameters=6144,fit_pixels=A.shape[0],
       build_normalized_RGB_MAE_prior=float(np.abs(A@prior-target).mean()),
       build_normalized_RGB_MAE_fitted=float(np.abs(A@colour-target).mean()),
       colour_inferred_points=int(inferred.sum()),observed_points=int((~inferred).sum()),
       seconds=time.time()-start,exact_asset_reload=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--collect-only',action='store_true');p.add_argument('--solve-only',action='store_true');args=p.parse_args()
    dest=RUN/'appearance';dest.mkdir(exist_ok=True)
    path=RUN/'contract.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(contract()))
    else:write(path,contract())
    assert (RUN/'smoke/receipt.json').exists(),'GPU entry/export/reload smoke required'
    if not args.solve_only:collect(dest)
    if not args.collect_only:solve(dest)

if __name__=='__main__':main()
