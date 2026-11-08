"""Inherited fixed-geometry RGB linear design and 1000-step deterministic FISTA."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,copy,json,sys,time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.sparse import vstack,save_npz,load_npz
from native_runtime import ROOT,load,camera
OLD=ROOT.parent/'scene008_a6_asset_controls_20261105/source'
BANK=Path('/root/autodl-tmp/codex_tmp/waymo_val029_cpu_20261108/inputs')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--design-source',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'revealed_material');options=parser.parse_args()
    import torch
    pipe,models,cfg=load(ROOT/'smoke_r5',5914)
    # Camera normalization is deterministic for the fixed input bank; no weights
    # from this diagnostic model enter the source-image material targets.
    del models
    sys.path.insert(0,str(OLD))
    from sh_asset import SHAsset,C0
    from linear_weights import design
    from deterministic_csr import Matrix
    out=options.output;out.mkdir(exist_ok=False);start=time.monotonic()
    asset=SHAsset.from_geometry(ROOT/'revealed_surface/initial_asset.npz').cuda()
    matrices=[];targets=[];records=[];model_from_world=np.loadtxt(Path(cfg['source_path'])/'transform_matrix.txt')
    names=[] if options.design_source else cfg['followup_scene_spec']['frame_ids']
    for name in names:
        raw=int(name[-3:]);labels=np.array(Image.open(Path(cfg['source_path'])/'mask/mask_0/Annotations'/f'{name}.png').resize((480,320),Image.Resampling.NEAREST));low=binary_erosion(labels==25,iterations=1)
        if low.sum()<4:continue
        mask=np.repeat(np.repeat(low,4,axis=0),4,axis=1)
        obs=json.loads((BANK/'observations'/f'{raw:03d}.json').read_text());body=next(b for b in obs['boxes'] if b['track']=='DwjZr_bYUSZOr-XDGregnA');pose=model_from_world@np.array(body['world_from_actor']).reshape(4,4)
        cam=camera(ROOT/'smoke_r5',cfg,raw);cam=copy.copy(cam);cam.image_width=1920;cam.image_height=1280
        for key in ('fx','fy','cx','cy'):setattr(cam,key,getattr(cam,key)*4)
        with torch.no_grad():
            lin=design(cam,pipe,asset.rows(torch.as_tensor(pose,device='cuda',dtype=torch.float32),cam,pipe,1,precomputed=True),torch.tensor(mask,device='cuda'))
        accepted=lin['alpha']>.9
        if not accepted.any():continue
        A=lin['matrix'][accepted].multiply(1/np.maximum(lin['alpha'][accepted,None],1e-6)).tocsr();A.eliminate_zeros();pixels=lin['pixels'][accepted]
        rgb=np.array(Image.open(BANK/obs['image_path']).convert('RGB')).reshape(-1,3)[pixels].astype(np.float32)/255
        matrices.append(A);targets.append(rgb);records.append(dict(raw=raw,rows=len(rgb),nnz=A.nnz));print(raw,len(rgb),flush=True)
        del lin
    if options.design_source:
        A=load_npz(options.design_source/'design.npz').astype(np.float32);target=np.load(options.design_source/'target.npy')
    else:
        A=vstack(matrices,format='csr').astype(np.float32);target=np.concatenate(targets);save_npz(out/'design.npz',A);np.save(out/'target.npy',target)
    M,N=A.shape;unknown=A.getnnz(axis=0)==0;prior=asset.rgb.detach().clone();target=torch.tensor(target,device='cuda');matrix=Matrix(A);transpose=Matrix(A.T)
    upper=np.asarray(A.T@np.asarray(A.sum(1)).ravel()).ravel();D=torch.tensor(2*(upper/M+.001/N),device='cuda',dtype=torch.float32)[:,None];x=prior.clone();y=x.clone();t=1.;trace=[];restarts=0
    def objective(value):return float((matrix(value)-target).double().square().sum()/M+.001*(value-prior).double().square().sum()/N)
    loss=objective(x)
    for step in range(1,1001):
        gradient=2*transpose(matrix(y)-target)/M+2*.001/N*(y-prior);z=(y-gradient/D).clamp(0,1);value=objective(z)
        if value>loss+1e-9:
            t=1.;gradient=2*transpose(matrix(x)-target)/M+2*.001/N*(x-prior);z=(x-gradient/D).clamp(0,1);value=objective(z);restarts+=1
        assert torch.isfinite(z).all()
        tn=(1+np.sqrt(1+4*t*t))/2;y=z+(t-1)/tn*(z-x);x=z;t=tn;loss=value
        if step%100==0:trace.append(dict(step=step,objective=loss));print(trace[-1],flush=True)
        if step in (500,1000):torch.save(dict(step=step,x=x.cpu(),y=y.cpu(),t=t,restarts=restarts,loss=loss),out/f'solver{step}.pt')
    assert torch.equal(x[torch.tensor(unknown,device='cuda')],prior[torch.tensor(unknown,device='cuda')])
    coefficients=asset.coefficients.clone();known=torch.tensor(~unknown,device='cuda');coefficients[known,0]=(x[known]-.5)/C0;asset.set_coefficients(coefficients);asset.metadata['colour_fit']=dict(steps=1000,ridge=.001,geometry_frozen=True,BUILD_only=True,reference_dataset='new street source RGB only');asset.export(out/'complete_asset.npz')
    (out/'receipt.json').write_text(json.dumps(dict(status='PASS_NEW_OBJECT_FROZEN_GEOMETRY_MATERIAL',frames=records,rows=M,columns=N,unknown_columns=int(unknown.sum()),steps=1000,ridge=.001,objective_trace=trace,solver_restarts=restarts,geometry_hash=asset.geometry_hash,holdout_used=False,seconds=time.monotonic()-start),indent=2));print('MATERIAL_COMPLETE',M,N,loss,flush=True)
if __name__=='__main__':main()
