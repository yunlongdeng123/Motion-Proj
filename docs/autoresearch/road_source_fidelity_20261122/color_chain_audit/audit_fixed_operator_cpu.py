"""只读复核已实测颜色矩阵、边界约束求解及 HF 支撑域；不调用 native/GPU。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.sparse import load_npz
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter,minimum_filter


def stats(x):
    x=np.asarray(x).ravel()
    return dict(min=float(x.min()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max()),mean=float(x.mean())) if x.size else {}


def spectral(a,b,mask,sigma):
    n=int(mask.sum())
    if not n:return dict(n=0)
    aa=(a-gaussian_filter(a,(sigma,sigma,0)))[mask].astype(np.float64).ravel()
    bb=(b-gaussian_filter(b,(sigma,sigma,0)))[mask].astype(np.float64).ravel()
    an,bn=np.linalg.norm(aa),np.linalg.norm(bb)
    return dict(n=n,HF_cosine=float(aa@bb/(an*bn)) if an*bn else None,
                HF_RMS_ratio=float(bn/an) if an else None,
                RGB_MAE=float(abs(a-b)[mask].mean()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_source_fidelity_20261122_r2'))
    ap.add_argument('--runs',nargs='+',default=['fixed_color_operator_02_115','fixed_color_operator_00_137'])
    args=ap.parse_args();rows=[];started=time.monotonic()
    for run in args.runs:
        folder=args.root/run
        receipt=json.loads((folder/'receipt.json').read_text())
        r=receipt['rows'][0];patch=r['patch'];data=folder/patch
        A=load_npz(data/'native_color_operator.npz').astype(np.float64)
        with np.load(data/'linear_state.npz',allow_pickle=False) as z:s={k:z[k] for k in z.files}
        m,n=A.shape;weight=1e-5*m/n
        x=s['fitted_core_RGB'].astype(np.float64);x0=s['direct_core_RGB'].astype(np.float64)
        y=s['target_minus_fixed'].astype(np.float64);fixed=s['fixed_core_image'].astype(np.float64)
        pred=A@x+fixed;target=y+fixed
        row=dict(run=run,patch=patch,reference=r['reference'],shape=[m,n],nnz=A.nnz,
                 prior_weight_in_unscaled_LS=weight,normalization_correct=True,
                 objective='mean RGB squared + 1e-5*mean prior squared; factor1/2 does not change minimizer',
                 A_values=stats(A.data),A_row_sums=stats(np.asarray(A.sum(1))),
                 zero_columns=int((np.asarray(A.power(2).sum(0)).ravel()==0).sum()),
                 fitted_min=float(x.min()),fitted_max=float(x.max()),
                 solver_receipt=r['solver'],channels=[],native_replay_checks={},HF_support_sensitivity=[])
        for channel in range(3):
            yy=y[:,channel];prior=x0[:,channel];xx=x[:,channel]
            def objective(z):
                residual=A@z-yy;diff=z-prior
                return .5*(residual@residual+weight*(diff@diff)),np.asarray(A.T@residual)+weight*diff
            cost,g=objective(xx)
            lower=xx<1e-6;upper=xx>1-1e-6
            kkt=g.copy();kkt[lower]=np.minimum(kkt[lower],0);kkt[upper]=np.maximum(kkt[upper],0)
            vertex=(g<0).astype(np.float64)
            fw_gap=float(g@(xx-vertex))
            # 同一有界凸目标的独立 CPU 精度诊断。不给源资产赋值，不宣称新增 native 结果。
            refined=minimize(objective,xx,jac=True,method='L-BFGS-B',bounds=[(0.,1.)]*n,
                options=dict(ftol=1e-14,gtol=1e-10,maxiter=6000,maxfun=10000,maxls=40,maxcor=20))
            refined_cost,refined_gradient=objective(refined.x)
            refined_fw=float(refined_gradient@(refined.x-(refined_gradient<0)))
            revised=A@refined.x+fixed[:,channel]
            row['channels'].append(dict(channel=channel,cost_from_saved_float32=cost,
                box_projected_gradient_inf=float(abs(kkt).max()),Frank_Wolfe_gap=fw_gap,
                normalized_channel_objective_gap_upper_bound=2*fw_gap/m,
                lower_bound_rows=int(lower.sum()),upper_bound_rows=int(upper.sum()),
                CPU_precision_replay=dict(success=bool(refined.success),status=int(refined.status),
                    message=str(refined.message),iterations=int(refined.nit),function_evals=int(refined.nfev),
                    cost=refined_cost,cost_decrease=cost-refined_cost,
                    normalized_channel_objective_decrease=2*(cost-refined_cost)/m,
                    Frank_Wolfe_gap=refined_fw,prediction_change=stats(abs(revised-pred[:,channel])),
                    original_reference_MAE=float(abs(pred[:,channel]-target[:,channel]).mean()),
                    refined_reference_MAE=float(abs(revised-target[:,channel]).mean()),
                    native_render_of_precision_replay=False)))
        for e in r['evaluations']:
            raw=e['raw'];path=data/f'{raw:03d}'/'source_float.npz'
            with np.load(path,allow_pickle=False) as z:f={k:z[k] for k in z.files}
            score=f['score_mask'];core=f['core'];alpha=f['alpha'];photo=f['observation'];fit=f['solved_rgb'];original=f['original_rgb']
            ys,xs=np.nonzero(score)
            crop_margin=min(ys.min(),xs.min(),score.shape[0]-1-ys.max(),score.shape[1]-1-xs.max())
            if raw==r['reference']:
                x0_,y0_,x1_,y1_=f['bbox_xyxy'];slice_score=s['score_mask'][y0_:y1_,x0_:x1_]
                row['native_replay_checks']=dict(score_order_mask_exact=bool(np.array_equal(slice_score,score)),
                    matrix_plus_fixed_vs_native_saved_max_abs=float(abs(pred-fit[score]).max()),
                    target_plus_fixed_vs_photo_max_abs=float(abs(target-photo[score]).max()),
                    source='existing native outputs only; matrix was measured by root GPU task')
            for sigma in (1.2,5/3,3.):
                radius=int(4*sigma+.5)
                score_support=minimum_filter(score.astype(np.uint8),size=2*radius+1,mode='constant',cval=0)>0
                alpha_support=minimum_filter(alpha,size=2*radius+1,mode='constant',cval=0)>=.95
                core_support=minimum_filter(core.astype(np.uint8),size=2*radius+1,mode='constant',cval=0)>0
                row['HF_support_sensitivity'].append(dict(raw=raw,sigma=sigma,gaussian_radius=radius,
                    minimum_score_to_crop_edge_px=int(crop_margin),crop_boundary_safe=bool(crop_margin>=radius),
                    original_score=dict(original=spectral(photo,original,score,sigma),solved=spectral(photo,fit,score,sigma)),
                    alpha_support_safe=dict(original=spectral(photo,original,score&alpha_support,sigma),solved=spectral(photo,fit,score&alpha_support,sigma)),
                    physical_core_support_safe=dict(original=spectral(photo,original,score&core_support,sigma),solved=spectral(photo,fit,score&core_support,sigma)),
                    complete_score_support_safe=dict(original=spectral(photo,original,score&score_support,sigma),solved=spectral(photo,fit,score&score_support,sigma))))
        rows.append(row)
    print(json.dumps(dict(schema='fixed_operator_independent_cpu_audit.v1',CPU_only=True,new_native_calls=0,
                          asset_modified=False,rows=rows,seconds=time.monotonic()-started),indent=2,ensure_ascii=True,allow_nan=False))


if __name__=='__main__':main()
