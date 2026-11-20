"""CPU 端点梯度诊断；不创建 optimizer，不更新参数，不调用 native/CUDA。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_cpu_chw(image,sigma,truncate=3.):
    if image.device.type!='cpu' or image.ndim!=3:
        raise ValueError('CPU CHW required')
    radius=int(truncate*sigma+.5)
    x=torch.arange(-radius,radius+1,dtype=torch.float64)
    kernel=torch.exp(-.5*(x/sigma)**2);kernel=kernel/kernel.sum()
    channels=image.shape[0];a=image.double()[None]
    # SciPy gaussian_filter依次处理vertical再horizontal，constant0边界。
    a=F.conv2d(F.pad(a,(0,0,radius,radius)),kernel[None,None,:,None].expand(channels,1,-1,1),groups=channels)
    a=F.conv2d(F.pad(a,(radius,radius,0,0)),kernel[None,None,None,:].expand(channels,1,1,-1),groups=channels)
    return a[0]


def masked_hf_cpu(image,context,sigma,truncate=3.):
    mask=torch.as_tensor(context,dtype=torch.bool,device='cpu')
    clean=torch.where(mask[None],image.double(),torch.zeros_like(image,dtype=torch.float64))
    mass=gaussian_cpu_chw(mask.double()[None],sigma,truncate)
    numerator=gaussian_cpu_chw(clean,sigma,truncate)
    low=torch.where(mass>0,numerator/mass.clamp_min(torch.finfo(torch.float64).tiny),torch.zeros_like(numerator))
    return torch.where(mask[None],clean-low,torch.zeros_like(clean)),mass[0]


def norm_stats(gradient):
    a=gradient.detach().double()
    return dict(L2=float(a.norm()),Linf=float(a.abs().max()),nonzero_scalars=int(torch.count_nonzero(a)))


def relation(rgb,hf):
    r,h=rgb.detach().double().ravel(),hf.detach().double().ravel()
    nr,nh=float(r.norm()),float(h.norm());dot=float(torch.dot(r,h))
    return dict(RGB_gradient_L2=nr,HF_gradient_L2=nh,dot_product=dot,
        cosine=dot/(nr*nh)if nr>0 and nh>0 else None,
        status='defined'if nr>0 and nh>0 else'zero_HF_gradient'if nh==0 else'zero_RGB_gradient',
        HF_derivative_along_negative_RGB_gradient=-dot,
        HF_derivative_along_unit_negative_RGB_gradient=-dot/nr if nr>0 else None,
        RGB_derivative_along_negative_HF_gradient=-dot,
        RGB_derivative_along_unit_negative_HF_gradient=-dot/nh if nh>0 else None,
        sign_interpretation='negative derivative: infinitesimal decrease; positive: infinitesimal increase; zero/N-A retained',
        finite_step_test_performed=False)


def arrays(path):
    with np.load(path,allow_pickle=False)as source:return {key:source[key].copy()for key in source.files}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main(bench,output):
    torch.set_num_threads(1);started=time.perf_counter()
    B=Path(bench);G=B.parent;Q=G.parent;P=Q.parent/'road_multiscale_material_20261118_r1'
    V=Q.parent/'road_capacity_cpu_20261117_r1/supervision_v4_candidate_r1'
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(Q/'code'),str(B/'metrics')]
    from world_texture_runtime import WorldTextureRuntime,FixedPixelReadLayout,compose_native_eligible
    from fixed_material_read_operator import FixedReadOperator
    from road_detail_metrics import _masked_hf,_strict_core
    manifest_path=G/'evaluation/frozen_r2/manifest.json';manifest=json.loads(manifest_path.read_text())
    metric_path=B/'metrics/road_detail_metrics.py';freeze=json.loads((B/'metric_freeze.json').read_text())
    if sha(metric_path)!=freeze['source_sha256']['metrics/road_detail_metrics.py']:
        raise RuntimeError('frozen metric source changed')
    files={Path(__file__),metric_path,manifest_path,B/'metric_freeze.json',B/'protocol.json',
           Q/'code/world_texture_runtime.py',Q/'code/fixed_material_read_operator.py',
           P/'execution_contract.json',G/'formal_r1/input_bindings.json',B/'lowres_r1/input_bindings.json'}
    result=dict(schema='RGB_HF_endpoint_gradient_alignment.v1',role='posthoc local diagnostic, not training or causal proof',
        material_space='straight R/alpha on fixed common baseline road alpha>=.95',
        RGB_space='original full A+.5C+S premult-sRGB L1, normalized within each source/view',
        HF_space='per-source squared signed HF residual, same frozen context/strict core as new benchmark',
        L_geometry_and_other_scene_parameters_frozen=True,arms={},input_bindings={},
        optimizer_updates=0,finite_parameter_steps=0,native_calls=0,GPU_operations=0,
        every_frame_source_and_zero_gradient_retained=True)
    for arm,gate in [('M01','observed_H'),('M11','observed_plus_teacher_H')]:
        field=G/'formal_r1'/arm/'C800/field.pt';files.add(field)
        model,metadata=WorldTextureRuntime.load(field,device='cpu')
        model.low.requires_grad_(False);model.high.requires_grad_(True)
        low_before=model.low.detach().clone();high_before=model.high.detach().clone()
        low,high=model.fields();gradient_bank={};arm_rows=[]
        for raw in (86,110):
            opdir=Q/'pixel_render/read_operator_box_r1'/('%03d'%raw)
            files.update(p for p in opdir.iterdir()if p.is_file())
            operator=FixedReadOperator.load(opdir)
            values=operator.torch_query(low,high,gate)
            layout=FixedPixelReadLayout(operator.row_ids,operator.mass,320,480,row_id_kind='pixel_flat')
            world_rgb,reader_valid=layout(values)
            basis_path=G/'native_qualification_r2/basis'/('%03d'%raw)/'road_basis.npz';basis=arrays(basis_path)
            pred=compose_native_eligible(torch.from_numpy(basis['baseline_rgb']),torch.from_numpy(basis['eligible_rgb']),
                torch.from_numpy(basis['eligible_weight']),world_rgb,reader_valid)
            saved_path=G/'formal_r1'/arm/'C800/rendered'/('%03d.npz'%raw);saved=arrays(saved_path)
            frozen_path=Path(manifest['frames'][str(raw)]['path']);frozen=arrays(frozen_path)
            if sha(frozen_path)!=manifest['frames'][str(raw)]['sha256']:raise RuntimeError('frozen frame binding differs')
            old_path=P/'stage0_native_r1'/('%03d_targets.npz'%raw);old=arrays(old_path)
            full_path=V/('%03d.npz'%raw);full=arrays(full_path)
            files.update((basis_path,saved_path,frozen_path,old_path,full_path))
            alpha=frozen['baseline_road_alpha'];stable=alpha>=.95
            if not np.array_equal(saved['road_alpha'].squeeze(),alpha):raise RuntimeError('method alpha differs from common alpha')
            pred_error=float(abs(pred.detach().numpy()-saved['road_rgb']).max())
            if pred_error>1e-6:raise RuntimeError('CPU original reader differs from saved RGB')
            safe_alpha=torch.from_numpy(np.where(stable,alpha,1.)).double()
            straight=torch.where(torch.from_numpy(stable)[None],pred.double()/safe_alpha[None],torch.zeros_like(pred,dtype=torch.float64))
            losses={};data={};RGB={}
            for source,weight in [('A',1.),('C',.5),('S',1.)]:
                if not np.array_equal(old[source].astype(bool),frozen[source].astype(bool)):raise RuntimeError('legacy source mask differs')
                mask=torch.from_numpy(old[source]).bool();target=torch.from_numpy(old['fixed_alpha'])[None]*torch.from_numpy(old['target_'+source])
                RGB[source]=weight*(pred-target).abs()[:,mask].mean()
                losses['RGB_'+source]=RGB[source]
            losses['RGB_full']=sum(RGB.values())
            for source in ('C','S'):
                score=frozen[source].astype(bool);context=full['coverage_'+source].astype(bool)&stable
                target=torch.from_numpy(frozen['target_'+source]).double()
                if not np.isfinite(target.numpy()[:,context]).all():raise RuntimeError('teacher invalid inside fixed context')
                for sigma in (1.,2.):
                    label='HF_'+source+'_sigma'+str(int(sigma))
                    report_path=B/'lowres_r1'/('%03d_%s_%s_straight_road_srgb_alpha95.json'%(raw,source,arm));report=json.loads(report_path.read_text());files.add(report_path)
                    benchmark=next(s for s in report['scales']if s['sigma_pixels']==sigma)
                    core=score&context&_strict_core(context,int(3*sigma+.5));n=int(core.sum())
                    if n!=benchmark['core_pixels']:raise RuntimeError('strict core denominator differs')
                    hp,_=masked_hf_cpu(straight,context,sigma);ht,_=masked_hf_cpu(target,context,sigma)
                    reference_numpy,_=_masked_hf(straight.detach().numpy().transpose(1,2,0),context,sigma,3.)
                    hf_operator_error=float(abs(hp.detach().numpy().transpose(1,2,0)-reference_numpy).max())
                    loss=(hp-ht).square()[:,torch.from_numpy(core)].mean()if n else None
                    RMSE=float(loss.detach().sqrt())if n else None
                    expected=benchmark['core']['HF_RMSE']
                    rmse_error=abs(RMSE-expected)if n else None
                    if hf_operator_error>1e-6 or (n and rmse_error>1e-6):raise RuntimeError('HF implementation does not match frozen metric')
                    losses[label]=loss
                    matched_label='matched_core_RGB_'+source+'_sigma'+str(int(sigma))
                    losses[matched_label]=(straight-target).abs()[:,torch.from_numpy(core)].mean()if n else None
                    data[label]=dict(source=source,sigma_pixels=sigma,core_pixels=n,
                        fixed_score_pixels=int(score.sum()),score_excluded_alpha95=int((score&~stable).sum()),
                        context_pixels=int(context.sum()),HF_RMSE=RMSE,benchmark_HF_RMSE=expected,
                        RMSE_validation_error=rmse_error,HF_operator_max_error=hf_operator_error,
                        benchmark_file=str(report_path),objective='HF_MSE; unweighted per-source mean',
                        status='defined'if n else'no_strict_core',matched_RGB_control=matched_label)
            # 完成全部数值/域验证后才求梯度；从不创建optimizer或执行参数步。
            gradients={};loss_rows={}
            for label,loss in losses.items():
                if loss is None:
                    gradients[label]=None;loss_rows[label]=dict(value=None,gradient=None,status='no_strict_core');continue
                grad=torch.autograd.grad(loss,model.high,retain_graph=True,allow_unused=True)[0]
                grad=torch.zeros_like(model.high)if grad is None else grad.detach()
                gradients[label]=grad;loss_rows[label]=dict(value=float(loss.detach()),gradient=norm_stats(grad),
                    status='zero_H_gradient'if not bool(torch.count_nonzero(grad))else'defined')
                outside=int(torch.count_nonzero(grad[model.latent_mask[...,0]==0]))
                loss_rows[label]['gradient']['nonzero_scalars_outside_common_latent']=outside
                if outside:raise RuntimeError('H gradient escaped common latent support')
            for label,entry in data.items():
                hf=gradients[label]
                entry['loss_and_gradient']=loss_rows[label]
                entry['comparisons']={}if hf is None else dict(
                    full_original_RGB=relation(gradients['RGB_full'],hf),
                    same_source_original_RGB=relation(gradients['RGB_'+entry['source']],hf),
                    matched_core_straight_RGB=relation(gradients[entry['matched_RGB_control']],hf))
            arm_rows.append(dict(raw=raw,cpu_render_saved_max=pred_error,
                old_fixed_alpha_vs_common_max=float(abs(old['fixed_alpha']-alpha).max()),
                original_RGB_objective={key:loss_rows[key]for key in ['RGB_A','RGB_C','RGB_S','RGB_full']},HF=data))
            gradient_bank[raw]=gradients
        crossview={};aggregates={}
        keys=['RGB_full','HF_C_sigma1','HF_C_sigma2','HF_S_sigma1','HF_S_sigma2']
        for key in keys:
            x,y=gradient_bank[86][key],gradient_bank[110][key]
            crossview[key]=relation(x,y)if x is not None and y is not None else dict(status='one_or_more_frames_no_core')
        rgb=(gradient_bank[86]['RGB_full']+gradient_bank[110]['RGB_full'])/2
        for sigma in (1,2):
            cs=[]
            for source,weight in [('C',.5),('S',1.)]:
                key='HF_'+source+'_sigma'+str(sigma);xs=[gradient_bank[raw][key]for raw in (86,110)]
                if all(x is not None for x in xs):
                    g=(xs[0]+xs[1])/2;aggregates[key]=relation(rgb,g);cs.append((weight,g))
                else:aggregates[key]=dict(status='one_or_more_frames_no_core')
            if len(cs)==2:aggregates['HF_C0.5_plus_S_sigma'+str(sigma)]=relation(rgb,sum(w*g for w,g in cs))
        if model.low.grad is not None or model.high.grad is not None:raise RuntimeError('unexpected gradient accumulation')
        if not torch.equal(model.low,low_before)or not torch.equal(model.high,high_before):raise RuntimeError('parameters changed')
        result['arms'][arm]=dict(rows=arm_rows,crossview_gradient_comparisons=crossview,
            balanced_view_gradient_comparisons=aggregates,parameters_exactly_unchanged=True,
            note='Crossview relation labels RGB/HF are generic vector slots; consult the metric key. Aggregate never replaces per-frame rows.')
    result['input_bindings']={str(p):sha(p)for p in sorted(files)}
    result['seconds']=time.perf_counter()-started
    result['limitations']=['Local endpoint/subgradient diagnostic only; no finite-step decrease or training success is inferred.',
        'Full original RGB and conditional straight HF have different domains/spaces; matched-core RGB control is reported.',
        'HF square residual is a diagnostic objective, not a new authorized training loss.',
        'Frozen pseudo teacher consistency is not hidden-road truth; no structure/lane truth invented.']
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output/'results.json'),seconds=result['seconds'],optimizer_updates=0,native_calls=0,GPU_operations=0,
        summary={arm:value['balanced_view_gradient_comparisons']for arm,value in result['arms'].items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bench',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        main(args.bench,args.output)
    except Exception:
        import traceback
        path=Path(args.output)
        if path.is_dir()and not(path/'results.json').exists():
            (path/'failed.json').write_text(json.dumps(dict(error=traceback.format_exc(),optimizer_updates=0,
                native_calls=0,GPU_operations=0,finite_parameter_steps=0),indent=2)+'\n')
        raise
