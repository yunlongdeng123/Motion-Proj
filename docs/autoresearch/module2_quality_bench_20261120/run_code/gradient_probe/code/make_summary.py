"""仅整理既有CPU梯度结果；不加载模型、不重跑求导。"""
from pathlib import Path
import argparse
import json


def main(source,output):
    d=json.loads(Path(source).read_text());summary=dict(schema='RGB_HF_gradient_probe_summary.v1',
        source_results=str(source),optimizer_updates=0,finite_parameter_steps=0,native_calls=0,GPU_operations=0,
        validation={},arms={},conditional_denominators={},interpretation=[
            'Nonzero H gradient establishes only a local descent direction, not finite-step success.',
            'Full RGB vs conditional HF differ in domain and color space; matched-core RGB control is retained.',
            'Near-orthogonal or conflicting gradients make absent HF supervision a candidate, not a proven sole bottleneck.',
            'Strict-core results do not represent all original C/S pixels, boundaries or native1920 details.'])
    rmse=[];operator=[];render=[];full_relations=[]
    for arm,a in d['arms'].items():
        per_frame=[]
        for row in a['rows']:
            render.append(row['cpu_render_saved_max'])
            for key,h in row['HF'].items():
                if h['RMSE_validation_error']is not None:rmse.append(h['RMSE_validation_error'])
                operator.append(h['HF_operator_max_error'])
                item=dict(raw=row['raw'],objective=key,source=h['source'],sigma_pixels=h['sigma_pixels'],
                    core_pixels=h['core_pixels'],status=h['status'],gradient=h['loss_and_gradient']['gradient'],
                    full_RGB=h['comparisons'].get('full_original_RGB'),
                    matched_core_RGB=h['comparisons'].get('matched_core_straight_RGB'))
                per_frame.append(item)
                if item['full_RGB']is not None:full_relations.append(dict(arm=arm,**item))
        summary['arms'][arm]=dict(per_frame=per_frame,
            balanced=d['arms'][arm]['balanced_view_gradient_comparisons'],
            crossview=d['arms'][arm]['crossview_gradient_comparisons'],
            parameters_exactly_unchanged=a['parameters_exactly_unchanged'])
    for source in ('C','S'):
        for sigma in (1,2):
            rows=[r['HF']['HF_'+source+'_sigma'+str(sigma)]for r in d['arms']['M11']['rows']]
            summary['conditional_denominators'][source+'_sigma'+str(sigma)]=dict(
                original_score_pixels=sum(r['fixed_score_pixels']for r in rows),
                excluded_alpha95=sum(r['score_excluded_alpha95']for r in rows),
                strict_core_pixels=sum(r['core_pixels']for r in rows))
    summary['validation']=dict(HF_RMSE_benchmark_max_abs_error=max(rmse)if rmse else None,
        differentiable_HF_scipy_max_abs_error=max(operator)if operator else None,
        CPU_reader_saved_RGB_max_abs_error=max(render)if render else None)
    summary['per_frame_full_RGB_to_HF']=dict(total=len(full_relations),
        negative_derivative=sum(x['full_RGB']['HF_derivative_along_negative_RGB_gradient']<0 for x in full_relations),
        positive_derivative=sum(x['full_RGB']['HF_derivative_along_negative_RGB_gradient']>0 for x in full_relations),
        zero_derivative=sum(x['full_RGB']['HF_derivative_along_negative_RGB_gradient']==0 for x in full_relations),
        NA=len(d['arms'])*8-len(full_relations))
    target=Path(output)
    if target.exists():raise FileExistsError('preserve previous summary')
    target.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(target),validation=summary['validation'],
        full_RGB_to_HF=summary['per_frame_full_RGB_to_HF'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.input,a.output)
