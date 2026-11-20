"""只运行解析合成负对照并输出数值；无训练、无native、无实际场景输入。"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from road_detail_metrics import ResolutionSpec,TileLayout,evaluate_detail,rank_cases


def evaluate(pred,teacher,**kwargs):
    h,w=teacher.shape[:2];mask=np.ones((h,w),bool)
    return evaluate_detail(pred,teacher,context_mask=mask,score_mask=mask,
        resolution=ResolutionSpec(w,h,'synthetic_input_pixels'),tile_layout=TileLayout(32,32),
        teacher_role='synthetic_known',**kwargs)


def compact(report):
    hf=report['scales'][0]
    return dict(HF=hf['core'],flat_spurious=hf['teacher_flat_spurious'],
        reference_detail_tiles=hf['tile_aggregate']['reference_detail_strata'][0]['eligible_tiles'],
        edge_curve=report['edges']['threshold_curves'][0]['curves'],
        all_fixed_tiles=hf['tile_aggregate']['total_fixed_tiles'])


def main(output):
    size=128;line=np.full((size,size,3),.3);line[:,63]=.8
    flat=np.full_like(line,.3)
    thin=dict(identity=evaluate(line,line),erased=evaluate(flat,line),
              blurred=evaluate(ndi.gaussian_filter(line,(2,2,0),mode='nearest'),line),
              flat_reference_NA=evaluate(flat,flat))
    yy,xx=np.indices((size,size));wave=.15*np.cos(2*np.pi*xx/16)
    target=np.repeat((.4+wave)[...,None],3,-1)
    phase=evaluate(np.repeat((.4+.15*np.sin(2*np.pi*xx/16))[...,None],3,-1),target)
    inverse=evaluate(np.repeat((.4-wave)[...,None],3,-1),target)
    teacher_flat=np.full_like(target,.4)
    spurious=evaluate(teacher_flat+.02*np.cos(2*np.pi*xx/6)[...,None],teacher_flat)
    alpha=.975+.025*np.cos(2*np.pi*xx/8);raster=alpha[...,None]*teacher_flat
    legacy=evaluate(raster,raster,color_space='premult_srgb')
    material=evaluate(raster/alpha[...,None],teacher_flat,color_space='stable_alpha_unpremult_srgb')
    report=dict(schema='road_detail_synthetic_controls.v1',
        role='analytic engineering controls, not real-scene quality acceptance',
        thin_line_color_MAE_erased=float(abs(flat-line).mean()),
        thin_line_color_MAE_blurred=float(abs(ndi.gaussian_filter(line,(2,2,0),mode='nearest')-line).mean()),
        thin_line={key:compact(value)for key,value in thin.items()},
        thin_line_case_ranking=rank_cases(thin,metric_path=('scales',0,'core','HF_relative_RMSE'),higher_is_worse=True),
        equal_energy_wrong_phase=compact(phase),inverted_polarity=compact(inverse),
        teacher_flat_spurious=compact(spurious),
        alpha_control=dict(legacy_reference_HF_RMS=legacy['scales'][0]['core']['reference_HF_RMS'],
            legacy_HF_error=legacy['scales'][0]['core']['HF_MAE'],
            material_reference_HF_RMS=material['scales'][0]['core']['reference_HF_RMS'],
            material_HF_error=material['scales'][0]['core']['HF_MAE']),
        training_updates=0,native_calls=0,GPU_operations=0,
        real_data_evaluated=False,automatic_quality_pass=False)
    output=Path(output)
    if output.exists():raise FileExistsError('preserve existing control summary')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),thin_erasure_MAE=report['thin_line_color_MAE_erased'],
        erased_relative_HF_error=thin['erased']['scales'][0]['core']['HF_relative_RMSE'],
        wrong_phase_cosine=phase['scales'][0]['core']['HF_cosine'],
        inverted_polarity_cosine=inverse['scales'][0]['core']['HF_cosine'],
        cases_retained=len(report['thin_line_case_ranking']['cases']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
