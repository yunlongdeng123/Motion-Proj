"""CPU构造真实冻结A0、B32独立pseudo层、Σ核与H门；不加载native。"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json
import numpy as np
from scipy.ndimage import maximum_filter
import torch
from world_material_field import gaussian_plan,WorldMaterialField

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')
COVER=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded')

def load(path,keys=None):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in (f.files if keys is None else keys)}

def main(output,chart_cache=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    a=load(BASE/'base_atlas.npz');r=load(COVER/'road.npz')
    p=load(COVER/'provenance.npz',['world_xyz','known','unknown','source','B32_new','effective_material_available','effective_material_source','effective_material_source_raw','effective_material_source_uv_native'])
    active=np.flatnonzero(r['active']);g=a['grid_world_row'];A0=a['rgb_srgb'].copy()
    prior_source=a['source'].copy();source=prior_source.copy();confidence=a['confidence'].copy();valid=a['legal']&~a['unknown']&(source>0)
    pseudo=p['B32_new'][g];A0[pseudo]=r['reference_srgb'][g[pseudo]];source[pseudo]=3;confidence[pseudo]=.25;valid[pseudo]=True
    A0[~valid]=0 # 未知占位不作为材质输入；unknown原台账另存。
    observed=(source==1)&a['known']&valid
    marks=load(BASE/'markings.npz',['observed_white'])['observed_white'][g]&observed
    detail=maximum_filter(marks.astype(np.float64),size=7,mode='constant')
    mH=observed*confidence*(.25+.75*detail)
    plan=gaussian_plan(p['world_xyz'][active],r['normal_world'][active],r['gaussian_sigma_m'][active],a['world_from_uv'],order=15)
    if chart_cache is not None:
        chart_cache=Path(chart_cache)
        if not (chart_cache/'receipt.json').exists():raise ValueError('真实chart缓存尚未完成')
        cached=load(chart_cache/'geometry_plan.npz')
        if not np.array_equal(cached['active_world_rows'],active):raise ValueError('真实chart active行身份不符')
        plan.update(cached);plan['sample_uv']=np.load(chart_cache/'sample_uv_f64.npy',mmap_mode='r')
    np.savez_compressed(output/'A0.npz',rgb_srgb=A0,valid=valid,source_prior=prior_source,source_effective=source,
        confidence_effective=confidence,unknown_prior=a['unknown'],unknown_effective=~valid,
        world_from_uv=a['world_from_uv'],grid_world_row=g,mH=mH,observed_marking_detail_support=detail,
        B32_frozen_pseudo_layer=pseudo)
    geometry={k:v for k,v in plan.items() if k not in ['sample_uv','center_receipt','sampling_receipt']}
    geometry.update(active_world_rows=active,world_xyz=p['world_xyz'][active],normal_world=r['normal_world'][active],sigma_m=r['gaussian_sigma_m'][active],
        row_material_available=p['effective_material_available'][active],legacy_rgb=r['reference_srgb'][active])
    np.savez_compressed(output/'geometry_kernel.npz',**geometry)
    model=WorldMaterialField(A0,valid,mH,plan)
    with torch.no_grad():
        filtered,mass=model.query();center,_=model.query(center_only=True)
    model.save(output/'field_initial.pt',dict(A0_source=str(BASE/'base_atlas.npz'),coverage_parent=str(COVER),
        pseudo_source='B32 frozen native teacher; prior ledger unchanged',world_from_uv=a['world_from_uv'].tolist(),
        kernel='fixedGH15 tangent Gaussian, normalized valid source mass',high_mask='source1 AND known *confidence*(.25+.75*trueobserved-white7x7support)',
        chart_cache=None if chart_cache is None else str(chart_cache),geometry_kernel=str(output/'geometry_kernel.npz'),
        numeric_filter_qualification='P/cpu_numeric_r1 truechart GH15 qualification; bind receipt separately',I0_colors_used=False))
    filtered=filtered.numpy();center=center.numpy();mass=mass.numpy();available=p['effective_material_available'][active]
    np.savez_compressed(output/'initial_row_queries.npz',active_world_rows=active,filtered_A0_rgb=filtered,center_A0_rgb=center,
        kernel_valid_mass=mass,row_material_available=available,legacy_fallback_required=(~available)|(mass<=1e-12))
    result=dict(schema='world_multiscale_material_CPU_preparation.v1',GPU_executed=False,native_calls=0,optimizer_updates=0,
        A0_source=str(BASE/'base_atlas.npz'),coverage_parent=str(COVER),active_rows=len(active),
        B32_pseudo_texels=int(pseudo.sum()),active_prior_unknown_rows=int((~available).sum()),
        center_vs_filtered_RGB_abs_max=float(np.abs(filtered-center)[available].max()),
        center_vs_filtered_RGB_mean=float(np.abs(filtered-center)[available].mean()),
        kernel_valid_mass_quantiles=np.quantile(mass[available],[0,.05,.5,.95,1]).tolist(),
        H_draft=dict(low_spacing_m=.12,high_spacing_m=.02,low_bound_sRGB=.30,high_bound_sRGB=.04,
            source_gate='source1 AND known *confidence*(.25+.75*trueobserved marking7x7 support)',highpass_sigma_m=.06,
            product_mH_H_inside_filter=True,strict_band_separation_claim=False),
        H_gate_source_counts={str(int(k)):int(((mH>0)&(source==k)).sum()) for k in [0,1,2,3]},
        H_parameter_scalar_upper_bound=int((mH>0).sum())*3,
        exact_chart_used=chart_cache is not None,
        kernel_numerical_qualification='PENDING; truechart helper and GH15 highfrequency qualification required',formal_authorized=False)
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--chart-cache',type=Path)
    args=parser.parse_args();main(args.output,args.chart_cache)
