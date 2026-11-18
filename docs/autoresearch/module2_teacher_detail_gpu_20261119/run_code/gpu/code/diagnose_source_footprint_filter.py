"""One geometry-chosen zero-update world-H filtering diagnostic, never a sigma sweep."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
R=Q/'gpu_20261119';P=Q.parent/'road_multiscale_material_20261118_r1'
sys.path[:0]=[str(R/'field_training/code'),str(Q/'code'),str(R)]
from training_factory import prepare_inputs
from world_texture_runtime import WorldTextureRuntime,FixedPixelReadLayout
from evaluation.material_evaluation import evaluate_arm

def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy() for k in z.files}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

out=R/'source_footprint_filter_r1';out.mkdir(exist_ok=False)
started=time.time();torch.set_num_threads(8)
samples=[]
for raw in (86,110):
    m=load(Q/'pixel_render/maps'/f'{raw:03d}.npz');t=load(P/'stage0_native_r1'/f'{raw:03d}_targets.npz')
    J=m['pixel_uv_jacobian'][t['C']|t['S']]
    # Fixed once from geometric source resolution; no final prediction, RGB, or metric is read.
    samples.append(np.sqrt(np.sum(J*J,axis=-1)/12))
std=np.concatenate(samples)
su,sv=np.median(std,axis=0)
contract=dict(schema='source_footprint_world_H_zero_update_diagnostic.v1',
    source_arm='M11',source_stage='C800',training_updates=0,source_masks='frozen teacher C union S, raw86/110',
    source_pixel_samples=len(std),sigma_uv=[float(su),float(sv)],
    nominal_sigma_m=[float(.02*su),float(.02*sv)],
    filter='diagonal Gaussian matching median source-pixel-box per-axis second moments; spatially constant diagnostic approximation',
    formula='sigma_axis=median(sqrt(sum_pixelaxes J_uv[axis,:]^2 / 12))',
    one_recipe_no_parameter_sweep=True,chosen_before_reading_filtered_outputs=True,
    L_A0_mH_geometry_coverage_readonly_unchanged=True,
    limitation='Ignores off-diagonal/spatial variation. Zero-update filtering control, not a trained method or a representation upper-limit test.')
write(out/'contract.json',contract)
model,metadata=WorldTextureRuntime.load(R/'formal_r1/M11/C800/field.pt',device='cpu')
with torch.no_grad():low,high=model.fields()
L=low.numpy();H=high.numpy()
filtered=gaussian_filter(H,sigma=(sv,su,0),mode='nearest',truncate=3)
np.savez_compressed(out/'physical_fields.npz',low=L,high=filtered,
    H_gate=model.H_gate.numpy(),valid=model.valid.numpy(),base=model.base.numpy())
prepared=prepare_inputs(device='cpu');rendered=out/'rendered';rendered.mkdir()
reload_error=0.
with np.load(out/'physical_fields.npz')as z:storedL=z['low'];storedH=z['high']
for raw in (86,110,134,179,198):
    op=prepared.pixel[raw]
    values=op.numpy_query(L,filtered,'observed_plus_teacher_H').astype(np.float32)
    replay=op.numpy_query(storedL,storedH,'observed_plus_teacher_H').astype(np.float32)
    reload_error=max(reload_error,float(np.max(np.abs(values-replay))))
    rgb=np.zeros((320*480,3),np.float32);rgb[op.row_ids]=values;rgb=rgb.reshape(320,480,3).transpose(2,0,1)
    valid=np.zeros(320*480,bool);valid[op.row_ids]=op.mass>1e-12;valid=valid.reshape(320,480)
    result={};K=prepared.basis[raw]['joint']['K']
    for mode in ('road','joint'):
        b=prepared.basis[raw][mode]
        replacement=b['eligible_weight'][None]*rgb
        result[mode+'_rgb']=b['baseline_rgb']+np.where(valid[None],replacement-b['eligible_rgb'],np.float32(0))
        result[mode+'_alpha']=b['alpha']
        Image.fromarray(np.rint(np.clip(result[mode+'_rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)).save(rendered/f'{raw:03d}_{mode}.png')
    result['K']=K
    assert np.max(np.abs(result['joint_rgb']-prepared.basis[raw]['joint']['baseline_rgb'])[:,K])==0
    np.savez_compressed(rendered/f'{raw:03d}.npz',**result)
metrics=evaluate_arm({raw:rendered/f'{raw:03d}.npz'for raw in (86,110,134,179,198)},
    R/'evaluation/frozen_r2/manifest.json',out/'evaluation',arm_id='M11_source_footprint_filter',stage='C800_zero_update')
baseline=json.loads((R/'evaluation/results_r2/M11/metrics.json').read_text())
summary=dict(contract=contract,seconds=time.time()-started,mechanical_pass=metrics['mechanical_pass'],
    field_reload_query_max=reload_error,optimizer_updates=0,native_calls=0,GPU_calls=0,
    pooled_before=baseline['pooled'],pooled_after=metrics['pooled'],
    detail_before=baseline['detail'],detail_after=metrics['detail'],
    diagnostic_stripe=[dict(raw=r['raw'],before=next(v['stripe']for v in baseline['rows']if v['raw']==r['raw']),after=r['stripe'])for r in metrics['rows']if r['raw']in(134,179,198)],
    field_H_RMS_before=float(np.sqrt(np.mean(H**2))),field_H_RMS_after=float(np.sqrt(np.mean(filtered**2))),
    state='DIAGNOSTIC_ONLY_NOT_PROMOTED',production_promoted=False,human_verdict=None)
write(out/'results.json',summary)
print(json.dumps(dict(sigma_uv=contract['sigma_uv'],photo_before=baseline['pooled'],photo_after=metrics['pooled'],
    C_HF_before=baseline['detail']['C']['1px'],C_HF_after=metrics['detail']['C']['1px'],
    S_HF_before=baseline['detail']['S']['1px'],S_HF_after=metrics['detail']['S']['1px'],
    stripe=summary['diagnostic_stripe']),indent=2),flush=True)
