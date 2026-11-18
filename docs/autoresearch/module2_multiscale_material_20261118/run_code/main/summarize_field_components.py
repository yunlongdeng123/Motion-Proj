"""正式结束后的CPU有界场分量审计；不训练、不调用native。"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse,hashlib,json,sys
import numpy as np
import torch
from world_material_field import blur

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def summarize(value,support):
    weight=support.to(value.dtype);den=weight.sum().clamp_min(1)
    return dict(weighted_DC=((value*weight).sum((0,1))/den).tolist(),
        weighted_RMS=float(((value.square()*weight).sum()/(3*den)).sqrt()),
        support_abs_max=float(value[weight[...,0]>0].abs().max()),
        support_nodes=int((weight[...,0]>0).sum()))
def main(run,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU-only analysis requires CUDA hidden')
    torch.set_num_threads(1)
    run=Path(run);output=Path(output)
    if output.exists():raise FileExistsError('preserve existing component audit')
    result=json.loads((run/'results.json').read_text())
    if result['formal_physical_updates']!=1400:
        raise ValueError('formal pair incomplete')
    events=[json.loads(line) for line in (run/'updates.jsonl').read_text().splitlines() if line.strip()]
    stage_updates={stage:sum(e['stage']==stage for e in events) for stage in ['A','L','B','C']}
    frame_updates={str(raw):sum(e['raw']==raw for e in events) for raw in [86,110]}
    if stage_updates!={'A':200,'L':600,'B':300,'C':300} or frame_updates!={'86':700,'110':700}:
        raise ValueError('physical update schedule mismatch')
    state={};files=[]
    with torch.no_grad():
        for stage in ['A200','L800','B500','ABC800']:
            path=run/stage/'asset/world_field.pt';files.append(path)
            d=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            s=d['state'];setting=d['settings']
            low=setting['low_bound']*torch.tanh(s['low'])*s['low_parameter_mask']
            raw=torch.tanh(s['high'])*s['high_parameter_mask']
            high_basis=.5*setting['high_bound']*(raw-blur(raw,setting['highpass_sigma_cells']))
            high=s['mH']*high_basis
            low_mask=s['low_parameter_mask'];high_mask=s['high_output_mask']
            leakage=blur(high,setting['highpass_sigma_cells'])
            with np.load(run/stage/'asset/road.npz',allow_pickle=False) as road:
                with np.load(run/stage/'asset/field_scope.npz',allow_pickle=False) as sc:
                    color=road['reference_srgb'][sc['active_world_rows'][sc['eligible']]]
            entry=dict(low=summarize(low,low_mask),effective_mH_high=summarize(high,high_mask),
                effective_high_blurred_lowfrequency=summarize(leakage,high_mask),
                low_tanh_abs_over_point99=int((torch.tanh(s['low']).abs()>.99).sum()),
                high_tanh_abs_over_point99=int((torch.tanh(s['high']).abs()>.99).sum()),
                eligible_shader_channels_at_zero=int((color==0).sum()),eligible_shader_channels_at_one=int((color==1).sum()),
                eligible_shader_rows_touching_zero_or_one=int(np.any((color==0)|(color==1),axis=1).sum()))
            state[stage]=dict(low=low,high=high,low_mask=low_mask,high_mask=high_mask,summary=entry)
        if not torch.equal(state['A200']['low'],state['B500']['low']):
            raise AssertionError('bounded L changed in stageB')
        if torch.count_nonzero(state['A200']['high']) or torch.count_nonzero(state['L800']['high']):
            raise AssertionError('L-only branch has nonzero H')
        drift=dict(C_low_bounded_spatial_delta=summarize(state['ABC800']['low']-state['B500']['low'],state['ABC800']['low_mask']),
            C_effective_mH_high_spatial_delta=summarize(state['ABC800']['high']-state['B500']['high'],state['ABC800']['high_mask']))
    receipt=dict(schema='multiscale_bounded_field_component_CPU_audit.v1',GPU_executed=False,native_calls=0,optimizer_updates=0,
        actual_physical_updates_by_stage=stage_updates,actual_physical_updates_by_frame=frame_updates,
        stages={key:v['summary'] for key,v in state.items()},drift=drift,B_bounded_low_exact_A=True,L_control_high_exact_zero=True,
        component_domain='bounded spatial field before S_Sigma filtering; low at12cm knots, effective mH*H at2cm nodes; statistics use qualified parameter/query support, not full-atlas zeros',
        clamp_boundary='cached shader counts identify exact0/1, not proof that an unclamped value exceeded the boundary',
        strict_band_orthogonality_claim=False,default_scene_modified=False,
        source_binding={str(path):sha(path) for path in [Path(__file__),run/'results.json',run/'updates.jsonl',*files]})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=P/'formal_r1');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run,a.output)

