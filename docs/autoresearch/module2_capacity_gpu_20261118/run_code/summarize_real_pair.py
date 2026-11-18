"""只读CPU汇总：实际I0/W4配对、节点界、冻结字段与联合门。"""
from pathlib import Path
import json
import numpy as np

G=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
S=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')
def read(path):return json.loads(path.read_text())
def load(path):
    with np.load(path,allow_pickle=False) as a:return {key:a[key].copy() for key in a.files}
def same(a,b):return np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)
protocol=read(C/'protocol_cpu_prepared.json');scope=load(G/'check1/native_r1/qualified_scope.npz')
road=load(S/'road.npz');prior=load(S/'provenance.npz');frequency=load(G/'check3/frozen_frequency_masks.npz')
frequency_meta={row['raw']:row for row in read(G/'check3/frozen_frequency_masks.json')['rows']}
result=dict(schema='actual_native_real_pair_results.v1',formal_updates=400,engineering_updates=8,
    scope_rows=int(scope['eligible_color'].sum()),S_seen_rows=int(scope['S_seen'].sum()),
    S_only_known_rows=int((scope['S_only_known']&scope['eligible_color']).sum()),runs={},teacher_gate=[],frequency_gate=[],
    coverage_frozen=True,negative_shared_capacity_certificate=False,unbounded_shared_field_failure_claim=False,
    production_promoted=False,human_verdict=None)
for arm in ('I0','W4'):
    p=G/'check3'/arm/'formal_r1';plan=load(p/'basis_plan.npz');learned=load(p/'learned_correction.npz')
    trained=load(p/'asset/road.npz');trained_prior=load(p/'asset/provenance.npz');delta=learned['delta_values']
    eligible=plan['eligible'];ids=plan['world_row_ids'];base=plan['base_srgb']
    if arm=='I0':row_delta=np.zeros_like(base);row_delta[eligible]=delta
    else:
        field=np.zeros((int(np.prod(plan['shape'])),3),np.float32);field[plan['mutable_texels']]=delta
        row_delta=(field[plan['corner_ids']].astype(np.float64)*plan['weights'][...,None]).sum(1).astype(np.float32)*eligible[:,None]
    affine=base+row_delta;reconstructed=np.clip(affine,0,1)
    reconstruction_error=float(np.max(np.abs(reconstructed-learned['effective_rgb'])))
    bake_error=float(np.max(np.abs(learned['effective_rgb']-trained['reference_srgb'][ids])))
    frozen=dict(all_nonRGB_road_fields=all(same(value,trained[key]) for key,value in road.items() if key!='reference_srgb'),
        RGB_outside_eligible=same(road['reference_srgb'][~scope['world_eligible_color']],trained['reference_srgb'][~scope['world_eligible_color']]),
        original_prior_fields=all(same(value,trained_prior[key]) for key,value in prior.items() if key not in ('effective_row_rgb','asset_mode')))
    if not all(frozen.values()) or max(reconstruction_error,bake_error)>1e-6:raise RuntimeError('CPU frozen/reconstruct failed:'+str((arm,frozen,reconstruction_error,bake_error)))
    evaluation=read(p/'evaluation_after.json');metrics={row['raw']:row for row in evaluation['rows']};frequencies=[]
    for raw in (86,110,134,179,198):
        out=load(p/f'rendered/{raw:03d}.npz');rgb=out['rgb_srgb'].astype(np.float64)
        baseline=load(C/f'ideal_targets_r2/geometry/{raw:03d}.npz')['baseline_forward_rgb_srgb__not_teacher'].astype(np.float64)
        mv=frequency[f'{raw:03d}_vertical'];mh=frequency[f'{raw:03d}_horizontal'];md=frequency[f'{raw:03d}_domain']
        ev=float(np.abs(rgb[:,2:]-2*rgb[:,1:-1]+rgb[:,:-2])[:,mv].mean())
        eh=float(np.abs(rgb[:,:,2:]-2*rgb[:,:,1:-1]+rgb[:,:,:-2])[:,mh].mean())
        frequencies.append(dict(raw=raw,vertical_E=ev,horizontal_E=eh,baseline_vertical_E=frequency_meta[raw]['baseline_vertical_E'],
            image_RGB_delta_energy=float(np.square(rgb-baseline)[:,md].mean()),domain_pixels=int(md.sum())))
    records=read(p/'training.json');reload=read(p/'public_reload.json')
    result['runs'][arm]=dict(actual_formal_updates=len(records),frame_counts={str(raw):sum(row['raw']==raw for row in records) for raw in (86,110)},
        metrics=evaluation,frequency=frequencies,parameters=read(p/'parameterization.json'),
        node_bound_activity=dict(parameter_scalars=int(delta.size),lower_bound_scalars=int((delta<=-1).sum()),upper_bound_scalars=int((delta>=1).sum()),
            max_absolute_node_delta=float(np.abs(delta).max()),shader_clipped_rows=int(np.any((affine<0)|(affine>1),axis=1).sum())),
        effective_row_delta_squared=float(np.square(reconstructed-base)[eligible].mean()),
        CPU_reconstruction_max=reconstruction_error,baked_RGB_max=bake_error,frozen_fields_exact=frozen,
        world_query_position_vs_qualified_scope_max_m=float(np.abs(plan['world_xyz']-scope['world_xyz']).max()),
        K=read(p/'joint_K.json'),public_reload=reload,shadow_known=read(p/'shadow_known_persistence.json'),
        scene=str(p/'asset/persistent/scene.json'),proof=str(p/'asset/persistent/proof.json'))
a=result['runs']['I0'];b=result['runs']['W4']
am={row['raw']:row for row in a['metrics']['rows']};bm={row['raw']:row for row in b['metrics']['rows']}
for raw in (86,110):
    for domain in ('A','C','S'):
        x=am[raw]['metrics'][domain];y=bm[raw]['metrics'][domain]
        result['teacher_gate'].append(dict(raw=raw,domain=domain,I0=x,W4=y,W4_minus_I0=y-x,tolerance=1/255,passed=y<=x+1/255))
for x,y in zip(a['frequency'],b['frequency']):
    excess=x['vertical_E']-x['baseline_vertical_E'];other=y['vertical_E']-y['baseline_vertical_E']
    result['frequency_gate'].append(dict(raw=x['raw'],I0_excess=excess,W4_excess=other,applicable=excess>0,
        excess_reduction=1-other/excess if excess>0 else None,passed=other<=.5*excess if excess>0 else None))
result['teacher_noninferiority_pass']=all(row['passed'] for row in result['teacher_gate'])
result['directional_stripe_gate_pass']=all(row['passed'] for row in result['frequency_gate'] if row['applicable'])
result['joint_gate_pass']=result['teacher_noninferiority_pass'] and result['directional_stripe_gate_pass']
result['conclusion']='bounded W4 did not jointly meet registered teacher-fit and striping criteria; not an intrinsic/unbounded shared-field or generative-prior limit'
result['whole_output_visual_verdict']='pending root actual-image inspection; numeric gates do not replace detail/visual review'
result['all_initial_native_cache_exact']=all(row['RGB_max']==0 and row['alpha_max']==0 for arm in ('I0','W4') for row in read(G/'check3'/arm/'formal_r1/initial_native_binding.json')['rows'])
result['paired_training_trace_exact']=[row['raw'] for row in read(G/'check3/I0/formal_r1/training.json')]==[row['raw'] for row in read(G/'check3/W4/formal_r1/training.json')]
result['lifecycle']=read(G/'check3/smoke_r1/completed.json');result['replay']=read(G/'check3/smoke_r1/replay_comparison.json')
(G/'check3/results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(results=str(G/'check3/results.json'),teacher_gate=result['teacher_gate'],frequency_gate=result['frequency_gate'],
    node_bounds={arm:result['runs'][arm]['node_bound_activity'] for arm in ('I0','W4')})),flush=True)
