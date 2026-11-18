"""匹配I0的CPU输入/协议与恢复fixture；没有native执行入口。"""
from pathlib import Path
import argparse,json
import numpy as np
from matched_i0_training import P,cpu_fixture,digest,sampler_trace


def load(path,keys=None):
    with np.load(path,allow_pickle=False)as z:return {k:z[k]for k in(keys if keys else z.files)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['fixture','prepare'],default='prepare');args=parser.parse_args()
    root=P/'matched_I0';root.mkdir(exist_ok=True)
    if args.mode=='fixture':
        result=cpu_fixture();(root/'cpu_fixture.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items()if k!='trace'}));return
    contract=json.loads((P/'execution_contract.json').read_text());coverage=Path(contract['coverage_parent'])
    scope=load(P/'method_scope_r1/scope.npz');query=load(P/'cpu_exact_r1/initial_row_queries.npz')
    road=load(coverage/'road.npz',['active','reference_srgb']);rows=np.flatnonzero(road['active'])
    assert np.array_equal(scope['active_world_rows'],rows)and np.array_equal(query['active_world_rows'],rows)
    eligible=scope['eligible'].astype(bool);assert int(eligible.sum())==84266
    c0=road['reference_srgb'][rows].copy();c0[eligible]=query['filtered_A0_rgb'][eligible]
    assert np.isfinite(c0).all()and((c0>=0)&(c0<=1)).all()
    output=root/'input_r1';output.mkdir(exist_ok=False)
    np.savez_compressed(output/'initialization.npz',c0=c0,eligible=eligible,active_world_rows=rows)
    target_counts={}
    for raw in(86,110):
        t=load(P/f'stage0_native_r1/{raw:03d}_targets.npz',['A','C','S','fullS'])
        target_counts[str(raw)]={k:int(v.sum())for k,v in t.items()}
    assert sum(v['S']for v in target_counts.values())==3131
    spec=dict(schema='matched_I0_preparation.v1',coverage_parent=str(coverage),
        filtered_A0_field=str(P/'cpu_exact_r1/field_initial.pt'),CPU_initialization=str(output/'initialization.npz'),
        initialization='same zero-correction filteredA0 on scientific84266 only; other colors readonly N parent',
        native_initialization_check='under future explicitlease requery authoritative zero field on same device, compareCPUcache<=1e-6; freeze actualGPUci0',
        active_rows=len(rows),eligible_rows=int(eligible.sum()),independent_parameters=int(eligible.sum())*3,
        target_counts=target_counts,q='binary1 per frozen source-valid mask; sourceS3131 includesnew909',
        targets=str(P/'stage0_native_r1'),sampler_trace=sampler_trace(),sampler='same Random101 balanced800 trace as L/ABC',
        parameterization='ci=clip(ci0+delta,0,1);delta starts0, projected[-1,1]',
        optimizer=dict(name='Adam',lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=0),
        objective='mean_abs_A +.5mean_abs_C+mean_abs_S+.001mean_eligible((ci-ci0)^2)',
        regularization_comparison='shared L/H regularizers are own method components; control is matched source/sampling/budget, not a perfect one-variable change',
        formal_updates=800,automatic_extra_cases=False,formal_GPU_authorized=False,GPU_executed=False,native_calls=0,
        required_native_lifecycle=['finite gradient through native raster','save/eval','completeparams/Adam/RNG/sampler restore','paired nextstep with immutable snapshot','actualjointK and publicreload'],
        source_bindings={str(x):digest(x)for x in [P/'cpu_exact_r1/field_initial.pt',P/'method_scope_r1/scope.npz',
            P/'cpu_exact_r1/initial_row_queries.npz',*[P/f'stage0_native_r1/{r:03d}_targets.npz'for r in(86,110)]]},
        no_human_or_production_pass=True,human_verdict=None)
    (root/'preparation_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(dict(spec=str(root/'preparation_spec.json'),initialization=str(output/'initialization.npz'),
        active_rows=len(rows),eligible_rows=int(eligible.sum()),S3131=True,GPU_executed=False)))


if __name__=='__main__':main()
