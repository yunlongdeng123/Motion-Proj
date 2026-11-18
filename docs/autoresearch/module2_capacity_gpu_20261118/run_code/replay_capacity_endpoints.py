"""只读补存原生颜色范围端点；同一冻结场景，两scope×五帧，0优化更新。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

G=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    sys.path.insert(0,str(G/'code'))
    from road_capacity_native import assert_gpu_lease,load_context
    assert_gpu_lease(G/'gpu_lease.json','stage_qa')
    original=read(G/'check2/native_primary_r1/binding.json')
    helper=G/'code/road_capacity_native.py'
    if sha(helper)!=original['helper_sha256']:raise RuntimeError('helper differs from original primary operator')
    source=G/'check2/native_primary_r1/runner_source.py'
    spec=importlib.util.spec_from_file_location('original_capacity_runner',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data={scope:module.Data(C/'ideal_targets_r2',scope) for scope in ('primary','relaxed')}
    for value in data.values():value.check_binding()
    out=G/'check2/endpoint_replay_r1';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();ctx=load_context(lease_file=G/'gpu_lease.json',lease_owner='stage_qa')
    import torch
    summaries=[];comparison=[]
    try:
        for scope in ('primary','relaxed'):
            d=data[scope];directory=out/scope;directory.mkdir()
            if not np.array_equal(ctx.world_row_ids,d.active_world):raise ValueError('active mapping differs')
            eligible=torch.as_tensor(d.eligible,device=ctx.base_rgb.device)
            lo=ctx.base_rgb.detach().clone();hi=ctx.base_rgb.detach().clone();lo[eligible]=0;hi[eligible]=1
            arrays={}
            with torch.no_grad():
                for raw in module.FRAMES:
                    lower=ctx.render_road(raw,lo);upper=ctx.render_road(raw,hi)
                    b=lower['rgb'].cpu().numpy();u=upper['rgb'].cpu().numpy()
                    a=lower['alpha'].squeeze().cpu().numpy();reference=d.geometry[raw]['fixed_alpha']
                    err=float(np.max(np.abs(a-reference)))
                    if err>1e-6:raise RuntimeError('frozen alpha differs')
                    np.savez_compressed(directory/f'{raw:03d}.npz',fixed_b=b,box_upper=u,fixed_alpha=a,D_color=d.masks[raw],
                        q=d.masks[raw].astype(np.float32),eligible_world=d.eligible_world,active_world_rows=d.active_world)
                    arrays[raw]=(b,u)
                    summaries.append(dict(scope=scope,raw=raw,D_color_pixels=int(d.masks[raw].sum()),alpha_max_error=err,
                        arrays=str(directory/f'{raw:03d}.npz'),keys=['fixed_b','box_upper','fixed_alpha','D_color','q','eligible_world','active_world_rows']))
            run='native_primary_r1' if scope=='primary' else 'native_relaxed_r1'
            for case in d.cases:
                target=d.target(case);total=0.
                for raw in module.FRAMES:
                    b,u=arrays[raw];M=d.masks[raw]
                    y=(target[raw]['target_rgb_srgb']*d.geometry[raw]['fixed_alpha'][None])[:,M].astype(np.float64)
                    violation=np.maximum(np.maximum(b[:,M].astype(np.float64)-y,y-u[:,M].astype(np.float64)),0.)
                    total+=float(np.sum(violation**2))
                floor=.5*total/d.denominator
                old=read(G/'check2'/run/case/'evaluation/evaluation.json')['certificate']['guarded_box_range_floor_numeric_proxy']
                comparison.append(dict(scope=scope,case=case,recomputed_floor_numeric_proxy=floor,original_logged_floor=old,absolute_difference=abs(floor-old)))
        ctx.assert_frozen()
        receipt=dict(schema='read_only_native_endpoint_postevaluation.v1',role='necessary engineering evidence replay, not additional solver experiment',
            root_authorization='2026-11-18 explicit reactivated stage_qa lease for20 read-only endpoint forwards',
            native_forward_calls=ctx.native_calls,optimizer_updates=0,solver_iterations=0,original_results_modified=False,
            source_operator_helper_sha256=sha(helper),original_runner_source_sha256=sha(source),source_binding_verified=True,
            frames=summaries,floor_comparison=comparison,all_floor_absolute_differences_le_1e_12=all(v['absolute_difference']<=1e-12 for v in comparison),
            strict_numeric_bound_established=False,negative_certificate_allowed=False,GPU_executed=True,
            elapsed_seconds=time.monotonic()-start,human_verdict=None)
        module.write(out/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    finally:ctx.close()


if __name__=='__main__':main()
