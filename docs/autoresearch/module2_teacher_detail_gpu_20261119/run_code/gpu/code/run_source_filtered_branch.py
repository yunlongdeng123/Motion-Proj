"""One prospective source-footprint parameterization control, same logical budget as M11."""
from pathlib import Path
import json,sys,time,traceback
import numpy as np
import torch
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
R=Q/'gpu_20261119';S=R/'source_filtered_training'
sys.path[:0]=[str(S/'code'),str(R/'field_training/code'),str(R/'code'),str(Q/'code')]
from training_factory import default_config,prepare_inputs,load_native_adapter
from source_filtered_field import SourceFootprintWorldTexture,make_source_filtered_state,rebase_zero_H_common_A
from field_training_state import max_tree_difference,cpu_clone
from run_formal_factorial import export

def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def tload(p):return torch.load(p,map_location='cpu',weights_only=False)

def main():
    out=S/'formal_r1';out.mkdir(exist_ok=False)
    start=time.time();adapter=None;smoke_updates=0;formal_updates=0
    contract=S/'prospective_contract.json';spec=json.loads(contract.read_text())
    assert spec['schedule']['additional_physical_updates']==600 and spec['additional_fallbacks_allowed']==0
    config=default_config();config['formal_launch_contract']=str(contract)
    config['additional_source_bindings']+= [str(p) for p in sorted((S/'code').glob('*.py'))]+[str(contract),__file__]
    torch.set_num_threads(8)
    write(out/'started.json',dict(time=start,PID=__import__('os').getpid(),formal_updates=0,smoke_updates=0))
    try:
        prepared=prepare_inputs(config,device='cuda');adapter=load_native_adapter(config)
        # Refresh one diagnostic camera through the actual rasterizer in this new context.
        sys.path.insert(0,str(Q/'pixel_render/code'))
        from native_basis_adapter import export_native_basis,scope_signature
        smoke_auth=dict(schema='pixel_native_basis_smoke.v1',phase='native_smoke',native_allowed=True,
            gpu_authorized=True,formal_allowed=False,frames=[179],
            scope_world_rows_sha256=scope_signature(prepared.active_world_rows,prepared.qualified),
            max_native_forward_calls=2,delete_actor_ids=[10])
        write(out/'native_smoke_contract.json',smoke_auth)
        export_native_basis(adapter.ctx,179,prepared.qualified,out/'fresh179',authorization=smoke_auth)
        for mode in ('road','joint'):
            with np.load(out/'fresh179'/f'{mode}_basis.npz')as z:
                for key in ('baseline_rgb','eligible_rgb','eligible_weight','alpha','K'):
                    assert np.array_equal(z[key],prepared.basis[179][mode][key]),('Fresh basis differs',mode,key)
        smoke=make_source_filtered_state(prepared,adapter,role='smoke',arm='M12',stage_counts=(2,2,2),config=config)
        trace=[]
        for _ in range(2):trace.append(smoke.step());smoke_updates+=1
        saved=smoke.save(out/'smoke_A2');a=tload(saved/'checkpoint.pt')
        trace.append(smoke.step());smoke_updates+=1;expected=smoke.snapshot()
        smoke.restore(a);restore_diff=max_tree_difference(a,smoke.snapshot())
        trace.append(smoke.step());smoke_updates+=1;next_diff=max_tree_difference(expected,smoke.snapshot())
        assert restore_diff==0 and next_diff<=1e-6
        for _ in range(3):trace.append(smoke.step());smoke_updates+=1
        saved=smoke.save(out/'smoke_ABC6')
        reloaded,_=SourceFootprintWorldTexture.load(saved/'field.pt',device='cuda')
        reload_error=0.
        with torch.no_grad():
            for raw in (86,110,134,179,198):
                actual=smoke.reader.render(smoke.model,raw,joint=True)
                replay=smoke.reader.render(reloaded,raw,joint=True)
                for key in ('road_rgb','joint_rgb','road_alpha','joint_alpha'):
                    reload_error=max(reload_error,float((actual[key]-replay[key]).abs().max()))
        assert reload_error==0
        smoke_receipt=dict(verified=True,native_GPU_executed=True,engineering_updates=smoke_updates,
            full_restore_max=restore_diff,next_step_full_snapshot_max=next_diff,
            specialized_field_reload_render_max=reload_error,stage_entries=['A','B','C'],
            all_gradients_finite=True,H_gradient_nonzero=all(r['gradients']['high']>0 for r in trace if r['stage']!='A'),
            formal_state_uses_smoke_weights=False,new_native_forward_calls=adapter.ctx.native_calls,
            fresh179_basis_exact=True,seconds=time.time()-start)
        assert smoke_receipt['H_gradient_nonzero']
        write(out/'smoke_receipt.json',smoke_receipt)
        del smoke,reloaded
        state=make_source_filtered_state(prepared,adapter,role='formal',arm='M12',config=config)
        parent=tload(R/'formal_r1/pixel_common_A200/checkpoint.pt')
        rebased=rebase_zero_H_common_A(parent,state,output_receipt=out/'A200_equivalence.json')
        state.fork_common_A(rebased)
        # Zero-H source filtering must preserve the original common A output exactly.
        with torch.no_grad():
            for raw in (86,110,134,179,198):
                pred=state.reader.render(state.model,raw,joint=True)
                with np.load(R/'formal_r1/pixel_common_A200/rendered'/f'{raw:03d}.npz')as z:
                    for key in ('road_rgb','joint_rgb'):
                        if float(np.abs(pred[key].cpu().numpy()-z[key]).max())!=0:raise RuntimeError('Zero-H A200 renderer equivalence failed')
        write(out/'input_bindings.json',prepared.bindings)
        lower=state.model.low.detach().clone();adamL=cpu_clone(state.optimizer.state[state.model.low]);trace=[]
        for end in (500,800):
            while state.cursor<end:
                r=state.step();formal_updates+=1;trace.append(r)
                if state.cursor%50==0:print(json.dumps(dict(arm='M12',physical=formal_updates,**r)),flush=True)
            if end==500:
                assert torch.equal(lower,state.model.low)
                assert max_tree_difference(adamL,cpu_clone(state.optimizer.state[state.model.low]))==0
            saved=state.save(out/('B500'if end==500 else'C800'))
            export(state,prepared,saved/'rendered');write(saved/'optimization.json',trace)
        final=out/'C800';loaded,_=SourceFootprintWorldTexture.load(final/'field.pt',device='cuda')
        field_max=0.
        with torch.no_grad():
            for raw in (86,110,134,179,198):
                a=state.reader.render(state.model,raw,joint=True);b=state.reader.render(loaded,raw,joint=True)
                field_max=max(field_max,float((a['road_rgb']-b['road_rgb']).abs().max()),float((a['joint_rgb']-b['joint_rgb']).abs().max()))
        assert formal_updates==600 and field_max==0
        adapter.assert_frozen()
        result=dict(schema='source_filtered_same_budget_completed.v1',completed=True,
            arm='M12',logical_updates=800,additional_formal_physical_updates=formal_updates,
            separate_engineering_updates=smoke_updates,original_factorial_physical_updates=3400,
            combined_formal_physical_updates=4000,combined_engineering_updates=17+smoke_updates,
            specialized_field_reload_render_max=field_max,source_sigma_uv=spec['source_sigma_uv'],
            native_forward_calls=adapter.ctx.native_calls,seconds=time.time()-start,
            all_geometry_coverage_and_ledgers_frozen=True,parameterization_only_changed=True,
            original_A200_output_exact=True,B_low_and_Adam_exact=True,
            GPU_keep_on=True,power_operations=0,production_promoted=False,human_verdict=None)
        write(out/'completed.json',result);print('COMPLETED',json.dumps(result),flush=True)
    except Exception:
        write(out/'failed.json',dict(error=traceback.format_exc(),smoke_updates=smoke_updates,
            formal_updates=formal_updates,seconds=time.time()-start));raise
    finally:
        if adapter is not None:adapter.ctx.close()

if __name__=='__main__':main()
