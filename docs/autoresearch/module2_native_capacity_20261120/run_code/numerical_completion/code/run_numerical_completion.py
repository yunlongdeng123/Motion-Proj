"""Separately registered same-objective numerical completion; no held labels.

CPU and actual short lifecycle gates precede a fresh2000FG run. All original
four arms and their protocol remain immutable. Default mode is inspection.
"""
from pathlib import Path
import argparse,copy,json,sys,time,traceback
import numpy as np
from checkpointed_lbfgsb import CheckpointedLBFGSB
from numerical_runtime import NumericalRuntime,ROOT,B,FRAMES,read,write,sha


RECIPE=dict(m=10,maxls=20,factr=1e7,pgtol=1e-8,maxFG=2000)


def checkpoint(solver,runtime,path):
    import torch
    solver.save(path.with_suffix('.npz'));torch.save(runtime.rng(),path.with_suffix('.rng.pt'))
    return dict(workspace=str(path.with_suffix('.npz')),workspace_sha256=sha(path.with_suffix('.npz')),
                RNG=str(path.with_suffix('.rng.pt')),RNG_sha256=sha(path.with_suffix('.rng.pt')))


def load_checkpoint(solver,runtime,info):
    import torch
    if sha(info['workspace'])!=info['workspace_sha256']or sha(info['RNG'])!=info['RNG_sha256']:raise RuntimeError('checkpoint files changed')
    solver.load(info['workspace']);runtime.restore_rng(torch.load(info['RNG'],map_location='cpu',weights_only=False))


def smoke_reference():
    root=ROOT/'smoke_r1';root.mkdir(exist_ok=False);out=root/'reference';runtime=None;t=time.monotonic()
    try:
        runtime=NumericalRuntime(out,role='smoke',owner='native_capacity_numerical_smoke',ledger_file=root/'physical_ledger.json',maxFG=20)
        precision=runtime.qualify_precision()
        solver=CheckpointedLBFGSB(runtime.x0,runtime.evaluate,identity=runtime.identity,**dict(RECIPE,maxFG=20))
        first=solver.next_accepted()
        if first is None:raise RuntimeError('smoke requires an actual NEW_X checkpoint')
        runtime.ledger['accepted']+=1;runtime._record()
        cp=checkpoint(solver,runtime,out/'accepted1');write(out/'accepted1_manifest.json',cp)
        second=solver.next_accepted()
        if second is None:raise RuntimeError('smoke requires next actual NEW_X after checkpoint')
        runtime.ledger['accepted']+=1;runtime._record()
        checkpoint(solver,runtime,out/'reference_accepted2')
        if not np.array_equal(runtime.cache['x'].reshape(-1),solver.x):raise RuntimeError('cached native output is not the accepted solver point')
        arrays={str(r)+'_rgb':runtime.cache['pred'][r]for r in FRAMES};arrays.update({str(r)+'_alpha':runtime.cache['alpha'][r]for r in FRAMES})
        np.savez_compressed(out/'reference_accepted2_native.npz',**arrays)
        write(out/'receipt.json',dict(completed=True,phase='reference',precision=precision,accepted1=first,accepted2=second,
            checkpoint=cp,reference_workspace=str(out/'reference_accepted2.npz'),native_reference=str(out/'reference_accepted2_native.npz'),
            identity=runtime.identity,physical=runtime.ledger,seconds=time.monotonic()-t,formal_FG=0))
        print('REFERENCE_COMPLETE',json.dumps(dict(seconds=time.monotonic()-t,physical=runtime.ledger)),flush=True)
    except Exception:
        write(root/'reference_failed.json',dict(error=traceback.format_exc(),seconds=time.monotonic()-t));raise
    finally:
        if runtime is not None:runtime.close()


def smoke_resume():
    root=ROOT/'smoke_r1';reference=read(root/'reference/receipt.json');out=root/'resume';runtime=None;t=time.monotonic()
    try:
        runtime=NumericalRuntime(out,role='smoke',owner='native_capacity_numerical_smoke_resume',ledger_file=root/'physical_ledger.json',maxFG=20)
        if runtime.identity!=reference['identity']:raise RuntimeError('fresh-process native inputs differ')
        solver=CheckpointedLBFGSB(runtime.x0,runtime.evaluate,identity=runtime.identity,**dict(RECIPE,maxFG=20))
        expected=CheckpointedLBFGSB(runtime.x0,runtime.evaluate,identity=runtime.identity,**dict(RECIPE,maxFG=20))
        expected.load(reference['reference_workspace'])
        load_checkpoint(solver,runtime,reference['checkpoint'])
        before=solver.snapshot();comparison=CheckpointedLBFGSB(runtime.x0,runtime.evaluate,identity=runtime.identity,**dict(RECIPE,maxFG=20))
        comparison.load(reference['checkpoint']['workspace']);baseline=comparison.snapshot()
        restored={k:bool(np.array_equal(before['arrays'][k],baseline['arrays'][k]))for k in solver.ARRAY_NAMES}
        if not all(restored.values()):raise RuntimeError('full Fortran workspace not exactly restored')
        row=solver.next_accepted()
        if row is None:raise RuntimeError('no NEW_X after actual fresh-process restore')
        runtime.ledger['accepted']+=1;runtime._record()
        ci_error=float(np.max(abs(solver.x-expected.x)));objective_relative=abs(float(solver.f)-float(expected.f))/max(abs(float(expected.f)),1e-12)
        actual_state=solver.snapshot();expected_state=expected.snapshot()
        workspace_errors={k:float(np.max(abs(actual_state['arrays'][k].astype(float)-expected_state['arrays'][k].astype(float))))
            for k in ('x','f','g','wa','iwa','lsave','isave','dsave')}
        native=[]
        with np.load(reference['native_reference'],allow_pickle=False)as z:
            for raw in FRAMES:
                error=float(np.max(abs(runtime.cache['pred'][raw]-z[str(raw)+'_rgb'])))
                alpha=float(np.max(abs(runtime.cache['alpha'][raw]-z[str(raw)+'_alpha'])))
                native.append(dict(raw=raw,RGB_max=error,alpha_max=alpha))
        if ci_error>1e-6 or objective_relative>1e-6 or any(r['RGB_max']>1e-6 or r['alpha_max']!=0 for r in native):
            raise RuntimeError('next NEW_X replay exceeds separate ci/native/function numerical gates')
        result=dict(completed=True,CPU_workspace_qualified=True,full_Fortran_restore_exact=restored,
            next_ci_max_abs=ci_error,next_objective_relative=objective_relative,next_native=native,next_workspace_errors_by_unit=workspace_errors,
            workspace_note='Immediate restore exact for ALL arrays. After native FG, gradient/curvature/timer buffers reported by unit, never conflated with ci/RGB error.',
            precision=reference['precision'],physical=runtime.ledger,identity=runtime.identity,
            reference_receipt=str(root/'reference/receipt.json'),reference_receipt_sha256=sha(root/'reference/receipt.json'),
            seconds_reference=reference['seconds'],seconds_resume=time.monotonic()-t,formal_FG=0,all_smoke_states_discard_for_formal=True)
        write(root/'receipt.json',result);print('SMOKE_PASSED',json.dumps({k:v for k,v in result.items()if k not in('identity','physical')}),flush=True)
    except Exception:
        write(root/'resume_failed.json',dict(error=traceback.format_exc(),seconds=time.monotonic()-t));raise
    finally:
        if runtime is not None:runtime.close()


def formal():
    import torch
    smoke=read(ROOT/'smoke_r1/receipt.json')
    if smoke.get('completed')is not True or smoke['physical']['FG']>20:raise RuntimeError('actual complete workspace smoke required')
    out=ROOT/'formal_r1'
    if out.exists():raise FileExistsError('preserve existing numerical phase; explicit full-state recovery needed')
    runtime=None;t=time.monotonic()
    try:
        runtime=NumericalRuntime(out,role='formal',owner='native_capacity_numerical_formal',ledger_file=out/'physical_ledger.json',maxFG=2000)
        if runtime.binding!=smoke['identity']['binding']:raise RuntimeError('source changed after actual numerical smoke')
        solver=CheckpointedLBFGSB(runtime.x0,runtime.evaluate,identity=runtime.identity,**RECIPE)
        while not solver.done:
            row=solver.next_accepted()
            if row is None:break
            runtime.ledger['accepted']+=1;runtime._record()
            with (out/'steps.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
            if solver.accepted%50==0:
                cp=checkpoint(solver,runtime,out/('checkpoint_%04d'%solver.accepted));write(out/'latest_checkpoint.json',cp)
                print('NUMERIC',solver.accepted,solver.fg_calls,float(solver.f),row['relative_FW_gap'],flush=True)
            if row['relative_FW_gap']<=.01:solver.stop_at_training_gate()
        cp=checkpoint(solver,runtime,out/'final_checkpoint');write(out/'latest_checkpoint.json',cp)
        x=solver.last_accepted_x.reshape(runtime.x0.shape)
        from gs_color_oracle import ColorLayout,BoundedGSColors
        layout=ColorLayout(runtime.ctx.active_world_rows,runtime.eligible);model=BoundedGSColors(runtime.seed,layout)
        with torch.no_grad():model.rgb.copy_(torch.as_tensor(x,device=model.rgb.device,dtype=model.rgb.dtype))
        model.save(out/'oracle_colors.pt');reloaded=BoundedGSColors.load(out/'oracle_colors.pt',device=model.rgb.device)
        if not torch.equal(model(),reloaded()):raise RuntimeError('bounded ci authority reload differs')
        full=np.zeros(len(runtime.ctx.road['active']),bool);full[layout.eligible_world_rows]=True
        asset=runtime.ctx.save_asset(reloaded(),full,out/'asset',training_record=dict(arm='G_NUMERIC',role='same_objective_numerical_capacity_control_not_production',
            source_identity=runtime.identity,solver_recipe=RECIPE,FGcount=solver.fg_calls,accepted_steps=solver.accepted))
        # Final accepted point is distinct from a pending line-search trial at cap.
        # Re-render it read-only for exact cross-precision objective reporting.
        with torch.no_grad():
            colors=reloaded();pred={}
            for raw in FRAMES:
                v=runtime.ctx.render_road(raw,colors);runtime.ledger['nativeF']+=1;pred[raw]=v['rgb']
            f32,_=runtime.original.objective(pred,'detail');f64,_=runtime.precise.objective(pred,'detail')
        gap=solver.gap(solver.last_accepted_x,solver.last_accepted_g)
        result=dict(arm='G_NUMERIC',completed=True,asset_dir=str(asset),authority=str(out/'oracle_colors.pt'),checkpoint=cp,
            source_identity=runtime.identity,physical=runtime.ledger,main_four_arms_unchanged=True,solver=RECIPE,
            extra=dict(accepted_steps=solver.accepted,FGcount=solver.fg_calls,last_training_FW_gap=gap,relative_training_FW_gap=gap/max(solver.last_accepted_f,1e-3),
                termination=solver.reason,original_FP32_objective=float(f32),FP64_reduction_objective=float(f64),
                best_evaluated_training_objective=solver.best_evaluated_f,export_point='last accepted NEW_X; best evaluated feasible point reported separately'),
            numeric_training_converged=gap/max(solver.last_accepted_f,1e-3)<=.01,strict_numerical_negative_certificate=False,
            post_solver_readonly_nativeF=2,post_public1920_fiveview_pending=True,quality_PASS=False,production_scene_modified=False,
            seconds=time.monotonic()-t,GPU_keep_on=True)
        runtime._record();write(out/'endpoint.json',result);write(out/'receipt.json',result)
        print('NUMERICAL_COMPLETED',json.dumps({k:v for k,v in result.items()if k not in('source_identity','physical')}),flush=True)
    except Exception:
        write(out/'failed.json',dict(error=traceback.format_exc(),seconds=time.monotonic()-t));raise
    finally:
        if runtime is not None:runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke_reference','smoke_resume','formal']);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(json.dumps(dict(default_GPU=False,recipe=RECIPE,registered_seed=str(B/'formal_r1/G_DETAIL/asset'),no_held=True)))
    elif a.stage=='smoke_reference':smoke_reference()
    elif a.stage=='smoke_resume':smoke_resume()
    elif a.stage=='formal':formal()
    else:p.error('explicit stage required')
