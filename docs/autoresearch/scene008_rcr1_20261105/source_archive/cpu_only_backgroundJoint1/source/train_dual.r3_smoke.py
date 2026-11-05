"""Native B10k diagnostic replay. Extra snapshots never invoke native evaluation.

The no_sgcm arm skips only refine; controller initialization and lineage remain.
All diagnostic I/O is checked for RNG neutrality. Formal updates remain 10000.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import argparse, gc, hashlib, inspect, json, random, signal, shutil, sys, time

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_only_B10k_20261103_r1')
ENGINE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
TOOLS = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/tools')
WATCH = [528407, 132314, 520617, 192988, 42783]

class Done(Exception): pass

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['native_ownership'], default='native_ownership')
    ap.add_argument('--mode', choices=['smoke', 'formal', 'resume'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--assignment', type=Path, required=True)
    ap.add_argument('--a6-masks', type=Path, required=True)
    ap.add_argument('--smoke-seconds', type=float, default=540)
    opt = ap.parse_args()
    out = opt.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    for filename in ('train_dual.py','diag_hooks.py','dual_actor_runtime.py','native_ownership.py','integrate_native.py','readout_native.py','early_accept.py'):
        shutil.copy2(Path(__file__).parent/filename,out/('executed_'+filename))
    cfg = json.loads((BASE/'config.json').read_text())
    cfg['dual_actor_assignment_npz'] = str(opt.assignment.resolve())
    cfg['dual_actor_assignment_sha256'] = hashlib.sha256(opt.assignment.read_bytes()).hexdigest()
    cfg['dual_actor_a6_masks'] = str(opt.a6_masks.resolve())
    cfg['dual_actor_schema_version'] = 1
    if opt.mode == 'resume':
        cfg = json.loads((opt.checkpoint.parent/'effective_config.json').read_text())
        assert json.loads((opt.checkpoint.parent/'diagnostic_contract.json').read_text())['arm'] == opt.arm
        cfg['start_checkpoint'] = str(opt.checkpoint)
        assert cfg['dual_actor_assignment_sha256']==hashlib.sha256(opt.assignment.read_bytes()).hexdigest()
        assert cfg['dual_actor_a6_masks']==str(opt.a6_masks.resolve())
    cfg['model_path'] = str(out)
    write(out/'effective_config.json', cfg)
    write(out/'diagnostic_contract.json', dict(arm=opt.arm, mode=opt.mode,
        changed_method='Native dual initialization; isolated instance RGB and coverage; joint ownership and measured depth; fixed measured static prefix; background-only ordinary static growth; native optimizer schedule and SGCM',
        formal_updates=10000, lr_horizon=30000, extra_checkpoints='RNG-neutral end-of-step; no extra evaluation',
        original_checkpoint_cadence_unchanged=True, PVD_changed=False, road_prior_changed=False,
        quality_accepted=False, human_verdict=None))
    diag = BASE.parent / 'scene008_static_origin_diag_20261104_r1/source'
    sys.path[:0] = [str(ENGINE), str(TOOLS), str(diag)]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import train
    from scene import GaussianModel
    from scene.paper_sgcm import SGCMController
    import scene.paper_sgcm as sgcm_module
    from utils import paper_checkpoint as cp, hybrid30k_runtime as runtime
    from utils.general_utils import seed_everything
    from run_v4_arm import bind_scene_contract
    from compare_v1_states import differences
    from dual_actor_runtime import (install_identity_serialization, install_union_masks,
                                    transfer_initial_rows, validate_partition)
    install_identity_serialization()
    install_union_masks(opt.a6_masks)
    from native_ownership import Ownership
    from integrate_native import integrate
    from readout_native import readout
    objective = Ownership()
    integrate(train, out, objective)
    with np.load(opt.assignment,allow_pickle=False) as assignment:
        moved_ids=assignment['original_static_indices'].astype(np.int64)
    assert hashlib.sha256(opt.assignment.read_bytes()).hexdigest()=='04671379a727ddca166b54030588cf865acf5aea2b947c9f6c9deeb3016bb0c1'
    assert len(moved_ids)==1023
    assert Path(train.__file__).resolve() == ENGINE/'train.py'
    assert torch.cuda.is_available()
    torch.set_num_threads(1); seed_everything(cfg['seed'])
    # Loader normally rewrites an identical transform into its input tree.
    # Redirect that publication while asserting the old bytes already agree.
    original_savetxt=np.savetxt
    transform_source=Path(cfg['source_path'])/'transform_matrix.txt'
    def protected_savetxt(filename, array, *a, **kw):
        if Path(filename).resolve()==transform_source.resolve():
            destination=out/'loader_transform_matrix.txt'
            result=original_savetxt(destination,array,*a,**kw)
            assert destination.read_bytes()==transform_source.read_bytes(), 'input transform changed'
            return result
        return original_savetxt(filename,array,*a,**kw)
    np.savetxt=protected_savetxt
    started = time.monotonic(); step = [0]; last_loss = {}; init = {}; handle = None
    stop_requested=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop_requested.__setitem__(0,True))

    def rng():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    def same_rng(a,b):
        return not differences(a,b,'rng')
    def restore_rng(s):
        random.setstate(s[0]); np.random.set_state(s[1]); torch.set_rng_state(s[2]); torch.cuda.set_rng_state_all(s[3])
    bind_scene_contract(train, cfg)
    original_scene = train.Scene
    def scene_hook(*a, **kw):
        scene = original_scene(*a, **kw)
        for field,expected in {'_xyz':'7598f1556be5c128330365999bcb072b89100b0448e5b570a4a92a43b8e8f471',
                               '_t':'7eb38e589a7f17cf67d7fe0e65231df449fbf24547dbcdb07eec462ec15354fa'}.items():
            assert hashlib.sha256(getattr(scene.gaussians,field).detach().cpu().numpy().tobytes()).hexdigest()==expected
        rng_before=rng()
        transfer_initial_rows(scene,moved_ids,out/'initial_parameter_transfer.json')
        assert same_rng(rng_before,rng()), 'Parameter transfer altered RNG'
        init['scene'] = scene
        hashes = {}
        for key in ('_xyz', '_t'):
            arr = getattr(scene.gaussians,key).detach().cpu().numpy()
            hashes[key] = hashlib.sha256(arr.tobytes()).hexdigest()
        write(out/'initialization.json', dict(static_rows=len(scene.gaussians.get_xyz),
            dynamic_rows=len(scene.dynamic_gaussians.get_xyz), initial_tensor_sha256=hashes,
            source_path=cfg['source_path'], camera_count=len(scene.getTrainCameras())))
        assert len(scene.gaussians.get_xyz)==591672-len(moved_ids)
        assert len(scene.dynamic_gaussians.get_xyz)==98452+len(moved_ids)
        return scene
    train.Scene = scene_hook
    # The new request only needs the dual-actor control, not another topology study.
    handle = type('Handle',(),{'assert_healthy':lambda _:None,'close':lambda _:None})()
    if opt.arm=='sgcm_no_split':
        original_guard=sgcm_module.guard_additions
        def split_off_guard(controller, model, clone, split):
            accepted_clone,accepted_split,stats=original_guard(controller,model,clone,split)
            stats=dict(stats,diagnostic_split_admission_disabled=int(accepted_split.sum()))
            return accepted_clone,torch.zeros_like(accepted_split),stats
        sgcm_module.guard_additions=split_off_guard
    original_refine = SGCMController.refine
    def refine(controller, model, radii, iteration, **kw):
        step[0] = int(iteration)
        if opt.arm=='no_sgcm': return None
        return original_refine(controller,model,radii,iteration,**kw)
    SGCMController.refine = refine
    original_record = runtime.record_step
    def record(args, iteration, camera, shift, static, dynamic, losses):
        step[0] = int(iteration)
        original_record(args,iteration,camera,shift,static,dynamic,losses)
        last_loss.clear(); last_loss.update({k:float(v) for k,v in losses.items()})
        if not all(np.isfinite(v) for v in last_loss.values()): raise RuntimeError('nonfinite loss')
    runtime.record_step = record
    original_adam = torch.optim.Adam.step
    def adam(optimizer,*a,**kw):
        if step[0]%100==0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None and not torch.isfinite(p.grad).all(): raise RuntimeError('nonfinite gradient')
        return original_adam(optimizer,*a,**kw)
    torch.optim.Adam.step = adam
    original_restore = cp.restore_training_state
    def restore(path, iteration, scene, models, args, sgcm=None):
        stack,ema = original_restore(path,iteration,scene,models,args,sgcm=sgcm)
        saved=torch.load(path,map_location='cpu'); current=cp.capture_training_state(iteration,scene,stack,models,args,sgcm=sgcm,ema=ema)
        for value in (saved,current):
            value['config']={k:v for k,v in value['config'].items() if k not in ('model_path','start_checkpoint')}
        errors=differences(saved,current,'continuation')
        for name,prefix in [('static','chkpnt'),('dynamic','dynamic_chkpnt'),('environment','env_light_chkpnt')]:
            old,_=torch.load(Path(path).parent/f'{prefix}{iteration}.pth',map_location='cpu')
            errors+=differences(old,models[name].capture(),name); del old; gc.collect()
        write(out/'restore_audit.json',dict(status='PASS_EXACT_FULL_RESTORE' if not errors else 'FAIL', differences=errors))
        if errors: raise RuntimeError(str(errors))
        return stack,ema
    cp.restore_training_state=restore

    def snapshot(iteration, loc):
        before=rng(); scene=loc['scene']
        models={'static':loc['gaussians'],'dynamic':loc['dynamic_gaussians'],'environment':loc['env_map']}
        cp.save_components(str(out),iteration,models)
        state=cp.capture_training_state(iteration,scene,loc['viewpoint_stack'],models,loc['args'],
            sgcm=loc['paper_sgcm'],ema=loc['ema_dict_for_log'])
        cp._atomic_write(out/f'training_state{iteration}.pth',lambda f:torch.save(state,f))
        assert same_rng(before,rng()), 'snapshot altered RNG'
        write(out/f'diagnostic_node_{iteration:06d}.json',dict(iteration=iteration,status='PASS_RNG_NEUTRAL_COMPLETE_STATE',
            elapsed_seconds=time.monotonic()-started, arm=opt.arm, static_rows=len(models['static'].get_xyz),
            dynamic_rows=len(models['dynamic'].get_xyz), native_evaluation_added=False))
    original_eval=train.complete_eval
    def evaluate(*a,**kw):
        result=original_eval(*a,**kw)
        loc=inspect.currentframe().f_back.f_locals
        iteration=int(loc['iteration']); step[0]=iteration
        static=loc['gaussians']
        if iteration%100==0:
            ids=static._sgcm_lineage_ids
            rec=dict(iteration=iteration,elapsed_seconds=time.monotonic()-started,
                static_rows=len(ids),dynamic_rows=len(loc['dynamic_gaussians'].get_xyz),
                watch={str(i):int((ids==i).sum()) for i in WATCH},no_time_split=bool(static.no_time_split),
                allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),losses=dict(last_loss))
            rec['actor_partition']=validate_partition(static,loc['dynamic_gaussians'],moved_ids)
            with (out/'runtime.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            write(out/'progress.json',rec); handle.assert_healthy()
        resume_step=int(opt.checkpoint.stem.replace('chkpnt','')) if opt.checkpoint else 0
        stop_smoke=opt.mode=='smoke' and iteration%100==0 and time.monotonic()-started>=opt.smoke_seconds
        stop_resume=opt.mode=='resume' and iteration>=resume_step+2
        due=iteration in (100, 500, 1000, 3000, 6000, 10000)
        if due or stop_smoke or stop_resume or stop_requested[0]:
            if iteration != 10000:
                snapshot(iteration,loc)
            before=rng()
            with torch.no_grad():
                metrics = readout(objective, loc['scene'], loc['args'], loc['background'], loc['env_map'], out, iteration)
            assert same_rng(before,rng()), 'readout altered RNG'
            if opt.mode=='formal' and iteration==1000:
                from early_accept import check_1k
                report = check_1k(metrics, out.parent/'control/probe_01000/summary.json', out/'early_accept_01000.json')
                if report['status'] != 'PASS':
                    raise RuntimeError("1K mechanism gate failed: " + str(report["checks"]))
        if stop_requested[0]:
            write(out/'returned.json',dict(status='STOPPED_WITH_COMPLETE_STATE',mode=opt.mode,
                iteration=iteration,seconds=time.monotonic()-started,quality_accepted=False,human_verdict=None))
            raise Done()
        if stop_smoke:
            before=rng()
            extra=list(a); extra[2]=[iteration]
            try: original_eval(*extra,**kw)
            finally: restore_rng(before)
            write(out/'smoke_evaluation.json',dict(iteration=iteration,status='PASS_NATIVE_EVALUATION_RNG_RESTORED'))
        if stop_smoke or stop_resume:
            write(out/'returned.json',dict(status='PASS_COMPLETE_BOUNDARY',mode=opt.mode,iteration=iteration,
                updates=2 if stop_resume else iteration,seconds=time.monotonic()-started,quality_accepted=False,human_verdict=None))
            raise Done()
        return result
    train.complete_eval=evaluate
    original_publish=cp.publish_checkpoint_event
    def publish(*a,**kw):
        result=original_publish(*a,**kw)
        if opt.mode=='formal' and int(a[1])==10000:
            write(out/'returned.json',dict(status='PASS_COMPLETE_BOUNDARY',mode=opt.mode,iteration=10000,
                updates=10000,seconds=time.monotonic()-started,quality_accepted=False,human_verdict=None))
            raise Done()
        return result
    cp.publish_checkpoint_event=publish
    os.chdir(ENGINE)
    try: train.training(OmegaConf.create(cfg))
    except Done: pass
    finally: handle.close()
    assert (out/'returned.json').exists()

if __name__=='__main__': main()


