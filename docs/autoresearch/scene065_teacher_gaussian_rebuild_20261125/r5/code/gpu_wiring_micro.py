from pathlib import Path
import hashlib,json,random,sys,time,traceback
import numpy as np
import torch
from PIL import Image

R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_gaussian_rebuild_20261125_r5')
sys.path.insert(0,str(R/'code'))
from teacher_gaussian_runtime import TeacherGaussianRuntime
from full_gaussian_lifecycle import BalancedSampler
import training_objective as objective

def main():
    out=R/'gpu_wiring_v1';assert not out.exists();out.mkdir();started=time.perf_counter();calls={'attempted':0,'completed':0};rt=None;error=None;record=None;event=None
    cfg=objective.read(R/'config/training.json');seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    try:
        rt=TeacherGaussianRuntime(R/'config/runtime.json',out/'native')
        cache_root=R/'fixed_training_cache_v1'
        if not cache_root.exists():objective.build_cache(rt,R/'config/targets8.remote.json',R.parent/'road_native30k_red_rebuild_20261125_r4/masks',cache_root,cfg)
        cache=objective.load_cache(torch,cache_root);optimizer=rt.make_optimizer(cfg);original=optimizer.step
        def counted(*a,**kw):
            calls['attempted']+=1;value=original(*a,**kw);calls['completed']+=1;return value
        optimizer.step=counted
        sampler=BalancedSampler(cfg['train_frames'],cfg['seed'],cfg['view_groups']);torch.cuda.synchronize();tick=time.perf_counter()
        record,event=objective.step(rt,optimizer,sampler,cache,cfg,1);torch.cuda.synchronize();update_seconds=time.perf_counter()-tick
        assert calls['completed']==1 and event is None
        assert all(x['finite']and x['l2']>0 for x in record['gradients_by_group'].values())
        assert all(x['l2']>0 for x in record['parameter_delta_by_group'].values())
        assert all(x['gradient_finite']and x['gradient_norm']>0 and x['nonzero_gradient_rows']>0 and x['visible_new_rows']>0 for x in record['screen_gradient_stats'])
        rt.assert_fixed();rt.export_field(out/'new_field_engineering_step000001.npz')
        with torch.no_grad():arrays={k:v.detach().cpu().numpy()for k,v in rt.render(110).items()}
        np.savez_compressed(out/'actual_raw110.npz',**arrays);Image.fromarray(np.rint(np.clip(arrays['rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)).save(out/'actual_raw110.png')
        objective.write(out/'update.json',record)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};raise
    finally:
        receipt=dict(schema='scene065.teacher_gaussian.real_gpu_wiring.v1',complete=error is None,error=error,engineering_optimizer_updates=calls['completed'],optimizer_attempts=calls['attempted'],
            formal_unique_updates=0,engineering_state_loaded_by_formal=False,source='fresh teacher_seed; no prior fixed-plane RGB/checkpoint',
            runtime_code=objective.pin(R/'code/teacher_gaussian_runtime.py'),objective_code=objective.pin(R/'code/training_objective.py'),
            configuration=objective.pin(R/'config/training.json'),runtime_contract=objective.pin(R/'config/runtime.json'),
            cache_manifest=objective.pin(R/'fixed_training_cache_v1/manifest.json')if(R/'fixed_training_cache_v1/manifest.json').exists()else None,
            parameter_groups=list(rt.named_new_parameters())if rt else [],native_original_parameters_frozen=bool(rt is not None and error is None),
            record=record,topology_event=event,native_forward_calls=rt.calls if rt else 0,native_backward_calls=rt.backward_calls if rt else 0,
            new_rows=len(rt.field.xyz)if rt else 0,wall_seconds=time.perf_counter()-started,peak_GPU_bytes=int(torch.cuda.max_memory_allocated()))
        if 'update_seconds'in locals():receipt['optimizer_update_seconds']=update_seconds
        objective.write(out/'receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
