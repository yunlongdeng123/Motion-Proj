"""Exact original10K restoration; 32-update smoke or1977-update bug replay."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / 'engine'
TOOLS = ROOT.parent / 'r15_gn11_stage1_v41_20261023/tools'
class CompleteBoundary(Exception):
    pass
def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['smoke', 'resume2step', 'replay'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    contract = json.loads((ROOT / 'isolated_fix_contract.json').read_text())
    assert ROOT.name == 'scene008_a5_single_bug_20261106_r1'
    current_source = {str(p.relative_to(ENGINE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ENGINE.rglob('*.py')}
    assert current_source == contract['isolated_engine_sha256'], 'Isolated engine bytes changed after preparation'
    start = int(a.checkpoint.stem.replace('chkpnt', ''))
    if a.mode != 'resume2step':
        assert start == 10000 and a.checkpoint.resolve() == (Path(contract['parent_model']) / 'chkpnt10000.pth').resolve()
    stop = start + 32 if a.mode == 'smoke' else 11977
    a.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((a.checkpoint.parent / 'effective_config.json').read_text())
    assert config['iterations'] == config['position_lr_max_steps'] == 30000
    config.update(model_path=str(a.output), start_checkpoint=str(a.checkpoint))
    write(a.output / 'effective_config.json', config)
    snapshot = a.output / 'executed_train_isolated_fix.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    write(a.output / 'runtime_source_manifest.json', dict(interpreter=sys.executable,
        engine_files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in ENGINE.rglob('*.py')},
        wrapper_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(), isolated_contract=contract))
    sys.path[:0] = [str(ENGINE), str(TOOLS)]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import train
    import gaussian_renderer
    import scene.gaussian_model as gaussian_models
    from utils import paper_checkpoint as cp
    from utils import hybrid30k_runtime as runtime
    from utils.general_utils import seed_everything
    from run_v4_arm import bind_scene_contract
    from compare_v1_states import differences
    assert Path(train.__file__).resolve() == ENGINE / 'train.py' and torch.cuda.is_available()
    for module, relative in ((gaussian_renderer, 'gaussian_renderer/__init__.py'),
                             (gaussian_models, 'scene/gaussian_model.py'),
                             (cp, 'utils/paper_checkpoint.py'), (runtime, 'utils/hybrid30k_runtime.py')):
        assert Path(module.__file__).resolve() == ENGINE / relative, (module.__name__, module.__file__)
    original_savetxt = np.savetxt
    transform = Path(config['source_path']) / 'transform_matrix.txt'
    transform_bytes = transform.read_bytes()
    def protected_savetxt(filename, array, *args, **kwargs):
        if isinstance(filename, (str, os.PathLike)) and Path(filename).resolve() == transform.resolve():
            target = a.output / 'loader_transform_matrix.txt'
            result = original_savetxt(target, array, *args, **kwargs)
            assert target.read_bytes() == transform_bytes and transform.read_bytes() == transform_bytes
            return result
        return original_savetxt(filename, array, *args, **kwargs)
    np.savetxt = protected_savetxt
    bind_scene_contract(train, config)
    base_scene = train.Scene
    def scene(*args, **kwargs):
        value = base_scene(*args, **kwargs)
        assert len(value.dynamic_gaussians.get_xyz) == 98452
        assert len(value.gaussians.get_xyz) == 591672
        return value
    train.Scene = scene
    seed_everything(config['seed'])
    torch.set_num_threads(1)
    restore0, publish0, record0 = cp.restore_training_state, cp.publish_checkpoint_event, runtime.record_step
    began = time.monotonic()
    def restore(path, iteration, scene, models, args, sgcm=None):
        stack, ema = restore0(path, iteration, scene, models, args, sgcm=sgcm)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        current = cp.capture_training_state(iteration, scene, stack, models, args, sgcm=sgcm, ema=ema)
        ignored = ('model_path', 'start_checkpoint')
        for item in (saved, current):
            item['config'] = {k: v for k, v in item['config'].items() if k not in ignored}
        errors = differences(saved, current, 'continuation')
        for name, prefix in [('static', 'chkpnt'), ('dynamic', 'dynamic_chkpnt'), ('environment', 'env_light_chkpnt')]:
            old, step = torch.load(a.checkpoint.parent / f'{prefix}{iteration}.pth', map_location='cpu', weights_only=False)
            assert step == iteration
            errors += differences(old, models[name].capture(), name)
            del old
            gc.collect()
        write(a.output / 'restore_audit.json', dict(status='PASS_EXACT_FULL_RESTORE' if not errors else 'FAIL', iteration=iteration, differences=errors))
        assert not errors, errors
        if a.mode != 'resume2step':
            for key in ('checkpoint_iterations', 'save_iterations', 'test_iterations'):
                args[key] = sorted(set(args[key]) | {stop})
                config[key] = list(args[key])
            write(a.output / 'effective_config.json', config)
        return stack, ema
    def record(args, iteration, camera, shift, static, dynamic, losses):
        record0(args, iteration, camera, shift, static, dynamic, losses)
        values = {k: float(v) for k, v in losses.items()}
        assert all(np.isfinite(v) for v in values.values())
        if iteration % 100 == 0:
            write(a.output / 'progress.json', dict(iteration=iteration, additional_updates=iteration-start,
                elapsed_seconds=time.monotonic()-began, allocated_bytes=torch.cuda.memory_allocated(),
                static_rows=len(static.get_xyz), dynamic_rows=len(dynamic.get_xyz), losses=values))
        if a.mode == 'resume2step' and iteration >= start + 3:
            write(a.output / 'resume_entry.json', dict(status='PASS_NEXT_TRAINING_ENTRY', completed_extra_updates=2, entered_iteration=iteration))
            raise CompleteBoundary()
    def publish(path, iteration, *args, **kwargs):
        event = publish0(path, iteration, *args, **kwargs)
        if iteration == stop:
            write(a.output / 'returned.json', dict(status='PASS_COMPLETE_BOUNDARY', mode=a.mode,
                source_iteration=start, iteration=iteration, additional_updates=iteration-start,
                accounting='ENGINEERING_BUG_REPLAY_NOT_FORMAL' if a.mode == 'replay' else 'EXTRA_REAL_GPU_SMOKE',
                elapsed_seconds=time.monotonic()-began, max_allocated_bytes=torch.cuda.max_memory_allocated(),
                max_reserved_bytes=torch.cuda.max_memory_reserved(), learning_rate_horizon=30000,
                formal_updates_added=0, automatic_shutdown_executed=False))
            raise CompleteBoundary()
        return event
    cp.restore_training_state, cp.publish_checkpoint_event, runtime.record_step = restore, publish, record
    adam0 = torch.optim.Adam.step
    def finite_step(optimizer, *args, **kwargs):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                    raise RuntimeError('Non-finite backward gradient')
        return adam0(optimizer, *args, **kwargs)
    torch.optim.Adam.step = finite_step
    os.chdir(ENGINE)
    try:
        train.training(OmegaConf.create(config))
    except CompleteBoundary:
        pass
    assert (a.output / ('resume_entry.json' if a.mode == 'resume2step' else 'returned.json')).exists()

if __name__ == '__main__':
    main()
