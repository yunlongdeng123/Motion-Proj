"""由原matrix worker提取三个函数；无原main、旧ROOT/scene17常量。"""
import argparse
import datetime
import functools
import faulthandler
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

def install_memory_hooks(training, monitor):
    """只包围已有调用记录资源；不改目标、采样或张量。"""
    original_render = training.render
    counter = {'iteration': 0}

    @functools.wraps(original_render)
    def observed_render(*args, **kwargs):
        if kwargs.get('is_training', False):
            counter['iteration'] += 1
            monitor.set_phase('training', iteration=counter['iteration'])
            if counter['iteration'] == 1 or counter['iteration'] % 100 == 0:
                counter.update(render_static_points=int(args[1].get_xyz.shape[0]), render_dynamic_points=int(args[2].get_xyz.shape[0]), render_width=int(args[0].image_width), render_height=int(args[0].image_height))
        return original_render(*args, **kwargs)
    training.render = observed_render

    def wrap(module_name, function_names, phase_name):
        module = importlib.import_module(module_name)
        for name in function_names:
            original = getattr(module, name)

            def measured(*args, _original=original, **kwargs):
                with monitor.phase(phase_name):
                    return _original(*args, **kwargs)
            setattr(module, name, functools.wraps(original)(measured))
    wrap('utils.hybrid30k_losses', ['diagnose_gradients', 'compare_objective_gradients'], 'gradient_diagnostics')
    wrap('utils.paper_checkpoint', ['save_components', 'capture_training_state', 'publish_checkpoint_event'], 'checkpoint')
    wrap('utils.hybrid30k_static_time', ['run_static_time_check'], 'static_time_eval')
    return counter

def write(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

def run_arm(arm):
    import fcntl
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    faulthandler.dump_traceback_later(120, repeat=False)
    cfg = check_config(arm)
    check_prepared()
    with (BASE / 'campaign/gpu_slot.lock').open('a+') as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
        if apps:
            raise RuntimeError('GPU已有计算任务: ' + apps)
        import torch
        from omegaconf import OmegaConf
        if not torch.cuda.is_available():
            raise RuntimeError('GPU尚未切回，不能训练')
        torch.cuda.set_device(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory < 30 * 1024 ** 3:
            raise RuntimeError('当前固定分辨率/点数计划要求至少30GiB GPU；不自动降配置')
        output = Path(cfg['model_path'])
        output.mkdir(parents=True, exist_ok=False)
        write(output / 'effective_config.json', cfg)
        args = OmegaConf.create(cfg)
        sys.path.insert(0, str(REPOSITORY))
        os.chdir(REPOSITORY)
        started = time.monotonic()
        status = {'status': 'initializing', 'arm': arm, 'iterations': 30000, 'pid': os.getpid(), 'started_singapore': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), 'GPU': torch.cuda.get_device_name(0), 'GPU_memory_bytes': gpu_memory, 'checkpoint_scope': 'model+Adam+RNG+paired sampler+SGCM lineage', 'evaluation_scope': '49 all-observed training views; not independent test'}
        write(output / 'status.json', status)
        from memory_monitor import MemoryMonitor
        monitor = MemoryMonitor(output, torch_module=torch, device=0, nvidia_device='0')
        monitor.start()
        monitor.set_phase('initialization')
        try:
            spec = importlib.util.spec_from_file_location('hybrid30k_training', REPOSITORY / 'train.py')
            training = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(training)
            training.seed_everything(0)
            bind_training_contract(training, cfg)
            from utils.hybrid30k_runtime import preserve_rng
            original_eval = training.complete_eval
            rendered = install_memory_hooks(training, monitor)
        except BaseException as error:
            status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            write(output / 'status.json', status)
            monitor.close(status='failed', error=repr(error))
            raise
        progress = {'iteration': 0, 'time': started, 'included_evaluation': False}

        def observed_eval(*positional, **keyword):
            (iteration, schedule, losses) = (positional[1], positional[2], positional[6])
            if iteration == 1:
                faulthandler.cancel_dump_traceback_later()
            numbers = {key: float(value.item()) if torch.is_tensor(value) else float(value) for (key, value) in losses.items()}
            if not all((math.isfinite(value) for value in numbers.values())):
                raise FloatingPointError('训练标量出现非finite值')
            if iteration == 1 or iteration % 100 == 0:
                now = time.monotonic()
                window_seconds = now - progress['time']
                status.update(status='training', iteration=iteration, wall_seconds=time.monotonic() - started, loss=numbers['loss'], psnr=numbers['psnr'], static_points=numbers['total_points'])
                status['speed_window'] = {'first_iteration': progress['iteration'] + 1, 'last_iteration': iteration, 'wall_seconds': window_seconds, 'wall_iterations_per_second': (iteration - progress['iteration']) / window_seconds, 'includes_initialization': progress['iteration'] == 0, 'includes_evaluation': progress['included_evaluation']}
                progress.update(iteration=iteration, time=now, included_evaluation=False)
                memory = monitor.update(iteration=iteration, extra={'static_points': numbers['total_points'], 'iter_forward_backward_ms': numbers['iter_time'], **rendered})
                status['memory'] = memory
                status['iter_forward_backward_ms'] = numbers['iter_time']
                write(output / 'status.json', status)
            if iteration in schedule:
                status.update(status='evaluating', iteration=iteration)
                progress['included_evaluation'] = True
                write(output / 'status.json', status)
                with preserve_rng(), monitor.phase('photo_eval'):
                    return original_eval(*positional, **keyword)
            return original_eval(*positional, **keyword)
        training.complete_eval = observed_eval
        try:
            training.training(args)
            torch.cuda.synchronize()
            for step in (10000, 20000, 30000):
                event = json.loads((output / 'checkpoint_events' / ('iteration_%06d.json' % step)).read_text())
                if event['status'] != 'complete' or len(event['checkpoints']) != 4 or len(event['frames']) != 49:
                    raise RuntimeError('完整检查点或49帧评价合同未满足')
                if any((not (output / 'eval' / ('train_%d_render' % step) / 'hybrid_raw' / (row['frame_id'].replace('/', '_') + '.npz')).is_file() for row in event['frames'])):
                    raise RuntimeError('原始alpha/FP32评价缺失')
                time_control = json.loads((output / ('static_time_%d' % step) / 'summary.json').read_text())
                if len(time_control['rows']) != 49 or time_control['anchor_pose_frame'] != cfg['followup_static_anchor_frame']:
                    raise RuntimeError('静态时间控制缺失或使用了不同固定pose')
            validate_completion(output, cfg)
            status.update(status='complete', iteration=30000, wall_seconds=time.monotonic() - started, memory=monitor.update(iteration=30000, extra={'arm_complete': True}))
            write(output / 'status.json', status)
        except BaseException as error:
            status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            write(output / 'status.json', status)
            raise
        finally:
            monitor.close(status=status['status'], error=status.get('error'))
