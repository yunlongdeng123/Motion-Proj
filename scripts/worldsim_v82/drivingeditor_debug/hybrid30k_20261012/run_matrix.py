"""CPU检查默认不启动；--run 在GPU开好后串行跑固定四臂。"""
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

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
ROOT = BASE / 'campaign/hybrid30k_20261012'
REPOSITORY = ROOT / 'frozen_repository'
HERE = Path(__file__).resolve().parent


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
                counter.update(render_static_points=int(args[1].get_xyz.shape[0]),
                               render_dynamic_points=int(args[2].get_xyz.shape[0]),
                               render_width=int(args[0].image_width), render_height=int(args[0].image_height))
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


def check_config(arm):
    from runtime import validate_config
    cfg = json.loads((ROOT / 'configs' / (arm + '.json')).read_text())
    validate_config(cfg)
    expected = {'C00': (False, False), 'G10': (True, False), 'N01': (False, True), 'GN11': (True, True)}
    if (cfg['paper_sgcm_enabled'], cfg['hybrid_nps_enabled']) != expected[arm]:
        raise ValueError('模块与arm标签不符')
    if Path(cfg['model_path']).parent != ROOT / 'runs' or Path(cfg['model_path']).exists():
        raise FileExistsError('必须使用未写过的本矩阵输出，旧检查点不会自动恢复')
    scene = Path(cfg['source_path'])
    files = sorted((scene / 'image/image_0').glob('*.png')) or sorted((scene / 'image/image_0').glob('*.jpg'))
    if len(files) != 49 or not (scene / 'points3d.ply').is_file() or not list(scene.glob('points3d_dynamic_*.ply')):
        raise ValueError('固定49视图或真实LiDAR初始化输入缺失')
    for frame in files:
        for directory in ('sky/sky_0', 'dynamic_masks/dynamic_mask_0'):
            if not (scene / directory / (frame.stem + '.png')).is_file():
                raise FileNotFoundError(directory + '/' + frame.stem)
    return cfg


def check_prepared():
    checks = json.loads((ROOT / 'integration_cpu_checks.json').read_text())
    if checks['status'] != 'PASS_CPU_INTEGRATION' or checks['GPU_used'] or checks['checkpoint_loaded']:
        raise RuntimeError('实际候选CPU集成检查未通过')


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
        if gpu_memory < 30 * 1024**3:
            raise RuntimeError('当前固定分辨率/点数计划要求至少30GiB GPU；不自动降配置')
        output = Path(cfg['model_path'])
        output.mkdir(parents=True, exist_ok=False)
        write(output / 'effective_config.json', cfg)
        args = OmegaConf.create(cfg)
        sys.path.insert(0, str(REPOSITORY))
        os.chdir(REPOSITORY)
        started = time.monotonic()
        status = {'status': 'initializing', 'arm': arm, 'iterations': 30000, 'pid': os.getpid(),
                  'started_singapore': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  'GPU': torch.cuda.get_device_name(0), 'GPU_memory_bytes': gpu_memory,
                  'checkpoint_scope': 'model+Adam+RNG+paired sampler+SGCM lineage',
                  'evaluation_scope': '49 all-observed training views; not independent test'}
        write(output / 'status.json', status)
        from memory_monitor import MemoryMonitor
        monitor = MemoryMonitor(output, torch_module=torch, device=0, nvidia_device='0')
        monitor.start()
        monitor.set_phase('initialization')

        # 从源码导入、点云/模型/Adam初始化起开始计峰值。
        try:
            spec = importlib.util.spec_from_file_location('hybrid30k_training', REPOSITORY / 'train.py')
            training = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(training)
            training.seed_everything(0)
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
            iteration, schedule, losses = positional[1], positional[2], positional[6]
            if iteration == 1:
                faulthandler.cancel_dump_traceback_later()
            numbers = {key: float(value.item()) if torch.is_tensor(value) else float(value) for key, value in losses.items()}
            if not all(math.isfinite(value) for value in numbers.values()):
                raise FloatingPointError('训练标量出现非finite值')
            if iteration == 1 or iteration % 100 == 0:
                now = time.monotonic()
                window_seconds = now - progress['time']
                status.update(status='training', iteration=iteration, wall_seconds=time.monotonic() - started,
                              loss=numbers['loss'], psnr=numbers['psnr'], static_points=numbers['total_points'])
                status['speed_window'] = {'first_iteration': progress['iteration'] + 1, 'last_iteration': iteration,
                    'wall_seconds': window_seconds,
                    'wall_iterations_per_second': (iteration - progress['iteration']) / window_seconds,
                    'includes_initialization': progress['iteration'] == 0,
                    'includes_evaluation': progress['included_evaluation']}
                progress.update(iteration=iteration, time=now, included_evaluation=False)
                memory = monitor.update(iteration=iteration, extra={'static_points': numbers['total_points'],
                                            'iter_forward_backward_ms': numbers['iter_time'], **rendered})
                status['memory'] = memory
                status['iter_forward_backward_ms'] = numbers['iter_time']
                write(output / 'status.json', status)
            if iteration in schedule:
                # LPIPS等评价初始化不能改变后续训练或完整恢复的随机状态。
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
                if any(not (output / 'eval' / ('train_%d_render' % step) / 'hybrid_raw' / (row['frame_id'].replace('/', '_') + '.npz')).is_file() for row in event['frames']):
                    raise RuntimeError('原始alpha/FP32评价缺失')
                time_control = json.loads((output / ('static_time_%d' % step) / 'summary.json').read_text())
                if len(time_control['rows']) != 49 or time_control['anchor_pose_frame'] != '0017010':
                    raise RuntimeError('静态时间控制缺失或使用了不同固定pose')
            status.update(status='complete', iteration=30000, wall_seconds=time.monotonic() - started,
                          memory=monitor.update(iteration=30000, extra={'arm_complete': True}))
            write(output / 'status.json', status)
        except BaseException as error:
            status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            write(output / 'status.json', status)
            raise
        finally:
            monitor.close(status=status['status'], error=status.get('error'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', choices=['C00', 'G10', 'N01', 'GN11'])
    cli = parser.parse_args()
    arms = ['C00', 'G10', 'N01', 'GN11']
    if cli.worker:
        if not cli.run:
            raise ValueError('worker需要--run；CPU检查不会启动')
        return run_arm(cli.worker)
    configs = {arm: check_config(arm) for arm in arms}
    check_prepared()
    if not cli.run:
        print(json.dumps({'status': 'PREPARED_WAITING_FOR_GPU', 'arms': arms, 'iterations_each': 30000,
                          'checkpoint_iterations': [10000, 20000, 30000], 'GPU_started': False}, ensure_ascii=False))
        return
    # controller也锁定，防止两个矩阵controller在独立worker间隙竞争同一队列。
    import fcntl
    with (ROOT / 'matrix.lock').open('a+') as matrix_lock:
        fcntl.flock(matrix_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []
        try:
            for index, arm in enumerate(arms):
                # 每臂三份完整检查点及媒体预留12GiB；未知膨胀不自动清理旧资产。
                if shutil.disk_usage(BASE).free < (len(arms) - index) * 12 * 1024**3:
                    raise RuntimeError('剩余实验的完整检查点/媒体空间不足')
                environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                                   OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
                                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                log = ROOT / (arm + '_training.log')
                if log.exists():
                    raise FileExistsError('拒绝覆盖训练日志或自动重跑失败臂')
                write(ROOT / 'matrix_status.json', {'status': 'running', 'current_arm': arm,
                       'controller_pid': os.getpid(), 'completed_arms': completed, 'queued_arms': arms[index + 1:]})
                with log.open('w', encoding='utf-8') as stream:
                    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--run', '--worker', arm],
                                   stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)
                completed.append(arm)
        except BaseException as error:
            write(ROOT / 'matrix_status.json', {'status': 'failed', 'current_arm': arm,
                   'completed_arms': completed, 'error': repr(error), 'traceback': traceback.format_exc()})
            raise
        write(ROOT / 'matrix_status.json', {'status': 'all_four_arms_complete', 'iterations_total': 120000})


if __name__ == '__main__':
    main()
