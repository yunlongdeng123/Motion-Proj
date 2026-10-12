"""CPU检查默认不启动；--run 在GPU开好后串行跑固定四臂。"""
import argparse
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
ROOT = BASE / 'campaign/hybrid30k_20261012'
REPOSITORY = ROOT / 'frozen_repository'
HERE = Path(__file__).resolve().parent


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
        spec = importlib.util.spec_from_file_location('hybrid30k_training', REPOSITORY / 'train.py')
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)
        training.seed_everything(0)
        from utils.hybrid30k_runtime import preserve_rng
        original_eval = training.complete_eval
        started = time.monotonic()
        status = {'status': 'initializing', 'arm': arm, 'iterations': 30000, 'pid': os.getpid(),
                  'started_singapore': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  'GPU': torch.cuda.get_device_name(0), 'GPU_memory_bytes': gpu_memory,
                  'checkpoint_scope': 'model+Adam+RNG+paired sampler+SGCM lineage',
                  'evaluation_scope': '49 all-observed training views; not independent test'}
        write(output / 'status.json', status)

        def observed_eval(*positional, **keyword):
            iteration, schedule, losses = positional[1], positional[2], positional[6]
            numbers = {key: float(value.item()) if torch.is_tensor(value) else float(value) for key, value in losses.items()}
            if not all(math.isfinite(value) for value in numbers.values()):
                raise FloatingPointError('训练标量出现非finite值')
            if iteration == 1 or iteration % 100 == 0:
                status.update(status='training', iteration=iteration, wall_seconds=time.monotonic() - started,
                              loss=numbers['loss'], psnr=numbers['psnr'], static_points=numbers['total_points'])
                write(output / 'status.json', status)
            if iteration in schedule:
                # LPIPS等评价初始化不能改变后续训练或完整恢复的随机状态。
                with preserve_rng():
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
                          GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            write(output / 'status.json', status)
        except BaseException as error:
            status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            write(output / 'status.json', status)
            raise


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
            with log.open('w', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--run', '--worker', arm],
                               stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)
        write(ROOT / 'matrix_status.json', {'status': 'all_four_arms_complete', 'iterations_total': 120000})


if __name__ == '__main__':
    main()
