"""持共享GPU锁等待scene8完整终态，再生成红车真实分割；不自动选择actor或训练。"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013')
REPRO = ROOT.parents[1]
OLD = REPRO / 'campaign/next_scene30k_20261012'
RED = ROOT / 'inputs/scene0029_069_117_red_cpu_v1'
SCENE8 = ROOT / 'runs/scene0008_023_071_hybrid_30000_seed0_GN11'
META = ROOT / 'metadata/red_masks_20261013'
SEG = OLD / 'segmentation_repository'
GPU_PY = '/root/autodl-tmp/envs/drivingeditor/bin/python'
SKY_PY = '/root/autodl-tmp/envs/dggt_inference/bin/python'
FRAME_IDS = ['0029%03d' % i for i in range(69, 118)]
STATE = {'schema_version': 1, 'task_id': 'WS-V82-DRIVINGEDITOR-GN11-GATE-OTHER-SCENES-20261013',
         'scene_id': 'scene0029_069_117', 'pid': os.getpid(), 'status': 'initializing',
         'training_started': False, 'actor_selection_complete': False, 'human_verdict': None,
         'checkpoint_loaded': False, 'restoration_verified': False, 'stages': []}
ACTIVE_CHILD = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(status, **extra):
    STATE.update(status=status, updated_utc=now(), **extra)
    temporary = META / 'status.json.tmp'
    temporary.write_text(json.dumps(STATE, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(META / 'status.json')
    print(json.dumps({'status': status, 'pid': os.getpid(), 'updated_utc': STATE['updated_utc']}, ensure_ascii=False), flush=True)


def validate_inputs():
    sys.path.insert(0, str(ROOT / 'tools/followup'))
    from contracts import gate_contract
    assert read(ROOT / 'tools/DRIVINGEDITOR_FOLLOWUP_20261013.json')['primary_gate']['result']['status'] == 'expected'
    gate = read(ROOT / 'metadata/GN11_gate_passed.json')
    assert gate['max_psnr_drop_db'] == 1.0
    proof = gate_contract(ROOT / 'metadata/GN11_gate_passed.json', {'scene_id': 'scene0029_069_117'},
                          read(gate['events']['GN11'])['config'], verify_steps=True)
    assert proof['status'] == 'GATE_PASSED_RECHECKED'
    manifest = read(RED / 'preparation_manifest.json')
    assert manifest['raw_training_range'] == [69, 117] and manifest['num_points'] == 600000
    assert [row['id'] for row in manifest['frames']] == FRAME_IDS
    assert manifest['training_ready'] is False
    assert [p.stem for p in sorted((RED / 'image/image_0').glob('*.png'))] == FRAME_IDS
    assert not list((RED / 'sky/sky_0').glob('*.png')) and not (RED / 'mask/mask_0').exists(), 'Preserve existing/partial segmentation; do not overwrite.'
    for path in [SKY_PY, GPU_PY, ROOT / 'tools/red_preparation/extract_sky_no_fingerprint.py',
                 REPRO / 'weights/DEVA-propagation.pth', REPRO / 'weights/sam_vit_h_4b8939.pth',
                 '/root/autodl-tmp/dggt/pretrained/segformer.b5.1024x1024.city.160k.pth',
                 '/root/autodl-tmp/external/dggt_segformer/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py',
                 REPRO / 'cache/torch/hub/checkpoints/resnet18-5c106cde.pth',
                 REPRO / 'cache/torch/hub/checkpoints/resnet50-19c8e357.pth',
                 SEG / 'demo/demo_automatic.py', REPRO / 'scripts/validate_deva_masks.py']:
        assert Path(path).is_file(), 'Required cached input/source missing: ' + str(path)
    patch = (SEG / 'deva/ext/SAM/automatic_mask_generator.py').read_text(encoding='utf-8')
    assert 'batch_data["masks"] = batch_data["masks"].cpu()' in patch
    return {'status': 'PASS_REAL_GATE_AND_RED49_CACHED_INPUTS_CPU_ONLY', 'GPU_used': False,
            'paired_steps_reloaded': True, 'frames': 49, 'actor_selection_pending': True}


def sample_gpu(child_pid, sample, stop):
    """独立stdlib父进程5秒采样；不导入Torch，采样峰仅为下界。"""
    import csv
    sample.update(interval_seconds=5, sampled_peaks_are_lower_bounds=True, process_peak_bytes=0,
                  device_peak_bytes=0, samples=0, errors=[], pytorch_allocated_reserved='unavailable_external_child')
    while not stop.is_set():
        try:
            device = subprocess.run(['/usr/bin/nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', '0'],
                                    capture_output=True, text=True, timeout=5, check=True, close_fds=False)
            apps = subprocess.run(['/usr/bin/nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5, check=True, close_fds=False)
            sample['device_peak_bytes'] = max(sample['device_peak_bytes'], int(float(device.stdout.strip().splitlines()[0]) * 2**20))
            for fields in csv.reader(apps.stdout.splitlines()):
                if int(fields[0]) == child_pid:
                    sample['process_peak_bytes'] = max(sample['process_peak_bytes'], int(float(fields[1]) * 2**20))
            sample['samples'] += 1
        except Exception as exc:
            sample['errors'].append(str(exc))
        stop.wait(5)


def stage(name, command, environment, lock_fd, cwd=None):
    global ACTIVE_CHILD
    entry = {'stage': name, 'command': command, 'cwd': str(cwd) if cwd else str(ROOT),
             'started_utc': now(), 'environment_overrides': environment, 'memory_sampling': {}}
    STATE['stages'].append(entry)
    started = time.monotonic()
    env = os.environ.copy()
    env.update(environment)
    with (META / (name + '.log')).open('xb') as log:
        child = subprocess.Popen(command, cwd=cwd or ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
        ACTIVE_CHILD = child
        entry['pid'] = child.pid
        save(name + '_running', active_child_pid=child.pid)
        stop = threading.Event()
        monitor = threading.Thread(target=sample_gpu, args=(child.pid, entry['memory_sampling'], stop), daemon=True)
        monitor.start()
        try:
            code = child.wait()
        finally:
            stop.set()
            monitor.join(timeout=12)
        ACTIVE_CHILD = None
    entry.update(returncode=code, completed_utc=now(), wall_seconds=time.monotonic() - started)
    assert code == 0, name + ' failed; preserve partial outputs/logs for engineering review'
    save(name + '_complete', active_child_pid=None)


def stop_owned(signum, frame):
    raise KeyboardInterrupt('Owned preprocessing controller stopped by signal%d' % signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', action='store_true')
    args = parser.parse_args()
    proof = validate_inputs()
    if not args.queue:
        print(json.dumps(proof))
        return
    META.mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM, stop_owned)
    signal.signal(signal.SIGINT, stop_owned)
    try:
        save('waiting_shared_GPU_lock_scene0008_completion', validation=proof)
        with (REPRO / 'campaign/gpu_slot.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            validate_inputs()
            status = read(SCENE8 / 'status.json')
            assert status['status'] == 'complete' and status['iteration'] == 30000, 'scene8 not complete; stop without models'
            from contracts import completion_contract
            complete = completion_contract(SCENE8, read(SCENE8 / 'effective_config.json'))
            assert complete['status'] == 'FOLLOWUP_COMPLETE_CONTRACT_PASS'
            # 原worker可能刚释放锁但仍在Python退出；只等待已核对owned PID，不碰无关进程。
            owner_pid = status['pid']
            owner_command = Path('/proc') / str(owner_pid) / 'cmdline'
            import select
            try:
                owner_fd = os.pidfd_open(owner_pid)
            except ProcessLookupError:
                owner_fd = None
            if owner_fd is not None:
                try:
                    if not select.select([owner_fd], [], [], 0)[0]:
                        try:
                            command = owner_command.read_bytes().decode('utf-8', errors='replace')
                        except FileNotFoundError:
                            command = ''
                        if command:
                            assert 'run_scene_gn11.py' in command and 'scene0008_spec.json' in command
                        assert select.select([owner_fd], [], [], 30)[0], 'Owned scene8 PID has not exited; stop without models'
                finally:
                    os.close(owner_fd)
            apps = subprocess.run(['/usr/bin/nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                  capture_output=True, text=True, check=True, timeout=10, close_fds=False)
            assert not apps.stdout.strip(), 'Unrelated/live GPU process present; stop without killing or starting models'
            save('lock_owned_scene0008_complete_starting_red_masks', scene8_completion_contract=complete)
            common = {'CUDA_VISIBLE_DEVICES': '0', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONOPTIMIZE': '0',
                      'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
                      'TORCH_HOME': str(REPRO / 'cache/torch'), 'PYTORCH_CUDA_ALLOC_CONF': '',
                      'DRIVINGEDITOR_PARENT_GPU_OWNER_PID': str(os.getpid()), 'DRIVINGEDITOR_GPU_LOCK_FD': str(lock.fileno()),
                      'PATH': '/root/autodl-tmp/envs/drivingeditor/bin:/usr/local/cuda-11.8/bin:' + os.environ.get('PATH', ''),
                      'LD_LIBRARY_PATH': '/usr/local/cuda-11.8/lib64:/root/autodl-tmp/envs/drivingeditor/lib:' + os.environ.get('LD_LIBRARY_PATH', '')}
            sky_env = dict(common, PYTHONPATH='/root/autodl-tmp/external/dggt_segformer:/root/autodl-tmp/envs/dggt_segformer/overlay')
            stage('sky', [SKY_PY, str(ROOT / 'tools/red_preparation/extract_sky_no_fingerprint.py'), '--scene', str(RED),
                         '--config', '/root/autodl-tmp/external/dggt_segformer/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py',
                         '--checkpoint', '/root/autodl-tmp/dggt/pretrained/segformer.b5.1024x1024.city.160k.pth', '--device', 'cuda:0'], sky_env, lock.fileno())
            from PIL import Image
            import numpy as np
            assert [p.stem for p in sorted((RED / 'sky/sky_0').glob('*.png'))] == FRAME_IDS
            for path in sorted((RED / 'sky/sky_0').glob('*.png')):
                with Image.open(path) as image:
                    assert image.size == (1920, 1280)
                    assert set(np.unique(np.asarray(image))).issubset({0, 255})
            deva_env = dict(common, PYTHONPATH=str(SEG))
            stage('deva', [GPU_PY, str(SEG / 'demo/demo_automatic.py'), '--model', str(REPRO / 'weights/DEVA-propagation.pth'),
                          '--SAM_CHECKPOINT_PATH', str(REPRO / 'weights/sam_vit_h_4b8939.pth'), '--SAM_ENCODER_VERSION', 'vit_h',
                          '--sam_variant', 'original', '--chunk_size', '4', '--img_path', str(RED / 'image/image_0'),
                          '--temporal_setting', 'semionline', '--size', '480', '--output', str(RED / 'mask/mask_0'),
                          '--use_short_id', '--suppress_small_objects', '--SAM_PRED_IOU_THRESHOLD', '0.7'], deva_env, lock.fileno(), SEG)
            stage('mask_validation', [GPU_PY, str(REPRO / 'scripts/validate_deva_masks.py'), '--scene', str(RED),
                                      '--output', str(RED / 'mask/mask_0')], dict(deva_env, CUDA_VISIBLE_DEVICES=''), lock.fileno(), SEG)
            validation = read(RED / 'mask/mask_0/mask_validation.json')
            assert validation['status'] == 'passed' and validation['frames'] == 49
            assert read(RED / 'preparation_manifest.json')['training_ready'] is False
            save('REAL_RED49_SKY_AND_DEVA_VALIDATED_ACTOR_SELECTION_PENDING', mask_validation=validation,
                 sky_frames=49, GPU_lock_released_on_exit=True, training_ready=False)
    except BaseException as exc:
        if ACTIVE_CHILD is not None and ACTIVE_CHILD.poll() is None:
            ACTIVE_CHILD.terminate()
            ACTIVE_CHILD.wait(timeout=30)
        save('engineering_stopped_assets_preserved', error=str(exc), training_ready=False)
        raise


if __name__ == '__main__':
    main()
