"""等待 P1 正式 10000 步及六窗退出后，独占运行 DGGT 官方重建和高斯编辑。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path('/root/autodl-tmp/motion_proj_v81')
RUN = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-DGGT-WAYMO-INFERENCE-20261011/r1')
P1 = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/r1/paper_bidirectional_m4')
EXTERNAL = Path('/root/autodl-tmp/dggt')
PYTHON = Path('/root/autodl-tmp/envs/dggt_inference/bin/python')
SEQUENCES = ('00f88c4f0a', '7e625db8c4', 'ff6eb95840')
SIDES = (.125, .33)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def p1_finished():
    state = read(P1/'review_cycles/states/step010000.json')
    if state.get('status') == 'review_cycle_engineering_failure':
        raise RuntimeError('P1 10000 阶段发生工程错误；保留现场，不能越过训练/验证失败启动 DGGT')
    verified = read(P1/'review_cycles/checkpoint_step010000_verified.json')
    ready = (state.get('status') == 'waiting_assistant_training_review'
            and state.get('training_step') == 10000
            and verified.get('step') == 10000 and verified.get('protocol_unchanged') is True
            and len(state.get('cases', [])) == 6)
    if not ready:
        return False
    checkpoint = P1/'review_cycles/train/p1-checkpoint-010000.pt'
    if (checkpoint.is_symlink() or not checkpoint.is_file()
            or checkpoint.stat().st_size != verified.get('full_checkpoint_bytes')
            or state.get('checkpoint') != str(checkpoint)
            or verified.get('updates') != 2500 or verified.get('source_step') != 7500
            or verified.get('scheduler_step') != 10000 or verified.get('rng_keys_present') is not True
            or verified.get('finite_loss') is not True or verified.get('frozen_grad_tensors') != 0
            or state.get('training_beyond_target_started') is not False):
        raise RuntimeError('10000 完整原件或生产者校验记录不匹配')
    latest = read(P1/'controller_state.json')
    later = [p for p in (P1/'review_cycles/states').glob('step*.json')
             if int(p.stem[4:]) > 10000]
    if later or latest.get('training_budget') != 10000:
        raise RuntimeError('检测到10000之后的P1任务，优先策略已变化；不把旧状态当当前空闲')
    # 复用 P1 生产者的固定六窗合同，不重算指标或加载GPU模型。
    sys.path.insert(0, str(REPO/'scripts/worldsim_v81'))
    import continue_p1_review_cycles as p1
    p1.fixed_windows(10000, checkpoint)
    return not p1.active_controllers()


def preparation(scene):
    state = read(RUN/'cpu_preparation_state.json')
    data = RUN/'processed/validation'
    masks = RUN/f'source/segformer_cpu_8/{scene}'
    paths = [data/scene/'images'/f'{i:03d}_0.jpg' for i in range(4)]
    paths += [masks/k/f'{i:03d}_0.png' for k in ('sky_masks', 'custom_masks', 'semantic_raw19') for i in range(4)]
    paths += [RUN/'source/official_mode2_runtime.py',
              REPO/'scripts/worldsim_v81/infer_dggt_waymo_edits.py',
              REPO/'scripts/worldsim_v81/refine_dggt_waymo_edits.py',
              EXTERNAL/'pretrained/model_latest_waymo.pth', EXTERNAL/'pretrained/diffusion_model.pth']
    paths += [data/scene/k/'0.txt' for k in ('intrinsics', 'extrinsics')]
    paths += [data/scene/'ego_pose'/f'{i:03d}.txt' for i in range(4)]
    missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size == 0]
    return (state.get('status') == 'complete' and str(state.get('scene_id')) == str(int(scene))
            and not missing), missing


def validate_media(folder, frames=4):
    from PIL import Image
    for branch in ('noop', 'delete', 'move', 'insert_copy'):
        for i in range(frames):
            with Image.open(folder/branch/f'{i:03d}.png') as image:
                image.load()
                if image.mode != 'RGB' or min(image.size) < 2:
                    raise RuntimeError(f'无效图像: {folder}/{branch}/{i:03d}.png')
        video = folder/f'{branch}.mp4'
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError(f'缺视频: {video}')


def validate_edits(folder):
    manifest = read(folder/'manifest.json')
    comparison = manifest.get('official_noop_comparison') or {}
    if (manifest.get('status') != 'completed' or manifest.get('frames') != 4
            or comparison.get('max_abs', 1) > 1e-4):
        raise RuntimeError('编辑结果未通过实际官方 no-op 一致性检验')
    validate_media(folder)


def validate_refined(folder):
    manifest = read(folder/'manifest.json')
    if (manifest.get('status') != 'completed' or manifest.get('timestep') != 199
            or manifest.get('mv_unet') is not False or manifest.get('model_loads') != 1):
        raise RuntimeError('Difix未完整完成或推理协议变化')
    validate_media(folder)


def preserve_partial(folder):
    if folder.exists():
        # 失败产物整体改名保留，既不删除证据，也不让新尝试覆盖已有文件。
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        folder.rename(folder.with_name(folder.name + '_failed_' + stamp))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='128')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume-failed', action='store_true', help='显式保留已成功分支，只续未完成工程阶段')
    args = parser.parse_args()
    if not args.scene.isdigit() or len(args.scene) != 3:
        raise ValueError('scene 必须是官方列表对应的三位编号')
    if args.check_only:
        ready, missing = preparation(args.scene)
        print(json.dumps({'status': 'cpu_preflight', 'preparation_ready': ready, 'missing': missing,
                          'p1_10000_finished': p1_finished(), 'gpu_processes': gpu_processes(),
                          'training_after_10000': False}, ensure_ascii=False))
        return
    if not args.worker:
        raise ValueError('仅显式 --worker 启动；调度器不会自行追加训练')
    RUN.mkdir(parents=True, exist_ok=True)
    state_path = RUN/'queue_state.json'
    previous = read(state_path)
    if previous and previous.get('scene') != args.scene:
        raise RuntimeError('同一run不能复用另一scene的已完成标志')
    if str(previous.get('status', '')).startswith('complete_'):
        raise RuntimeError('已经完成推理，不重复跑；继续同步和审核')
    if previous.get('status') == 'engineering_failure' and not args.resume_failed:
        raise RuntimeError('先读原错误栈并修复；显式 --resume-failed 只续缺失阶段')
    with (RUN/'queue.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {**previous, 'scene': args.scene, 'controller_pid': os.getpid(), 'updated_at': now(),
                 'training_after_10000': False, 'human_verdict': None, 'commands': previous.get('commands', [])}
        if args.resume_failed and state.get('error'):
            # 保留修复前的错误；它不再表示本次恢复后的活动故障。
            history = list(state.get('resolved_engineering_failures', []))
            history.append({'error': state.pop('error'), 'traceback': state.pop('traceback', None),
                            'preserved_at': now(), 'scope': 'previous_failed_attempt'})
            state['resolved_engineering_failures'] = history

        def update(**fields):
            state.update(fields, updated_at=now())
            write(state_path, state)

        try:
            update(status='waiting_p1_10000_and_cpu_preparation')
            deadline = time.monotonic() + 8*3600
            while True:
                ready, missing = preparation(args.scene)
                if ready and p1_finished() and not gpu_processes():
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('等待超过8小时；保留队列状态，检查依赖')
                update(preparation_ready=ready, missing=missing, gpu_processes=gpu_processes())
                time.sleep(30)
            # 和 P1 使用相同的控制器锁，并保护下一段 launch，确保只有一个 GPU 作业。
            with (P1/'controller.lock').open('a') as ctl, (P1/'review_cycles/launch.lock').open('a') as launch:
                fcntl.flock(ctl, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(launch, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if gpu_processes() or not p1_finished():
                    raise RuntimeError('取得锁后 GPU/P1 状态发生变化；不启动竞争任务')
                data = RUN/'processed/validation'
                for kind in ('sky_masks', 'custom_masks'):
                    src = RUN/f'source/segformer_cpu_8/{args.scene}/{kind}'
                    dst = data/args.scene/kind
                    if dst.exists():
                        if any((dst/p.name).read_bytes() != p.read_bytes() for p in src.glob('*.png')):
                            raise RuntimeError(f'拒绝覆盖不同的 RGB 预测掩码: {dst}')
                    else:
                        shutil.copytree(src, dst)
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           CUDA_HOME='/usr/local/cuda', MAX_JOBS='1', TORCH_CUDA_ARCH_LIST='8.6',
                           TMPDIR=str(RUN/'tmp'), TORCH_HOME=str(EXTERNAL/'.cache/torch'),
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           PYTHONPATH=str(EXTERNAL),
                           PATH=f'{PYTHON.parent}:/root/autodl-tmp/envs/motionproj/bin:/usr/local/cuda/bin:' + env['PATH'])
                env.pop('CUDA_VISIBLE_DEVICES', None)

                def command(name, argv):
                    if gpu_processes():
                        raise RuntimeError(f'{name} 启动前 GPU 未闲')
                    entry = {'stage': name, 'argv': argv, 'started_at': now()}
                    state['commands'].append(entry)
                    update(status=name, child_pid=None)
                    with (RUN/f'logs/{name}.log').open('a') as log:
                        child = subprocess.Popen(argv, cwd=EXTERNAL, env=env, stdout=log, stderr=subprocess.STDOUT)
                        update(child_pid=child.pid)
                        code = child.wait()
                    entry.update(returncode=code, ended_at=now())
                    update(child_pid=None)
                    if code:
                        raise RuntimeError(f'{name} 返回 {code}；读 {RUN}/logs/{name}.log')

                baseline = RUN/'baseline_official'
                trace = baseline/'official_trace.pt'
                if state.get('official_baseline_complete') and not trace.is_file():
                    raise RuntimeError('已完成基线的actual trace丢失；保留现场重新核对')
                if not state.get('official_baseline_complete'):
                    preserve_partial(baseline)
                    command('official_baseline', [str(PYTHON), str(RUN/'source/official_mode2_runtime.py'),
                        '--image_dir', str(data), '--scene_names', str(int(args.scene)), '--input_views', '1',
                        '--sequence_length', '4', '--start_idx', '0', '--mode', '2',
                        '--ckpt_path', str(EXTERNAL/'pretrained/model_latest_waymo.pth'),
                        '--output_path', str(baseline), '-images', '-depth', '-metrics'])
                    if not trace.is_file():
                        raise RuntimeError('官方基线未保存实际 render trace')
                    update(official_baseline_complete=True)
                config = RUN/'source/edit_config.json'
                if not config.exists():
                    write(config, {'selection': {'source': 'semantic_masks', 'labels': [13],
                          'directory': str(RUN/f'source/segformer_cpu_8/{args.scene}/semantic_raw19'),
                          'roi_box_normalized_xyxy': [.52604, .53906, .79688, .82813],
                          'anchor_normalized_xy': [.67, .70], 'provenance': 'rgb_manual_roi_and_anchor'},
                          'move': {'right_widths': 1.0}, 'save_gaussians': True})
                edited = RUN/'gaussian_edits'
                if state.get('gaussian_edits_complete'):
                    validate_edits(edited)
                if not state.get('gaussian_edits_complete'):
                    preserve_partial(edited)
                    command('gaussian_edits', [str(PYTHON), str(REPO/'scripts/worldsim_v81/infer_dggt_waymo_edits.py'),
                        '--external', str(EXTERNAL), '--data-root', str(data), '--scene', args.scene,
                        '--checkpoint', str(EXTERNAL/'pretrained/model_latest_waymo.pth'),
                        '--output-dir', str(edited), '--sequence-length', '4', '--edit-config', str(config),
                        '--official-trace', str(trace)])
                    validate_edits(edited)
                    update(gaussian_edits_complete=True)
                refined = RUN/'difix_refined'
                if state.get('difix_complete'):
                    validate_refined(refined)
                if not state.get('difix_complete'):
                    preserve_partial(refined)
                    command('difix_refinement', [str(PYTHON), str(REPO/'scripts/worldsim_v81/refine_dggt_waymo_edits.py'),
                        '--external', str(EXTERNAL), '--edits-dir', str(edited),
                        '--output-dir', str(refined), '--sequence-length', '4', '--seed', '1234'])
                    validate_refined(refined)
                    update(difix_complete=True)
                update(status='complete_assistant_review_pending', controller_pid=None,
                       gpu_processes=gpu_processes(),
                       raw_edit_manifest=str(edited/'manifest.json'),
                       refined_manifest=str(RUN/'difix_refined/manifest.json'),
                       inference_only=True, official_weights_unchanged=True)
        except Exception as exc:
            update(status='engineering_failure', controller_pid=None, error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
