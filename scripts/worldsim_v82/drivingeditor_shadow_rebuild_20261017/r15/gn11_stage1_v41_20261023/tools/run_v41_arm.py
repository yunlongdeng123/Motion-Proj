"""默认 dry-run；正式训练须真实生命周期 smoke 通过。旧 RP 不在此入口。"""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def configuration(campaign, arm, mode, label=None, stop_after=None, resume=None):
    if arm != 'R':
        raise ValueError('当前仅固定几何 V4.1-R；RP/几何补全须待质量与有效覆盖验收后')
    plan = read(campaign/'plan.json')
    baseline = read(plan['baseline_config'])
    original = read(campaign/'configs'/f'{arm}.json')
    expected = dict(baseline, model_path=str(campaign/'arms'/arm/'run30k'), v4_arm=arm,
        v4_bundle=plan['road_bundle'], v41_dense_manifest=plan['dense_manifest'])
    if original != expected:
        raise ValueError('原 V1 配置或方法配置被修改')
    config = dict(original)
    if mode == 'smoke':
        if not label or Path(label).name != label or label in ('.', '..') or stop_after not in (200, 500, 1000):
            raise ValueError('smoke 使用独立命名目录及固定 200/500/1000 节点')
        config.update(model_path=str(campaign/'smoke'/label))
        for name in ('save_iterations', 'checkpoint_iterations', 'test_iterations'):
            config[name] = sorted(set(config[name]) | {200, 500, 1000})
    elif stop_after or label:
        raise ValueError('正式训练保持30K，无短程开关')
    if resume is not None:
        if mode != 'smoke':
            raise ValueError('正式恢复另行核对实际更新账，禁止隐式恢复')
        resume = Path(resume).resolve()
        run = Path(config['model_path'])
        step = int(resume.stem.removeprefix('chkpnt'))
        if resume.parent != run or resume.name != f'chkpnt{step}.pth' or step >= stop_after:
            raise ValueError('smoke 恢复必须在同一目录内向后继续')
        manifest = read(run/f'v41_checkpoint{step}.json')
        if manifest['V1_updates'] != step or manifest['road_updates'] != step:
            raise ValueError('恢复节点更新数不一致')
        if not all((run/name).is_file() and (run/name).stat().st_size > 0 for name in manifest['files']):
            raise ValueError('恢复节点五件不齐')
        config['start_checkpoint'] = str(resume)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('R',))
    parser.add_argument('--mode', choices=('smoke', 'train'), default='train')
    parser.add_argument('--label')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    campaign = options.campaign.resolve()
    config = configuration(campaign, options.arm, options.mode, options.label, options.stop_after, options.resume)
    print(json.dumps(dict(mode=options.mode, arm=options.arm, output=config['model_path'],
        native_schedule_iterations=config['iterations'], stop_after=options.stop_after,
        execute=options.execute, resume=config['start_checkpoint']), indent=2), flush=True)
    if not options.execute:
        return
    validation = read(campaign/'cpu_validation.json')
    if validation['status'] != 'PASS_CPU_PRE_GPU':
        raise ValueError('CPU 检查未通过')
    if options.mode == 'train':
        receipt = read(campaign/'smoke'/'receipt.json')
        if receipt['status'] != 'PASS_REAL_GPU_LIFECYCLE_SMOKE' or receipt['wall_seconds'] < 600:
            raise ValueError('先完成约10分钟真实 GPU 生命周期 smoke')
        sys.path.insert(0, str(campaign/'engine'))
        from utils.v41_objectives import VERSION, POLICY
        if receipt['version'] != VERSION or receipt['policy'] != POLICY or receipt['formal_config'] != config:
            raise ValueError('smoke 与待启动正式方法/配置不一致')
    output = Path(config['model_path'])
    if output.exists() and not options.resume:
        raise FileExistsError('不覆盖、不自动重放')
    # 检查发生在 CUDA 初始化之前；本次交付只调用 dry-run。
    sys.path.insert(0, str(campaign/'engine'))
    import torch
    from omegaconf import OmegaConf
    import train
    from run_v4_arm import bind_scene_contract
    from utils.general_utils import seed_everything
    from utils.v41_road import SmokeStop
    from utils import v41_road
    v41_road.STOP_AFTER = options.stop_after
    if Path(train.__file__).resolve() != campaign/'engine/train.py':
        raise RuntimeError('错误 engine')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke/训练需要 CUDA，不降级 CPU 实验')
    output.mkdir(parents=True, exist_ok=bool(options.resume))
    if options.resume:
        from v41_restore_audit import install
        install(output)
        # 原 Scene 绑定凭据保留；恢复仍逐项检查，写新凭据。
        existing = output/'actual_scene_binding.json'
        previous = output/f'actual_scene_binding_before_resume_{Path(options.resume).stem}.json'
        if previous.exists():
            raise FileExistsError('同一个恢复入口已有现场')
        existing.rename(previous)
    bind_scene_contract(train, config)
    name = 'effective_config.json' if not options.resume else f'effective_config_resume_{Path(options.resume).stem}.json'
    (output/name).write_text(json.dumps(config, indent=2), encoding='utf-8')
    os.chdir(campaign/'engine')
    torch.cuda.set_device(0); seed_everything(config['seed'])
    started = time.perf_counter()
    stopped = None
    try:
        train.training(OmegaConf.create(config))
    except SmokeStop as exc:
        stopped = int(exc.args[0])
    expected = options.stop_after or 30000
    node = read(output/f'v41_checkpoint{expected}.json')
    if node['V1_updates'] != expected or node['road_updates'] != expected:
        raise RuntimeError('训练返回但完整节点不匹配')
    if options.mode == 'smoke' and stopped != expected:
        raise RuntimeError('smoke 未在目标完整节点退出')
    previous_step = int(Path(options.resume).stem.removeprefix('chkpnt')) if options.resume else 0
    (output/f'phase_{previous_step}_{expected}_returned.json').write_text(json.dumps(dict(
        status='RETURNED_AT_COMPLETE_NODE', iteration=expected,
        phase_V1_updates=expected-previous_step, phase_road_updates=expected-previous_step,
        wall_seconds=time.perf_counter()-started, mode=options.mode,
        quality_accepted=False, human_verdict=None), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
