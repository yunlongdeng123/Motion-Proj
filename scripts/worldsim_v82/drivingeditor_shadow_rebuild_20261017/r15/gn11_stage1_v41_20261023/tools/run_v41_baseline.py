"""原V1直接入口：不安装道路 hook，原配置仅改变输出目录。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import subprocess
from v41_formal_contract import read, validate_authorization


def check_evaluator_entry(c):
    """用既有smoke的原生四件验证新增B评价入口；零更新，明确不是B训练。"""
    started = time.perf_counter()
    fixture = c/'smoke/native_evaluator_entry1000'
    output = c/'smoke/native_evaluator_entry1000_results'
    source_run = c/'smoke/continuous_R'
    plan = read(c/'plan.json')
    fixture.mkdir(exist_ok=False)
    config = dict(read(plan['baseline_config']), model_path=str(fixture))
    (fixture/'effective_config.json').write_text(json.dumps(config, indent=2))
    files = [f'{prefix}1000.pth' for prefix in ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state')]
    for name in files:
        (fixture/name).symlink_to(source_run/name)
    (fixture/'v1_checkpoint1000.json').write_text(json.dumps(dict(iteration=1000, files=files)))
    (fixture/'evaluation_fixture.json').write_text(json.dumps(dict(
        kind='ENGINEERING_EVALUATION_FIXTURE_NOT_B_TRAINING', source_run=str(source_run),
        optimizer_updates=0, reused_components='native smoke static/dynamic/environment only; road omitted'), indent=2))
    with (fixture/'evaluation.log').open('x') as stream:
        subprocess.run([sys.executable, str(c/'tools/evaluate_v41.py'), '--campaign', str(c),
            '--arm', 'B', '--run', str(fixture), '--iteration', '1000', '--output', str(output)],
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = read(output/'summary.json')
    original = read(c/'smoke/continuous_evaluation1000/summary.json')
    assert result['aggregate49']['frame_count'] == 49
    assert result['absent_road_arrays_are_zero'] and result['V1_native_replay_max_abs'] <= 2e-5
    differences = {key: result['aggregate49']['regions']['v1']['full'][key]-
        original['aggregate49']['regions']['v1']['full'][key] for key in ('psnr','ssim','lpips')}
    assert all(abs(value) < 1e-5 for value in differences.values()), differences
    (fixture/'returned.json').write_text(json.dumps(dict(status='PASS_NATIVE_ONLY_EVALUATOR_ENTRY',
        optimizer_updates=0, wall_seconds=time.perf_counter()-started,
        metric_differences=differences), indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    options = p.parse_args()
    c = options.campaign.resolve()
    auth = validate_authorization(c)
    plan = read(c/'plan.json')
    config = dict(read(plan['baseline_config']), model_path=str(c/'arms/B/run30k'))
    assert config == auth['B_config'] and config['start_checkpoint'] is None
    assert config['iterations'] == 30000
    for key in ('checkpoint_iterations', 'save_iterations', 'test_iterations'):
        assert config[key] == [10000, 20000, 30000]
    print(json.dumps(dict(arm='B', output=config['model_path'], execute=options.execute)), flush=True)
    if not options.execute:
        return
    check_evaluator_entry(c)
    output = Path(config['model_path'])
    if output.exists():
        raise FileExistsError('原V1对照只允许fresh新目录，不覆盖旧现场')
    source = Path(plan['V1_source'])
    native_train = (source/'train.py').read_text()
    assert 'V41_BEGIN' not in native_train and 'V4_BEGIN' not in native_train
    sys.path.insert(0, str(source))
    import torch
    from omegaconf import OmegaConf
    import train
    from run_v4_arm import bind_scene_contract
    from utils.general_utils import seed_everything
    assert Path(train.__file__).resolve() == source/'train.py'
    assert torch.cuda.is_available()
    output.mkdir(parents=True, exist_ok=False)
    (output/'effective_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    bind_scene_contract(train, config)
    os.chdir(source)
    torch.cuda.set_device(0)
    seed_everything(config['seed'])
    started = time.perf_counter()
    train.training(OmegaConf.create(config))
    (output/'cuda_memory_peak.json').write_text(json.dumps(dict(
        allocated_bytes=torch.cuda.max_memory_allocated(0),
        reserved_bytes=torch.cuda.max_memory_reserved(0),
        scope='entire native training() including native evaluation and checkpoint',
        reset_peak_memory_stats_called=False), indent=2))
    for step in (10000, 20000, 30000):
        files = [f'{prefix}{step}.pth' for prefix in ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state')]
        assert all((output/f).is_file() and (output/f).stat().st_size > 0 for f in files)
        (output/f'v1_checkpoint{step}.json').write_text(json.dumps(dict(
            iteration=step, V1_updates=step, road_updates=0, files=files), indent=2))
    (output/'phase_0_30000_returned.json').write_text(json.dumps(dict(
        status='RETURNED_AT_COMPLETE_NODE', iteration=30000, phase_V1_updates=30000,
        phase_road_updates=0, wall_seconds=time.perf_counter()-started,
        source=str(source), quality_accepted=False, human_verdict=None), indent=2))


if __name__ == '__main__':
    main()
