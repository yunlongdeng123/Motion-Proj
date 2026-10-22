"""从 V1 源码复制 V4；仅准备文件，不启动训练或渲染。"""

from pathlib import Path
import argparse
import json
import shutil
import sys
from v4_train_hooks import install_hooks, strip_hooks


def validate_config(config):
    if (config['seed'] != 0 or config['iterations'] != 30000 or config['start_checkpoint'] is not None
            or config['resolution_scales'] != [4] or config['source_path'] != config['followup_scene_spec']['source_path']):
        raise ValueError('必须从同49帧 V1 fresh 初始化，保持 seed0/30K/原生分辨率')
    if any(key.startswith(('v2_', 'v3_', 'v4_')) for key in config):
        raise ValueError('只允许原 V1 配置')
    if (not all(config[key] for key in ('including_dynamic', 'paper_method', 'hybrid_method',
                                      'paper_sgcm_enabled', 'hybrid_nps_enabled', 'sgcm_guard_enabled'))
            or config['hybrid_lambda_dynamic_opacity'] != 100.):
        raise ValueError('V1 SGCM/NPS/完整动态 mask opacity 监督必须保留')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline-config', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    options = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    config = json.loads(options.baseline_config.read_text(encoding='utf-8'))
    validate_config(config)
    source = options.source.resolve()
    overlay = package/'engine_overlay'
    native = (source/'train.py').read_text(encoding='utf-8')
    modified = install_hooks(native)
    if strip_hooks(modified) != native:
        raise ValueError('去除两处 V4 hook 后必须逐字等于 V1 train.py')
    for name in ('scene/gaussian_model.py', 'gaussian_renderer/__init__.py',
                 'utils/hybrid30k_losses.py', 'scene/paper_sgcm.py'):
        if not (source/name).is_file():
            raise FileNotFoundError(name)
    sys.path.insert(0, str(overlay))
    from utils.v4_road import load_bundle, classify_units, VERSION, POLICY
    meta, arrays = load_bundle(options.bundle)
    observed, prior = classify_units(arrays)
    if options.destination.exists():
        raise FileExistsError('V4 只创建新目录，不能覆盖旧版本或已有运行')
    destination = options.destination.resolve()
    shutil.copytree(source, destination/'engine', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (destination/'engine/train.py').write_text(modified, encoding='utf-8')
    for file in overlay.rglob('*.py'):
        target = destination/'engine'/file.relative_to(overlay)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    shutil.copytree(package/'tools', destination/'tools', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(package/'tests', destination/'tests', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('README.md', 'architecture.svg', 'evaluation_contract.json'):
        shutil.copy2(package/name, destination/name)
    configs = destination/'configs'
    configs.mkdir()
    for arm in ('B', 'R', 'RP'):
        values = dict(config, model_path=str(destination/'arms'/arm/'run30k'),
                      v4_arm=arm, v4_bundle=str(options.bundle.resolve()))
        (configs/f'{arm}.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
        (destination/'arms'/arm).mkdir(parents=True)
    plan = dict(task='WS-V82-R15-GN11-V4-20261022', version=VERSION,
        V1_source=str(source), baseline_config=str(options.baseline_config.resolve()),
        road_bundle=str(options.bundle.resolve()), campaign=str(destination),
        arms=['B', 'R', 'RP'], iterations_each=30000, second_stage_updates=0,
        policy=POLICY, road_points=int(arrays['surface_supported'].sum()),
        observed_units=int(observed.sum()), connected_prior_units=int(prior.sum()),
        unsupported_units=int((~arrays['surface_supported']).sum()),
        unknown_supported_units=int((arrays['surface_supported'] & ~observed & ~prior).sum()),
        equal_steps_not_equal_compute=True, training_started=False,
        GPU_used=False, optimizer_updates=0, failure_ledger_refs=['V82-F01', 'V82-F06'],
        failure_ledger_delta='none', quality_accepted=False, human_verdict=None)
    (destination/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
