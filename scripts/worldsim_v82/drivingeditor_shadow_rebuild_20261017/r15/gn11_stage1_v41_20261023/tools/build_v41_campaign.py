"""从原 V1 创建独立 V4.1 目录；仅 CPU 准备，不启动任何 GPU 任务。"""

import argparse
import json
from pathlib import Path
import shutil
import sys
from v41_train_hooks import install_hooks


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'baseline-config', 'bundle', 'dense-manifest', 'destination', 'v4-package'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    if args.destination.exists():
        raise FileExistsError('保留全部旧版本；只创建新 campaign')
    baseline = json.loads(args.baseline_config.read_text())
    sys.path.insert(0, str(args.v4_package/'tools'))
    from build_v4_campaign import validate_config
    validate_config(baseline)
    dense = json.loads(args.dense_manifest.read_text())
    if not dense['dense_domain_ready'] or dense['source_bundle'] != str(args.bundle.resolve()):
        raise ValueError('密集证据不完整或道路输入不一致')
    destination = args.destination.resolve()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(args.source, destination/'engine', ignore=ignore)
    (destination/'engine/train.py').write_text(install_hooks((args.source/'train.py').read_text()), encoding='utf-8')
    for overlay in (args.v4_package/'engine_overlay', package/'engine_overlay'):
        for file in overlay.rglob('*.py'):
            target = destination/'engine'/file.relative_to(overlay)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(file, target)
    shutil.copytree(package/'tools', destination/'tools', ignore=ignore)
    shutil.copytree(package/'tests', destination/'tests', ignore=ignore)
    for name in ('run_v4_arm.py', 'v4_eval_metrics.py', 'compare_v1_states.py'):
        shutil.copy2(args.v4_package/'tools'/name, destination/'tools'/name)
    (destination/'configs').mkdir()
    for arm in ('R',):
        config = dict(baseline, model_path=str(destination/'arms'/arm/'run30k'),
            v4_arm=arm, v4_bundle=str(args.bundle.resolve()),
            v41_dense_manifest=str(args.dense_manifest.resolve()))
        (destination/'configs'/f'{arm}.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (destination/'arms'/arm).mkdir(parents=True)
    for name in ('README.md', 'architecture.svg'):
        if (package/name).exists():
            shutil.copy2(package/name, destination/name)
    plan = dict(task='WS-V82-R15-GN11-V41-PRE-GPU-20261023', version='GN11_V41_DENSE_COMPOSITE_ROAD_R5_20261024',
        V1_source=str(args.source.resolve()), baseline_config=str(args.baseline_config.resolve()),
        road_bundle=str(args.bundle.resolve()), dense_manifest=str(args.dense_manifest.resolve()),
        campaign=str(destination), arms=['R'], deferred=['RP', 'a6_geometry_completion'], iterations_each=30000,
        formal_checkpoint_steps=[10000, 20000, 30000], stage='CPU_PREPARATION_GPU_NOT_STARTED',
        V4_execution='old RP remains user-paused; no old controller reused',
        GPU_used=False, optimizer_updates=0, quality_accepted=False, human_verdict=None,
        failure_ledger_refs=['V82-F01', 'V82-F06'], failure_ledger_delta='none')
    (destination/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
