"""只准备新目录，不启动GPU；保留失败现场和原49帧道路输入。"""
import argparse
import datetime
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--reviewed-source', type=Path, required=True)
    parser.add_argument('--user-authorized-fresh-restart', action='store_true')
    args = parser.parse_args()
    old, new, source = args.previous.resolve(), args.campaign.resolve(), args.reviewed_source.resolve()
    if not args.user_authorized_fresh_restart:
        raise RuntimeError('human fresh restart approval required')
    if new.exists() or old == new or old in new.parents:
        raise FileExistsError('a distinct fresh sibling campaign is required')
    registry = json.loads((old/'pair_registry.json').read_text())
    if registry['status'] != 'FAILED_PRESERVED_NO_RESTART' or registry.get('active'):
        raise RuntimeError('predecessor must be terminal and preserved')
    new.mkdir()
    # 仅复制代码，旧checkpoint、日志、评价、注册表一律不搬动/覆盖。
    shutil.copytree(old/'engine', new/'engine')
    shutil.copytree(source/'engine_overlay', new/'engine', dirs_exist_ok=True)
    for path in old.glob('*.py'):
        shutil.copy2(path, new/path.name)
    for path in (source/'tools').glob('*.py'):
        shutil.copy2(path, new/path.name)
    shutil.copytree(source/'tests', new/'tests', ignore=shutil.ignore_patterns('__pycache__'))
    for path in old.glob('*.svg'):
        shutil.copy2(path, new/path.name)
    for arm in ['full', 'real_only', 'prior_only', 'ownership_only']:
        (new/'arms'/arm).mkdir(parents=True)
    (new/'monitor').mkdir()
    plan = json.loads((old/'plan.json').read_text())
    approval = json.loads((old/'GPU_APPROVAL.json').read_text())
    restart = dict(restart_policy='user_authorized_fresh_after_unresumable_publication_failure',
        preserved_failed_campaign=str(old), failed_updates_separately_accounted=10000,
        execution_start_mode='fresh', publication_fix='seven_channel_road_float_and_v3_runtime_schema',
        automatic_restart=False, human_verdict=None)
    plan.update(restart, campaign=str(new), training_updates_executed=0)
    approval.update(restart, campaign=str(new),
        authorization_quote='不能保证从checkpoints精确续训的话，可以从头开始',
        authorized_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for name, record in [('plan.json', plan), ('GPU_APPROVAL.json', approval)]:
        (new/name).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(campaign=str(new), previous=str(old), input_bundle=plan['input_bundle'],
                         GPU_started=False, future_training_updates=120000), indent=2))


if __name__ == '__main__':
    main()
