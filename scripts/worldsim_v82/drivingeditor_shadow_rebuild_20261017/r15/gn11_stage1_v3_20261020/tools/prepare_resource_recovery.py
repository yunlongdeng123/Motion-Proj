"""仅准备显存准入恢复目录；没有GPU调用、进程操作或关机动作。"""
import argparse
import datetime
import json
from pathlib import Path
import shutil

from v3_resource_recovery import RESOURCE_RECOVERY_POLICY, failed_updates, validate_resource_recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--reviewed-source', type=Path, required=True)
    parser.add_argument('--user-authorized-autonomous-repair', action='store_true')
    args = parser.parse_args()
    if not args.user_authorized_autonomous_repair:
        raise RuntimeError('fresh resource recovery requires current human authorization')
    old, new, source = args.previous.resolve(), args.campaign.resolve(), args.reviewed_source.resolve()
    if new.exists() or new == old or old in new.parents:
        raise FileExistsError('a distinct new sibling campaign is required')
    registry = json.loads((old/'pair_registry.json').read_text())
    if registry['status'] != 'FAILED_PRESERVED_NO_RESTART' or registry.get('active'):
        raise RuntimeError('predecessor must be terminal with no active slots')
    plan = json.loads((old/'plan.json').read_text())
    approval = json.loads((old/'GPU_APPROVAL.json').read_text())
    counts = {}
    for arm in plan['arm_order']:
        path = old/'arms'/arm/'run30k/status.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state.get('status') != 'failed_preserved_no_restart' or 'OutOfMemoryError' not in state.get('error', ''):
                raise RuntimeError('predecessor is not the audited late-stage OOM failure')
            counts[arm] = failed_updates(state)
    if set(counts) != {'full', 'real_only'}:
        raise RuntimeError('unexpected completed/failed predecessor arms')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    authorization = dict(scope='V3_main_four_arms_only', authorized_utc=now,
        human_quote='4实验你全部顺利完成后帮我把远端autodl关机，不用我审批。如果你两次30分钟检查都没问题，可以把定时任务改成每60分钟（如果新批次或出现风险时恢复 10 分钟，以此类推），所有bug你自己去修好然后让实验全部完整顺利跑好。',
        autonomous_engineering_repair=True, preserve_failed_assets=True,
        physical_schedule_authorization_quote='嗯，要是没法2+2，就只能1+1+1+1了',
        physical_max_concurrent_training=1,
        checkpoint_resume_only_after_strict_validation=True, fresh_fallback_authorized=True,
        shutdown_after_all_four_complete=True, no_additional_shutdown_confirmation_required=True,
        shutdown_requires=['four_actual_30k_three_Adam_counts', 'six_complete_four_part_nodes_each',
                           'all_49_frame_formal_reconstruction_and_deletion_evaluations',
                           'finite_assets_and_road_ID_checks', 'reports_and_code_pushed',
                           'no_training_evaluation_data_render_or_pending_launch_controller_on_host'],
        never_infer_completion_from_GPU_idle=True, never_kill_unrelated_work=True,
        no_199_extension_or_counterfactual=True, no_method_or_quality_verdict_fabrication=True,
        human_verdict=None)
    updates = int(plan.get('failed_updates_separately_accounted', 0))+sum(counts.values())
    patch = dict(campaign=str(new), max_concurrent_training=1,
        resource_recovery_policy=RESOURCE_RECOVERY_POLICY, execution_start_mode='fresh',
        preserved_failed_campaign=str(old), previous_OOM_actual_updates=counts,
        failed_updates_separately_accounted=updates, training_updates_executed=0,
        automatic_restart=False, human_verdict=None,
        restart_policy='user_authorized_fresh_after_resource_admission_failure')
    for record in [plan, approval]:
        record.pop('pair_policy', None); record.update(patch)
    approval.update(authorization_quote=authorization['human_quote'], authorized_utc=now)
    validate_resource_recovery(plan, approval)
    new.mkdir()
    shutil.copytree(old/'engine', new/'engine', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(source/'engine_overlay', new/'engine', dirs_exist_ok=True)
    for path in old.glob('*.py'): shutil.copy2(path, new/path.name)
    for path in (source/'tools').glob('*.py'): shutil.copy2(path, new/path.name)
    shutil.copytree(source/'tests', new/'tests', ignore=shutil.ignore_patterns('__pycache__'))
    for path in old.glob('*.svg'): shutil.copy2(path, new/path.name)
    for arm in plan['arm_order']: (new/'arms'/arm).mkdir(parents=True)
    (new/'monitor').mkdir()
    for name, record in [('plan.json', plan), ('GPU_APPROVAL.json', approval), ('OPERATOR_AUTHORIZATION.json', authorization)]:
        (new/name).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(campaign=str(new), input_bundle=plan['input_bundle'],
        failed_actual_updates=updates, future_training_updates=120000,
        physical_slots=1, GPU_started=False, shutdown_executed=False), indent=2))


if __name__ == '__main__':
    main()
