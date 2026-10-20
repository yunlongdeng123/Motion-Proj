"""资源恢复只改调度，不改任何训练方法；两个逻辑批次保留、物理槽为1。"""
RESOURCE_RECOVERY_POLICY = dict(
    version='V3_SINGLE_SLOT_RESOURCE_RECOVERY_R1_20261021',
    logical_batches=[['full', 'real_only'], ['prior_only', 'ownership_only']],
    max_concurrent_training=1,
    reason='two_late_stage_workers_exhausted_actual_32GiB_device',
    scientific_configuration_changed=False,
    failed_assets='preserve_all_no_overwrite',
    restart='fresh_seed0_no_unvalidated_resume',
    power='only_after_all_four_and_formal_evaluations_and_asset_checks_complete',
)


def validate_resource_recovery(plan, approval):
    if plan.get('resource_recovery_policy') != RESOURCE_RECOVERY_POLICY:
        raise RuntimeError('resource recovery policy differs from reviewed single-slot protocol')
    if approval.get('resource_recovery_policy') != RESOURCE_RECOVERY_POLICY:
        raise RuntimeError('single-slot resource recovery lacks matching human authorization')
    if plan.get('max_concurrent_training') != 1 or approval.get('max_concurrent_training') != 1:
        raise RuntimeError('late-stage concurrent OOM must not be replayed with two active workers')
    if 'pair_policy' in plan or 'pair_policy' in approval:
        raise RuntimeError('stale physical 2-slot policy must not coexist with resource recovery')
    if plan.get('arm_order') != ['full', 'real_only', 'prior_only', 'ownership_only']:
        raise RuntimeError('four-arm paired order changed')


def exact_30k(state):
    return (state.get('status') == 'complete' and state.get('iteration') == 30000
            and state.get('optimizer_step_counts') == dict(static=30000, dynamic=30000, environment=30000))


def failed_updates(state):
    counts = state.get('optimizer_step_counts', {})
    if set(counts) != {'static', 'dynamic', 'environment'} or len(set(counts.values())) != 1:
        raise ValueError('failure accounting requires three matching actual Adam counts')
    count = counts['static']
    if type(count) is not int or not 0 <= count <= 30000:
        raise ValueError('invalid actual failed update count')
    return count
