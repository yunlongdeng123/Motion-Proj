"""控制器和单组GPU入口共用的、默认拒绝的授权校验。纯CPU，无外部动作。"""

import json
from pathlib import Path

GPU_ENTRY_VERSION = 'V3_GPU_ENTRY_R2_20261021'


def require_gpu_authorization(plan_path, plan, acknowledged, arm=None, mode=None):
    if not acknowledged:
        raise RuntimeError('explicit post-review GPU execution flag is required at every entry')
    campaign = Path(plan['campaign']).resolve()
    if Path(plan_path).resolve() != campaign/'plan.json':
        raise RuntimeError('main GPU approval cannot authorize alternate or counterfactual plans')
    approval_path = campaign/'GPU_APPROVAL.json'
    if not approval_path.is_file():
        raise RuntimeError('GPU_APPROVAL.json is absent; direct single-arm entry also requires approval')
    approval = json.loads(approval_path.read_text())
    expected = dict(status='USER_AUTHORIZED_GPU_V3', scope='zero_update_gates_and_four_fresh_30k',
        method_version=plan['method_version'], road_version=plan['road_version'],
        supervision_version=plan['common_v3_policy']['supervision_version'],
        render_audit_version=plan['render_audit_version'], gpu_entry_version=GPU_ENTRY_VERSION,
        input_bundle=plan['input_bundle'], arm_order=plan['arm_order'])
    if any(approval.get(key) != value for key, value in expected.items()):
        raise RuntimeError('GPU approval does not match reviewed input/version/four-arm scope')
    if arm is not None and (arm not in plan['arm_order'] or mode not in {'gate', 'train'}):
        raise RuntimeError('unregistered GPU arm/mode')
    if plan.get('road_source_exclusions'):
        raise RuntimeError('main approval excludes source counterfactual experiments')
    return approval


def require_reviewed_invocation(arguments, plan):
    """不能拿正确plan的审批，配上CLI里的另一套engine/input或额外run。"""
    campaign = Path(plan['campaign']).resolve()
    expected = dict(engine=campaign/'engine', bundle=Path(plan['input_bundle']),
        baseline_config=Path(plan['baseline_config']), readiness=Path(plan['readiness']),
        binding_tools=Path(plan['binding_tools']), gpu_lock=Path(plan['gpu_lock']),
        output=campaign/'arms'/arguments.arm/('gate_001' if arguments.mode == 'gate' else 'run30k'))
    if arguments.mode == 'train':
        expected['gpu_gate'] = campaign/'arms'/arguments.arm/'gate_001/gpu_gate_receipt.json'
    for key, path in expected.items():
        actual = getattr(arguments, key, None)
        if actual is None or Path(actual).resolve() != path.resolve():
            raise RuntimeError('GPU invocation differs from approved main run: '+key)
