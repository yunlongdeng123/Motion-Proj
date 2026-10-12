"""Prepare isolated S sources/config and genuine long references; never launch.

Only148 small original source files are copied. No checkpoint, image/NPZ or data
copy is made. Default reruns are byte-idempotent and refuse conflicting output.
The reviewed component owner separately prepares the actual S runtime plan.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
ROOT = BASE / 'campaign/vgpu_bugfix_20261012'
HERE = ROOT / 'loss_audit/next_diagnostics_cpu/observed_static_rgb_S'
PLAN = HERE / 'materialized_plan'
LONG = ROOT / 'loss_audit/long_pair_30k_32000'
ORIGINAL = BASE / 'campaign/resume_vgpu_baseline_20261011/frozen_repository'
REPLICA = HERE / 'frozen_repository'
OUTPUT = ROOT / 'continuation_diagnostic/observed_static_rgb30k_to32000'
B1 = ROOT / 'continuation_diagnostic/original30k_to32000'
B2 = ROOT / 'continuation_diagnostic/original30k_to32000_noise_reference'
EXTRA = {'diagnostic_observed_static_rgb_enabled': True,
         'diagnostic_lambda_observed_static_rgb': 1.0}
PINS = {
    LONG / 'materialized_plan/prepared_plan.json': '359b9d62ee9aec28951532cedbaed45401838dd6ea2b961d5704bec2d303a2d5',
    LONG / 'materialized_plan/continuation_harness.py': '88a2df38e5ac4df5905f1de030701189763d469f9eb3daf9e3339f1931aba829',
    LONG / 'materialized_plan/baseline_noise_reference/prepared_plan.json': '0b07086c6d425b7dc8526266b8520d5bd8d8d4736697833574029b2c22681329',
    LONG / 'materialized_plan/baseline_noise_reference/continuation_harness.py': 'ede8591eb6ffbc4cca970439d1e8dcd5b75a2b008b408fc03cab73c145e7dd3a',
    LONG / 'v_binding_cpu/approved_original_source_sha256.json': '459e41773f8200f35db6812e72ea13b673d0a576abe588422f7be7b3d3e8c0c9',
    LONG / 'evaluation_protocol_locked.json': 'c725cd2153d1104ddaf81078ac753f4e7f474e0be325ca9fbb5b1fccb342f658',
    ROOT / 'component_diagnostic/continuation_plan/runtime_safety_parameterized.py': 'f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1',
    ROOT / 'component_diagnostic/scripts/component_diagnostic.py': '0170283337ce3e2a1ea7a016e6f8a8e2b534101eebfe7b3ba4ecdcbcc1903baa',
    ROOT / 'sgcm_data_audit/outside_opacity_review/common49_input_manifest_review_cpu.json': '1b7161a2b30a08fb7fd8354a71b257cf99ad39bef16732a5f24a367174c24d50',
    B1 / 'receipts/full49_summary.json': '36879999ab39e208b0cb031e7557c83a3dabc11f32e361687796b8e641eced31',
    B2 / 'receipts/full49_summary.json': '2ded7dda0952e0bf8605154a642576049ce45b41e8c1df1ddda3bec62085c92e',
    B2 / 'receipts/B1_B2_observed_noise.json': 'e4ae32c07b63ed7be0171a727f8a12ec153f66e63d6ac89f4594fa4985145f08',
    B2 / 'receipts/common49_inputs_after_B2_cpu.json': 'dfd33893200805b4aebf147121b2be72270163a2ec61674e7704d2053de7f381',
    HERE / 'candidate/train.py': 'c4ca7ee8de35d748aac404394a52f65f4f12ff3a1b4d069e31d0c5bb2b8dd3b2',
    HERE / 'observed_static_rgb.patch': '79fe61a729a545d783d8ef807866b9754d3ec019c901908b0f1a6af82104099b',
    HERE / 'static_anchor_guard.py': '0a879a7a2574c877eb6464a29f7c91948e40e09b29310e34a02a86a5c7c23764',
}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def payload(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()


def write_once(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        require(p.read_bytes() == data, 'Refuse conflicting existing S artifact: ' + str(p))
        return
    with p.open('xb') as stream:
        stream.write(data)


def load_guard_module():
    spec = importlib.util.spec_from_file_location('independent_observed_static_RGB_guard', HERE / 'static_anchor_guard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_refs():
    refs = dict(PINS)
    selected = []
    for arm, output in [('B1', B1), ('B2', B2)]:
        status, summary = read(output / 'status.json'), read(output / 'receipts/full49_summary.json')
        require(status['status'] == 'complete' and status['actual_exit_code'] == 0 and status['iteration'] == 32000 and not status['gpu_lock_held'], arm + ' actual complete gate')
        require(summary['status'] == 'all_long_original_receipts_verified_CPU' and summary['arm'] == arm and summary['step_count'] == 2000 and summary['all4_node_eval_RNG_preserved'] and summary['first_step_original10_exact'], arm + ' actual original CPU export')
        rows = [json.loads(line) for line in (output / 'step_losses.jsonl').read_text().splitlines()]
        require([r['iteration'] for r in rows] == list(range(30001, 32001)), arm + ' exactly2000 actual steps')
        selected.append([{k: r[k] for k in ('iteration', 'camera_id', 'time_shift', 'learning_rates')} for r in rows])
        evaluation = read(output / 'extra_eval/32000/diagnostic_eval.json')
        require(len(evaluation['frames']) == 49 and evaluation['all_rng_unchanged'], arm + ' genuine49 eval')
        manifest = read(output / 'terminal_32000/diagnostic_snapshot_manifest.json')
        require(manifest['status'] == 'complete' and manifest['iteration'] == 32000 and len(manifest['checkpoints']) == 4 and all(x['cpu_load_iteration_schema_validated'] for x in manifest['checkpoints'].values()), arm + ' terminal four-state evidence')
        for relative in ('status.json', 'step_losses.jsonl', 'continuation_diagnostic.json', 'first_step_reference_comparison.json', 'runtime_import_pins.json', 'receipts/full49_summary.json', 'receipts/gpu_release_handoff.json', 'terminal_32000/diagnostic_snapshot_manifest.json', 'extra_eval/32000/diagnostic_eval.json'):
            p = output / relative
            refs[p] = sha(p)
    require(selected[0] == selected[1], 'Actual B1/B2 all2000 sampling/LR differ')
    noise = read(B2 / 'receipts/B1_B2_observed_noise.json')
    require(noise['status'] == 'completed_B1_B2_observed_original_repeat_variation' and len(noise['primary_metrics']) == 11, 'Use actual canonical original-repeat noise, no replacement metrics')
    common = read(B2 / 'receipts/common49_inputs_after_B2_cpu.json')
    require(common['status'] == 'PASS_common49_original_input_SHA_after_B2', 'Actual completed common49 evidence')
    return refs, selected[0]


def main():
    guard_module = load_guard_module()
    for p, expected in PINS.items():
        require(sha(p) == expected, 'Reviewed source/common reference pin differs: ' + str(p))
    require(not OUTPUT.exists(), 'Training output already exists; S preparation cannot overwrite any run')
    refs, rows = completed_refs()
    plan = read(LONG / 'materialized_plan/prepared_plan.json')
    original_inventory = read(LONG / 'v_binding_cpu/approved_original_source_sha256.json')
    require(guard_module.source_inventory(ORIGINAL) == original_inventory and len(original_inventory) == 148, 'Original exact148 source inventory')
    expected_inventory = dict(original_inventory)
    expected_inventory['train.py'] = sha(HERE / 'candidate/train.py')
    for relative in original_inventory:
        source = HERE / 'candidate/train.py' if relative == 'train.py' else ORIGINAL / relative
        write_once(REPLICA / relative, source.read_bytes())
    require(guard_module.source_inventory(REPLICA) == expected_inventory, 'Only one train.py S delta and exact148 source')
    parent = BASE / 'runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline'
    event = read(parent / 'checkpoint_events/iteration_030000.json')
    require(event['status'] == 'complete' and event['iteration'] == 30000, 'Original saved30k event')
    cfg = copy.deepcopy(plan['training_config'])
    cfg['model_path'] = str(OUTPUT)
    cfg.update(EXTRA)
    require(cfg['start_checkpoint'] == str(parent / 'chkpnt30000.pth') and cfg['iterations'] == cfg['position_lr_max_steps'] == 50000 and cfg['seed'] == 0 and cfg['including_dynamic'] is True and cfg['paper_method'] is True and type(cfg['lambda_dssim']) is float and cfg['lambda_dssim'] == 0.2, 'Fixed S initial/model/math schedule')
    from omegaconf import OmegaConf
    config_path = PLAN / 'static_anchor_effective_config.yaml'
    write_once(config_path, OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True).encode())
    protocol_path = HERE / 'evaluation_protocol_S_locked.json'
    require(read(protocol_path)['allow_training'] is False and read(protocol_path)['coefficient'] == 1.0, 'S preregistration coefficient/gate')
    refs.update({config_path: sha(config_path), protocol_path: sha(protocol_path), Path(__file__): sha(__file__), HERE / 'source_preparation_cpu.json': sha(HERE / 'source_preparation_cpu.json')})
    first = json.loads((B1 / 'step_losses.jsonl').read_text().splitlines()[0])
    original10 = {k: first['losses'][k] for k in ('loss_l1', 'loss_ssim', 'loss_paper_global_rgb', 'loss_paper_dynamic_rgb', 'loss_paper_sky', 'loss_paper_nps', 'loss_paper_dynamic_opacity', 'loss_paper_depth', 'loss', 'psnr')}
    observer = ROOT / 'component_diagnostic/continuation_plan/runtime_safety_parameterized.py'
    helper = ROOT / 'component_diagnostic/scripts/component_diagnostic.py'
    binding = {
        'schema_version': 1, 'status': 'actual_original_long_refs_S_source_config_bound_CPU_runtime_preparation_pending',
        'training_started': False, 'allow_training': False, 'experiment_kind': 'diagnostic_observed_static_rgb_complement',
        'scientific_status': 'Paper-unlisted observed-static RGB diagnostic, exact2024 coefficient1 .8L1+.2DSSIM fullHW on complementary actor mask. No V outside-opacity. Recovery from identical original30k, not from-zero or certified paper formula.',
        'parent_iteration': 30000, 'first_iteration': 30001, 'stop_iteration': 32000, 'steps': 2000, 'horizon': 50000, 'seed': 0,
        'output': str(OUTPUT), 'repository': str(REPLICA), 'original_repository': str(ORIGINAL), 'parent_static_checkpoint': cfg['start_checkpoint'],
        'parent_training_config': event['config'], 'training_config': cfg, 'added_training_fields': EXTRA,
        'training_config_changed_original_fields': ['model_path', 'start_checkpoint'], 'training_config_added_fields': sorted(EXTRA), 'strict_original_config': False,
        'algorithm_changed_files': ['train.py'], 'algorithm_patch': str(HERE / 'observed_static_rgb.patch'), 'algorithm_patch_sha256': sha(HERE / 'observed_static_rgb.patch'),
        'original_source_sha256': original_inventory, 'replica_source_sha256': expected_inventory,
        'protected_original_file_sha256': plan['original_file_sha256'],
        'baseline_binding': {'plan_path': str(LONG / 'materialized_plan/prepared_plan.json'), 'plan_sha256': sha(LONG / 'materialized_plan/prepared_plan.json'), 'harness_path': str(LONG / 'materialized_plan/continuation_harness.py'), 'harness_sha256': sha(LONG / 'materialized_plan/continuation_harness.py'), 'observer_path': str(observer), 'observer_sha256': sha(observer), 'output': str(B1)},
        'guard_path': str(HERE / 'static_anchor_guard.py'), 'guard_sha256': sha(HERE / 'static_anchor_guard.py'), 'auxiliary_file_sha256': {str(p): v for p, v in refs.items()},
        'paired_protocol': {'path': str(protocol_path), 'sha256': sha(protocol_path), 'preregistered_before_S_results': True, 'defined_after_long_V_results': True, 'unchanged_original_endpoint_protocol': {'path': str(LONG / 'evaluation_protocol_locked.json'), 'sha256': sha(LONG / 'evaluation_protocol_locked.json')}},
        'runtime_helper_import_contract': {'component_diagnostic': {'path': str(helper), 'sha256': sha(helper), 'functions': ['stats', 'rgba_package', 'region_stats', 'rng_capture', 'rng_restore', 'rng_pins']}, 'runtime_safety_parameterized': {'path': str(observer), 'sha256': sha(observer)}},
        'evaluation': {'nodes': [30001, 30300, 31000, 32000], 'early_fixed_camera_ids': ['image_0/0017' + str(i).zfill(3) for i in (0, 10, 20, 30, 48)], 'terminal_eval_camera_count': 49, 'all_extra_eval_rng_preserved': True},
        'restore_policy': plan['restore_policy'], 'restore_verification_status': 'S restore not executed; actual original30k restore is a mandatory runtime gate. CPU preparation is not CUDA restore proof.',
        'baseline_completion': {'status': 'complete', 'B1_actual_exit_code': 0, 'B2_actual_exit_code': 0, 'both_terminal49_and_four_validated_checkpoints': True},
        'first_step_pair_reference': {'iteration': 30001, 'values': original10, 'original_objective_log_key': 'loss_original_objective_before_diagnostic', 'comparison_phase': 'Actual original compose return before any S computation or addition; repeat captured original10 and FP32 total decomposition before actual first backward.', 'source_path': str(B1 / 'step_losses.jsonl'), 'source_sha256': sha(B1 / 'step_losses.jsonl'), 'policy': 'All10 exact against actual B1 before S and before backward; subsequent losses may differ.'},
        'all_iteration_pair_reference': {'rows': rows, 'source_path': str(B1 / 'step_losses.jsonl'), 'source_sha256': sha(B1 / 'step_losses.jsonl'), 'policy': 'All2000 actual camera/time_shift/LR exactly paired with B1 and its already validated B2 repeat; no seed after restore.'},
        'original_repeat_noise_reference': {'path': str(B2 / 'receipts/B1_B2_observed_noise.json'), 'sha256': sha(B2 / 'receipts/B1_B2_observed_noise.json'), 'B2_output': str(B2), 'B2_steps_path': str(B2 / 'step_losses.jsonl'), 'B2_steps_sha256': sha(B2 / 'step_losses.jsonl'), 'B2_full49_summary_path': str(B2 / 'receipts/full49_summary.json'), 'B2_full49_summary_sha256': sha(B2 / 'receipts/full49_summary.json'), 'is_confidence_interval_or_CUDA_noise_bound': False},
        'common49_completed_reference': {'path': str(B2 / 'receipts/common49_inputs_after_B2_cpu.json'), 'sha256': sha(B2 / 'receipts/common49_inputs_after_B2_cpu.json'), 'runtime_mask_NPZ_or_actualScene_pose_snapshot_available': False},
        'extra_eval_rng_policy': plan['extra_eval_rng_policy'], 'harness_integration_status': 'Independent S harness/launcher owned by component agent; actual CPU plan and independent review still required.',
        'launch_gates': ['Actual S CPU plan with genuine original10-before-S and before-backward decomposition gates', 'Independent final S source/config/guard/harness/plan review', 'Root review of completed read-only fixed-pose/time scan', 'Root exact S approval receipt and component-only exclusive GPU ownership'],
        'unchanged_algorithms': ['All original paper_losses functions and compose/weights', 'foreground Eq13 and prediction-masked Eq14', 'shared environment/ray jitter', 'static temporal fields and optimizer', 'depth/zbuffer and original input masks', 'SGCM/reset/topology', 'original30k four-state RNG/sampler/LR50k']}
    guard = guard_module.ObservedStaticRgbExperimentGuard(binding)
    checks = {'sources': guard.verify_sources(), 'auxiliary': guard.verify_auxiliary_files()}
    # No large protected checkpoint rehash here. The real S CPU runtime plan and
    # later restore still apply the complete unchanged protected-file contract.
    write_once(PLAN / 'experiment_binding.json', payload(binding))
    receipt = {'status': 'S_SOURCE_CONFIG_REAL_LONG_REFERENCES_BOUND_CPU_ONLY', 'training_started': False, 'CUDA_import_or_use': False, 'large_checkpoint_or_data_copy': False, 'source_files': 148, 'algorithm_changed_files': ['train.py'], 'binding_sha256': sha(PLAN / 'experiment_binding.json'), 'config_sha256': sha(config_path), 'guard_sha256': sha(HERE / 'static_anchor_guard.py'), 'candidate_train_sha256': sha(REPLICA / 'train.py'), 'source_diff_sha256': sha(HERE / 'observed_static_rgb.patch'), 'source_and_auxiliary_checks': checks, 'protected_original_pin_count': len(plan['original_file_sha256']), 'large_protected_hashes_here': 'Reused approved actual B1 pins; mandatory complete protected verification remains in real S CPU plan and runtime. No new CUDA restore proof claimed.', 'actual_runtime_CPU_plan_ready': False, 'GPU_launch_authorized': False}
    write_once(HERE / 'binding_preparation_cpu.json', payload(receipt))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
