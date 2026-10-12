"""CPU-only, idempotent long V binding; completed real B1 AND B2 are mandatory.

Default checks only. --materialize-and-cpu-prepare creates the isolated V source,
actual binding and CPU plan after both completion exports. No launch API exists.
The original long harness, guard, math and evaluation protocol stay unchanged.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
ROOT = BASE / 'campaign/vgpu_bugfix_20261012'
LONG = ROOT / 'loss_audit/long_pair_30k_32000'
PLAN = LONG / 'materialized_plan'
HERE = LONG / 'v_binding_cpu'
ORIGINAL = BASE / 'campaign/resume_vgpu_baseline_20261011/frozen_repository'
REPLICA = LONG / 'variant_frozen_repository'
OUTPUT = ROOT / 'continuation_diagnostic/opacity_outside30k_to32000'
COMPONENT = ROOT / 'component_diagnostic/continuation_plan'
PAIR_KEYS = ('iteration', 'camera_id', 'time_shift', 'learning_rates')
FRAMES = [f'image_0/0017{i:03}' for i in range(49)]
NODES = [30001, 30300, 31000, 32000]
PINNED = {
    PLAN / 'prepared_plan.json': '359b9d62ee9aec28951532cedbaed45401838dd6ea2b961d5704bec2d303a2d5',
    PLAN / 'continuation_harness.py': '88a2df38e5ac4df5905f1de030701189763d469f9eb3daf9e3339f1931aba829',
    PLAN / 'baseline_noise_reference/prepared_plan.json': '0b07086c6d425b7dc8526266b8520d5bd8d8d4736697833574029b2c22681329',
    PLAN / 'baseline_noise_reference/continuation_harness.py': 'ede8591eb6ffbc4cca970439d1e8dcd5b75a2b008b408fc03cab73c145e7dd3a',
    PLAN / 'continuation_variant_harness.py': '93df22a4b8db482af596657ba1d3e86e61a0ad0893dac3af35f85adf872ba1d4',
    PLAN / 'outside_experiment_guard.py': '8356201320c757079a5dfaf4cb4968e22bc537fbf8f1dd7e0e6d1b6242f0d42c',
    PLAN / 'launch_opacity_outside_continuation.py': '633019f97b800ea3a942a1b44dd41d7972b2e332ee714caddb21a0e69e443898',
    PLAN / 'algorithm_overrides/train.py': 'ff82c0d86b385bcdf28feff550163cd3ec9c7bec92e939306330ba4af335e99a',
    PLAN / 'algorithm_overrides/utils/paper_losses.py': '3832a7ced1387e1ff33f5e3aee0b56bc533303f80f5be840269236139fc866ab',
    PLAN / 'outside_effective_config.yaml': '97fa2db569466c9a903ad5d83ac788a1b2afeb5fff8804875ffd39c37851ad3c',
    LONG / 'evaluation_protocol_locked.json': 'c725cd2153d1104ddaf81078ac753f4e7f474e0be325ca9fbb5b1fccb342f658',
    ROOT / 'sgcm_data_audit/long_pair_review/interpretation_preregistered.json': 'af9726931762fde14c00684ed1004ce2a6464a46299e832f969cdce3c916a75d',
    COMPONENT / 'runtime_safety_parameterized.py': 'f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1',
    ROOT / 'component_diagnostic/scripts/component_diagnostic.py': '0170283337ce3e2a1ea7a016e6f8a8e2b534101eebfe7b3ba4ecdcbcc1903baa',
    ROOT / 'component_diagnostic/scripts/prepare_long_original_receipts_cpu.py': 'c82c4273c3032812f72298c2259bc744267f8ae1628c982b2528d965e2f49ea3',
    ROOT / 'component_diagnostic/scripts/quantify_long_original_repeat_cpu.py': 'd2a5d00ff35f352a186b2f6fa5bfcc7d3036b92eecd0cf7a889122f88e51a705',
    ROOT / 'component_diagnostic/scripts/verify_long_repeat_common_inputs_cpu.py': '38361760284ebc04b49829e274d69fa3c6a6d21319c5f51db957a0b1a05f1ec8',
    ROOT / 'sgcm_data_audit/outside_opacity_review/common49_input_manifest_review_cpu.json': '1b7161a2b30a08fb7fd8354a71b257cf99ad39bef16732a5f24a367174c24d50',
    HERE / 'approved_original_source_sha256.json': '459e41773f8200f35db6812e72ea13b673d0a576abe588422f7be7b3d3e8c0c9',
    ROOT / 'continuation_diagnostic/original30k_to32000_noise_reference/receipts/full49_summary.json': '2ded7dda0952e0bf8605154a642576049ce45b41e8c1df1ddda3bec62085c92e',
    ROOT / 'continuation_diagnostic/original30k_to32000_noise_reference/receipts/B1_B2_observed_noise.json': 'e4ae32c07b63ed7be0171a727f8a12ec153f66e63d6ac89f4594fa4985145f08',
    ROOT / 'continuation_diagnostic/original30k_to32000_noise_reference/receipts/common49_inputs_after_B2_cpu.json': 'dfd33893200805b4aebf147121b2be72270163a2ec61674e7704d2053de7f381',
    ROOT / 'continuation_diagnostic/original30k_to32000_noise_reference/receipts/gpu_release_handoff.json': '3486faf0bfe3bcc85949094a0d538a829766504b1eddd63414c10008144df4eb',
    ROOT / 'continuation_diagnostic/original30k_to32000_noise_reference/terminal_32000/diagnostic_snapshot_manifest.json': 'cce918f4349ea6d2dc0cc996f73bde75d4841d45cb15f286a10a06b90eecd90e',
    ROOT / 'continuation_diagnostic/original30k_to32000_noise_reference/step_losses.jsonl': '71759c95cdaf4193651d065c0dd81aa4507c3b58319602629679cb36caa8c05b',
}


class Incomplete(Exception):
    """Expected prerequisite waiting, not permission to use short/placeholder refs."""


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def payload(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def write_once(path, data):
    """Identical reruns reuse bytes; conflicting or partial prior output fails closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'Refuse conflicting existing generated file: ' + str(path))
        return False
    with path.open('xb') as stream:
        stream.write(data)
    return True


def window(plan):
    for key, expected in (('parent_iteration', 30000), ('first_iteration', 30001), ('stop_iteration', 32000), ('steps', 2000), ('horizon', 50000), ('seed', 0)):
        require(type(plan[key]) is type(expected) and plan[key] == expected, 'Long window invariant: ' + key)
    require(plan['algorithm_changes'] == [] and plan['repository'] == str(ORIGINAL), 'Original arm algorithm/source changed')
    require(plan['extra_eval_nodes'] == NODES and plan['terminal_eval_camera_count'] == 49, 'Original eval schedule changed')


def validate_release(arm, release):
    # B1 and B2 were exported by their owner's different receipt writers. Each
    # actual schema is checked explicitly; no permissive alternate-field fallback.
    if arm == 'B1':
        require(release['actual_iteration'] == 32000 and release['actual_step_count'] == 2000 and release['actual_worker_exit_code'] == 0 and release['actual_launcher_exit_code'] == 0 and release['both_completed_processes_absent'] is True, 'B1 release receipt failed')
    elif arm == 'B2':
        require(release['schema_version'] == 1 and release['status'] == 'B2_complete_exit0_GPU_released_and_CPU_pair_noise_verified' and release['completed_arm'] == 'B2_original_repeat_30k_to32000' and Path(release['output']) == ROOT / 'continuation_diagnostic/original30k_to32000_noise_reference', 'Actual B2 release schema/arm differs')
        require(release['worker_exit_code'] == 0 and release['launcher_exit_code'] == 0 and release['both_actual_processes_absent'] is True and release['GPU_compute_apps'] == [] and release['exclusive_lock_available_and_probe_released'] is True, 'Actual B2 worker/launcher/release failed')
        require(release['all2000_steps_finite_logged'] is True and release['all2000_camera_time_shift_LR_exact_to_B1'] is True and release['actual_full_saved30k_restore_PASS'] is True and release['terminal_four_states_hash_bytes_CPU_schema_validated'] is True, 'Actual B2 release long trajectory/restore/four-state evidence failed')
    else:
        raise ValueError('Unknown original arm')


def completed(arm):
    plan_dir = PLAN if arm == 'B1' else PLAN / 'baseline_noise_reference'
    plan = read(plan_dir / 'prepared_plan.json')
    window(plan)
    output = Path(plan['output'])
    expected = ROOT / ('continuation_diagnostic/original30k_to32000' + ('' if arm == 'B1' else '_noise_reference'))
    require(output == expected, 'Actual long original output differs')
    status_path = output / 'status.json'
    if not status_path.exists():
        raise Incomplete(arm + ' has not started/completed')
    status = read(status_path)
    require(status['status'] != 'failed', arm + ' failed; preserve evidence and stop')
    if status['status'] != 'complete':
        raise Incomplete(arm + ' is not complete: ' + status['status'])
    require(status['actual_exit_code'] == 0 and status['gpu_lock_held'] is False and status['iteration'] == 32000, arm + ' worker exit/lock/terminal failed')
    for relative in ('continuation_diagnostic.json', 'receipts/full49_summary.json', 'receipts/gpu_release_handoff.json'):
        if not (output / relative).exists():
            raise Incomplete(arm + ' completion export/handoff pending: ' + relative)
    report = read(output / 'continuation_diagnostic.json')
    summary = read(output / 'receipts/full49_summary.json')
    release = read(output / 'receipts/gpu_release_handoff.json')
    launcher = read(plan_dir / 'continuation_launcher_status.json')
    require(int((output / 'actual_exit_code.txt').read_text()) == 0 and launcher['status'] == 'complete' and launcher['actual_exit_code'] == 0, arm + ' actual launcher/worker exit failed')
    validate_release(arm, release)
    require(report['first_iteration'] == 30001 and report['stop_iteration'] == 32000 and report['step_count'] == 2000 and report['horizon'] == 50000 and report['seed'] == 0, arm + ' not the actual long report')
    require(report['original_algorithm_changed'] is False and report['genuine_full_loss_backward_Adam'] is True and report['restore']['status'] == 'actual_restore_pass' and report['all_node_eval_rng_preserved'] is True and report['terminal_full49_evaluation'] is True, arm + ' real train/restore/eval gate failed')
    require(summary['status'] == 'all_long_original_receipts_verified_CPU' and summary['arm'] == arm and summary['step_count'] == 2000 and summary['stop_iteration'] == 32000 and summary['all_protected_SHA_unchanged'] is True and summary['all4_node_eval_RNG_preserved'] is True and summary['first_step_original10_exact'] is True and summary['gpu_lock_released_by_harness'] is True, arm + ' actual completed export gate failed')
    if arm == 'B1':
        require(sha(output / 'receipts/full49_summary.json') == '36879999ab39e208b0cb031e7557c83a3dabc11f32e361687796b8e641eced31', 'Root supplied actual B1 summary pin changed')
    else:
        pair = summary['paired_original_repeat_evidence']
        require(pair['B1_B2_complete_pair_validated'] is True and pair['all2000_camera_time_shift_LR_exact'] is True and pair['all49_GT_float32_dimensions_region_counts_exact'] is True, 'Actual B2 export did not validate long B1/B2 pairing')
    require(read(output / 'diagnostic_plan.json') == plan, arm + ' actual training plan differs from CPU-approved plan')
    approval_path = plan_dir / 'root_coordination_receipt.json'
    require(read(output / 'root_coordination_receipt.json') == read(approval_path), arm + ' approval lineage differs')
    approval = read(approval_path)
    require(approval['decision'] == 'APPROVED_CONTROLLED_30K_TO_32000' and approval['prepared_plan_sha256'] == sha(plan_dir / 'prepared_plan.json') and approval['harness_sha256'] == plan['harness_sha256'], arm + ' actual original root approval pin differs')
    rows_path = output / 'step_losses.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    require([r['iteration'] for r in rows] == list(range(30001, 32001)) and all(math.isfinite(v) for r in rows for v in r['losses'].values()), arm + ' needs exactly2000 finite true steps')
    first = read(output / 'first_step_reference_comparison.json')
    require(len(first['comparisons']) == 10 and all(r['exact_equal'] is True for r in first['comparisons'].values()), arm + ' original first10 failed')
    snap_path = output / 'terminal_32000/diagnostic_snapshot_manifest.json'
    snap = read(snap_path)
    require(snap['status'] == 'complete' and snap['iteration'] == 32000 and snap['parent_iteration'] == 30000 and snap['horizon_retained'] == 50000 and snap['training_config'] == plan['training_config'] and snap['not_a_formal_paper_checkpoint_event'] is True and snap['continuation_RNG_restored_after_extra_eval'] is True and set(snap['checkpoints']) == {'static', 'dynamic', 'environment', 'training_state'}, arm + ' terminal4 manifest gate failed')
    for branch, row in snap['checkpoints'].items():
        p = Path(row['path'])
        require(p.parent == output / 'terminal_32000' and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'] and row['cpu_load_iteration_schema_validated'] is True, arm + ' terminal4 file gate failed: ' + branch)
        require(summary['checkpoint_validation'][branch]['sha256'] == row['sha256'], arm + ' exported snapshot pin differs')
    final_path = output / 'extra_eval/32000/diagnostic_eval.json'
    final = read(final_path)
    require(final['iteration'] == 32000 and [f['frame_id'] for f in final['frames']] == FRAMES and final['all_rng_unchanged'] is True, arm + ' full49 final frame/RNG gate failed')
    paths = [plan_dir / n for n in ('prepared_plan.json', 'continuation_harness.py', 'root_coordination_receipt.json', 'continuation_launcher_status.json')]
    paths += [output / n for n in ('status.json', 'continuation_diagnostic.json', 'diagnostic_plan.json', 'root_coordination_receipt.json', 'step_losses.jsonl', 'first_step_reference_comparison.json', 'runtime_import_pins.json', 'actual_exit_code.txt', 'receipts/full49_summary.json', 'receipts/gpu_release_handoff.json')]
    paths += [snap_path, final_path]
    for artifact in summary['artifact_pins']:
        p = Path(artifact['path'])
        require(p.resolve().is_relative_to(output.resolve()) and p.stat().st_size == artifact['bytes'] and sha(p) == artifact['sha256'], arm + ' terminal49 artifact pin differs: ' + str(p))
        if artifact.get('artifact') == 'raw_float32_arrays':
            paths.append(p)
    return {'plan': plan, 'output': output, 'rows': rows, 'final': final, 'pins': {str(p): sha(p) for p in paths}}


def paired(b1, b2):
    import numpy as np
    require(b1['plan']['original_file_sha256'] == b2['plan']['original_file_sha256'], 'B1/B2 source/kernel/data/initial4 pins differ')
    a, b = copy.deepcopy(b1['plan']['training_config']), copy.deepcopy(b2['plan']['training_config'])
    a.pop('model_path'); b.pop('model_path')
    require(a == b, 'B1/B2 original config differs beyond output')
    refs = [{k: r[k] for k in PAIR_KEYS} for r in b1['rows']]
    require(refs == [{k: r[k] for k in PAIR_KEYS} for r in b2['rows']], 'Actual long2000 sample/time_shift/LR mismatch')
    values = {k: b1['rows'][0]['losses'][k] for k in b1['plan']['first_step_reference']['values']}
    require(len(values) == 10 and values == {k: b2['rows'][0]['losses'][k] for k in values}, 'Actual first10 original scalars differ')
    domains = []
    for f1, f2 in zip(b1['final']['frames'], b2['final']['frames']):
        counts = {k: v['pixel_count'] for k, v in f1['regions'].items()}
        require(counts == {k: v['pixel_count'] for k, v in f2['regions'].items()}, 'Actual49 region counts differ')
        with np.load(f1['float32_arrays'], allow_pickle=False) as x, np.load(f2['float32_arrays'], allow_pickle=False) as y:
            require(x['gt'].dtype == y['gt'].dtype == np.float32 and x['gt'].shape == y['gt'].shape == (3, 320, 480) and np.array_equal(x['gt'], y['gt']), 'Actual49 GT differs')
        domains.append({'frame_id': f1['frame_id'], 'GT_float32_exact': True, 'region_pixel_counts': counts})
    return refs, values, domains


def noise_reference(b1, b2):
    """Bind the component owner's one canonical noise receipt; do not regenerate it."""
    path = b2['output'] / 'receipts/B1_B2_observed_noise.json'
    if not path.exists():
        raise Incomplete('Actual completed B1/B2 noise export is pending')
    noise = read(path)
    require(noise['schema_version'] == 1 and noise['status'] == 'completed_B1_B2_observed_original_repeat_variation' and noise['CPU_only_saved_evidence_analysis'] is True and noise['training_or_CUDA_started'] is False and noise['endpoint_iteration'] == 32000, 'Real long canonical noise receipt gate failed')
    script = ROOT / 'component_diagnostic/scripts/quantify_long_original_repeat_cpu.py'
    require(noise['analysis_script']['path'] == str(script) and noise['analysis_script']['sha256'] == sha(script), 'Actual noise analysis source differs')
    pinned = {row['path']: row['sha256'] for row in noise['input_pins']}
    for output in (b1['output'], b2['output']):
        for name in ('receipts/full49_summary.json', 'status.json', 'continuation_diagnostic.json', 'step_losses.jsonl', 'extra_eval/32000/diagnostic_eval.json', 'terminal_32000/diagnostic_snapshot_manifest.json'):
            require(pinned.get(str(output / name)) == sha(output / name), 'Noise input not actual long evidence: ' + str(output / name))
    require(pinned.get(str(PLAN / 'evaluation_protocol_locked.json')) == PINNED[LONG / 'evaluation_protocol_locked.json'], 'Noise protocol pin differs')
    for row in noise['input_pins']:
        require(Path(row['path']).stat().st_size == row['bytes'] and sha(row['path']) == row['sha256'], 'Canonical noise input pin differs')
    # Reuse the exact pinned compare function for checks. Never write another
    # noise analysis or introduce a different aggregation/threshold/empty rule.
    spec = importlib.util.spec_from_file_location('canonical_long_noise_CPU', script)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    primary = [(d['domain'], d['array'], k) for d in read(LONG / 'evaluation_protocol_locked.json')['primary_metrics'] for k in d['metrics']]
    require([(r['region'], r['array'], r['metric']) for r in noise['primary_metrics']] == primary and len(primary) == 11, 'Noise must cover exactly locked11 primary metrics in order')
    for row in noise['primary_metrics']:
        actual = helper.compare_metric(b1['final']['frames'], b2['final']['frames'], row['region'], row['array'], row['metric'], primary=True)
        require(row == actual, 'Canonical primary noise differs from saved actual49 metrics')
    context = noise['context_metrics_all_six_domains']
    require(len(context) == 126 and len({(r['region'], r['array'], r['metric']) for r in context}) == 126 and {r['region'] for r in context} == set(helper.REGIONS), 'Canonical noise must retain all6 domains/126 unique context metrics')
    for row in context:
        require(row == helper.compare_metric(b1['final']['frames'], b2['final']['frames'], row['region'], row['array'], row['metric']), 'Canonical context noise differs from actual49 metrics')
    input_path = b2['output'] / 'receipts/common49_inputs_after_B2_cpu.json'
    if not input_path.exists():
        raise Incomplete('Actual post-B2 common49 input/mask receipt is pending')
    inputs = read(input_path)
    original_common_path = ROOT / 'sgcm_data_audit/outside_opacity_review/common49_input_manifest_review_cpu.json'
    original_common = read(original_common_path)
    require(inputs['status'] == 'PASS_common49_original_input_SHA_after_B2' and inputs['CPU_only'] is True and inputs['GPU_or_training_started'] is False and inputs['actual467_file_SHA_matches_original'] is True and inputs['actual245_required_common49_SHA_match'] is True and inputs['new_shadow_mask_absent'] is True and inputs['same_original_source_path_and_repository'] is True, 'Actual post-B2 input/mask receipt failed')
    require(inputs['actual_input_SHA'] == original_common['file_sha256'] and len(inputs['actual_input_SHA']) == 467 and inputs['common_independent_receipt_sha256'] == sha(original_common_path) and inputs['input_manifest_sha256'] == original_common['input_manifest_sha256'], 'Actual post-B2 full467 input provenance differs')
    masks = inputs['common49_original_decoded_mask_receipts']
    require(len(inputs['common49_required_relative_paths']) == 245 and len(masks) == 98 and {(r['frame_id'], r['kind']) for r in masks} == {(f, k) for f in FRAMES for k in ('sky', 'actor')}, 'Actual98 original mask receipts incomplete')
    for row in masks:
        require(row['encoded_sha256'] == inputs['actual_input_SHA'][row['source_relative_path']] and len(row['decoded_original_L_uint8_sha256']) == 64, 'Actual original mask source receipt differs')
    return path


def materialize(b1, b2, refs, first, domains, noise_path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Explicit empty CUDA visibility and no bytecode required for CPU materialization')
    require(not OUTPUT.exists(), 'V output already exists; never prepare over an attempted run')
    guard_spec = importlib.util.spec_from_file_location('frozen_long_outside_guard', PLAN / 'outside_experiment_guard.py')
    guard = importlib.util.module_from_spec(guard_spec); guard_spec.loader.exec_module(guard)
    original = guard.source_inventory(ORIGINAL)
    require(original == read(HERE / 'approved_original_source_sha256.json'), 'Actual148 original files differ from independently reviewed frozen source contract')
    config = read(PLAN / 'outside_effective_config.yaml')
    baseline = copy.deepcopy(b1['plan']['training_config']); baseline['model_path'] = str(OUTPUT)
    require(guard.same_value({k: v for k, v in config.items() if k not in guard.EXTRA_CONFIG}, baseline) and guard.same_value({k: config[k] for k in guard.EXTRA_CONFIG}, guard.EXTRA_CONFIG), 'Exact original config plus two fields/.05 required')
    expected_inventory = dict(original)
    for relative in guard.ALGORITHM_FILES:
        expected_inventory[relative] = sha(PLAN / 'algorithm_overrides' / relative)
    if not REPLICA.exists():
        staging = REPLICA.with_name(REPLICA.name + '.cpu_staging')
        require(not staging.exists(), 'Partial replica staging exists; inspect before recovery')
        staging.mkdir()
        for relative in original:
            target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True)
            source = PLAN / 'algorithm_overrides' / relative if relative in guard.ALGORITHM_FILES else ORIGINAL / relative
            shutil.copyfile(source, target)
        require(guard.source_inventory(staging) == expected_inventory, 'Copied V source inventory failed')
        staging.rename(REPLICA)
    require(guard.source_inventory(REPLICA) == expected_inventory and {k for k in original if original[k] != expected_inventory[k]} == guard.ALGORITHM_FILES, 'Only the reviewed two-file algorithm delta allowed')
    patch = ''.join(line for relative in sorted(guard.ALGORITHM_FILES) for line in difflib.unified_diff((ORIGINAL / relative).read_text().splitlines(keepends=True), (REPLICA / relative).read_text().splitlines(keepends=True), fromfile='a/' + relative, tofile='b/' + relative)).encode()
    patch_path = HERE / 'actual_algorithm_diff.patch'; write_once(patch_path, patch)
    pair_report = {'status': 'actual_completed_B1_B2_long_pair_validated', 'steps': 2000, 'all2000_sampling_time_shift_LR_exact': True, 'actual_first10_original_scalars_exact_between_baselines': first, 'all49_domains': domains, 'B1_file_sha256': b1['pins'], 'B2_file_sha256': b2['pins'], 'actual_checkpoint_schema_validation': 'Completed unchanged harness CPU-loaded all four; completed CPU receipt exporter and this generator rehash all four. This generator does not restore CUDA/model state.', 'V_actual_restore': 'Not performed; original runtime restore observer remains mandatory.'}
    write_once(HERE / 'pair_reference_validation.json', payload(pair_report))
    from omegaconf import OmegaConf
    parent = Path(b1['plan']['parent'])
    parent_config = OmegaConf.to_container(OmegaConf.load(parent / 'effective_config.yaml'), resolve=True)
    aux = {str(path): sha(path) for path in PINNED}
    aux.update(b1['pins']); aux.update(b2['pins'])
    for path in (Path(__file__).resolve(), patch_path, HERE / 'pair_reference_validation.json', noise_path, b2['output'] / 'receipts/common49_inputs_after_B2_cpu.json'):
        aux[str(path)] = sha(path)
    binding = {'schema_version': 1, 'status': 'actual_long_B1_B2_bound_CPU_preparation_pending_independent_review', 'training_started': False, 'allow_training': False, 'experiment_kind': 'hypothesis_complement_background_domain', 'scientific_status': 'Explicit complementary background-domain hypothesis .05; not certified author Eq13, not strict original config, only recovery from degenerated30k. Eq14 prediction masking/shared environment, static temporal semantics and nearest-depth remain unchanged.', 'parent_iteration': 30000, 'first_iteration': 30001, 'stop_iteration': 32000, 'steps': 2000, 'horizon': 50000, 'seed': 0, 'output': str(OUTPUT), 'repository': str(REPLICA), 'original_repository': str(ORIGINAL), 'parent_static_checkpoint': str(parent / 'chkpnt30000.pth'), 'parent_training_config': parent_config, 'training_config': config, 'added_training_fields': guard.EXTRA_CONFIG, 'training_config_changed_original_fields': ['model_path', 'start_checkpoint'], 'training_config_added_fields': sorted(guard.EXTRA_CONFIG), 'strict_original_config': False, 'algorithm_changed_files': sorted(guard.ALGORITHM_FILES), 'algorithm_patch': str(patch_path), 'algorithm_patch_sha256': sha(patch_path), 'original_source_sha256': original, 'replica_source_sha256': expected_inventory, 'protected_original_file_sha256': b1['plan']['original_file_sha256'], 'baseline_binding': {'plan_path': str(PLAN / 'prepared_plan.json'), 'plan_sha256': sha(PLAN / 'prepared_plan.json'), 'harness_path': str(PLAN / 'continuation_harness.py'), 'harness_sha256': sha(PLAN / 'continuation_harness.py'), 'observer_path': str(COMPONENT / 'runtime_safety_parameterized.py'), 'observer_sha256': sha(COMPONENT / 'runtime_safety_parameterized.py'), 'output': str(b1['output'])}, 'guard_path': str(PLAN / 'outside_experiment_guard.py'), 'guard_sha256': sha(PLAN / 'outside_experiment_guard.py'), 'auxiliary_file_sha256': aux, 'paired_protocol': {'path': str(LONG / 'evaluation_protocol_locked.json'), 'sha256': sha(LONG / 'evaluation_protocol_locked.json'), 'preregistered_before_long_results': True}, 'runtime_helper_import_contract': {'component_diagnostic': {'path': str(ROOT / 'component_diagnostic/scripts/component_diagnostic.py'), 'sha256': sha(ROOT / 'component_diagnostic/scripts/component_diagnostic.py'), 'functions': ['stats', 'rgba_package', 'region_stats', 'rng_capture', 'rng_restore', 'rng_pins']}, 'runtime_safety_parameterized': {'path': str(COMPONENT / 'runtime_safety_parameterized.py'), 'sha256': sha(COMPONENT / 'runtime_safety_parameterized.py')}}, 'evaluation': {'nodes': NODES, 'early_fixed_camera_ids': b1['plan']['early_fixed_camera_ids'], 'terminal_eval_camera_count': 49, 'all_extra_eval_rng_preserved': True}, 'restore_policy': b1['plan']['restore_policy'], 'restore_verification_status': 'CPU only; V model/optimizer/RNG not actually restored. Original runtime observer is still required.', 'baseline_completion': {'status': 'complete', 'B1_actual_exit_code': 0, 'B2_actual_exit_code': 0, 'both_terminal49_and_four_validated_checkpoints': True, 'pair_validation_path': str(HERE / 'pair_reference_validation.json'), 'pair_validation_sha256': sha(HERE / 'pair_reference_validation.json')}, 'first_step_pair_reference': {'iteration': 30001, 'values': first, 'original_objective_log_key': 'loss_original_objective_before_diagnostic', 'comparison_phase': 'Before real first Tensor.backward, compare captured original total without subtracting added loss.', 'source_path': str(b1['output'] / 'step_losses.jsonl'), 'source_sha256': sha(b1['output'] / 'step_losses.jsonl'), 'policy': 'All10 exact against actual B1 and verified B2 before first backward; later losses may differ.'}, 'all_iteration_pair_reference': {'rows': refs, 'source_path': str(b1['output'] / 'step_losses.jsonl'), 'source_sha256': sha(b1['output'] / 'step_losses.jsonl'), 'policy': 'All2000 actual sample/time_shift/LR exact against B1; B2 actual2000 was independently compared and bound.'}, 'original_repeat_noise_reference': {'path': str(noise_path), 'sha256': sha(noise_path), 'B2_output': str(b2['output']), 'B2_steps_path': str(b2['output'] / 'step_losses.jsonl'), 'B2_steps_sha256': sha(b2['output'] / 'step_losses.jsonl'), 'B2_full49_summary_path': str(b2['output'] / 'receipts/full49_summary.json'), 'B2_full49_summary_sha256': sha(b2['output'] / 'receipts/full49_summary.json'), 'is_confidence_interval_or_CUDA_noise_bound': False}, 'extra_eval_rng_policy': b1['plan']['extra_eval_rng_policy'], 'harness_integration_status': 'Reuse unchanged long93df harness /835620 isolated guard. No launch authorization.', 'launch_gates': ['Actual completed B1 and B2 with full2000, full49, original source/config and four-state evidence', 'Independent final V source/binding/CPUplan review', 'Root exact V receipt and exclusive GPU ownership handoff'], 'unchanged_algorithms': ['foreground Eq13', 'prediction-masked Eq14', 'original compose/weights/backward/Adam', 'static temporal fields', 'depth and nearest-surface policy', 'SGCM/reset', 'original sampler/RNG/LR50k']}
    checks = guard.OutsideExperimentGuard(binding)
    checks.verify_sources(); checks.verify_auxiliary_files(); checks.verify_protected_files()
    tree = ast.parse((ORIGINAL / 'utils/paper_checkpoint.py').read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_config_dict', '_validate_resume_configuration', '_same_config_value')]
    namespace = {}; exec(compile(ast.Module(body=funcs, type_ignores=[]), 'unchanged_formal_validator', 'exec'), namespace)
    checks.validate_config(parent_config, config, namespace['_validate_resume_configuration'])
    binding_path = PLAN / 'experiment_binding.json'; write_once(binding_path, payload(binding))
    plan_path = PLAN / 'opacity_outside_prepared_plan.json'
    existing_plan = plan_path.exists()
    if existing_plan:
        spec = importlib.util.spec_from_file_location('unchanged_long_V_harness_CPU', PLAN / 'continuation_variant_harness.py')
        harness = importlib.util.module_from_spec(spec); spec.loader.exec_module(harness)
        current, _, _ = harness.prepare(OUTPUT, binding_path)
        require(read(plan_path) == current, 'Existing V CPUplan conflicts with actual unchanged prepare; never overwrite')
    command = [sys.executable, str(PLAN / 'continuation_variant_harness.py'), '--experiment-binding', str(binding_path)]
    if not existing_plan:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
        write_once(HERE / 'cpu_prepare.log', result.stdout.encode())
        require(result.returncode == 0 and plan_path.exists(), 'Original unchanged V CPU prepare failed; inspect cpu_prepare.log, no launch')
    prepared = read(plan_path)
    require(prepared['experiment_binding_sha256'] == sha(binding_path) and prepared['allow_training'] is False and prepared['all467_original_inputs_hash_verified'] is True, 'Actual V CPUplan binding/input gate failed')
    receipt = {'status': 'actual_long_V_CPUplan_prepared_pending_independent_review', 'binding_sha256': sha(binding_path), 'prepared_plan_sha256': sha(plan_path), 'harness_sha256': sha(PLAN / 'continuation_variant_harness.py'), 'guard_sha256': sha(PLAN / 'outside_experiment_guard.py'), 'generator_sha256': sha(__file__), 'actual_CPU_prepare_exit_code': 0, 'training_started': False, 'CUDA_initialized': False, 'allow_training': False, 'actual_GPU_restore': False, 'command': command, 'pair_reference_validation_sha256': sha(HERE / 'pair_reference_validation.json'), 'canonical_noise_reference_sha256': sha(noise_path)}
    write_once(HERE / 'cpu_prepare_receipt.json', payload(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialize-and-cpu-prepare', action='store_true')
    args = parser.parse_args()
    require(Path(__file__).resolve().parent == HERE, 'Run only from the canonical independent remote CPU directory')
    for path, expected in PINNED.items():
        require(sha(path) == expected, 'Approved source/config/protocol pin changed: ' + str(path))
    # Inspect B2 readiness first: while it trains no large B1/B2 artifact rehashes.
    b2_output = ROOT / 'continuation_diagnostic/original30k_to32000_noise_reference'
    require(not (b2_output / 'status.json').exists() or read(b2_output / 'status.json')['status'] != 'failed', 'B2 failed; no V binding can be generated')
    if not (b2_output / 'receipts/gpu_release_handoff.json').exists():
        result = {'status': 'WAIT_REAL_B2_COMPLETION_AND_EXPORT', 'training_started': False, 'V_binding_created_by_this_call': False, 'V_replica_created_by_this_call': False, 'V_CPUplan_created_by_this_call': False, 'actual_B2_status': read(b2_output / 'status.json')['status'] if (b2_output / 'status.json').exists() else 'not_started', 'policy': 'No placeholders or short30300 references; real complete B1/B2 long outputs are required.'}
    else:
        try:
            b1, b2 = completed('B1'), completed('B2')
            refs, first, domains = paired(b1, b2)
            noise_path = noise_reference(b1, b2)
            result = materialize(b1, b2, refs, first, domains, noise_path) if args.materialize_and_cpu_prepare else {'status': 'REAL_B1_B2_READY_CPU_CHECK_ONLY', 'B1_sha256': b1['pins'], 'B2_sha256': b2['pins'], 'canonical_noise_reference_sha256': sha(noise_path), 'steps': len(refs), 'all49_domain_count': len(domains), 'training_started': False, 'V_binding_created_by_this_call': False, 'V_replica_created_by_this_call': False, 'V_CPUplan_created_by_this_call': False}
        except Incomplete as error:
            result = {'status': 'WAIT_REAL_B1_B2_COMPLETION_AND_EXPORT', 'reason': str(error), 'training_started': False, 'V_binding_created_by_this_call': False, 'V_replica_created_by_this_call': False, 'V_CPUplan_created_by_this_call': False}
    # Only this generator's own mutable status is overwritten. Bound artifacts
    # and every approved B1/B2/current source/config/plan are never overwritten.
    (HERE / 'gate_status.json').write_bytes(payload(result))
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, allow_nan=False))


if __name__ == '__main__':
    main()
