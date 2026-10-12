"""Build one small nearest-depth CPU plan draft using existing tested source.

No training import, launch API, replica materialization or checkpoint copy.
All current V, baseline, formal source and data remain read-only.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
WORK = AUDIT.parent
BASE = PurePosixPath('/root/autodl-tmp/drivingeditor_repro')
REMOTE = BASE / 'campaign/vgpu_bugfix_20261012'
ORIGINAL = BASE / 'campaign/resume_vgpu_baseline_20261011/frozen_repository'
LONG = AUDIT / 'long_pair_30k_32000'
REMOTE_LONG = REMOTE / 'loss_audit/long_pair_30k_32000'
CANDIDATE = AUDIT / 'candidates/nearest_depth'
REMOTE_CANDIDATE = REMOTE / 'loss_audit/candidates/nearest_depth'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    checks = []
    def check(name, result):
        if not result:
            raise AssertionError(name)
        checks.append({'check': name, 'pass': True})
    base_path = AUDIT / 'source_frozen/utils/camera_utils.py'
    source_path = CANDIDATE / 'utils/camera_utils.py'
    original = ast.parse(base_path.read_text(encoding='utf-8'))
    candidate = ast.parse(source_path.read_text(encoding='utf-8'))
    removed_imports = 0
    restored_assignments = 0
    class Restore(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            nonlocal removed_imports
            if node.module == 'utils.sparse_depth':
                check('only_expected_nearest_helper_import', [n.name for n in node.names] == ['nearest_positive_depth'])
                removed_imports += 1
                return None
            return node
        def visit_Assign(self, node):
            nonlocal restored_assignments
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'nearest_positive_depth':
                check('only_expected_depth_target_reduction', ast.unparse(node) == 'pts_depth = nearest_positive_depth(uv, uvz[:, 2], h, w)')
                restored_assignments += 1
                return ast.parse('pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]').body[0]
            return node
    restored = Restore().visit(copy.deepcopy(candidate))
    check('exact_one_import_and_one_depth_assignment_only', removed_imports == restored_assignments == 1)
    check('candidate_AST_restores_complete_original_camera_utils', ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False))
    approved_source = read(LONG / 'v_binding_cpu/approved_original_source_sha256.json')
    check('base_camera_actual_pin_matches_approved_original', sha(base_path) == approved_source['utils/camera_utils.py'])
    helper_path = CANDIDATE / 'utils/sparse_depth.py'
    helper = ast.parse(helper_path.read_text())
    check('helper_is_numpy_only', [(n.names[0].name) for n in helper.body if isinstance(n, ast.Import)] == ['numpy'] and not any(isinstance(n, ast.ImportFrom) for n in helper.body))
    old_tests_path = AUDIT / 'candidates/candidates_tests_remote_cpu.json'
    old_tests = read(old_tests_path)
    depth_tests = [row for row in old_tests['tests'] if row['test'] in {'collision_nearest', 'all24_input_permutations_exact_equal', 'noncollision_identical_to_original', 'zero_nan_inf_behind_camera_and_invalid_uv_ignored', 'empty_returns_allzero', 'exact_candidate_loadCam_collision_integration', 'exact_candidate_loadCam_zero_nan_behind_integration'}]
    check('reuse_all7_existing_remote_depth_tests_passed', len(depth_tests) == 7 and all(row['passed'] for row in depth_tests))
    b1_plan = read(LONG / 'materialized_plan/prepared_plan.json')
    b1_output_local = WORK / 'continuation_diagnostic/original30k_to32000'
    b1_rows_path = b1_output_local / 'step_losses.jsonl'
    rows = [json.loads(line) for line in b1_rows_path.read_text().splitlines() if line]
    check('actual_B1_reference_has_full2000_steps', [r['iteration'] for r in rows] == list(range(30001, 32001)))
    original10 = b1_plan['first_step_reference']['values']
    unchanged = sorted(set(original10) - {'loss', 'loss_paper_depth'})
    check('first8_gate_excludes_exactly_intended_depth_and_total', len(unchanged) == 8 and set(unchanged) | {'loss', 'loss_paper_depth'} == set(original10))
    config = copy.deepcopy(b1_plan['training_config'])
    output = REMOTE / 'continuation_diagnostic/nearest_depth_only30k_to32000'
    config['model_path'] = str(output)
    check('no_outside_flags_or_extra_training_config', not any(k.startswith('diagnostic_') for k in config))
    check('same_horizon_coefficient_and_reset_free_window', config['iterations'] == config['position_lr_max_steps'] == 50000 and config['paper_lambda_depth'] == .1 and b1_plan['stop_iteration'] == 32000 < 33000)
    patch_path = AUDIT / 'candidates/nearest_depth.patch'
    references = {
        'existing_source_diff': {'path': str(REMOTE / 'loss_audit/candidates/nearest_depth.patch'), 'sha256': sha(patch_path)},
        'existing_candidate_source': {relative: {'path': str(REMOTE_CANDIDATE / relative), 'sha256': sha(CANDIDATE / relative)} for relative in ('utils/camera_utils.py', 'utils/sparse_depth.py')},
        'existing_remote_CPU_depth_tests': {'path': str(REMOTE / 'loss_audit/candidates/candidates_tests_remote_cpu.json'), 'sha256': sha(old_tests_path), 'tested_depth_cases': depth_tests},
        'existing_real_collision_evidence': {'path': str(REMOTE / 'loss_audit/sparse_depth_collisions_cpu.json'), 'sha256': sha(AUDIT / 'sparse_depth_collisions_cpu.json')},
        'original_frozen_source_inventory': {'path': str(REMOTE_LONG / 'v_binding_cpu/approved_original_source_sha256.json'), 'sha256': sha(LONG / 'v_binding_cpu/approved_original_source_sha256.json')},
        'B1_actual_steps': {'path': str(PurePosixPath(b1_plan['output']) / 'step_losses.jsonl'), 'sha256': sha(b1_rows_path)},
        'B1_actual_plan': {'path': str(REMOTE_LONG / 'materialized_plan/prepared_plan.json'), 'sha256': sha(LONG / 'materialized_plan/prepared_plan.json')},
        'B2_actual_noise': {'path': str(REMOTE / 'continuation_diagnostic/original30k_to32000_noise_reference/receipts/B1_B2_observed_noise.json'), 'sha256': sha(WORK / 'continuation_diagnostic/original30k_to32000_noise_reference/receipts/B1_B2_observed_noise.json')},
    }
    parent = PurePosixPath(b1_plan['parent'])
    parent4 = {path: value for path, value in b1_plan['original_file_sha256'].items() if PurePosixPath(path).parent == parent and path.endswith('30000.pth')}
    check('references_original30k_four_states_only_no_checkpoint_copy', len(parent4) == 4)
    check('remote_paths_are_absolute_POSIX_from_Windows_generation', str(output).startswith('/root/autodl-tmp/') and '\\' not in str(output) and all('\\' not in v['path'] for v in references.values() if isinstance(v, dict) and 'path' in v))
    plan = {'schema_version': 1, 'status': 'CPU_PLAN_DRAFT_ONLY_NOT_LAUNCHABLE', 'allow_training': False, 'GPU_launch_authorized': False, 'actual_CUDA_restore': False, 'materialized_source_or_config': False, 'checkpoint_copies_created': 0, 'current_V_source_config_plan_modified': False, 'experiment_kind': 'single_variable_nearest_positive_LiDAR_depth_target_only', 'scientific_status': 'Confirmed inherited order-dependent target bug; nearest finite positive z is an explicit visibility policy omitted by paper. Branch-collapse causality and rendered benefit are unproven.', 'original_repository': str(ORIGINAL), 'new_replica': None, 'new_harness': None, 'new_experiment_guard': None, 'future_output_unmaterialized': str(output), 'changed_existing_source_files': ['utils/camera_utils.py'], 'added_source_files': ['utils/sparse_depth.py'], 'training_or_renderer_or_loss_source_changed': False, 'outside_complement_enabled': False, 'training_config': config, 'training_config_extra_fields': [], 'configuration_bookkeeping_changes_only': ['model_path', 'start_checkpoint'], 'parent_iteration': 30000, 'first_iteration': 30001, 'stop_iteration': 32000, 'steps': 2000, 'horizon': 50000, 'parent_four_file_sha256': parent4, 'baseline_arms': ['actual_original_B1_30k_to32000', 'actual_original_B2_30k_to32000'], 'same_sampler_rng_optimizer_restore_as_baselines': True, 'same_all2000_camera_time_shift_LR_required': True, 'evaluation_nodes': [30001, 30300, 31000, 32000], 'fixed_early_camera_indices': [0, 10, 20, 30, 48], 'terminal_all49': True, 'all_extra_eval_RNG_preserved_required': True, 'no33000_reset': True, 'SGCM_stopped15000_not_exercised': True, 'first_backward_gate': {'actual_saved30k_full_restore_must_pass': True, 'unchanged8_scalar_values_exact_to_actual_B1': {k: rows[0]['losses'][k] for k in unchanged}, 'intended_changed_scalars': ['loss_paper_depth', 'loss'], 'policy': 'Do not reuse V all10 gate: corrected depth target can change depth and total at first forward. Require exact unchanged8 and exact camera/time_shift/LR. Log both old-target and nearest-target Eq16 on the same rendered depth; apply unchanged compose directly to common terms and each depth value rather than recovering a total by floating subtraction.'}, 'CPU_target_acceptance': ['All49 source/camera/input hashes stay original; only derived sparse-depth arrays differ', 'Nearest target finite positive at valid pixels, zero at missing pixels, exact permutation invariance', 'Noncollision pixels identical to original last-write map; changed collision count and per-frame values disclosed', 'Independent nearest reduction reference and collision/noncollision inverse-depth residuals evaluated on same support'], 'GPU_effect_acceptance_if_authorized_later': ['Original four parameters/optimizer moments/options/devices, RNG/sampler, schedule and original finite checks pass', 'All2000 actual sampling/time_shift/LR pair with both originals; full49 GT/source masks/region counts and RNG-safe nodes match', 'Report original locked11 metrics and actor/context evidence against both originals and observed repeat differences; do not select new success thresholds', 'Also score each model against both last-write and nearest depth targets on fixed collision/noncollision/full-HW domains; improved own objective alone is insufficient', 'Complete32000 exit0, all49 and isolated four-state snapshot with actual CPU load/schema/SHA verification'], 'paper_basis': {'Eq16_pages': [1702, 1703], 'inverse_definition_explicit': True, 'zbuffer_collision_policy_unpublished': True, 'harmonic900_alpha_normalization_validity_eps_lambda_unchanged': True}, 'references': references, 'launch_gates_missing': ['Root decides whether this isolated original-math nearest experiment is the next question', 'Create new minimal source/guard/harness profile; preserve current93df/835620 and V source/plan', 'Independent review exact source/config/guard/real prepared plan', 'Root exact receipt and exclusive GPU owner handoff'], 'interpretation_limit': 'This warm-start test addresses inherited target order dependence and recovery, not author implementation certification, from-zero success, SGCM repair or static time-invariant enforcement.'}
    (HERE / 'nearest_depth_only_cpu_plan_draft.json').write_text(json.dumps(plan, indent=2, allow_nan=False))
    receipt = {'status': 'PASS_CPU_DRAFT_SOURCE_INVARIANTS', 'checks': checks, 'check_count': len(checks), 'existing_depth_source_and_test_reused_not_reengineered': True, 'GPU_or_training_started': False, 'checkpoint_copies_created': 0, 'source_config_data_or_current_V_modified': False, 'draft_sha256': sha(HERE / 'nearest_depth_only_cpu_plan_draft.json'), 'source_diff_sha256': sha(patch_path), 'generator_sha256': sha(__file__)}
    (HERE / 'nearest_depth_draft_preflight_cpu.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in receipt.items() if k != 'checks'}))


if __name__ == '__main__':
    main()
