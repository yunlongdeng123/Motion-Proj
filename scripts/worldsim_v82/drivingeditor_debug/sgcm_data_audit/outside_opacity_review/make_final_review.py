"""Bind the completed independent source/execution-path review to final bytes."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent.parent
CONTROL = TASK / 'component_diagnostic/continuation_plan'
REFS = HERE / 'refs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def function_source(path, name):
    text = Path(path).read_text(encoding='utf8')
    return next(ast.get_source_segment(text, node) for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.FunctionDef) and node.name == name)


plan_path = REFS / 'opacity_outside_prepared_plan.json'
plan = read(plan_path)
binding_path = TASK / 'loss_audit/controlled_opacity_outside/experiment_binding.json'
binding = read(binding_path)
preflight_path = REFS / 'opacity_outside_cpu_preflight_receipt.json'
preflight = read(preflight_path)
candidate = read(HERE / 'candidate_independent_review_cpu.json')
guard_review = read(HERE / 'guard_bound_independent_review_cpu.json')
common = read(HERE / 'common49_input_manifest_review_cpu.json')
baseline = read(CONTROL / 'prepared_plan.json')
harness_path = CONTROL / 'continuation_variant_harness.py'
launcher_path = CONTROL / 'launch_opacity_outside_continuation.py'
expected = {'harness_sha256': '93bb8df2cca142b3ea0b08e42ab8c3fe5defa214641537b7f3935eb300444359', 'prepared_plan_sha256': 'ef4b873994b8a8d21fcbc96efbce1c6d764249f5b8fdf43be19c60bedbc7153e', 'launcher_sha256': '426f99b3b1a9c75ffcf8e225933065c5fa1ca898dfa566aa7b5e42b6715a94dc', 'experiment_binding_sha256': 'b96ec92c0694a251e39da85a3ae4abab131366a75ad52cd47098ec0df953b214', 'guard_sha256': 'f3a86b6fb128b83c6a189b95787ed001da2ac84a4444ecc68bba283fae5477f9'}
actual = {'harness_sha256': sha(harness_path), 'prepared_plan_sha256': sha(plan_path), 'launcher_sha256': sha(launcher_path), 'experiment_binding_sha256': sha(binding_path), 'guard_sha256': sha(TASK / 'loss_audit/outside_experiment_guard.py')}
assert actual == expected
assert all(preflight[key] == value for key, value in expected.items())
assert preflight['status'] == 'PASS' and not preflight['training_started'] and not preflight['CUDA_initialized']
assert all(plan[key] == expected[key] for key in ('harness_sha256', 'launcher_sha256', 'experiment_binding_sha256', 'guard_sha256'))
assert plan['training_config'] == binding['training_config']
assert plan['repository'] == binding['repository'] and plan['output'] == binding['output']
assert plan['parent_iteration'] == 30000 and plan['first_iteration'] == 30001 and plan['stop_iteration'] == 30300 and plan['steps'] == 300 and plan['horizon'] == 50000 and plan['position_lr_max_steps'] == 50000 and plan['seed'] == 0
assert candidate['candidate_only_status'] == 'PASS' and all(row['passed'] for row in candidate['checks'])
assert candidate['pins']['loss_audit/candidates/opacity_outside/train.py'] == binding['replica_source_sha256']['train.py'] == 'ff82c0d86b385bcdf28feff550163cd3ec9c7bec92e939306330ba4af335e99a'
assert candidate['pins']['loss_audit/candidates/opacity_outside/utils/paper_losses.py'] == binding['replica_source_sha256']['utils/paper_losses.py'] == '3832a7ced1387e1ff33f5e3aee0b56bc533303f80f5be840269236139fc866ab'
assert function_source(CONTROL / 'continuation_harness.py', 'diagnostic_eval') == function_source(harness_path, 'diagnostic_eval')
assert sha(CONTROL / 'continuation_harness.py') == 'ac9745d5c0cbab7701ae8be5beec9587d57ea79288a3167259d0af1d0e164458'
assert sha(CONTROL / 'runtime_safety_parameterized.py') == plan['observer_sha256'] == 'f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1'
assert guard_review['all_passed'] and guard_review['guard_sha256'] == expected['guard_sha256']
assert guard_review['formal_validator_sha256'] == binding['original_source_sha256']['utils/paper_checkpoint.py'] == binding['replica_source_sha256']['utils/paper_checkpoint.py']
assert sha(HERE / 'common49_input_manifest_review_cpu.json') == '1b7161a2b30a08fb7fd8354a71b257cf99ad39bef16732a5f24a367174c24d50'
assert common['all_original_input_sha_match'] and common['full_input_file_count'] == 467
assert plan['original_file_sha256'][common['input_manifest']] == common['input_manifest_sha256']
assert len(plan['all_iteration_pair_reference']['rows']) == 300
assert plan['all_iteration_pair_reference'] == binding['all_iteration_pair_reference']
assert [row['iteration'] for row in plan['all_iteration_pair_reference']['rows']] == list(range(30001, 30301))
assert plan['first_step_reference']['values'] == binding['first_step_pair_reference']['values'] == baseline['first_step_reference']['values'] and len(plan['first_step_reference']['values']) == 10
assert plan['extra_eval_nodes'] == baseline['extra_eval_nodes'] and plan['early_fixed_camera_ids'] == baseline['early_fixed_camera_ids'] and plan['terminal_eval_camera_count'] == 49
assert {path: plan['original_file_sha256'][path] for path in binding['protected_original_file_sha256']} == baseline['original_file_sha256']
assert len(plan['runtime_source_sha256']) == 148
assert {str(Path(binding['repository']) / relative).replace('\\', '/'): value for relative, value in binding['replica_source_sha256'].items()} == plan['runtime_source_sha256']
remote_base = '/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012'
report = {'status': 'PASS', 'decision': 'CPU preparation can run as the exact controlled single-variable experiment with root coordination receipt and exclusive GPU ownership.', **expected, 'review_scope': 'Independent final source/config/actual execution-path and CPU-preflight review; this reviewer did not launch GPU or train.', 'harness_path': remote_base + '/component_diagnostic/continuation_plan/continuation_variant_harness.py', 'prepared_plan_path': remote_base + '/component_diagnostic/continuation_plan/opacity_outside_prepared_plan.json', 'experiment_binding_path': remote_base + '/loss_audit/controlled_opacity_outside/experiment_binding.json', 'remote_actual_SHA_verified': True, 'blocking_findings': [], 'allowed_algorithm_delta': {'formula': '-mean((1-Md)*alpha_d*log1p(-alpha_d/(1+1e-4)))', 'coefficient': 0.05, 'changed_source_files': ['train.py', 'utils/paper_losses.py'], 'additional_train_delta': 'Pure pre-add original objective loss.item() observation only.', 'added_config': binding['added_training_fields'], 'bookkeeping_config_changes': ['model_path', 'start_checkpoint']}, 'reviewed_execution': {'actual_replica_repository_used': plan['repository'], 'actual_training_loss_mask_and_metric_module_import_paths_SHA_checked': True, 'actual_SSIM_and_PSNR_function_binding_checked': True, 'exact_same_original30k_four_files': True, 'unchanged_full_restore_observer': True, 'strict_original_config_validator_bytes_preserved': True, 'isolated_adapter_only_two_explicit_config_fields': True, 'all_original_loss_functions_and_train_math_preserved': True, 'first30001_original10_scalars_exact_BEFORE_backward': 'Hard runtime gate; original captured loss, no subtractive recovery or tolerance.', 'all300_camera_time_shift_LR_exact': 'Hard per-step runtime gate against actual completed original baseline.', 'extra_eval_rng_restoration': 'Unchanged entire original diagnostic_eval function; capture, finally restore, hash verify.', 'all49_original_GT_dimensions_region_counts_pairing': 'Hard per-node runtime gate; terminal all49.', 'terminal_state': '30300 after full backward/Adam and RNG-safe49eval; static/dynamic/env/complete sidecar hashed and CPU-load checked; no formal event.', 'full467_original_inputs_SHA_checked': True, 'new_shadow_masks_rejected': True, 'all_source_original_and_replica_inventories_verified': True, 'original_and_isolated_validator_hooks_restored_in_finally': True, 'Tensor_backward_hook_restored_in_finally': True, 'iterations_and_position_lr_max_steps': 50000}, 'evidence_sha256': {'candidate_independent_review_cpu.json': sha(HERE / 'candidate_independent_review_cpu.json'), 'guard_bound_independent_review_cpu.json': sha(HERE / 'guard_bound_independent_review_cpu.json'), 'common49_input_manifest_review_cpu.json': sha(HERE / 'common49_input_manifest_review_cpu.json'), 'opacity_outside_cpu_preflight_receipt.json': sha(preflight_path), 'opacity_outside_cpu_preflight.log': sha(REFS / 'opacity_outside_cpu_preflight.log'), 'opacity_outside_harness_delta.patch': sha(REFS / 'opacity_outside_harness_delta.patch')}, 'actual_variant_GPU_restore_executed_by_this_reviewer': False, 'effectiveness_or_final_paper_formula_certified': False, 'specific_limitation': 'No independent all49 actualScene pose receipt was stored for baseline; same pinned loader/camera/normalization, all467 real inputs and exact49GT/mask-domain pairing are used and this limitation remains explicit.'}
(HERE / 'final_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
text = '''独立最终审查 PASS：可按该精确计划运行隔离单变量300步实验。
实验仅增加掩膜外dynamic opacity项，λ=.05；保留原paper loss、depth、SGCM、reset、50k horizon。
真实执行路径、配置及CPU preflight已核对；正式源/guard未弱化。无需再修复。

同一30k四文件 + 467原输入
              |
              v
原完整restore + 隔离两字段guard
              |
       +------+------+
       v             v
 原paper loss  原paper loss + .05 Loutside
       |             |
       +------+------+
              v
原300采样轨迹 + RNG-safe49相机/掩膜评估
              |
              v
区域指标 + 30300四文件完整state

硬检查：30001 backward前原10标量精确相等；每一步camera/time_shift/LR一致；
49帧GT浮点数组、尺寸、掩膜区域像素数一致；相同eval函数源码；全部输入和源SHA一致；
30300完成真实Adam后保存static/dynamic/env/完整sidecar，核SHA与CPU可加载schema。

本PASS是CPU准备与执行协议可运行结论，未执行variant GPU，也不是效果或作者公式认证。
baseline未独立记录每帧实际Scene pose；同原loader/cameras/normalization+全部真实输入
及49GT/区域配对作为明确替代证据。原组件/RNG/optimizer的实际restore仍由GPU硬守卫验证。
'''
text += '\n' + '\n'.join(f'{key}: {value}' for key, value in expected.items()) + '\n'
(HERE / 'final_review.txt').write_text(text, encoding='utf8')
print(json.dumps({'status': report['status'], 'final_review_sha256': sha(HERE / 'final_review.json'), **expected}, indent=2))
