"""CPU-only source/evaluation-code receipt after real variant preparation."""
import ast
import hashlib
import json
from pathlib import Path

PLAN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/component_diagnostic/continuation_plan')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def routine_source(path, name):
    source = Path(path).read_text()
    tree = ast.parse(source)
    selected = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(selected) != 1:
        raise RuntimeError('Ambiguous source routine: ' + name)
    return ast.get_source_segment(source, selected[0])


def main():
    preflight = json.loads((PLAN / 'opacity_outside_cpu_preflight.log').read_text())
    prepared = json.loads((PLAN / 'opacity_outside_prepared_plan.json').read_text())
    if preflight['training_started'] or preflight['CUDA_initialized'] or preflight['allow_training']:
        raise RuntimeError('Variant CPU preparation entered training/GPU stage')
    if preflight['prepared_plan_sha256'] != sha(PLAN / 'opacity_outside_prepared_plan.json') or preflight['harness_sha256'] != sha(PLAN / 'continuation_variant_harness.py'):
        raise RuntimeError('Actual CPU-prepared code/plan changed')
    source_original = routine_source(PLAN / 'continuation_harness.py', 'diagnostic_eval')
    source_variant = routine_source(PLAN / 'continuation_variant_harness.py', 'diagnostic_eval')
    if source_original != source_variant:
        raise RuntimeError('Variant changed actual original eval/camera/mask/statistics routine')
    ast.parse((PLAN / 'launch_opacity_outside_continuation.py').read_text())
    variant_tree = ast.parse((PLAN / 'continuation_variant_harness.py').read_text())
    dependency_loops = [node for node in ast.walk(variant_tree) if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and [item.id for item in node.target.elts if isinstance(item, ast.Name)] == ['module', 'expected']]
    if len(dependency_loops) != 1:
        raise RuntimeError('Ambiguous actual runtime dependency check loop')
    dependency_pins = {}
    for entry in dependency_loops[0].iter.elts:
        relative = entry.elts[1].right.value
        path = str(Path(prepared['repository']) / relative)
        if path not in prepared['runtime_source_sha256'] or not Path(path).is_file() or sha(path) != prepared['runtime_source_sha256'][path]:
            raise RuntimeError('Runtime dependency absent from actual148 inventory: ' + relative)
        dependency_pins[relative] = prepared['runtime_source_sha256'][path]
    train_tree = ast.parse((Path(prepared['repository']) / 'train.py').read_text())
    if not any(isinstance(node, ast.ImportFrom) and node.module == 'utils.loss_utils' and {'psnr', 'ssim'}.issubset({item.name for item in node.names}) for node in train_tree.body):
        raise RuntimeError('Actual frozen training imports PSNR/SSIM differently from checked loss_utils')
    if sha(PLAN / 'runtime_safety_parameterized.py') != 'f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1' or sha(PLAN / 'continuation_harness.py') != 'ac9745d5c0cbab7701ae8be5beec9587d57ea79288a3167259d0af1d0e164458':
        raise RuntimeError('Approved baseline harness/restore observer bytes changed')
    receipt = {'status': 'PASS', 'scope': 'Actual variant default CPU preparation plus AST/same-source verification; no training or CUDA', 'training_started': False, 'CUDA_initialized': False, 'harness_sha256': sha(PLAN / 'continuation_variant_harness.py'), 'prepared_plan_sha256': sha(PLAN / 'opacity_outside_prepared_plan.json'), 'launcher_sha256': sha(PLAN / 'launch_opacity_outside_continuation.py'), 'experiment_binding_sha256': prepared['experiment_binding_sha256'], 'guard_sha256': prepared['guard_sha256'], 'approved_baseline_harness_bytes_unchanged': True, 'approved_restore_observer_bytes_unchanged': True, 'diagnostic_eval_entire_function_source_exactly_equal': True, 'diagnostic_eval_function_source_sha256': hashlib.sha256(source_original.encode()).hexdigest(), 'unchanged_metric_camera_helper_sha256': prepared['runtime_helper_import_contract']['component_diagnostic']['sha256'], 'actual_source_config_guard_checks': prepared['preparation_checks'], 'all467_original_inputs_hash_verified': prepared['all467_original_inputs_hash_verified'], 'all300_reference_records': len(prepared['all_iteration_pair_reference']['rows']), 'first10_reference_scalars': len(prepared['first_step_reference']['values']), 'actual_variant_GPU_restore_not_yet_executed': True, 'GPU_launch_authorized_by_this_receipt': False}
    receipt['actual_runtime_dependency_paths_148_inventory_verified'] = dependency_pins
    receipt['actual_train_imports_PSNR_and_SSIM_from_loss_utils_AST_verified'] = True
    destination = PLAN / 'opacity_outside_cpu_preflight_receipt.json'
    destination.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
