"""Small focused checks of the actual9-line S block and strict S config guard."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/drivingeditor_repro')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    require = lambda ok: ok or (_ for _ in ()).throw(AssertionError('CPU check failed'))
    binding_path = HERE / 'materialized_plan/experiment_binding.json'
    binding = json.loads(binding_path.read_text())
    repo = Path(binding['repository'])
    original = Path(binding['original_repository'])
    source = (repo / 'train.py').read_bytes()
    tree = ast.parse(source.decode())
    block = [n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Attribute) and n.test.func.attr == 'get' and n.test.args and isinstance(n.test.args[0], ast.Constant) and n.test.args[0].value == 'diagnostic_observed_static_rgb_enabled']
    require(len(block) == 1)
    block = block[0]
    lines = source.splitlines(keepends=True)
    removed = b''.join(lines[:block.lineno - 1] + lines[block.end_lineno:])
    require(removed == (original / 'train.py').read_bytes())
    require(sha(repo / 'utils/paper_losses.py') == sha(original / 'utils/paper_losses.py'))
    loss_utils = load('S_original_loss_utils_CPU', repo / 'utils/loss_utils.py')
    statement = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), '<actual S source block>', 'exec')
    generator = torch.Generator(device='cpu').manual_seed(31)
    pred = torch.rand((3, 13, 17), generator=generator, requires_grad=True)
    truth = torch.rand((3, 13, 17), generator=generator)
    dynamic_mask = torch.zeros((1, 13, 17)); dynamic_mask[:, 3:10, 6:13] = 1
    dynamic_alpha = torch.full_like(dynamic_mask, 0.75, requires_grad=True)
    baseline = torch.tensor(0.2980644106864929, dtype=torch.float32)
    namespace = {'args': OmegaConf.create(binding['training_config']), 'log_dict': {}, 'loss': baseline,
                 'static_image': pred, 'gt_image': truth, 'dynamic_mask': dynamic_mask,
                 'dynamic_alpha': dynamic_alpha, 'F': F, 'ssim': loss_utils.ssim}
    exec(statement, namespace)
    old_tree = ast.parse((BASE / 'DrivingEditor-run/train.py').read_text())
    old_mask = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_object_pixels')
    old_namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[old_mask], type_ignores=[])), '<actual2024 mask>', 'exec'), old_namespace)
    old_prediction = old_namespace['extract_object_pixels'](pred, 1 - dynamic_mask)
    old_truth = old_namespace['extract_object_pixels'](truth, 1 - dynamic_mask)
    old_rgb = 0.8 * F.l1_loss(old_prediction, old_truth) + 0.2 * (1.0 - loss_utils.ssim(old_prediction, old_truth))
    require(torch.equal(namespace['observed_static_rgb'], old_rgb))
    require(torch.equal(namespace['loss'], baseline + 1.0 * old_rgb))
    require(namespace['log_dict']['loss_original_objective_before_diagnostic'] == baseline.item())
    new_grad, dynamic_grad = torch.autograd.grad(namespace['observed_static_rgb'], [pred, dynamic_alpha], allow_unused=True, retain_graph=True)
    old_grad = torch.autograd.grad(old_rgb, pred)[0]
    require(torch.equal(new_grad, old_grad) and torch.count_nonzero(new_grad * dynamic_mask).item() == 0 and torch.count_nonzero(new_grad * (1 - dynamic_mask)).item() > 0 and dynamic_grad is None)
    disabled = dict(namespace); disabled['args'] = OmegaConf.create(dict(binding['training_config'], diagnostic_observed_static_rgb_enabled=False)); disabled['loss'] = baseline; disabled['log_dict'] = {}
    exec(statement, disabled)
    require(disabled['loss'] is baseline and disabled['log_dict'] == {})
    guard_module = load('independent_S_guard_CPU', HERE / 'static_anchor_guard.py')
    formal = load('unchanged_formal_guard_CPU', repo / 'utils/paper_checkpoint.py')
    guard = guard_module.ObservedStaticRgbExperimentGuard(binding)
    guard.validate_config(binding['parent_training_config'], binding['training_config'], formal._validate_resume_configuration)
    guard.validate_config(binding['training_config'], binding['training_config'], formal._validate_resume_configuration)
    negative = []
    for key, value in [('diagnostic_observed_static_rgb_enabled', 1), ('diagnostic_lambda_observed_static_rgb', 1), ('diagnostic_lambda_observed_static_rgb', 0.05), ('iterations', 32000), ('paper_lambda_dynamic_rgb', 5.0), ('diagnostic_background_domain_complement', True), ('model_path', '/tmp/unbound-S')]:
        changed = copy.deepcopy(binding['training_config']); changed[key] = value
        try:
            guard.validate_config(binding['parent_training_config'], changed, formal._validate_resume_configuration)
        except ValueError:
            negative.append({'changed_field': key, 'value': value, 'rejected': True})
        else:
            raise AssertionError('S guard accepted changed field/type: ' + key)
    try:
        formal._validate_resume_configuration(binding['parent_training_config'], binding['training_config'])
    except ValueError:
        formal_rejects_extra = True
    else:
        raise AssertionError('Formal original guard unexpectedly accepted S')
    require(not torch.cuda.is_initialized())
    receipt = {'status': 'PASS_FOCUSED_S_SOURCE_MATH_CONFIG_CPU', 'GPU_or_training_started': False,
               'actual_S_source_only9_lines_inverse_bytes_original': True, 'original_paper_losses_bytes_unchanged': True,
               'fullHW_static_photo_value_and_prediction_gradient_exact_2024_term': True,
               'actual_FP32_total_decomposition_exact': True, 'captured_original_objective_exact': True,
               'disabled_S_same_original_loss_tensor': True, 'actor_pixel_direct_static_prediction_gradient_zero': True,
               'added_term_has_no_direct_dynamic_alpha_graph': True,
               'gradient_limit': 'Static render includes shared ENV; the real renderer has indirect shared-environment/composite interactions. This CPU image-tensor test does not claim exclusive static-model updates.',
               'guard_valid_original_saved_and_same_bound_S_saved_configs': True, 'guard_rejected_changes': negative,
               'formal_guard_bytes_unchanged_and_rejects_extra_S_fields': formal_rejects_extra,
               'actual_cuda_restore_proved_here': False,
               'pins': {str(p): sha(p) for p in [Path(__file__), binding_path, repo / 'train.py', repo / 'utils/paper_losses.py', HERE / 'static_anchor_guard.py', BASE / 'DrivingEditor-run/train.py']}}
    (HERE / 'focused_S_checks_cpu.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'status': receipt['status'], 'training_started': False, 'CUDA_initialized': False, 'receipt': str(HERE / 'focused_S_checks_cpu.json'), 'receipt_sha256': sha(HERE / 'focused_S_checks_cpu.json')}))


if __name__ == '__main__':
    main()
