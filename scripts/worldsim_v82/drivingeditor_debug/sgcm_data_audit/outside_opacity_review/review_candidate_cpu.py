"""Independent CPU/AST review of the isolated outside-opacity candidate.

This script never imports the renderer, train.py, or a CUDA extension and
does not change the candidate, the formal source, or any checkpoint.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import torch

torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
HERE = Path(__file__).resolve().parent
TASK = HERE.parent.parent
BASE = TASK / 'loss_audit/source_frozen'
CANDIDATE = TASK / 'loss_audit/candidates/opacity_outside'
PLAN = TASK / 'component_diagnostic/continuation_plan'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(node):
    return ast.dump(node, include_attributes=False)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checks = []


def check(name, condition, details=None):
    checks.append({'test': name, 'passed': bool(condition), 'details': details})
    if not condition:
        raise AssertionError(name)


original_losses = ast.parse((BASE / 'utils/paper_losses.py').read_text(encoding='utf8'))
candidate_losses = ast.parse((CANDIDATE / 'utils/paper_losses.py').read_text(encoding='utf8'))
new_name = 'diagnostic_dynamic_background_opacity_loss'
removed = [node for node in candidate_losses.body if isinstance(node, ast.FunctionDef) and node.name == new_name]
without_new = copy.deepcopy(candidate_losses)
without_new.body = [node for node in without_new.body if not (isinstance(node, ast.FunctionDef) and node.name == new_name)]
check('all_original_paper_losses_exact_ast_preserved', dump(original_losses) == dump(without_new) and len(removed) == 1)


class RemoveOnlyAuthorizedAdditions(ast.NodeTransformer):
    """Reject hidden changes by rebuilding the complete frozen train AST."""
    def __init__(self):
        self.counts = {}

    def counted(self, name):
        self.counts[name] = self.counts.get(name, 0) + 1

    def visit_ImportFrom(self, node):
        if node.module == 'utils.paper_losses' and any(alias.name == new_name for alias in node.names):
            assert all(alias.asname is None for alias in node.names if alias.name == new_name)
            node.names = [alias for alias in node.names if alias.name != new_name]
            self.counted('import')
        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {'diagnostic_background_opacity', 'diagnostic_lambda_background'}:
            expected = ('zero' if node.targets[0].id == 'diagnostic_background_opacity'
                        else 'float(args.get("diagnostic_lambda_dynamic_background_opacity", 0.0))')
            assert dump(node.value) == dump(ast.parse(expected, mode='eval').body)
            self.counted(node.targets[0].id)
            return None
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'loss' and isinstance(node.value, ast.BinOp) and dump(node.value) == dump(ast.parse('loss + diagnostic_lambda_background * diagnostic_background_opacity', mode='eval').body):
            self.counted('loss_addition')
            return None
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript) and dump(node.targets[0]) == dump(ast.parse("log_dict['loss_diagnostic_dynamic_background_opacity'] = None").body[0].targets[0]):
            assert dump(node.value) == dump(ast.parse('diagnostic_background_opacity.item()', mode='eval').body)
            self.counted('diagnostic_log')
            return None
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript) and dump(node.targets[0]) == dump(ast.parse("log_dict['loss_original_objective_before_diagnostic'] = None").body[0].targets[0]) and dump(node.value) == dump(ast.parse('loss.item()', mode='eval').body):
            self.counted('pre_add_original_objective_log')
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if dump(node.test) == dump(ast.parse('diagnostic_lambda_background > 0.0', mode='eval').body):
            assert len(node.body) == 1 and not node.orelse
            assert dump(node.body[0]) == dump(ast.parse('diagnostic_background_opacity = diagnostic_dynamic_background_opacity_loss(dynamic_alpha, dynamic_mask)').body[0])
            self.counted('background_evaluation_gate')
            return None
        return self.generic_visit(node)


original_train = ast.parse((BASE / 'train.py').read_text(encoding='utf8'))
candidate_train = ast.parse((CANDIDATE / 'train.py').read_text(encoding='utf8'))
stripper = RemoveOnlyAuthorizedAdditions()
stripped = stripper.visit(copy.deepcopy(candidate_train))
expected_counts = {'import': 1, 'diagnostic_background_opacity': 1, 'diagnostic_lambda_background': 1, 'background_evaluation_gate': 1, 'loss_addition': 1, 'diagnostic_log': 1}
if 'pre_add_original_objective_log' in stripper.counts:
    expected_counts['pre_add_original_objective_log'] = 1
check('complete_train_ast_unchanged_except_authorized_additions', dump(stripped) == dump(original_train) and stripper.counts == expected_counts, stripper.counts)

baseline = load(BASE / 'utils/paper_losses.py', 'independent_original_losses')
variant = load(CANDIDATE / 'utils/paper_losses.py', 'independent_variant_losses')
alpha = torch.tensor([[[0., .001, .1, .8, .99, 1.]]], dtype=torch.float64, requires_grad=True)
foreground = torch.ones_like(alpha)
background = torch.zeros_like(alpha)
fg_value = variant.diagnostic_dynamic_background_opacity_loss(alpha, foreground)
fg_gradient, = torch.autograd.grad(fg_value, alpha, retain_graph=True)
check('new_term_has_exact_zero_value_and_gradient_on_foreground', fg_value.item() == 0. and bool((fg_gradient == 0).all()))
outside_value = variant.diagnostic_dynamic_background_opacity_loss(alpha, background)
outside_gradient, = torch.autograd.grad(outside_value, alpha, retain_graph=True)
check('finite_background_penalty_and_positive_high_alpha_gradient', bool(torch.isfinite(outside_value)) and bool(torch.isfinite(outside_gradient).all()) and bool((outside_gradient[0, 0, 1:] > 0).all()))
single = torch.tensor([[[.8]]], dtype=torch.float64)
single_loss = variant.diagnostic_dynamic_background_opacity_loss(single, torch.zeros_like(single))
wide = torch.tensor([[[.8, .8, .8, .8]]], dtype=torch.float64)
wide_mask = torch.tensor([[[0., 1., 1., 1.]]], dtype=torch.float64)
check('normalization_by_full_image_area', torch.equal(variant.diagnostic_dynamic_background_opacity_loss(wide, wide_mask), single_loss / 4.))
reference_alpha = torch.tensor([[[.2, .8]]], dtype=torch.float64, requires_grad=True)
reference_mask = torch.tensor([[[1., 0.]]], dtype=torch.float64)
baseline_loss = baseline.dynamic_opacity_loss(reference_alpha, reference_mask)
variant_original_loss = variant.dynamic_opacity_loss(reference_alpha, reference_mask)
check('foreground_eq13_value_and_gradient_exact_equal', torch.equal(baseline_loss, variant_original_loss) and torch.equal(torch.autograd.grad(baseline_loss, reference_alpha, retain_graph=True)[0], torch.autograd.grad(variant_original_loss, reference_alpha, retain_graph=True)[0]))
zero_coefficient = variant_original_loss + 0. * variant.diagnostic_dynamic_background_opacity_loss(reference_alpha, reference_mask)
check('zero_complement_coefficient_preserves_value_and_gradient', torch.equal(zero_coefficient, baseline_loss) and torch.equal(torch.autograd.grad(zero_coefficient, reference_alpha)[0], torch.autograd.grad(baseline_loss, reference_alpha)[0]))

paths = [BASE / 'utils/paper_losses.py', BASE / 'train.py', CANDIDATE / 'utils/paper_losses.py', CANDIDATE / 'train.py', PLAN / 'continuation_harness.py', PLAN / 'prepared_plan.json', PLAN / 'runtime_safety_parameterized.py', TASK / 'component_diagnostic/scripts/component_diagnostic.py']
pins = {str(path.relative_to(TASK)).replace('\\', '/'): sha(path) for path in paths}
check('approved_baseline_harness_and_plan_sha_match', pins['component_diagnostic/continuation_plan/continuation_harness.py'] == 'ac9745d5c0cbab7701ae8be5beec9587d57ea79288a3167259d0af1d0e164458' and pins['component_diagnostic/continuation_plan/prepared_plan.json'] == 'f12d6c0801997d0a09020617c7629f5f41abfab4083ebbf0400f1e715be6d205')
check('no_cuda_initialized', not torch.cuda.is_initialized())
report = {'review_role': 'Independent candidate source reviewer; CPU only', 'candidate_only_status': 'PASS', 'experiment_status': 'PENDING_CONTROLLED_GUARD_HARNESS_PLAN_AND_RUNTIME_EVIDENCE', 'checks': checks, 'pins': pins, 'numeric_evidence': {'sample_alpha': alpha.detach().tolist(), 'outside_raw_loss': outside_value.item(), 'outside_gradient': outside_gradient.detach().tolist()}, 'torch': torch.__version__, 'CUDA_initialized': torch.cuda.is_initialized(), 'formal_files_modified': False, 'limitations': 'PASS covers only these exact candidate bytes and baseline references. No actual checkpoint restore, CUDA execution, finite3D realizability, quality improvement, or paper formula certification is asserted.'}
(HERE / 'candidate_independent_review_cpu.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({'candidate_only_status': report['candidate_only_status'], 'checks': len(checks), 'candidate_pins': {k: v for k, v in pins.items() if k.startswith('loss_audit/candidates/')}, 'experiment_status': report['experiment_status']}, indent=2))
