"""Independent configuration boundary review; no GPU/model imports.

Default fixture uses the real saved30k config already in CPU audit metadata.
With --binding, it checks the exact prepared experiment binding instead.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binding', type=Path)
parser.add_argument('--verify-files', action='store_true')
args = parser.parse_args()
guard_path = TASK / 'loss_audit/outside_experiment_guard.py'
guard_module = load(guard_path, 'independent_outside_guard')
if args.binding:
    binding = json.loads(args.binding.read_text(encoding='utf8'))
else:
    parent = json.loads((TASK / 'component_diagnostic/cpu_audit.json').read_text(encoding='utf8'))['records']['30000']['training_state']['config']
    current = copy.deepcopy(parent)
    current['model_path'] = '/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/continuation_diagnostic/opacity_outside30k_to30300'
    current['start_checkpoint'] = '/root/autodl-tmp/drivingeditor_repro/runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline/chkpnt30000.pth'
    current.update(guard_module.EXTRA_CONFIG)
    binding = {'experiment_kind': 'hypothesis_complement_background_domain', 'added_training_fields': copy.deepcopy(guard_module.EXTRA_CONFIG), 'algorithm_changed_files': ['train.py', 'utils/paper_losses.py'], 'parent_iteration': 30000, 'first_iteration': 30001, 'stop_iteration': 30300, 'horizon': 50000, 'seed': 0, 'parent_training_config': parent, 'training_config': current, 'output': current['model_path'], 'parent_static_checkpoint': current['start_checkpoint'], 'repository': '/isolated/replica'}

guard = guard_module.OutsideExperimentGuard(binding)
formal_path = (Path(binding['original_repository']) / 'utils/paper_checkpoint.py'
               if args.binding and args.verify_files
               else TASK / 'loss_audit/source_frozen/utils/paper_checkpoint.py')
formal_ast = ast.parse(formal_path.read_text(encoding='utf8'))
nodes = [node for node in formal_ast.body if isinstance(node, ast.FunctionDef) and node.name in {'_validate_resume_configuration', '_same_config_value'}]
namespace = {'_config_dict': lambda value: dict(value)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(formal_path), 'exec'), namespace)
original_validator = namespace['_validate_resume_configuration']
calls = []


def tracked_validator(saved, current):
    calls.append((copy.deepcopy(saved), copy.deepcopy(current)))
    original_validator(saved, current)


checks = []


def check(name, condition, detail=None):
    checks.append({'test': name, 'passed': bool(condition), 'details': detail})
    if not condition:
        raise AssertionError(name)


def rejected(operation):
    try:
        operation()
    except (ValueError, RuntimeError):
        return True
    return False


saved = binding['parent_training_config']
current = binding['training_config']
before_saved, before_current = copy.deepcopy(saved), copy.deepcopy(current)
receipt = guard.validate_config(saved, current, tracked_validator)
check('exact_parent_to_hypothesis_calls_unchanged_formal_validation', receipt['original_validator_called'] and len(calls) == 1 and calls[0][0] == saved and not (set(calls[0][1]) & set(guard_module.EXTRA_CONFIG)))
check('configuration_inputs_not_mutated', saved == before_saved and current == before_current)
receipt = guard.validate_config(current, current, tracked_validator)
check('same_exact_saved_hypothesis_revalidates_all_original_fields', receipt['saved_configuration_kind'] == 'same_bound_hypothesis' and len(calls) == 2)

mutated_fields = []
deleted_fields = []
for key, value in current.items():
    changed = copy.deepcopy(current)
    if isinstance(value, bool):
        changed[key] = not value
    elif isinstance(value, (int, float)):
        changed[key] = value + 1
    elif isinstance(value, str):
        changed[key] = value + '__unauthorized__'
    elif isinstance(value, list):
        changed[key] = value + ['__unauthorized__']
    elif value is None:
        changed[key] = '__unauthorized__'
    else:
        raise AssertionError('Uncovered field type: ' + key)
    check('reject_changed_field_' + key, rejected(lambda c=changed: guard.validate_config(saved, c, tracked_validator)))
    mutated_fields.append(key)
    changed = copy.deepcopy(current)
    del changed[key]
    check('reject_deleted_field_' + key, rejected(lambda c=changed: guard.validate_config(saved, c, tracked_validator)))
    deleted_fields.append(key)
changed = copy.deepcopy(current)
changed['unapproved_new_field'] = 0
check('reject_unapproved_new_field', rejected(lambda: guard.validate_config(saved, changed, tracked_validator)))
for key, value in current.items():
    if type(value) not in (bool, int, float):
        continue
    changed = copy.deepcopy(current)
    changed[key] = int(value) if type(value) in (bool, float) else float(value)
    check('reject_equal_numeric_value_wrong_type_' + key, rejected(lambda c=changed: guard.validate_config(saved, c, tracked_validator)))
for key in saved:
    altered_saved = copy.deepcopy(saved)
    del altered_saved[key]
    check('reject_missing_saved_field_' + key, rejected(lambda s=altered_saved: guard.validate_config(s, current, tracked_validator)))
check('no_failed_case_reaches_original_validator', len(calls) == 2)
fake_formal = types.SimpleNamespace(__file__=str(formal_path), _validate_resume_configuration=original_validator)
check('adapter_rejects_formal_module_path', rejected(lambda: guard.install_on_isolated_module(fake_formal)))
check('formal_validator_binding_unmodified', fake_formal._validate_resume_configuration is original_validator)
if args.verify_files:
    file_receipt = guard.verify_sources()
    file_receipt.update(guard.verify_protected_files())
    file_receipt.update(guard.verify_auxiliary_files())
else:
    file_receipt = {'not_run': 'Boundary test only; source/config binding and protected file checks remain mandatory on prepared remote artifact.'}

report = {'guard_boundary_status': 'PASS', 'prepared_artifact_status': 'FINAL_BOUND_ARTIFACT_CHECKED' if args.binding and args.verify_files else 'PRELIMINARY_CONFIG_BOUNDARY_ONLY', 'guard_sha256': sha(guard_path), 'formal_validator_sha256': sha(formal_path), 'binding_sha256': sha(args.binding) if args.binding else None, 'original_field_count': len(saved), 'current_field_count': len(current), 'checks_count': len(checks), 'all_passed': all(item['passed'] for item in checks), 'checks': checks, 'file_receipt': file_receipt, 'type_sensitive_configuration_converter': 'Dictionary inputs passed through _config_dict; the two actual frozen validator function ASTs execute unchanged.', 'GPU_used': False, 'actual_checkpoint_restore_claimed': False, 'full_experiment_launch_review': 'PENDING_HARNESS_PLAN_AND_RUNTIME_EVIDENCE'}
(HERE / ('guard_independent_review_cpu.json' if not args.binding else 'guard_bound_independent_review_cpu.json')).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
print(json.dumps({key: report[key] for key in ('guard_boundary_status', 'prepared_artifact_status', 'guard_sha256', 'binding_sha256', 'original_field_count', 'current_field_count', 'checks_count')}, indent=2))
