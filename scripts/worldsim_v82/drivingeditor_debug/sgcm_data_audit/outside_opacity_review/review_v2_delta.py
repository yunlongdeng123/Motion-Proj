"""Targeted v2 runtime-import correction review; no old test suite rerun."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def dump(node):
    return ast.dump(node, include_attributes=False)


HERE = Path(__file__).resolve().parent
TASK = HERE.parent.parent
CONTROL = TASK / 'component_diagnostic/continuation_plan'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--verify-files', action='store_true')
args = parser.parse_args()
binding_path = TASK / 'loss_audit/controlled_opacity_outside/experiment_binding.json'
binding = read(binding_path)
old_harness = HERE / 'refs/continuation_variant_harness_v1.py'
new_harness = CONTROL / 'continuation_variant_harness.py'
old_plan = read(HERE / 'refs/opacity_outside_prepared_plan.json')
plan_path = HERE / 'refs_v2/opacity_outside_prepared_plan.json'
new_plan = read(plan_path)
receipt_path = HERE / 'refs_v2/opacity_outside_cpu_preflight_receipt.json'
preflight = read(receipt_path)
original_review = read(HERE / 'final_review.json')
expected_harness_sha = '3a2f9add3cab0763c4b0ef612fd5b735afd7f2d337b151306da1a4ceffcb6154'
expected_plan_sha = 'be9e385b37805457d2d9986985c9b7bd65af3fcde66445d3f15d156db5c90a93'
assert sha(old_harness) == original_review['harness_sha256'] == '93bb8df2cca142b3ea0b08e42ab8c3fe5defa214641537b7f3935eb300444359'
assert sha(new_harness) == preflight['harness_sha256'] == new_plan['harness_sha256'] == expected_harness_sha
assert sha(plan_path) == preflight['prepared_plan_sha256'] == expected_plan_sha
assert sha(binding_path) == new_plan['experiment_binding_sha256'] == original_review['experiment_binding_sha256'] == 'b96ec92c0694a251e39da85a3ae4abab131366a75ad52cd47098ec0df953b214'
assert sha(TASK / 'loss_audit/outside_experiment_guard.py') == original_review['guard_sha256'] == new_plan['guard_sha256']
assert preflight['status'] == 'PASS' and not preflight['training_started'] and not preflight['CUDA_initialized']
assert [key for key in sorted(set(old_plan) | set(new_plan)) if old_plan.get(key) != new_plan.get(key)] == ['harness_sha256']


class OnlyThreeImportCorrections(ast.NodeTransformer):
    def __init__(self):
        self.counts = {'remove_import': 0, 'remove_pin_tuple': 0, 'correct_PSNR_identity': 0}

    def visit_Import(self, node):
        if len(node.names) == 1 and node.names[0].name == 'utils.image_utils' and node.names[0].asname == 'image_utils':
            self.counts['remove_import'] += 1
            return None
        return node

    def visit_List(self, node):
        target = ast.parse("[(image_utils, REPO / 'utils/image_utils.py')]", mode='eval').body.elts[0]
        kept = []
        for item in node.elts:
            if dump(item) == dump(target):
                self.counts['remove_pin_tuple'] += 1
            else:
                kept.append(item)
        node.elts = kept
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'image_utils' and node.attr == 'psnr':
            node.value.id = 'loss_utils'
            self.counts['correct_PSNR_identity'] += 1
        return self.generic_visit(node)


old_ast = ast.parse(old_harness.read_text(encoding='utf8'))
new_ast = ast.parse(new_harness.read_text(encoding='utf8'))
correction = OnlyThreeImportCorrections()
assert dump(correction.visit(copy.deepcopy(old_ast))) == dump(new_ast)
assert correction.counts == {'remove_import': 1, 'remove_pin_tuple': 1, 'correct_PSNR_identity': 1}
assert not any(isinstance(node, ast.Import) and any(item.name == 'utils.image_utils' for item in node.names) for node in ast.walk(new_ast))

expected_runtime = {}
for node in ast.walk(new_ast):
    if not isinstance(node, ast.For) or not isinstance(node.target, ast.Tuple) or not isinstance(node.iter, ast.List):
        continue
    if [item.id for item in node.target.elts if isinstance(item, ast.Name)] != ['module', 'expected']:
        continue
    for item in node.iter.elts:
        module, location = item.elts
        assert isinstance(module, ast.Name) and isinstance(location, ast.BinOp) and isinstance(location.left, ast.Name) and location.left.id == 'REPO' and isinstance(location.op, ast.Div) and isinstance(location.right, ast.Constant)
        relative = location.right.value
        assert relative in binding['replica_source_sha256'], relative
        expected_runtime[relative] = binding['replica_source_sha256'][relative]
assert len(expected_runtime) == 13
assert expected_runtime == preflight['actual_runtime_dependency_paths_148_inventory_verified']
physical = {}
if args.verify_files:
    for relative, expected in expected_runtime.items():
        path = Path(binding['repository']) / relative
        assert path.is_file() and sha(path) == expected, relative
        physical[relative] = expected
    training_path = Path(binding['repository']) / 'train.py'
    loss_path = Path(binding['repository']) / 'utils/loss_utils.py'
else:
    training_path = TASK / 'loss_audit/candidates/opacity_outside/train.py'
    loss_path = TASK / 'loss_audit/source_frozen/utils/loss_utils.py'
assert sha(training_path) == binding['replica_source_sha256']['train.py'] == 'ff82c0d86b385bcdf28feff550163cd3ec9c7bec92e939306330ba4af335e99a'
assert sha(loss_path) == expected_runtime['utils/loss_utils.py']
train_ast = ast.parse(training_path.read_text(encoding='utf8'))
loss_ast = ast.parse(loss_path.read_text(encoding='utf8'))
imports = [node for node in train_ast.body if isinstance(node, ast.ImportFrom) and node.module == 'utils.loss_utils']
assert len(imports) == 1 and {item.name for item in imports[0].names} >= {'psnr', 'ssim'} and all(item.asname is None for item in imports[0].names)
assert {'psnr', 'ssim'} <= {node.name for node in loss_ast.body if isinstance(node, ast.FunctionDef)}
identity = ast.parse('training.ssim is not loss_utils.ssim or training.psnr is not loss_utils.psnr', mode='eval').body
assert any(isinstance(node, ast.If) and dump(node.test) == dump(identity) for node in ast.walk(new_ast))
assert new_plan['runtime_source_sha256'] == old_plan['runtime_source_sha256']
assert new_plan['training_config'] == old_plan['training_config'] and new_plan['original_file_sha256'] == old_plan['original_file_sha256']
assert new_plan['guard_sha256'] == old_plan['guard_sha256'] and new_plan['observer_sha256'] == old_plan['observer_sha256'] and new_plan['launcher_sha256'] == old_plan['launcher_sha256']

report = copy.deepcopy(original_review)
report.update(review_version=2, status='PASS', decision='The exact v2 import correction can run under a fresh root receipt and exclusive GPU handoff.', harness_sha256=expected_harness_sha, prepared_plan_sha256=expected_plan_sha, previous_review_sha256=sha(HERE / 'final_review.json'), previous_preparation_import_failure='Attempt1 failed on nonexistent utils.image_utils before training/state restoration/optimizer; v1 review missed this inventory membership check.', reviewed_revision={'only_harness_AST_changes': correction.counts, 'plan_changed_fields': ['harness_sha256'], 'all13_runtime_dependency_paths_exist_in_actual148_inventory': True, 'actual13_runtime_dependency_file_sha256': physical, 'actual_training_AST_import': 'from utils.loss_utils import psnr, ssim', 'actual_loss_utils_defines_both_functions': True, 'runtime_identity_gate': 'training.psnr is loss_utils.psnr; training.ssim is loss_utils.ssim', 'binding_guard_candidate_loss_source_config_restore_eval_and_sampler_unchanged': True, 'old_51_512_CPU_cases_rerun': False, 'CPU_only': True, 'new_variant_GPU_restore_or_training_executed_by_this_reviewer': False}, v2_cpu_preflight_receipt_sha256=sha(receipt_path), v2_cpu_preflight_log_sha256=sha(HERE / 'refs_v2/opacity_outside_cpu_preflight.log'), source_physical_SHA_rechecked_for_this_revision=args.verify_files)
report['evidence_sha256']['final_review_v1.json'] = sha(HERE / 'final_review.json')
report['review_scope'] = 'Targeted independent v2 AST delta, real13 import-file inventory/SHA, actual training import and revised CPU-preflight review only. Earlier valid algorithm/config/restore/evaluation evidence remains unchanged; v1 nonexistent import claim is superseded.'
(HERE / 'final_review_v2.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
text = '''v2独立审查 PASS：可在新的root receipt与GPU交接后重试该精确计划。
v1漏查utils.image_utils不在实际inventory，attempt1在任何训练/restore/optimizer之前失败。
修正仅3处：删除不存在的import与pin tuple；PSNR identity改为loss_utils.psnr。
真实train.py从utils.loss_utils导入psnr/ssim，该文件同时定义两函数。

同30k四文件 + 原输入 -> 原完整restore -> 原paper loss + .05 Loutside
                                             |
                                             v
                              同300轨迹 / RNG-safe49评估 -> 四文件state

v2新增runtime loop的13个源路径都存在真实148inventory，逐SHA通过；
新plan只有harness_sha256改变。原binding/guard/算法源/config/restore/eval/采样和launcher全不变。
旧51/512cases未重跑，旧final_review/common49记录未覆盖。未启动GPU。
此PASS仅用于准备可运行，不是实际恢复或实验效果认证。
'''
text += '\nharness_sha256: ' + expected_harness_sha + '\nprepared_plan_sha256: ' + expected_plan_sha + '\nexperiment_binding_sha256: ' + new_plan['experiment_binding_sha256'] + '\n'
(HERE / 'final_review_v2.txt').write_text(text, encoding='utf8')
print(json.dumps({'status': 'PASS', 'harness_sha256': expected_harness_sha, 'prepared_plan_sha256': expected_plan_sha, 'physical13_file_SHA_checked': args.verify_files, 'final_review_v2_sha256': sha(HERE / 'final_review_v2.json')}, indent=2))
