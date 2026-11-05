"""CPU-only source review; never import Torch, scene modules or old scene code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGINAL = {
    'calibrate.py': 'c6f9a09735fac05db3e97284462f3b4de527e3c1216fc5588610561ce6cefe2f',
    'refine_colour.py': '1f7b6ebb47efd16aed645143cfa1b971a5d2e8005f5a788e7355f1b92c1d17bc',
    'scene_runtime.py': '76f2aeb66feda3ccf4ecc02246cc2f3789f34f3bf06c3c4bca2651b93df97d25',
    'sh_asset.py': 'ff9ee4b0f6a0bc90fccedfd4be17404d2df129072d6da1562b211a8aa6469bd4',
}


def tree(path):
    text = Path(path).read_text(encoding='utf-8')
    compile(text, str(path), 'exec')
    return ast.parse(text)


def function(parsed, name, owner=None):
    nodes = parsed.body
    if owner:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    return next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def audit():
    checks = {}
    for name, expected in ORIGINAL.items():
        actual = hashlib.sha256((ROOT/'original_source'/name).read_bytes()).hexdigest()
        assert actual == expected, f'Original snapshot changed: {name}'
        checks['original/'+name] = actual
    original = tree(ROOT/'original_source/calibrate.py')
    fine = tree(ROOT/'original_source/refine_colour.py')
    extracted = tree(ROOT/'calibration_method.py')
    mapping = [('cache_design', 'Experiment', original, 'CoarseMethod'),
               ('solve', 'Experiment', original, 'CoarseMethod'),
               ('solve', 'Experiment', fine, 'FineMethod'),
               ('world_basis', None, original, None), ('grid_reg', None, original, None)]
    equal = []
    for name, old_owner, old_tree, new_owner in mapping:
        assert dump(function(old_tree, name, old_owner)) == dump(function(extracted, name, new_owner)), name
        equal.append(f'{new_owner+"." if new_owner else ""}{name}')
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in extracted.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id in
                   ('RAWS', 'HOLDOUT', 'GRID', 'MAX_DELTA', 'MIN_WEIGHT')
                   and not isinstance(node.value, ast.Call)}
    assert assignments == dict(HOLDOUT=(89, 95, 101, 105, 113, 121, 125, 130, 137),
                               GRID=(6, 5), MAX_DELTA=.09, MIN_WEIGHT=.02)
    forbidden = ('scene008_a6_ray_switch_20261105', 'scene008_ray_road_colour_20261105',
                 'scene008_ray_road_colour_fine_20261105', 'scene008_native_road1k_20261105_r1')
    candidate_files = ('final_runtime.py', 'calibration_adapter.py', 'calibration_method.py',
                       'evaluate_outputs.py', 'run_finalize.py')
    for name in candidate_files:
        text = (ROOT/name).read_text(encoding='utf-8')
        assert not any(token in text for token in forbidden), f'Legacy candidate binding: {name}'
        parsed = tree(ROOT/name)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                assert not (call.endswith('.backward') or call.endswith('.step')
                            or call.endswith('torch.save')), f'Unexpected optimization/model save: {name}:{node.lineno}'
    for path in ROOT.glob('*.py'):
        tree(path)
        checks[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(status='PASS_CPU_STATIC_AUDIT', original_snapshots_sha256_verified=True,
                exact_original_AST_methods=equal, original_calibration_constants_unchanged=True,
                candidate_hardcoded_old_scene_or_road_or_coeff_paths=False,
                backward_optimizer_step_or_torch_save_in_candidate=False,
                all_private_python_sources_compile=True, source_sha256=checks,
                limitations='Static review only. No new CUDA execution is claimed; real preflight remains required.')


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
