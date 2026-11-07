"""CPU reproduction against the actual frozen renderer, without importing CUDA.

The function is extracted with AST, not reimplemented. A tiny CPU point fixture
uses the exact SHM/marginal/instant-velocity methods from gaussian_model.py.
The candidate fix only removes unconditional `time_shift = None` assignments.
This proves interface behavior; it does not prove an a5 quality improvement.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--renderer', type=Path, required=True)
    ap.add_argument('--model-source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--patch-output', type=Path, required=True)
    ap.add_argument('--sampler-log', type=Path)
    opt = ap.parse_args()
    original = opt.renderer.read_text()
    tree = ast.parse(original)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_rasterization')
    assignments = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'time_shift' for t in n.targets)
                   and isinstance(n.value, ast.Constant) and n.value.value is None]
    assert len(assignments) == 2, [n.lineno for n in assignments]
    drop = {n.lineno for n in assignments}
    fixed_text = ''.join(line for i, line in enumerate(original.splitlines(True), 1) if i not in drop)
    fixed_fn = next(n for n in ast.parse(fixed_text).body if isinstance(n, ast.FunctionDef) and n.name == fn.name)
    namespace = dict(torch=torch, np=np, GaussianModel=object)
    def compile_fn(node):
        scope = dict(namespace)
        module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]))
        exec(compile(module, str(opt.renderer), 'exec'), scope)
        return scope['prepare_rasterization']
    before, after = compile_fn(fn), compile_fn(fixed_fn)
    model_tree = ast.parse(opt.model_source.read_text())
    cls = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == 'GaussianModel')
    names = {'get_xyz_SHM', 'get_marginal_t', 'get_inst_velocity'}
    methods = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == 3
    model_namespace = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(opt.model_source), 'exec'), model_namespace)

    class Fixture:
        def __init__(self):
            self._xyz = torch.tensor([[1., 2., 3.], [2., 1., 4.]])
            self._velocity = torch.tensor([[2., 1., .5], [1., 0., .4]])
            self._t = torch.tensor([[0.], [.1]])
            self.T, self.velocity_decay = .2, 1.
            self.get_scaling_t = torch.tensor([[.3], [.4]])
            self.get_t = self._t
            self.get_xyz = self._xyz
            self.get_opacity = torch.tensor([[.8], [.7]])
            self.get_scaling = torch.ones((2, 3)) * .03
            self.get_rotation = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
            self.get_features = torch.zeros((2, 16, 3))
    for name in names:
        setattr(Fixture, name, model_namespace[name])
    camera = SimpleNamespace(timestamp=.07, world_view_transform=torch.eye(4))
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False)
    fields = ('means3D', 'opacities', 'features', 'mask')
    def outputs(function, dynamic, shift):
        return function(camera, Fixture(), pipe, time_shift=shift, dynamic=dynamic)
    def difference(a, b):
        return {key: float((a[key].to(torch.float64) - b[key].to(torch.float64)).abs().max()) for key in fields}
    rows = []
    for dynamic in (False, True):
        base = outputs(before, dynamic, None)
        shifted = outputs(before, dynamic, .013)
        fixed_base = outputs(after, dynamic, None)
        fixed_shifted = outputs(after, dynamic, .013)
        row = dict(branch='dynamic' if dynamic else 'static',
                   original_shift_delta=difference(base, shifted),
                   fixed_shift_delta=difference(fixed_base, fixed_shifted),
                   unshifted_fix_delta=difference(base, fixed_base))
        assert all(v == 0 for v in row['unshifted_fix_delta'].values())
        assert row['fixed_shift_delta']['means3D'] > 0 and row['fixed_shift_delta']['opacities'] > 0
        if dynamic:
            assert row['original_shift_delta']['means3D'] > 0
            assert all(v == 0 for v in difference(shifted, fixed_shifted).values())
        else:
            assert all(v == 0 for v in row['original_shift_delta'].values())
        rows.append(row)
    assert all(v == 0 for v in difference(outputs(after, False, .013), outputs(after, True, .013)).values())
    sampler = None
    if opt.sampler_log:
        records = [json.loads(line) for line in opt.sampler_log.read_text().splitlines() if line.strip()]
        used = [r for r in records if r.get('time_shift') is not None]
        sampler = dict(source=str(opt.sampler_log), rows=len(records), nonnull_time_shifts=len(used),
                       nonzero_time_shifts=sum(r['time_shift'] != 0 for r in used),
                       sha256=digest(opt.sampler_log))
    patch = ''.join(difflib.unified_diff(original.splitlines(True), fixed_text.splitlines(True),
                                       fromfile='a/gaussian_renderer/__init__.py', tofile='b/gaussian_renderer/__init__.py'))
    opt.patch_output.parent.mkdir(parents=True, exist_ok=True)
    opt.patch_output.write_text(patch)
    result = dict(status='PASS_REPRODUCED_STATIC_TIME_SHIFT_ARGUMENT_DROPPED', GPU_used=False,
                  original_engine_modified=False, optimization_updates=0,
                  renderer=dict(path=str(opt.renderer), sha256=digest(opt.renderer)),
                  gaussian_model=dict(path=str(opt.model_source), sha256=digest(opt.model_source)),
                  removed_assignment_lines=sorted(drop), branches=rows, sampler_log=sampler,
                  candidate_patch=str(opt.patch_output),
                  interpretation='Original active static path drops a nonzero time_shift; dynamic honors it. Removing two unconditional assignments restores the existing shifted formula, leaves every None-shift output unchanged, and leaves the dynamic branch unchanged.',
                  limitation='CPU interface proof only. Existing trained checkpoint renders normally use time_shift=None, so this fix does not repair saved a5 geometry by itself. A controlled training replay is required to test a5 causality.')
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
