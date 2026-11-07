"""Strict CPU tests of the delivered screen-radius candidate; no engine writes.

Run with the existing drivingeditor Python. The candidate patch is applied to
source text in memory. Source methods are executed with device='cuda' adapted
to 'cpu' only; real lineage helpers, optimizer remapping, clone/split, and SGCM
refine are retained. No optimizer.step or CUDA initialization is requested.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import ast
import copy
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import types
import numpy as np
import torch
from torch import nn

ORIGINAL_SHA = 'e5da10984f8836da1ac51c975823c3ca89bbf8c56f45ec4d4dcb071c06aafa72'
CANDIDATE_SHA = '3b3c5f43ea46342f0e36b461af04fd1d2e52afead24d9953673894dff788ce1a'
PATCH_SHA = '939cad8c1568eb0bbf5a70faf350dceb2f2962b37260066181cc1713ff72f58e'
ENGINE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
FIELDS = dict(xyz='_xyz', f_dc='_features_dc', f_rest='_features_rest',
              scaling='_scaling', rotation='_rotation', opacity='_opacity',
              t='_t', scaling_t='_scaling_t', velocity='_velocity')
BUFFERS = ['max_radii2D', 'xyz_gradient_accum', 't_gradient_accum', 'denom',
           '_sgcm_lineage_ids', '_sgcm_clone_blocked']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def apply_unified(original, patch):
    """Apply this single-file unified diff in memory, checking every context."""
    rows, changes = original.splitlines(True), patch.splitlines(True)
    assert changes[0].strip() == '--- a/scene/gaussian_model.py'
    assert changes[1].strip() == '+++ b/scene/gaussian_model.py'
    output, old_cursor, index = [], 0, 2
    while index < len(changes):
        match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\r\n]*\r?\n?', changes[index])
        assert match, ('Invalid hunk header', changes[index])
        start, old_count, new_start, new_count = (int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1))
        assert start - 1 >= old_cursor
        output.extend(rows[old_cursor:start - 1])
        old_cursor = start - 1
        assert len(output) == new_start - 1
        used_old = used_new = 0
        index += 1
        while index < len(changes) and not changes[index].startswith('@@ '):
            line = changes[index]
            kind, body = line[:1], line[1:]
            assert kind in [' ', '+', '-'], ('Unsupported patch line', line)
            if kind in [' ', '-']:
                assert rows[old_cursor] == body, ('Patch context differs', old_cursor + 1)
                old_cursor += 1
                used_old += 1
            if kind in [' ', '+']:
                output.append(body)
                used_new += 1
            index += 1
        assert (used_old, used_new) == (old_count, new_count)
    output.extend(rows[old_cursor:])
    return ''.join(output)


class CPUDevice(ast.NodeTransformer):
    def visit_keyword(self, node):
        node = self.generic_visit(node)
        if node.arg == 'device' and isinstance(node.value, ast.Constant) and node.value.value == 'cuda':
            node.value = ast.Constant(value='cpu')
        return node


def load_methods(text, engine, name):
    module = types.ModuleType(name)
    namespace = module.__dict__
    namespace.update(torch=torch, np=np, nn=nn, math=math, Path=Path, dataclass=dataclass,
                     asdict=asdict, BasicPointCloud=object,
                     inverse_sigmoid=lambda value: torch.log(value / (1 - value)))
    sys.modules[name] = module
    helpers = ast.parse((engine / 'scene/hybrid_sgcm_helpers.py').read_text())
    helper_nodes = [node for node in helpers.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    utility = ast.parse((engine / 'utils/general_utils.py').read_text())
    rotation = next(node for node in utility.body if isinstance(node, ast.FunctionDef) and node.name == 'build_rotation')
    model = next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == 'GaussianModel')
    sgcm = ast.parse((engine / 'scene/paper_sgcm.py').read_text())
    sgcm_nodes = [node for node in sgcm.body if isinstance(node, ast.ClassDef)]
    selected = helper_nodes + [rotation, model] + sgcm_nodes
    adapted = CPUDevice().visit(ast.Module(body=copy.deepcopy(selected), type_ignores=[]))
    exec(compile(ast.fix_missing_locations(adapted), name + ':CPU_DEVICE_ONLY', 'exec'), namespace)
    return namespace


def make(namespace, n=3, radii=None):
    cls = namespace['GaussianModel']
    model = cls.__new__(cls)
    model.setup_functions()
    model.contract = False
    model.percent_dense = .01
    model.t_grad = False
    model.no_time_split = True
    model.big_point_threshold = .1
    model.T = .2
    model.velocity_decay = 1.
    values = dict(xyz=torch.tensor([[0., 0., .35], [.1, 0., .1], [.2, 0., .8]])[:n].clone(),
                  f_dc=torch.arange(n * 3, dtype=torch.float32).reshape(n, 1, 3) / 10,
                  f_rest=torch.zeros(n, 15, 3), scaling=torch.full((n, 3), math.log(.001)),
                  rotation=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1), opacity=torch.zeros(n, 1),
                  t=torch.zeros(n, 1), scaling_t=torch.full((n, 1), math.log(.3)), velocity=torch.zeros(n, 3))
    groups = []
    for name, value in values.items():
        parameter = nn.Parameter(value)
        setattr(model, FIELDS[name], parameter)
        groups.append(dict(params=[parameter], name=name))
    model.optimizer = torch.optim.Adam(groups, lr=.001)
    for group_index, group in enumerate(model.optimizer.param_groups):
        parameter = group['params'][0]
        moment = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape) + group_index * 100 + 1
        model.optimizer.state[parameter] = dict(step=torch.tensor(7.), exp_avg=moment.clone(), exp_avg_sq=moment.square())
    model.max_radii2D = torch.tensor(radii or [11., 22., 33.][:n])
    model.xyz_gradient_accum = torch.arange(1, n + 1, dtype=torch.float32)[:, None]
    model.t_gradient_accum = torch.arange(2, n + 2, dtype=torch.float32)[:, None]
    model.denom = torch.ones(n, 1)
    model._sgcm_lineage_ids = torch.arange(10, 10 + n, dtype=torch.int64)
    model._sgcm_clone_blocked = torch.tensor([False, True, False][:n])
    return model


def snapshot(model):
    state = dict(parameters={name: getattr(model, attr).detach().clone() for name, attr in FIELDS.items()},
                 buffers={name: getattr(model, name).detach().clone() for name in BUFFERS}, optimizer={})
    for group in model.optimizer.param_groups:
        state['optimizer'][group['name']] = {key: value.clone() if torch.is_tensor(value) else value
                                             for key, value in model.optimizer.state[group['params'][0]].items()}
    return state


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(equal(a[key], b[key]) for key in a)
    return a == b


def check(model):
    n = len(model.get_xyz)
    assert model.max_radii2D.shape == (n,)
    assert model.max_radii2D.dtype == torch.float32 and model.max_radii2D.device.type == 'cpu'
    for name in ['xyz_gradient_accum', 't_gradient_accum', 'denom']:
        assert getattr(model, name).shape == (n, 1)
    for group in model.optimizer.param_groups:
        parameter = group['params'][0]
        assert parameter is getattr(model, FIELDS[group['name']]) and parameter.shape[0] == n
        state = model.optimizer.state[parameter]
        assert state['exp_avg'].shape == state['exp_avg_sq'].shape == parameter.shape
        assert state['step'].item() == 7
    assert model._sgcm_lineage_ids.shape == model._sgcm_clone_blocked.shape == (n,)
    return n


def no_admission(model):
    model.xyz_gradient_accum.zero_()
    model.t_gradient_accum.zero_()


def generic(model, threshold, prune_only=False):
    model.densify_and_prune(1., .005, 1., threshold, .002, prune_only=prune_only)
    return check(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, default=ENGINE)
    parser.add_argument('--patch', type=Path, default=Path(__file__).with_name('screen_radius_candidate.patch'))
    options = parser.parse_args()
    source_path = options.engine / 'scene/gaussian_model.py'
    original_bytes = source_path.read_bytes()
    assert digest(original_bytes) == ORIGINAL_SHA
    override = globals().get('PATCH_TEXT_OVERRIDE')
    patch_bytes = override.encode('utf-8') if override is not None else options.patch.read_bytes()
    assert digest(patch_bytes) == PATCH_SHA
    original = original_bytes.decode('utf-8')
    candidate = apply_unified(original, patch_bytes.decode('utf-8'))
    assert digest(candidate.encode()) == CANDIDATE_SHA
    def functions(text):
        cls = next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == 'GaussianModel')
        return {node.name: ast.dump(node, include_attributes=False) for node in cls.body if isinstance(node, ast.FunctionDef)}
    left, right = functions(original), functions(candidate)
    changed = {name for name in left if left[name] != right[name]}
    assert changed == {'densification_postfix', 'densify_and_prune'}
    namespaces = [load_methods(original, options.engine, 'screen_original_cpu'),
                  load_methods(candidate, options.engine, 'screen_candidate_cpu')]
    reports = []

    # Original two-point failure becomes the requested threshold behavior.
    counts, models, rng = [], [], []
    for namespace in namespaces:
        model = make(namespace, n=2, radii=[100., 1.]); no_admission(model)
        before = snapshot(model)
        torch.manual_seed(31)
        counts.append(generic(model, 20.))
        rng.append(torch.get_rng_state().clone()); models.append(model)
        if namespace is namespaces[1]:
            assert model.max_radii2D.tolist() == [0.]
            assert model._sgcm_lineage_ids.tolist() == [11]
            for group in model.optimizer.param_groups:
                name = group['name']; state = model.optimizer.state[group['params'][0]]
                assert torch.equal(state['exp_avg'], before['optimizer'][name]['exp_avg'][1:])
    assert counts == [2, 1] and torch.equal(*rng)
    reports.append(dict(case='original_failure_fixed', original_survivors=counts[0], candidate_survivors=counts[1],
                        radii=[100, 1], threshold=20, RNG_equal=True, optimizer_survivor_moments_preserved=True))

    # None/0: no extra screen/world geometric pruning; normal window still resets.
    for threshold in [None, 0.]:
        states = []
        for namespace in namespaces:
            model = make(namespace, n=2, radii=[100., 1.]); no_admission(model)
            model._scaling.data.fill_(math.log(.5))
            assert generic(model, threshold) == 2
            assert model.max_radii2D.tolist() == [0., 0.]
            states.append(snapshot(model))
        assert equal(*states)
        reports.append(dict(case='normal_disabled_threshold', threshold=threshold, survivors=2,
                            no_extra_geometric_prune=True, normal_window_reset=True, all_state_equal_to_original=True))

    model = make(namespaces[1], n=2, radii=[100., 1.]); no_admission(model)
    generic(model, None)  # e.g. old normal3000 node, where screen pruning is off.
    model.max_radii2D = torch.maximum(model.max_radii2D, torch.tensor([5., 3.]))
    assert generic(model, 20.) == 2
    reports.append(dict(case='disabled_to_enabled_threshold_no_old_window_inheritance',
                        prior_interval_radii=[100, 1], next_interval_radii=[5, 3], threshold=20,
                        survivors=2, prior_interval_did_not_trigger_late_prune=True))

    # prune_only keeps original survivor radius history, for every threshold.
    for threshold in [None, 0., 20.]:
        states = []
        for namespace in namespaces:
            model = make(namespace, n=2, radii=[100., 1.])
            generic(model, threshold, prune_only=True)
            states.append(snapshot(model))
        assert equal(*states)
        expected = [1.] if threshold else [100., 1.]
        assert states[1]['buffers']['max_radii2D'].tolist() == expected
        reports.append(dict(case='prune_only_original_semantics', threshold=threshold,
                            survivor_radii=expected, no_new_window_reset=True, all_state_equal_to_original=True))

    # Empty and real append: old rows keep exact radii; new rows have zero.
    for parents in [[], [2, 0]]:
        states, rng = [], []
        for namespace in namespaces:
            model = make(namespace); before = snapshot(model)
            parent = torch.tensor(parents, dtype=torch.int64)
            children = [getattr(model, FIELDS[name])[parent].detach().clone() for name in
                        ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 't', 'scaling_t', 'velocity']]
            torch.manual_seed(39)
            initial_rng = torch.get_rng_state().clone()
            model.densification_postfix(*children, sgcm_parent_indices=parent)
            check(model)
            assert torch.equal(initial_rng, torch.get_rng_state())
            states.append(snapshot(model)); rng.append(torch.get_rng_state().clone())
            if namespace is namespaces[1]:
                assert model.max_radii2D.tolist() == [11., 22., 33.] + [0.] * len(parents)
                assert model._sgcm_lineage_ids.tolist() == [10, 11, 12] + [10 + p for p in parents]
                for group in model.optimizer.param_groups:
                    state = model.optimizer.state[group['params'][0]]
                    assert torch.equal(state['exp_avg'][:3], before['optimizer'][group['name']]['exp_avg'])
                    assert torch.count_nonzero(state['exp_avg'][3:]).item() == 0
                    assert torch.count_nonzero(state['exp_avg_sq'][3:]).item() == 0
        states[0]['buffers'].pop('max_radii2D'); states[1]['buffers'].pop('max_radii2D')
        assert equal(*states) and torch.equal(*rng)
        reports.append(dict(case='direct_postfix_append', added_rows=len(parents),
                            candidate_radii=[11, 22, 33] + [0] * len(parents),
                            old_moments_preserved_new_moments_zero=True, unrelated_buffers_and_RNG_unchanged=True))

    # Actual clone followed by split: preserve padding/order and filter parent rows.
    for temporal in [False, True]:
        states, rng = [], []
        expected_lineage = [10, 12, 10] + [11] * (3 if temporal else 2)
        expected_radii = [11., 33.] + [0.] * (4 if temporal else 3)
        for namespace in namespaces:
            model = make(namespace); model._scaling.data[1].fill_(math.log(.02))
            model.t_grad = temporal; model.no_time_split = not temporal
            torch.manual_seed(41)
            grads = torch.tensor([[2.], [2.], [0.]])
            model.densify_and_clone(grads, 1., 1., torch.zeros(3, 1), .002, time_clone=temporal)
            check(model)
            model.densify_and_split(grads, 1., 1., torch.zeros(3, 1), .002, time_split=temporal)
            assert check(model) == len(expected_lineage)
            assert model._sgcm_lineage_ids.tolist() == expected_lineage
            if namespace is namespaces[1]:
                assert model.max_radii2D.tolist() == expected_radii
            states.append(snapshot(model)); rng.append(torch.get_rng_state().clone())
        states[0]['buffers'].pop('max_radii2D'); states[1]['buffers'].pop('max_radii2D')
        assert equal(*states) and torch.equal(*rng)
        reports.append(dict(case='actual_clone_then_split_parent_delete', temporal=temporal, old_rows=3,
                            final_rows=len(expected_lineage), lineage_ids=expected_lineage, candidate_radii=expected_radii,
                            row_mapping_and_RNG_equal_except_intended_radius_buffer=True))

    # Normal path with actual clone/split consumes radii only after topology.
    for temporal in [False, True]:
        model = make(namespaces[1], radii=[100., 10., 15.]); model._scaling.data[1].fill_(math.log(.02))
        model.t_grad = temporal; model.no_time_split = not temporal
        model.xyz_gradient_accum = torch.tensor([[2.], [2.], [0.]])
        model.t_gradient_accum.zero_()
        old = snapshot(model); torch.manual_seed(43)
        expected_lineage = [12, 10] + [11] * (3 if temporal else 2)
        assert generic(model, 20.) == len(expected_lineage)
        assert model._sgcm_lineage_ids.tolist() == expected_lineage
        assert model.max_radii2D.tolist() == [0.] * len(expected_lineage)
        for group in model.optimizer.param_groups:
            state = model.optimizer.state[group['params'][0]]
            assert torch.equal(state['exp_avg'][:1], old['optimizer'][group['name']]['exp_avg'][2:3])
            assert torch.count_nonzero(state['exp_avg'][1:]).item() == 0
        reports.append(dict(case='normal_actual_additions_screen_prune_and_reset', temporal=temporal,
                            final_rows=len(expected_lineage), lineage_ids=expected_lineage,
                            survivor_radii=[0] * len(expected_lineage), old_survivor_moments_and_new_zeros_correct=True))

    # Execute actual SGCM refine: clone row0, split+remove row1, prune row2.
    for guard_enabled, no_addition in [(False, False), (False, True), (True, False), (True, True)]:
        states, rng = [], []
        for namespace in namespaces:
            model = make(namespace)
            cfg = namespace['SGCMConfig'](guard_enabled=guard_enabled, max_new_points=10, max_points=100)
            region = np.array([[10., 10., 10.], [11., 11., 11.]]) if no_addition else np.array([[-2., -2., -2.], [2., 2., 2.]])
            controller = namespace['SGCMController'](cfg, 1., np.array([0., 0., 1., 0.]), np.eye(4), [region], {})
            if guard_enabled:
                world_view = np.eye(4); world_view[3, 2] = 2.
                controller.training_views = (dict(world_view=world_view, full_proj=np.eye(4),
                                                   timestamp=0., width=480, height=320, id='tiny_cpu_view'),)
            radii = torch.tensor([5., 30., 2.])
            torch.manual_seed(47); initial_rng = torch.get_rng_state().clone()
            receipt = controller.refine(model, radii, 550, visibility_filter=radii > 0)
            assert torch.equal(initial_rng, torch.get_rng_state())
            check(model); states.append(snapshot(model)); rng.append(torch.get_rng_state().clone())
        assert equal(*states) and torch.equal(*rng)
        expected = [11., 22., 33.] if no_addition else [11., 0., 0., 0.]
        assert states[1]['buffers']['max_radii2D'].tolist() == expected
        reports.append(dict(case='actual_SGCM_refine', guard_enabled=guard_enabled, no_addition=no_addition, receipt=receipt,
                            candidate_radii=expected, every_parameter_buffer_optimizer_and_RNG_equal_to_original=True))

    model = make(namespaces[1]); no_admission(model); model._opacity.data.fill_(-20.)
    assert generic(model, 20.) == 0 and model.max_radii2D.shape == (0,)
    reports.append(dict(case='all_rows_pruned_empty_buffer_alignment', rows=0, optimizer_shapes_valid=True))

    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASS_STRICT_CPU_SCREEN_RADIUS_CANDIDATE', GPU_used=False,
                         cuda_initialized=False, optimization_updates=0, engine_files_modified=False,
                         original_sha256=ORIGINAL_SHA, candidate_sha256=CANDIDATE_SHA, patch_sha256=PATCH_SHA,
                         changed_methods=sorted(changed), cases=reports, case_count=len(reports),
                         policy='Postfix preserves old-row radii and appends zeros; normal generic pruning defers its existing window reset until after all predicates and row filtering, including None/0; prune_only retains its original survivor history.',
                         native_callers=[dict(path=str(options.engine / 'train.py'), line=779, role='normal generic caller; prune_only defaults False'),
                                         dict(path=str(source_path), lines=[652, 693], role='split and clone call postfix; their row order and gradient padding are unchanged'),
                                         dict(path=str(options.engine / 'scene/paper_sgcm.py'), line=317, role='SGCM append restores its saved old statistics, then filters removed parents; tested with guard on and off')],
                         untouched='Threshold values, loss, sampling, clone/split order, other buffer-reset rules, parameter values and optimizer remapping code are unchanged.',
                         source_helper_sha256={str(path): digest(path.read_bytes()) for path in
                             [source_path, options.engine / 'scene/hybrid_sgcm_helpers.py', options.engine / 'scene/paper_sgcm.py', options.engine / 'utils/general_utils.py']},
                         limitations=['CPU interface/bookkeeping validation only; no GPU replay or a5 causal result is claimed.',
                                      'Restoring the existing20px criterion may substantially alter trained scene topology; candidate is not applied to any old or running engine.']), indent=2))


if __name__ == '__main__':
    main()
