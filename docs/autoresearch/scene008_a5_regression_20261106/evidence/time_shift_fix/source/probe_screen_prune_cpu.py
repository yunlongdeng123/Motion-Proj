"""Execute retained Gaussian topology methods on tiny CPU tensors.

The AST adapter changes device='cuda' to device='cpu' only. It imports no scene
package or CUDA extension and leaves topology/order/conditions unchanged.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn

SOURCE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137/scene/gaussian_model.py')


class CPUDevice(ast.NodeTransformer):
    def visit_keyword(self, node):
        node = self.generic_visit(node)
        if node.arg == 'device' and isinstance(node.value, ast.Constant) and node.value.value == 'cuda':
            node.value = ast.Constant(value='cpu')
        return node


def main():
    source = SOURCE.read_text()
    original_ast = ast.parse(source)
    definition = next(node for node in original_ast.body if isinstance(node, ast.ClassDef) and node.name == 'GaussianModel')
    utility_path = SOURCE.parent.parent / 'utils/general_utils.py'
    utility_ast = ast.parse(utility_path.read_text())
    rotation = next(node for node in utility_ast.body if isinstance(node, ast.FunctionDef) and node.name == 'build_rotation')
    module = ast.fix_missing_locations(ast.Module(body=[CPUDevice().visit(copy.deepcopy(rotation)),
                                                       CPUDevice().visit(copy.deepcopy(definition))], type_ignores=[]))
    namespace = dict(torch=torch, np=np, nn=nn, math=math, BasicPointCloud=object,
                     inverse_sigmoid=lambda value: torch.log(value / (1 - value)),
                     lineage_after_append=lambda *a: None, lineage_after_prune=lambda *a: None)
    exec(compile(module, str(SOURCE) + ':CPU_DEVICE_ONLY_ADAPTER', 'exec'), namespace)
    Model = namespace['GaussianModel']

    def model():
        m = Model.__new__(Model)
        m.setup_functions()
        m.contract = False
        m.percent_dense = .01
        m.t_grad = False
        m.no_time_split = True
        m.big_point_threshold = .1
        m.T = .2
        m.velocity_decay = 1.
        m._sgcm_lineage_ids = None
        values = dict(xyz=torch.tensor([[0., 0., 0.], [.1, 0., 0.]]),
                      f_dc=torch.zeros(2, 1, 3), f_rest=torch.zeros(2, 15, 3),
                      scaling=torch.full((2, 3), math.log(.001)),
                      rotation=torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                      opacity=torch.zeros(2, 1), t=torch.zeros(2, 1),
                      scaling_t=torch.full((2, 1), math.log(.3)), velocity=torch.zeros(2, 3))
        mapping = dict(xyz='_xyz', f_dc='_features_dc', f_rest='_features_rest',
                       scaling='_scaling', rotation='_rotation', opacity='_opacity',
                       t='_t', scaling_t='_scaling_t', velocity='_velocity')
        groups = []
        for name, value in values.items():
            parameter = nn.Parameter(value)
            setattr(m, mapping[name], parameter)
            groups.append(dict(params=[parameter], name=name))
        m.optimizer = torch.optim.Adam(groups, lr=.001)
        m.xyz_gradient_accum = torch.zeros(2, 1)
        m.t_gradient_accum = torch.zeros(2, 1)
        m.denom = torch.ones(2, 1)
        m.max_radii2D = torch.tensor([100., 1.])
        return m

    outcomes = []
    for prune_only in [False, True]:
        m = model()
        before = m.max_radii2D.clone()
        m.densify_and_prune(max_grad=1., min_opacity=.005, extent=1., max_screen_size=20., max_grad_t=.002, prune_only=prune_only)
        outcomes.append(dict(prune_only=prune_only, before_radii=before.tolist(),
                             expected_screen_size_pruned_rows=int((before > 20).sum()),
                             actual_pruned_rows=2 - len(m.get_xyz), survivor_rows=len(m.get_xyz),
                             after_radii=m.max_radii2D.tolist()))
    assert outcomes[0]['actual_pruned_rows'] == 0
    assert outcomes[1]['actual_pruned_rows'] == 1

    print(json.dumps(dict(status='PASS_CPU_SCREEN_PRUNE_BUG_REPRODUCED', GPU_used=False,
                         cuda_initialized=torch.cuda.is_initialized(), source=str(SOURCE),
                         source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                         adapter='Source AST unchanged except device=cuda -> cpu; no-lineage hooks are no-ops.',
                         screen_radius_bookkeeping=outcomes,
                         limitation='Shared baseline bug; this test does not establish a5 reconstruction causality.'), indent=2))


if __name__ == '__main__':
    main()
