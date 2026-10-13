import os
import json
from pathlib import Path
import sys
import time
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013')
sys.path.insert(0,str(BASE/'infra'))
from priority import require_gpu_slot
slot,admission=require_gpu_slot('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
try:
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['OPENBLAS_NUM_THREADS']='4'
    sys.path.insert(0,str(BASE/'validation_package'))
    import ast
    import importlib.util
    import math
    from types import SimpleNamespace
    import torch
    import torch.nn.functional as F
    started=time.monotonic()
    # Reuse only tiny analytic input fixtures. Do not import pytest or replace
    # the installed CUDA rasterizer; this environment is deliberately unchanged.
    source=ast.parse((BASE/'infra/tests/test_renderer_contract.py').read_text(encoding='utf-8'))
    names={'camera','pipe','StaticPVG','WorldState','Instances'}
    definitions=ast.Module(body=[node for node in source.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names],type_ignores=[])
    namespace={'torch':torch,'F':F,'math':math,'SimpleNamespace':SimpleNamespace}
    exec(compile(definitions,'analytic_input_fixtures','exec'),namespace)
    spec=importlib.util.spec_from_file_location('actual_instance_renderer',BASE/'infra/renderer.py')
    renderer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    static=namespace['StaticPVG'](depth=1.,opacity=.6)
    model=namespace['Instances']()
    for obj in (static,model.state):
        for name,value in vars(obj).copy().items():
            if torch.is_tensor(value):
                setattr(obj,name,value.detach().cuda().requires_grad_(value.requires_grad))
    static.get_xyz_SHM=lambda timestamp:static.get_xyz
    result=renderer.render(namespace['camera'](device='cuda'),static,model,namespace['pipe'](),torch.zeros(3,device='cuda'))
    visible=result['dynamic_visible_alpha']
    assert visible.is_cuda and bool(torch.isfinite(result['render']).all())
    assert float(visible.max())>0 and float(visible.max())<float(result['dynamic_alpha'].max())
    torch.testing.assert_close(visible,result['instance_visible_contribution'].sum(0,keepdim=True),atol=2e-6,rtol=2e-5)
    visible.sum().backward()
    for gradient in (static.get_opacity.grad,model.state.opacity.grad):
        assert gradient is not None and bool(torch.isfinite(gradient).all()) and bool((gradient.abs()>0).any())
    receipt={'test':'actual_cuda_rasterizer_joint_occlusion_contract','status':'PASS','wall_seconds':time.monotonic()-started,
             'CUDA_initialized':torch.cuda.is_initialized(),'admission':admission,'scope':'small analytic occlusion ABI/gradient test; not scene quality',
             'visible_union':float(visible.max()),'isolated_union':float(result['dynamic_alpha'].max()),
             'static_opacity_gradient':static.get_opacity.grad.cpu().tolist(),'instance_opacity_gradient':model.state.opacity.grad.cpu().tolist(),
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    with (BASE/'metadata/actual_cuda_contract_r1.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2)
    print(json.dumps(receipt))
finally:
    slot.close()
