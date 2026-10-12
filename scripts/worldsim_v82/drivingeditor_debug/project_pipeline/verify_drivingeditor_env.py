import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
from simple_knn._C import distCUDA2
import diff_gaussian_rasterization
import nvdiffrast.torch as dr

base = Path('/root/autodl-tmp/drivingeditor_repro')
torch.manual_seed(0)
x = torch.rand(1024, 3, device='cuda')
d = distCUDA2(x)
assert d.shape == (1024,) and torch.isfinite(d).all() and (d >= 0).all()
texture = torch.rand(1, 16, 16, 3, device='cuda', requires_grad=True)
uv = torch.rand(1, 8, 8, 2, device='cuda', requires_grad=True)
sample = dr.texture(texture, uv, filter_mode='linear')
sample.mean().backward()
assert torch.isfinite(texture.grad).all() and torch.isfinite(uv.grad).all()
sys.path.insert(0, str(base / 'DrivingEditor'))
from scene.envlight import EnvLight
light = EnvLight(32).cuda()
directions = torch.randn(1, 16, 16, 3, device='cuda')
light(directions).mean().backward()
assert light.base.grad is not None and torch.isfinite(light.base.grad).all()
from gaussian_renderer import render
from scene import Scene, GaussianModel, Dynamic_GaussianModel
torch.cuda.synchronize()
result = {
    'python': sys.version, 'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
    'platform': platform.platform(), 'simple_knn_cuda': 'pass',
    'nvdiffrast_texture_backward': 'pass', 'envlight_cube_backward': 'pass',
    'official_scene_renderer_import': 'pass',
    'upstream_commit': subprocess.check_output(['git', '-C', str(base / 'DrivingEditor'), 'rev-parse', 'HEAD'], text=True).strip(),
    'dependencies': {},
}
for path in (base / 'deps').iterdir():
    if (path / '.git').exists():
        result['dependencies'][path.name] = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
(base / 'logs/environment_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
