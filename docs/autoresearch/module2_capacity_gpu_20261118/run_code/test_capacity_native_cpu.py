"""单GPU租约与固定packet契约；不导入Torch/native或加载真实场景。"""
from pathlib import Path
import importlib.util
import json
import os
import sys
from types import SimpleNamespace
import numpy as np
import pytest

path=Path(os.environ.get('ROAD_CAPACITY_NATIVE_PATH',Path(__file__).with_name('road_capacity_native.py')))
spec=importlib.util.spec_from_file_location('road_capacity_native_cpu_test',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_lease_owner_required_and_revoked_before_native_import(tmp_path):
    path=tmp_path/'lease.json';path.write_text(json.dumps(dict(gpu_authorized=True,active=True,owner='static_handoff')))
    with pytest.raises(RuntimeError):m.load_context(lease_file=path,lease_owner='road_asset')
    assert m.assert_gpu_lease(path,'/root/static_handoff')['owner']=='static_handoff'
    path.write_text(json.dumps(dict(gpu_authorized=True,active=False,owner='static_handoff')))
    with pytest.raises(RuntimeError):m.assert_gpu_lease(path,'static_handoff')
    with pytest.raises(RuntimeError):m.assert_gpu_lease(None,'road_asset')

def test_import_does_not_load_native_package():
    assert 'drivingeditor_dynamic_edit' not in sys.modules
    assert 'diff_gaussian_rasterization' not in sys.modules

def test_color_forward_substitutes_only_rgb_without_mutating_template():
    ctx=m.NativeRoadContext.__new__(m.NativeRoadContext)
    ctx.native_calls=0;ctx.base_rgb=np.full((6,3),.2,np.float32)
    ctx._packets={86:dict(colors_precomp=ctx.base_rgb,means3D=np.zeros((6,3)),opacities=np.ones((6,1)),scales=np.ones((6,3)))}
    ctx.require_lease=lambda:None;ctx.camera=lambda raw:'camera'
    received=[]
    def raster(camera,pipe,unused,packet,background):received.append(packet);return {'rgb':packet['colors_precomp'].copy()}
    ctx.scene=SimpleNamespace(_bindings=lambda:SimpleNamespace(raster=raster),pipe='pipe',_background='black')
    value=np.full((6,3),.7,np.float32)
    result=ctx.render_road(86,value)
    assert np.array_equal(result['rgb'],value)
    assert ctx._packets[86]['colors_precomp'] is ctx.base_rgb
    for key in ('means3D','opacities','scales'):assert received[0][key] is ctx._packets[86][key]
    with pytest.raises(ValueError):ctx.render_road(86,np.zeros((5,3)))

def test_selected_npz_read_does_not_require_large_checkpoint(tmp_path):
    file=tmp_path/'sample.npz';np.savez(file,active=np.ones(5,bool),reference_srgb=np.zeros((5,3)),unused=np.arange(50))
    arrays=m.load_npz(file,('active',))
    assert set(arrays)=={'active'} and arrays['active'].all()
