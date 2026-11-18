"""共享材质持久化的行为测试：world查询/保护边界/过期缓存；不加载CUDA。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pytest
import torch
from drivingeditor_dynamic_edit.road_multiscale_material import WorldMaterialField
from drivingeditor_dynamic_edit.road_material_chart import make_exact_sample_plan
from drivingeditor_dynamic_edit.road_multiscale_asset import load_shared_road_material


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_asset(root):
    y,x=np.mgrid[:4,:4];keys=np.stack((x,y),-1).reshape(-1,2)
    surface=np.stack((x*.02,y*.02,np.zeros_like(x)),axis=-1)
    grid=np.arange(16).reshape(4,4);affine=np.eye(4);affine[:2,:2]*=.02
    plan=make_exact_sample_plan(surface,grid,keys,np.arange(16),np.tile([0.,0.,1.],(16,1)),
        np.tile([.013,.013,.003],(16,1)),affine,order=15,world_xyz=surface.reshape(-1,3))
    model=WorldMaterialField(np.full((4,4,3),.2),np.ones((4,4),bool),np.full((4,4),.25),plan)
    with torch.no_grad():model.low.fill_(.5)
    model.save(root/'world_field.pt',dict(role='synthetic CPU behavior fixture'))
    eligible=np.arange(16)<8;legacy=np.full((16,3),.125,np.float32)
    colors=legacy.copy()
    with torch.no_grad():colors[eligible]=model.query()[0].numpy()[eligible]
    np.savez(root/'road.npz',active=np.ones(16,bool),reference_srgb=colors)
    np.savez(root/'provenance.npz',capacity_source_prior_reference_srgb=legacy,capacity_color_eligible=eligible)
    np.savez(root/'field_scope.npz',active_world_rows=np.arange(16),eligible=eligible)
    receipt=dict(field_sha256=sha(root/'world_field.pt'),derived_road_sha256=sha(root/'road.npz'),field_scope_sha256=sha(root/'field_scope.npz'))
    (root/'field_receipt.json').write_text(json.dumps(receipt))
    return colors,legacy,eligible


def test_authoritative_world_field_requery_and_readonly_scope(tmp_path):
    colors,legacy,eligible=make_asset(tmp_path)
    asset=load_shared_road_material(tmp_path)
    out=asset.query_colors().numpy()
    np.testing.assert_allclose(out,colors,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(out[~eligible],legacy[~eligible])
    assert {k for k,_ in asset.model.named_parameters()}=={'low','high'}
    assert not any(p.requires_grad for p in asset.model.parameters())


def test_stale_cache_requires_explicit_reexport(tmp_path):
    _,legacy,eligible=make_asset(tmp_path);asset=load_shared_road_material(tmp_path)
    with torch.no_grad():asset.model.low.add_(.1)
    with pytest.raises(ValueError,match='派生Gaussian缓存'):asset.query_colors()
    out=asset.query_colors(verify_derived=False).numpy()
    np.testing.assert_array_equal(out[~eligible],legacy[~eligible])


def test_readonly_change_rejected_even_if_body_digest_updated(tmp_path):
    colors,_,_=make_asset(tmp_path);colors[15]=.8
    np.savez(tmp_path/'road.npz',active=np.ones(16,bool),reference_srgb=colors)
    r=json.loads((tmp_path/'field_receipt.json').read_text());r['derived_road_sha256']=sha(tmp_path/'road.npz')
    (tmp_path/'field_receipt.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='只读颜色'):load_shared_road_material(tmp_path)
