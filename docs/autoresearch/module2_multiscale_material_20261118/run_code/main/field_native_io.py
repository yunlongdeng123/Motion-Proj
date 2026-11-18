"""显式opt-in共享field公共入口；road.npz仅为派生GS颜色缓存。"""
from pathlib import Path
import gc,json,sys
import numpy as np
import torch
from world_material_field import WorldMaterialField
from train_world_material import P,G,digest


def save_field_asset(state,output,record):
    output=Path(output);colors=state.colors().detach()
    world_mask=np.zeros(len(state.ctx.road['active']),bool);world_mask[state.ctx.active_world_rows]=state.eligible
    state.ctx.save_asset(colors,world_mask,output,training_record=dict(record,authoritative_shared_field=True,baked_ci_is_derived=True))
    metadata=dict(state.field_metadata,coverage_parent=state.contract['coverage_parent'],world_scope_file=str(output/'field_scope.npz'),
        source_input_binding=state.binding,shader_role='S_Sigma actual world field; cachedci derived; nonqualified/K legacy override')
    state.model.save(output/'world_field.pt',metadata)
    np.savez_compressed(output/'field_scope.npz',active_world_rows=state.ctx.active_world_rows,eligible=state.eligible)
    receipt=dict(schema='multiscale_authoritative_field_asset.v1',field='world_field.pt',field_sha256=digest(output/'world_field.pt'),
        derived_road='road.npz',derived_road_sha256=digest(output/'road.npz'),geometry_sigma_opacity_frozen=True,
        source_A0=str(P/'cpu_exact_r1/A0.npz'),source_A0_sha256=digest(P/'cpu_exact_r1/A0.npz'),
        source_geometry=str(P/'cpu_exact_r1/geometry_kernel.npz'),source_geometry_sha256=digest(P/'cpu_exact_r1/geometry_kernel.npz'),
        field_scope_sha256=digest(output/'field_scope.npz'),opt_in=True,default_scene_modified=False)
    (output/'field_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return colors


def verify_geometry_queries(model,metadata):
    """从原world chart/Σ重新构造固定抽样的query；不以bakedci当field真值。"""
    from road_chart_sampling import make_exact_sample_plan
    source=Path(metadata['coverage_parent'])
    with np.load(metadata['geometry_kernel']) as d:geo={k:d[k].copy() for k in ['active_world_rows','world_xyz','normal_world','sigma_m']}
    with np.load(metadata['A0_source']) as d:a={k:d[k].copy() for k in ['surface_world_xyz','grid_world_row','world_from_uv']}
    with np.load(source/'provenance.npz') as d:keys=d['world_keys'].copy()
    chosen=np.linspace(0,len(geo['active_world_rows'])-1,192,dtype=int)
    query=make_exact_sample_plan(a['surface_world_xyz'],a['grid_world_row'],keys,geo['active_world_rows'][chosen],
        geo['normal_world'][chosen],geo['sigma_m'][chosen],a['world_from_uv'],order=15,world_xyz=geo['world_xyz'][chosen])
    center=float(np.abs(model.center_uv[chosen].cpu().numpy()-query['center_uv']).max())
    quad=float(np.abs(model.sample_uv[chosen].cpu().numpy()-query['sample_uv']).max())
    valid=np.array_equal(model.geometric_sample_valid[chosen].cpu().numpy(),query['geometric_sample_valid'])
    if max(center,quad)>1e-7 or not valid:raise RuntimeError('serializedfield的physical query未重构一致')
    return dict(rebuilt_from_original_world_geometry=True,rows=192,center_UV_max=center,quad_UV_max=quad,geometric_valid_exact=valid)


class LoadedFieldContext:
    def __init__(self,base,model,colors,metadata,query_receipt):
        self.base=base;self.model=model;self.colors=colors;self.metadata=metadata;self.query_receipt=query_receipt
    def render_road(self,raw):return self.base.render_road(raw,self.colors)
    def render_joint(self,raw):return self.base.render_joint(raw,self.colors)
    def close(self):
        self.base.close();self.model=None;self.colors=None;gc.collect();torch.cuda.empty_cache()


def load_field_context(asset,*,lease_file,lease_owner):
    sys.path.insert(0,str(G/'code'))
    from road_capacity_native import load_context,load_npz
    asset=Path(asset);receipt=json.loads((asset/'field_receipt.json').read_text())
    if digest(asset/'world_field.pt')!=receipt['field_sha256'] or digest(asset/'road.npz')!=receipt['derived_road_sha256']:
        raise ValueError('field authority或派生缓存binding损坏')
    if digest(asset/'field_scope.npz')!=receipt['field_scope_sha256']:raise ValueError('field资格绑定损坏')
    if digest(receipt['source_A0'])!=receipt['source_A0_sha256'] or digest(receipt['source_geometry'])!=receipt['source_geometry_sha256']:
        raise ValueError('权威field原A0/geometry来源绑定损坏')
    model,metadata=WorldMaterialField.load(asset/'world_field.pt',device='cpu')
    geometry_receipt=verify_geometry_queries(model,metadata)
    ctx=load_context(asset_dir=asset,lease_file=lease_file,lease_owner=lease_owner)
    scope=load_npz(asset/'field_scope.npz')
    if not np.array_equal(scope['active_world_rows'],ctx.active_world_rows):raise ValueError('field/native world-row身份不同')
    parent=load_npz(Path(metadata['coverage_parent'])/'road.npz')
    for key in parent:
        if key!='reference_srgb' and not np.array_equal(parent[key],ctx.road[key]):raise ValueError('field/cache几何或覆盖不同')
    model=model.to(ctx.base_rgb.device).requires_grad_(False)
    indices=torch.as_tensor(np.flatnonzero(scope['eligible']),device=ctx.base_rgb.device)
    colors=torch.as_tensor(parent['reference_srgb'][ctx.active_world_rows],device=ctx.base_rgb.device).clone()
    with torch.no_grad():colors[indices]=model.query(indices=indices)[0]
    error=float((colors-ctx.base_rgb).abs().max())
    if error>1e-6:raise RuntimeError('从权威field重新query的ci与派生GS缓存不一致')
    return LoadedFieldContext(ctx,model,colors,metadata,dict(**geometry_receipt,requeried_ci_max_vs_baked_cache=error,
        world_field_is_authority=True,public_scene_loader_still_baked_GS_cache=True))
