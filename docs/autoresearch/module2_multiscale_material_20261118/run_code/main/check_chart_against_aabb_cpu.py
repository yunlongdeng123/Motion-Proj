"""实现测试+外部AABB locator交叉核；不代替root/main对helper的同行审。"""
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
from road_chart_sampling import StructuredRoadChart, exact_center_uv, local_chart_plan

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')
COVER=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded')


def load(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in keys}


def bilinear(field,uv):
    H,W=field.shape[:2];xy=np.asarray(uv).reshape(-1,2);lo=np.floor(xy).astype(np.int64);f=xy-lo
    x,y=lo.T;u,v=f.T;out=np.zeros((len(x),field.shape[-1]),np.float64)
    for dx,dy,w in [(0,0,(1-u)*(1-v)),(1,0,u*(1-v)),(0,1,(1-u)*v),(1,1,u*v)]:
        ok=(x+dx>=0)&(x+dx<W)&(y+dy>=0)&(y+dy<H)
        out[ok]+=field[y[ok]+dy,x[ok]+dx]*w[ok,None]
    return out


def main():
    start=time.monotonic()
    a=load(BASE/'base_atlas.npz',['surface_world_xyz','grid_world_row','world_from_uv'])
    p=load(COVER/'provenance.npz',['world_keys','world_xyz'])
    r=load(COVER/'road.npz',['active','normal_world','gaussian_sigma_m'])
    A0=load(P/'cpu_initial_r1/A0.npz',['rgb_srgb'])['rgb_srgb']
    chart=StructuredRoadChart(a['surface_world_xyz'],a['world_from_uv'])
    active=np.flatnonzero(r['active']);center,center_receipt=exact_center_uv(p['world_keys'],a['grid_world_row'],active)
    node=bilinear(A0,center);u,v=center.astype(np.int64).T
    node_exact=np.array_equal(node,A0[v,u].astype(np.float64))
    # 独立locator只建一个真实128²局部patch的canonical faces，不用global affine。
    x0,x1,y0,y1=0,128,78,206
    patch=a['surface_world_xyz'][y0:y1,x0:x1];H,W=patch.shape[:2]
    yy,xx=np.mgrid[:H-1,:W-1];aa=(yy*W+xx).ravel()
    faces=np.stack((np.stack((aa,aa+1,aa+W+1),1),np.stack((aa,aa+W+1,aa+W),1)),1).reshape(-1,3)
    modpath=P/'trellis2/code/exact_mesh_uv.py';spec=importlib.util.spec_from_file_location('independent_AABB_locator',modpath)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    locator=mod.TriangleLocatorXY(patch.reshape(-1,3),faces)
    inside=(center[:,0]>=x0+8)&(center[:,0]<=x1-9)&(center[:,1]>=y0+8)&(center[:,1]<=y1-9)
    candidates=active[inside];rows=candidates[np.linspace(0,len(candidates)-1,256,dtype=np.int64)]
    plan=local_chart_plan(a['surface_world_xyz'],a['grid_world_row'],p['world_keys'],rows,
        r['normal_world'][rows],r['gaussian_sigma_m'][rows],a['world_from_uv'],order=15,world_xyz=p['world_xyz'][rows])
    offset=np.einsum('nij,qj->nqi',plan['chol_world_xy'],plan['standard_nodes'])
    world=plan['world_xyz'][:,None,:2]+offset
    fast=chart.query(world,return_geometry=True);external=locator.locate(world)
    both=fast['valid']&external['valid'];assert both.all()
    vertex_uv=np.stack(np.meshgrid(np.arange(x0,x1),np.arange(y0,y1),indexing='xy'),-1).reshape(-1,2)
    true_uv=np.einsum('...i,...ij->...j',external['barycentric'],vertex_uv[faces[external['fixed_face_id']]])
    uv_error=np.max(np.abs(fast['uv']-true_uv),axis=-1)
    world_error=np.max(np.abs(fast['world_xyz']-external['world_xyz']),axis=-1)
    colors_fast=bilinear(A0,fast['uv']);colors_external=bilinear(A0,true_uv)
    inverse=np.linalg.inv(plan['local_worldXY_from_uv'])
    local=plan['center_uv'][:,None,:]+np.einsum('nij,nqj->nqi',inverse,offset)
    local_err=np.max(np.abs(local-true_uv),axis=-1)
    affine=(world-a['world_from_uv'][:2,3])@np.linalg.inv(a['world_from_uv'][:2,:2]).T
    affine_err=np.max(np.abs(affine-true_uv),axis=-1)
    q=lambda z:np.quantile(z,[0,.5,.95,.99,1]).tolist()
    result=dict(schema='true_chart_helper_external_locator_CPU_crosscheck.v1',GPU_executed=False,native_calls=0,
        review_role='implementation tests plus external algorithm crosscheck; root/main are peer reviewers of this fix',
        external_locator=str(modpath),node_center_rows=len(active),node_center_colors_exact=node_exact,
        center_receipt=center_receipt,source_tangent_sigmas_equal=bool(np.allclose(r['gaussian_sigma_m'][active,0],r['gaussian_sigma_m'][active,1],atol=1e-12,rtol=0)),
        rows=len(rows),actual_GH15_points=int(both.size),all_valid_in_both=True,
        fast_vs_AABB_UV_error_quantiles=q(uv_error),fast_vs_AABB_world_error_max_m=float(world_error.max()),
        fast_vs_AABB_A0_RGB_error_max=float(abs(colors_fast-colors_external).max()),
        exact_world_reconstruction_max_m=float(fast['worldXY_reconstruction_error'].max()),
        local_J_linearization_UV_error_quantiles=q(local_err),global_affine_UV_error_quantiles=q(affine_err),
        local_approximation_is_not_authoritative_query=True,formal_authorized=False,
        elapsed_seconds=time.monotonic()-start)
    assert node_exact and float(world_error.max())<1e-7 and float(uv_error.max())<1e-7
    out=P/'qa/chart_external_locator_crosscheck.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
