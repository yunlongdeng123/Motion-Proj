"""仅重用冻结geometry缓存，三角bary→真实chart UV及局部pixel微分；不读任何teacherRGB。"""
from pathlib import Path
import argparse,hashlib,json,importlib.util,sys
import numpy as np

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
DEFAULT_GEOMETRY=B/'road_capacity_cpu_20261117_r1/ideal_targets_r2/geometry'
DEFAULT_ATLAS=B/'road_asset_20261116_r1/phase2/formal_r1/asset/base_atlas.npz'
Q=B/'road_teacher_detail_cpu_20261119_r1'
P=B/'road_multiscale_material_20261118_r1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for value in iter(lambda:f.read(4*1024**2),b''):h.update(value)
    return h.hexdigest()


def prepare_frame(source,world_xyz,world_uv):
    allowed=['surface_world_xyz','canonical_mesh_hit_valid','triangle_world_rows','triangle_weights',
        'pixel_world_xy_jacobian','camera_K','world_to_camera','fixed_evaluation_domain','plane_fallback','K']
    with np.load(source,allow_pickle=False)as z:d={k:z[k]for k in allowed}
    valid=d['canonical_mesh_hit_valid'].copy();yy,xx=np.nonzero(valid);ids=d['triangle_world_rows'][valid];w=d['triangle_weights'][valid]
    if np.any(ids<0)or np.any(ids>=len(world_xyz))or not np.allclose(w.sum(1),1,atol=1e-12,rtol=0):raise ValueError('无效canonical三角形身份/weights')
    vertices=world_xyz[ids];uv_vertices=world_uv[ids]
    X=(vertices*w[...,None]).sum(1);uv=(uv_vertices*w[...,None]).sum(1)
    reconstruction=float(np.abs(X-d['surface_world_xyz'][valid]).max())
    M=d['world_to_camera'];R=M[:3,:3];O=-R.T@M[:3,3];cam=X@R.T+M[:3,3];K=d['camera_K']
    projected=cam[:,:2]/cam[:,2:]*np.diag(K)[:2]+K[:2,2]
    offset=float(np.median(projected-np.stack((xx,yy),1)))
    offset=round(offset*2)/2
    reprojection=float(np.abs(projected-(np.stack((xx,yy),1)+offset)).max())
    direction=np.c_[((np.stack((xx,yy),1)+offset)-K[:2,2])/np.diag(K)[:2],np.ones(len(X))]@R
    normal=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]);den=(normal*direction).sum(1)
    t=(normal*(vertices[:,0]-O)).sum(1)/den;hit=O+t[:,None]*direction
    ray_error=float(np.abs(hit-X).max());Jxy=np.empty((len(X),2,2))
    for axis in(0,1):
        dr=R.T[:,axis]/K[axis,axis];dt=-t*(normal@dr)/den
        Jxy[:,:,axis]=(t[:,None]*dr+direction*dt[:,None])[:,:2]
    Ex=np.stack((vertices[:,1,:2]-vertices[:,0,:2],vertices[:,2,:2]-vertices[:,0,:2]),-1)
    Eu=np.stack((uv_vertices[:,1]-uv_vertices[:,0],uv_vertices[:,2]-uv_vertices[:,0]),-1)
    Juv=np.einsum('nij,njk->nik',Eu@np.linalg.inv(Ex),Jxy)
    if max(reconstruction,ray_error)>1e-7 or reprojection>1e-4 or not np.isfinite(Juv).all():raise ValueError('真G_s缓存重建/投影/微分未过')
    h,width=valid.shape;xyz=np.zeros((h,width,3),np.float64);center=np.full((h,width,2),-4.,np.float64);jac=np.zeros((h,width,2,2));depth=np.zeros((h,width))
    xyz[valid]=X;center[valid]=uv;jac[valid]=Juv;depth[valid]=cam[:,2]
    data=dict(surface_world_xyz=xyz,center_uv=center,pixel_uv_jacobian=jac,canonical_mesh_hit_valid=valid,
        triangle_world_rows=d['triangle_world_rows'],triangle_weights=d['triangle_weights'],camera_K=K,world_to_camera=M,
        fixed_evaluation_domain=d['fixed_evaluation_domain'],plane_fallback=d['plane_fallback'],K=d['K'],camera_depth=depth,
        pixel_center_offset=np.asarray(offset))
    receipt=dict(canonical_full=int(valid.sum()),fixed_D=int(d['fixed_evaluation_domain'].sum()),
        exact_D=int((valid&d['fixed_evaluation_domain']).sum()),fallback_D=int((d['plane_fallback']&d['fixed_evaluation_domain']).sum()),
        triangle_XYZ_reconstruction_max_m=reconstruction,pinhole_reprojection_max_px=reprojection,ray_triangle_plane_hit_max_m=ray_error,
        cached_Jxy_transposed_vs_actual_local_triangle_max=float(np.abs(Jxy-d['pixel_world_xy_jacobian'][valid].swapaxes(-1,-2)).max()),
        Jacobian_axes='new map[J_uv output,pixel input]; old cache[pixel input,worldXY output] transposed before comparison',
        pixel_center_offset=offset,read_keys_only=allowed,teacher_RGB_or_ideal_marking_labels_read=False)
    return data,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-root',type=Path,default=DEFAULT_GEOMETRY);parser.add_argument('--atlas',type=Path,default=DEFAULT_ATLAS)
    parser.add_argument('--output',type=Path,default=Q/'pixel_render/maps');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('保留旧地图，输出须为新目录')
    with np.load(args.atlas,allow_pickle=False)as a:surface=a['surface_world_xyz'];grid=a['grid_world_row'];affine=a['world_from_uv']
    vv,uu=np.indices(grid.shape);n=int(grid.max())+1
    good=grid.ravel()>=0
    world_xyz=np.full((n,3),np.nan);world_uv=np.full((n,2),np.nan);world_xyz[grid.ravel()[good]]=surface.reshape(-1,3)[good];world_uv[grid.ravel()[good]]=np.stack((uu.ravel(),vv.ravel()),1)[good]
    args.output.mkdir(parents=True);rows=[];authority={str(args.atlas):sha(args.atlas)}
    # 仅160个真实点用既有全scene 9邻居chart反查交叉核，不遍历大mesh。
    helper=P/'code/road_chart_sampling.py';spec=importlib.util.spec_from_file_location('road_chart_sampling_geometry_only',helper)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);chart=mod.StructuredRoadChart(surface,affine)
    masks_path=B/'road_activation_20261118_r1/qa/material_masks.npz'
    with np.load(masks_path,allow_pickle=False)as material:
        for raw in(86,110,134,179,198):
            source=args.geometry_root/f'{raw:03d}.npz';data,receipt=prepare_frame(source,world_xyz,world_uv)
            valid=data['canonical_mesh_hit_valid'];where=np.flatnonzero(valid);chosen=where[np.linspace(0,len(where)-1,32,dtype=int)]
            xy=data['surface_world_xyz'].reshape(-1,3)[chosen,:2];check=chart.query(xy)
            receipt['chart_spot_rows']=32;receipt['chart_spot_valid_all']=bool(check['valid'].all())
            receipt['chart_spot_UV_max']=float(np.abs(check['uv']-data['center_uv'].reshape(-1,2)[chosen]).max())
            if not receipt['chart_spot_valid_all']or receipt['chart_spot_UV_max']>1e-7:raise ValueError('canonical barycentric UV与exact chart不一致')
            if raw in(86,110):
                for label,key in [('fullS','full_S'),('new909','current_new_support_909')]:
                    mask=material[f'{raw:03d}_{key}'];receipt[label]=int(mask.sum());receipt[label+'_canonical_hit']=int((mask&valid).sum())
            receipt.update(raw=raw,source_geometry=str(source),source_sha256=sha(source));authority[str(source)]=receipt['source_sha256']
            np.savez_compressed(args.output/f'{raw:03d}.npz',**data);rows.append(receipt)
    result=dict(schema='pixel_world_canonical_geometry_CPU_maps.v1',frames=rows,source_geometry_sha256=authority,
        geometry_only=True,teacher_RGB_or_ideal_marking_labels_read=False,GPU_executed=False,native_calls=0,optimizer_updates=0,
        fixed_D=sum(r['fixed_D']for r in rows),exact_D=sum(r['exact_D']for r in rows),fallback_D=sum(r['fallback_D']for r in rows),
        cached_full_G_s_center_maps_ready=True,local_triangle_pixel_UV_differential_ready=True,
        actual_field_pixel_filter_convergence_qualified=False,native_basis_prepared=False,native_runtime_validated=False,formal_allowed=False,
        invalid_geometry_fallback='retain original eligible average-ci contribution; never replace with fitted plane',
        footprint_boundary='local canonical-face derivative; face-boundary/subpixel footprint qualification remains a separate future smoke gate')
    (args.output/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),fixed_D=result['fixed_D'],exact_D=result['exact_D'],fallback_D=result['fallback_D'],maps_ready=True,native_ready=False)))


if __name__=='__main__':main()
