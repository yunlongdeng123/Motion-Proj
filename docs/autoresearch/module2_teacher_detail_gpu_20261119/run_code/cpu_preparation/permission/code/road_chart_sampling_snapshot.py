"""真实道路chart的CPU采样计划；global affine仅用于候选cell索引，不作为真坐标。"""
import numpy as np


def exact_center_uv(world_keys, grid_world_row, world_row_ids):
    keys = np.asarray(world_keys, np.int64)
    grid = np.asarray(grid_world_row, np.int64)
    rows = np.asarray(world_row_ids, np.int64)
    origin = keys[grid[0, 0]]
    uv = keys[rows] - origin
    H, W = grid.shape
    if np.any(uv[:, 0] < 0) or np.any(uv[:, 0] >= W) or np.any(uv[:, 1] < 0) or np.any(uv[:, 1] >= H):
        raise ValueError('world_keys不符合原atlas key-origin/布局，禁止静默重排')
    if not np.array_equal(grid[uv[:, 1], uv[:, 0]], rows):
        raise ValueError('world_keys到grid_world_row不是同一个世界节点')
    return uv.astype(np.float64), dict(key_origin=origin.tolist(), rows=len(rows), center_is_exact_node=True)


def local_chart_jacobian(surface_world_xyz, center_uv):
    surface = np.asarray(surface_world_xyz, np.float64)[..., :2]
    H, W = surface.shape[:2]
    uv = np.asarray(center_uv)
    if not np.array_equal(uv, np.rint(uv)):
        raise ValueError('局部node Jacobian要求精确整数center_uv')
    u, v = np.rint(uv).astype(np.int64).T
    ul, ur = np.maximum(0, u-1), np.minimum(W-1, u+1)
    vl, vr = np.maximum(0, v-1), np.minimum(H-1, v+1)
    du = (surface[v, ur]-surface[v, ul])/(ur-ul)[:, None]
    dv = (surface[vr, u]-surface[vl, u])/(vr-vl)[:, None]
    J = np.stack((du, dv), axis=-1)
    if not np.isfinite(J).all() or np.any(np.abs(np.linalg.det(J)) < 1e-10):
        raise ValueError('真实局部chart Jacobian退化')
    return J


def world_tangent_covariance(normal_world, sigma_m, tangent_basis_world=None):
    normal = np.array(normal_world, np.float64, copy=True)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    sigma = np.asarray(sigma_m, np.float64)
    if tangent_basis_world is None:
        if not np.allclose(sigma[:, 0], sigma[:, 1], atol=1e-12, rtol=0):
            raise ValueError('各向异性切向Σ必须提供与native一致的tangent_basis_world，禁止忽略旋转')
        C = sigma[:, 0, None, None]**2*(np.eye(2)[None]-normal[:, :2, None]*normal[:, None, :2])
    else:
        B = np.asarray(tangent_basis_world, np.float64)
        if B.shape != (len(normal), 3, 2):
            raise ValueError('tangent_basis_world形状须N×3×2')
        C = np.einsum('nik,nk,njk->nij', B[:, :2], sigma[:, :2]**2, B[:, :2])
    if not np.isfinite(C).all() or np.any(np.linalg.det(C) <= 0):
        raise ValueError('worldXY tangent covariance不是正定')
    return C


def local_chart_plan(surface_world_xyz, grid_world_row, world_keys, world_row_ids,
                     normal_world, sigma_m, world_from_uv, *, order=15, world_xyz=None,
                     tangent_basis_world=None):
    center, receipt = exact_center_uv(world_keys, grid_world_row, world_row_ids)
    surface = np.asarray(surface_world_xyz, np.float64)
    ui, vi = center.astype(np.int64).T
    means = surface[vi, ui]
    if world_xyz is not None and not np.allclose(means, world_xyz, atol=1e-8, rtol=0):
        raise ValueError('当前Gaussian世界节点与source surface_world_xyz不同，不能借atlas改几何')
    J = local_chart_jacobian(surface, center)
    inverse = np.linalg.inv(J)
    Cworld = world_tangent_covariance(normal_world, sigma_m, tangent_basis_world)
    Cuv = np.einsum('nij,njk,nlk->nil', inverse, Cworld, inverse)
    x, w = np.polynomial.hermite.hermgauss(int(order))
    xx, yy = np.meshgrid(x, x, indexing='xy')
    wx, wy = np.meshgrid(w, w, indexing='xy')
    return dict(center_uv=center, chol_uv=np.linalg.cholesky(Cuv),
                local_worldXY_from_uv=J, world_covariance_xy=Cworld,
                chol_world_xy=np.linalg.cholesky(Cworld), world_xyz=means,
                standard_nodes=np.stack((xx.ravel(), yy.ravel()), axis=1)*np.sqrt(2),
                weights=(wx*wy).ravel()/np.pi,
                world_from_uv=np.asarray(world_from_uv, np.float64), order=int(order),
                center_receipt=receipt)


def _barycentric(p, a, b, c):
    e1, e2, d = b-a, c-a, p-a
    det = e1[:, 0]*e2[:, 1]-e1[:, 1]*e2[:, 0]
    safe = np.where(np.abs(det)>1e-14, det, 1.)
    w1 = (d[:, 0]*e2[:, 1]-d[:, 1]*e2[:, 0])/safe
    w2 = (e1[:, 0]*d[:, 1]-e1[:, 1]*d[:, 0])/safe
    w0 = 1-w1-w2
    valid = (np.abs(det)>1e-14)&(np.minimum(np.minimum(w0,w1),w2)>=-1e-9)
    return np.stack((w0,w1,w2),axis=1), valid


class StructuredRoadChart:
    """和canonical G_s一致的(00,10,11)/(00,11,01)实际worldXY三角形。"""
    def __init__(self, surface_world_xyz, world_from_uv):
        self.surface = np.asarray(surface_world_xyz, np.float64)
        self.H, self.W = self.surface.shape[:2]
        self.affine = np.asarray(world_from_uv, np.float64)
        self.seed_inverse = np.linalg.inv(self.affine[:2, :2])
        yy, xx = np.mgrid[:self.H, :self.W]
        expected = np.stack((xx,yy),axis=-1)@self.affine[:2,:2].T+self.affine[:2,3]
        residual_uv = (self.surface[...,:2]-expected)@self.seed_inverse.T
        self.seed_error_max_uv = float(np.max(np.abs(residual_uv)))
        if self.seed_error_max_uv >= .99:
            raise ValueError('global affine只能作邻cell seed；当前偏差>=1cell，需扩大独立加速索引')
        a,b,c,d = self.surface[:-1,:-1,:2],self.surface[:-1,1:,:2],self.surface[1:,1:,:2],self.surface[1:,:-1,:2]
        det0=(b-a)[...,0]*(c-a)[...,1]-(b-a)[...,1]*(c-a)[...,0]
        det1=(c-a)[...,0]*(d-a)[...,1]-(c-a)[...,1]*(d-a)[...,0]
        if not (np.all(det0>1e-12) and np.all(det1>1e-12)):
            raise ValueError('实际worldXY chart有退化/反转cell，局部邻cell索引不适用')

    def query(self, world_xy, *, chunk=262144, return_geometry=False):
        value = np.asarray(world_xy, np.float64)
        shape = value.shape[:-1]
        points = value.reshape(-1, value.shape[-1])[:, :2]
        uv = np.full((len(points),2), -4., np.float64)
        valid = np.zeros(len(points),bool)
        faces = np.full(len(points),-1,np.int64)
        weights = np.zeros((len(points),3),np.float64) if return_geometry else None
        hits = np.full((len(points),3),np.nan) if return_geometry else None
        errors = np.zeros(len(points),np.float64)
        offsets=((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1))
        for begin in range(0,len(points),chunk):
            p=points[begin:begin+chunk]
            seed=(p-self.affine[:2,3])@self.seed_inverse.T
            finite=np.isfinite(seed).all(1)
            cell=np.floor(np.where(finite[:,None],seed,0)).astype(np.int64)
            pending=finite.copy()
            for dx,dy in offsets:
                selected=np.flatnonzero(pending)
                if not len(selected):break
                x,y=cell[selected,0]+dx,cell[selected,1]+dy
                inside=(x>=0)&(x<self.W-1)&(y>=0)&(y<self.H-1)
                selected,x,y=selected[inside],x[inside],y[inside]
                if not len(selected):continue
                a=self.surface[y,x];b=self.surface[y,x+1];c=self.surface[y+1,x+1];d=self.surface[y+1,x]
                w0,v0=_barycentric(p[selected],a[:,:2],b[:,:2],c[:,:2])
                w1,v1=_barycentric(p[selected],a[:,:2],c[:,:2],d[:,:2])
                use0=v0;use1=~v0&v1;found=use0|use1
                if not found.any():continue
                sid=selected[found];gid=begin+sid
                w=np.where(use0[found,None],w0[found],w1[found])
                u=x[found]+np.where(use0[found],w[:,1]+w[:,2],w[:,1])
                v=y[found]+np.where(use0[found],w[:,2],w[:,1]+w[:,2])
                mapped=np.stack((u,v),axis=1)
                nearest=np.rint(mapped);mapped=np.where(np.abs(mapped-nearest)<5e-10,nearest,mapped)
                uv[gid]=mapped;valid[gid]=True
                faces[gid]=2*(y[found]*(self.W-1)+x[found])+use1[found].astype(np.int64)
                triangles=np.stack((a[found],np.where(use0[found,None],b[found],c[found]),
                                    np.where(use0[found,None],c[found],d[found])),axis=1)
                reconstructed=np.einsum('ni,nij->nj',w,triangles)
                errors[gid]=np.max(np.abs(reconstructed[:,:2]-p[sid]),axis=1)
                if return_geometry:weights[gid]=w;hits[gid]=reconstructed
                pending[sid]=False
        result=dict(uv=uv.reshape(*shape,2),valid=valid.reshape(shape),
                    triangle_face_id=faces.reshape(shape),worldXY_reconstruction_error=errors.reshape(shape))
        if return_geometry:result.update(barycentric=weights.reshape(*shape,3),world_xyz=hits.reshape(*shape,3))
        return result


def exact_triangle_uv(world_xy, surface_world_xyz, world_from_uv, *, chunk=262144, return_geometry=False):
    return StructuredRoadChart(surface_world_xyz,world_from_uv).query(world_xy,chunk=chunk,return_geometry=return_geometry)


def make_exact_sample_plan(surface_world_xyz, grid_world_row, world_keys, world_row_ids,
                           normal_world, sigma_m, world_from_uv, *, order=15, world_xyz=None,
                           tangent_basis_world=None, chunk_rows=1024, sample_uv_path=None):
    plan=local_chart_plan(surface_world_xyz,grid_world_row,world_keys,world_row_ids,normal_world,sigma_m,
                          world_from_uv,order=order,world_xyz=world_xyz,tangent_basis_world=tangent_basis_world)
    chart=StructuredRoadChart(surface_world_xyz,world_from_uv)
    n,q=len(plan['center_uv']),len(plan['weights'])
    samples=(np.empty((n,q,2),np.float64) if sample_uv_path is None else
             np.lib.format.open_memmap(sample_uv_path,mode='w+',dtype=np.float64,shape=(n,q,2)))
    geometry_valid=np.zeros((n,q),bool)
    max_error=0.
    for begin in range(0,n,chunk_rows):
        stop=min(n,begin+chunk_rows)
        world=plan['world_xyz'][begin:stop,None,:2]+np.einsum('nij,qj->nqi',plan['chol_world_xy'][begin:stop],plan['standard_nodes'])
        result=chart.query(world)
        samples[begin:stop]=result['uv'];geometry_valid[begin:stop]=result['valid']
        max_error=max(max_error,float(result['worldXY_reconstruction_error'].max(initial=0)))
        if order%2:
            middle=q//2;samples[begin:stop,middle]=plan['center_uv'][begin:stop];geometry_valid[begin:stop,middle]=True
    if hasattr(samples,'flush'):samples.flush()
    plan.update(sample_uv=samples,geometric_sample_valid=geometry_valid,
                sampling_receipt=dict(center_is_world_key_node=True,source='actual canonical G_s XY triangles',
                                      global_affine_role='local candidate seed only',seed_error_max_uv=chart.seed_error_max_uv,
                                      mapped_points=n*q,unmapped_points=int((~geometry_valid).sum()),
                                      worldXY_reconstruction_max_m=max_error,dtype='float64',
                                      invalid_query_uv_sentinel=[-4.,-4.],
                                      local_chart_covariance_role='diagnostic/local approximation only; sample_uv uses exact physical-world quadrature'))
    return plan
