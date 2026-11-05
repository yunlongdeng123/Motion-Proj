"""One frozen HR-normalized chart-opacity viewport control.

Usage:
    wrapped = RelativeAreaOpacityAsset(base_asset)
    scene = runtime.scene(raw, wrapped, delete=False)

The wrapped base is never modified. Its current rows/colours/camera are used
unchanged; only chart-row opacities in a render copy can differ below 1920x1280.
This is NOT the complete Mip-Splatting method or a geometry-accuracy claim.
The native legacy neg_fov convention, off-axis clamp, and +0.3 I are reproduced.
No dependence on any particular asset row count, retained-row cut, or colour.
"""
import copy
import math
import torch

REFERENCE_HW=(1280,1920)
NATIVE_FLOOR_VARIANCE=.3


def covariance_world(rows, select):
    """Reproduce native computeCov3D, including its unnormalised quaternion."""
    packed=rows.get('cov3D_precomp')
    if packed is not None:
        p=packed[select].double()
        if p.shape[1]!=6:raise ValueError('Expected packed symmetric cov3D [N,6]')
        return torch.stack((p[:,0],p[:,1],p[:,2],p[:,1],p[:,3],p[:,4],p[:,2],p[:,4],p[:,5]),-1).reshape(-1,3,3)
    scales=rows['scales'][select].double()
    q=rows['rotations'][select].double()
    r,x,y,z=q.unbind(-1)
    # GLM's nine-scalar mat3 constructor fills columns. S*R is then used by
    # forward.cu, followed by transpose(M)*M; do not silently normalise q.
    columns=torch.stack((1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),
                         2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),
                         2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y)),-1)
    R=columns.reshape(-1,3,3).transpose(-1,-2)
    M=scales[:,:,None]*R  # active native wrapper has scale_modifier=1
    return M.transpose(-1,-2)@M


def projected_covariance(rows,camera,pipe,select):
    """Unfiltered G in native pixel units, with native legacy FoV semantics."""
    means=rows['means3D'][select].double()
    view=camera.world_view_transform.double()
    t=means@view[:3,:3]+view[3,:3]
    z=t[:,2]
    if not bool(torch.isfinite(t).all()) or not bool((z>0).all()):
        raise ValueError('Chart control requires finite positive camera-z rows')
    tx=math.tan(-.5) if pipe.neg_fov else math.tan(float(camera.FoVx)*.5)
    ty=math.tan(-.5) if pipe.neg_fov else math.tan(float(camera.FoVy)*.5)
    fx=camera.image_width/(2*tx);fy=camera.image_height/(2*ty)
    # Preserve the exact min(lim,max(-lim,x/z)) order even for negative lim.
    qx=torch.minimum(z.new_tensor(1.3*tx),torch.maximum(z.new_tensor(-1.3*tx),t[:,0]/z))
    qy=torch.minimum(z.new_tensor(1.3*ty),torch.maximum(z.new_tensor(-1.3*ty),t[:,1]/z))
    J=means.new_zeros((len(means),2,3))
    J[:,0,0]=fx/z;J[:,0,2]=-fx*qx/z
    J[:,1,1]=fy/z;J[:,1,2]=-fy*qy/z
    # With the native row-vector view storage, W is view[:3,:3].
    T=view[:3,:3][None]@J.transpose(-1,-2)
    return T.transpose(-1,-2)@covariance_world(rows,select)@T


def relative_area_factors(rows,camera,pipe,chart):
    """Return one factor per asset row, HR exactly 1, nonchart exactly 1."""
    n=len(rows['opacities'])
    if chart.shape!=(n,) or chart.dtype!=torch.bool:
        raise ValueError('Chart mask must match the current base.rows row order')
    factor=rows['opacities'].new_ones(n)
    h,w=int(camera.image_height),int(camera.image_width)
    rh,rw=REFERENCE_HW
    if (h,w)==REFERENCE_HW or not bool(chart.any()):return factor
    if h<=0 or w<=0 or h>rh or w>rw or h*rw!=w*rh:
        raise ValueError('Control supports same-aspect viewports up to reference 1920x1280')
    reference=copy.copy(camera);reference.image_height=rh;reference.image_width=rw
    G=projected_covariance(rows,camera,pipe,chart)
    H=projected_covariance(rows,reference,pipe,chart)
    I=torch.eye(2,device=G.device,dtype=G.dtype)[None]*NATIVE_FLOOR_VARIANCE
    det=lambda v:torch.linalg.det(v).clamp_min(1e-24)
    k=torch.sqrt(det(G)/det(G+I))/torch.sqrt(det(H)/det(H+I))
    if not bool(torch.isfinite(k).all()) or not bool((k>0).all()) or not bool((k<=1+1e-8).all()):
        raise ValueError('Invalid area-normalization factor')
    factor[chart]=k.to(factor.dtype)
    return factor


class RelativeAreaOpacityAsset:
    """Delegate a persistent asset, changing only chart opacity render copies."""
    def __init__(self,base_asset,chart_kind=1):
        kind=getattr(base_asset,'representation_kind',None)
        if kind is None or kind.ndim!=1 or len(kind)!=len(base_asset.xyz):
            raise ValueError('Base asset must retain per-row representation_kind provenance')
        self.base_asset=base_asset
        self.chart_kind=chart_kind
        self.last_area_factors=None
        self.last_area_record=None

    def __getattr__(self,name):return getattr(self.base_asset,name)

    @torch.no_grad()
    def rows(self,matrix,camera,pipe,channels,precomputed=False):
        rows=self.base_asset.rows(matrix,camera,pipe,channels,precomputed=precomputed)
        chart=self.base_asset.representation_kind==self.chart_kind
        factor=relative_area_factors(rows,camera,pipe,chart)
        self.last_area_factors=factor.detach()
        self.last_area_record=dict(reference_hw=list(REFERENCE_HW),
            viewport_hw=[int(camera.image_height),int(camera.image_width)],
            asset_rows=len(chart),chart_rows=int(chart.sum()),chart_kind=self.chart_kind,
            native_neg_fov=bool(pipe.neg_fov),native_filter_variance_px2=NATIVE_FLOOR_VARIANCE,
            source_asset_opacity_unchanged=True,geometry_colours_camera_unchanged=True,
            formula='k_R=sqrt(det(G_R)/det(G_R+0.3I))/sqrt(det(G_HR)/det(G_HR+0.3I))')
        if (int(camera.image_height),int(camera.image_width))==REFERENCE_HW:
            assert torch.equal(factor,torch.ones_like(factor))
            self.last_area_record['HR_exact_identity']=True
            return rows  # exact same dictionary and tensors as the base rows
        if not bool(chart.any()):return rows
        op=rows['opacities'].clone()
        op[chart]=op[chart]*factor[chart,None]
        assert torch.equal(op[~chart],rows['opacities'][~chart])
        self.last_area_record['chart_factor_min']=float(factor[chart].min())
        self.last_area_record['chart_factor_mean']=float(factor[chart].double().mean())
        self.last_area_record['chart_factor_max']=float(factor[chart].max())
        return dict(rows,opacities=op)
