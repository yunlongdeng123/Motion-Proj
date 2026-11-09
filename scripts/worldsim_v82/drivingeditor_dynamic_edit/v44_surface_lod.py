"""Camera-frequency sampling of one immutable C2 surface and RGBA material.

SurfaceLODAsset(base_asset, posterior_npz) implements the legacy asset.rows API.
S is the frozen reference-UV cubic depth spline. F is the final canonical chart
RGB and opacity/.99 on that same UV grid, not a new image or fitted texture.
The camera only chooses du=1/sqrt(median(abs(det d(screenXY)/d(UV)))).
No edit/delete state, brightness mask, area-opacity factor, or global renderer
change enters this operator. Native neg_fov affects the unchanged raster only.
"""
import hashlib,math
from functools import wraps
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
try:
    from .v44_manifest import _positive_hw, _matrix, _pose, _finite_number
except ImportError:
    from v44_manifest import _positive_hw, _matrix, _pose, _finite_number

C0=np.float32(.28209479177387814)

def arr(x):return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.array(x,copy=True)
def _uv(value):
    value=np.asarray(value)
    if value.ndim!=2 or value.shape[1]!=2:raise ValueError('UV must be a real (N,2) array')
    return _matrix(value,value.shape,'UV')
def _no_grad(function):
    @wraps(function)
    def call(*args,**kwargs):
        import torch
        with torch.no_grad():return function(*args,**kwargs)
    return call
def hashes(values):
    h=hashlib.sha256()
    for a in values:
        a=np.ascontiguousarray(a);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()
def stats(a):
    a=np.asarray(a).ravel()
    if not len(a):return dict(count=0,min=None,p10=None,median=None,p90=None,max=None,mean=None)
    if not np.isfinite(a).all():raise ValueError('Statistics require finite samples')
    q=np.quantile(a,[0,.1,.5,.9,1])
    return dict(count=len(a),min=float(q[0]),p10=float(q[1]),median=float(q[2]),p90=float(q[3]),max=float(q[4]),mean=float(a.mean()))
def select(rows,mask):return {k:None if v is None else v[mask].contiguous() for k,v in rows.items()}
def merge(a,b):
    import torch
    return {k:None if a[k] is None else torch.cat((a[k],b[k]),0) for k in a}


class SurfaceLODAsset:
    def __init__(self,base_asset,posterior_npz,reference_hw):
        self.base_asset=base_asset;self.posterior_path=Path(posterior_npz)
        self.posterior_sha256=hashlib.sha256(self.posterior_path.read_bytes()).hexdigest()
        with np.load(self.posterior_path,allow_pickle=False) as z:
            required={'u_knots_pixels','v_knots_pixels','control_delta_z_m','z_reference_m','reference_K','reference_camera_from_local_m'}
            if not required<=set(z.files):raise ValueError('Surface posterior NPZ lacks required spline/camera fields')
            self.tu=z['u_knots_pixels'].copy();self.tv=z['v_knots_pixels'].copy()
            for knots in (self.tu,self.tv):
                if (knots.ndim!=1 or len(knots)<8 or knots.dtype.kind not in 'iuf' or not np.isfinite(knots).all()
                        or (np.diff(knots)<0).any() or knots[3]>=knots[-4]):
                    raise ValueError('Cubic knots must be finite ordered 1D arrays with a nonempty support interval')
            size=(len(self.tu)-4,len(self.tv)-4);coef=z['control_delta_z_m']
            if coef.shape==(int(np.prod(size)),):coef=coef.reshape(size)
            self.coef=_matrix(coef,size,'surface control_delta_z_m')
            scalar=np.asarray(z['z_reference_m'])
            if scalar.shape!=() or scalar.dtype.kind not in 'iuf':raise ValueError('z_reference_m must be a real scalar')
            self.zref=_finite_number(scalar.item(),'z_reference_m',positive=True)
            self.K=_matrix(z['reference_K'],(3,3),'reference_K')
            if (min(self.K[0,0],self.K[1,1])<=0 or not np.allclose(self.K[2],[0,0,1],atol=1e-10,rtol=0)
                    or abs(self.K[0,1])>1e-10 or abs(self.K[1,0])>1e-10):
                raise ValueError('Reference pinhole K needs positive focal lengths and zero skew')
            self.camera_from_local=_pose(z['reference_camera_from_local_m'],'reference_camera_from_local_m')
        self.invR=np.linalg.inv(self.camera_from_local[:3,:3]);self.invK=np.linalg.inv(self.K)
        geometry=arr(base_asset.xyz);kinds=arr(base_asset.representation_kind);n=len(geometry)
        if geometry.shape!=(n,3) or not n or not np.isfinite(geometry).all() or kinds.shape!=(n,) or kinds.dtype.kind not in 'iu':
            raise ValueError('Base asset needs finite (N,3) geometry and explicit integer representation kinds')
        self.chart=kinds==1;self.retained=~self.chart
        if not self.chart.any():raise ValueError('Surface LOD requires actual canonical chart rows')
        self.uv=arr(base_asset.resample_reference_pixel_xy[self.chart]).astype(np.float64)
        self.local=arr(base_asset.xyz[self.chart]).astype(np.float64)
        self.reference_hw=_positive_hw(reference_hw,'reference_hw')
        reference_h,reference_w=self.reference_hw
        if (self.uv.shape!=(int(self.chart.sum()),2) or not np.isfinite(self.uv).all()
                or (self.uv<0).any() or (self.uv[:,0]>=reference_w).any() or (self.uv[:,1]>=reference_h).any()
                or not np.array_equal(np.rint(self.uv),self.uv)):
            raise ValueError('Chart UV must be unique integer pixels inside declared reference_hw')
        coefficients=arr(base_asset.coefficients)
        if coefficients.ndim!=3 or coefficients.shape[0]!=n or coefficients.shape[1]<1 or coefficients.shape[2]!=3 or not np.isfinite(coefficients).all() or not (coefficients[:,1:]==0).all():
            raise ValueError('F requires finite fixed SH0 material coefficients')
        xyz,zu,zv,z=self.surface(self.uv,derivatives=True)
        self._dlocal=np.stack((zu,zv),-1)
        self.spline_source_local_maxabs=float(np.abs(xyz.astype(np.float32)-self.local).max())
        if self.spline_source_local_maxabs>=2e-6:raise ValueError('Frozen spline geometry disagrees with canonical chart points')
        sigma=(.6*z/self.K[0,0]).astype(np.float32)
        base_sigma=arr(base_asset.sigma)
        if base_sigma.shape!=(n,) or not np.isfinite(base_sigma).all() or (base_sigma<=0).any():raise ValueError('Base sigma must be finite positive (N,)')
        self.spline_source_sigma_maxabs=float(abs(sigma-base_sigma[self.chart]).max())
        if self.spline_source_sigma_maxabs>=2e-7:raise ValueError('Frozen spline and canonical Gaussian sigma disagree')
        base_opacity=arr(base_asset.opacity);base_rgb=arr(base_asset.rgb)
        if base_opacity.shape!=(n,) or base_rgb.shape!=(n,3) or not np.isfinite(base_opacity).all() or not np.isfinite(base_rgb).all():
            raise ValueError('Base opacity/RGB shapes must match finite geometry rows')
        alpha=arr(base_asset.opacity[self.chart])/np.float32(.99)
        if not np.all((alpha>=0)&(alpha<=1+2e-7)):raise ValueError('Canonical chart opacity must be in [0,.99]')
        alpha=np.minimum(alpha,1)
        rgb=arr(base_asset.rgb[self.chart]).clip(min=0)
        ij=np.rint(self.uv).astype(np.int64)
        if len(np.unique(ij[:,1]*reference_w+ij[:,0]))!=len(ij):raise ValueError('Canonical chart UV pixels must be unique')
        tex=np.zeros((reference_h,reference_w,4),np.float32)
        tex[ij[:,1],ij[:,0],:3]=rgb*alpha[:,None]
        tex[ij[:,1],ij[:,0],3]=alpha
        self.mips=[tex]
        while min(self.mips[-1].shape[:2])>1:
            a=self.mips[-1];h,w=a.shape[:2]
            a=np.pad(a,((0,h%2),(0,w%2),(0,0)))
            self.mips.append(a.reshape(a.shape[0]//2,2,a.shape[1]//2,2,4).mean((1,3)))
        self._persistent=[self.tu,self.tv,self.coef,self.K,self.camera_from_local,self.uv,self.local,self.invR,self.invK,self._dlocal,*self.mips]
        self._zref_snapshot,self._reference_hw_snapshot=self.zref,self.reference_hw
        for a in self._persistent:a.setflags(write=False)
        self.persistent_SF_hash=hashes(self._persistent)
        self._render_xyz=None;self._render_rgb=None;self._render_kind=None
        self.last_lod_record=None;self._cache={}

    def __getattr__(self,k):return getattr(self.base_asset,k)
    @property
    def xyz(self):return self.base_asset.xyz if self._render_xyz is None else self._render_xyz
    @property
    def rgb(self):return self.base_asset.rgb if self._render_rgb is None else self._render_rgb
    @property
    def representation_kind(self):return self.base_asset.representation_kind if self._render_kind is None else self._render_kind

    def surface(self,uv,derivatives=False):
        uv=_uv(uv)
        u=BSpline(self.tu,np.eye(len(self.tu)-4),3,extrapolate=False)
        v=BSpline(self.tv,np.eye(len(self.tv)-4),3,extrapolate=False)
        bu=u(uv[:,0]);bv=v(uv[:,1])
        z=self.zref+np.sum((bu@self.coef)*bv,axis=1)
        rays=np.column_stack((uv+.5,np.ones(len(uv))))@self.invK.T
        xyz=(rays*z[:,None]-self.camera_from_local[:3,3])@self.invR.T
        if not np.isfinite(xyz).all() or not (z>0).all():raise ValueError('Surface query lies outside valid positive-depth spline support')
        if not derivatives:return xyz,z
        dzdu=np.sum((u(uv[:,0],nu=1)@self.coef)*bv,axis=1)
        dzdv=np.sum((bu@self.coef)*v(uv[:,1],nu=1),axis=1)
        dcdu=rays*dzdu[:,None];dcdu[:,0]+=z/self.K[0,0]
        dcdv=rays*dzdv[:,None];dcdv[:,1]+=z/self.K[1,1]
        return xyz,dcdu@self.invR.T,dcdv@self.invR.T,z

    def screen_area(self,matrix,camera):
        pose=_pose(arr(matrix),'surface query pose',similarity=True);P=_matrix(arr(camera.full_proj_transform),(4,4),'full_proj_transform')
        _positive_hw((camera.image_height,camera.image_width),'camera viewport')
        world=self.local@pose[:3,:3].T+pose[:3,3]
        clip=np.column_stack((world,np.ones(len(world))))@P
        d_world=np.einsum('ij,njk->nik',pose[:3,:3],self._dlocal)
        dq=np.einsum('ji,njk->nik',P[:3,:],d_world)
        w=clip[:,3]
        dx=(dq[:,0,:]*w[:,None]-clip[:,0,None]*dq[:,3,:])/w[:,None]**2*(camera.image_width/2)
        dy=(dq[:,1,:]*w[:,None]-clip[:,1,None]*dq[:,3,:])/w[:,None]**2*(camera.image_height/2)
        area=abs(dx[:,0]*dy[:,1]-dx[:,1]*dy[:,0])
        valid=np.isfinite(area)&(area>1e-12)&(w>1e-5)
        if not valid.any():raise ValueError('No finite front-facing chart differential')
        return area[valid],int((~valid).sum())

    def material(self,uv,du):
        uv=_uv(uv);du=_finite_number(du,'du',positive=True)
        lod=max(0.,math.log2(du));lo=min(int(math.floor(lod)),len(self.mips)-1);hi=min(lo+1,len(self.mips)-1)
        frac=lod-math.floor(lod) if hi!=lo else 0.
        def sample(level):
            t=self.mips[level];xy=(uv+.5)/(2**level)-.5
            ij=np.floor(xy).astype(np.int64);f=xy-ij;result=np.zeros((len(uv),4),np.float64)
            for oy in (0,1):
                for ox in (0,1):
                    x=ij[:,0]+ox;y=ij[:,1]+oy
                    valid=(x>=0)&(x<t.shape[1])&(y>=0)&(y<t.shape[0])
                    weight=(f[:,0] if ox else 1-f[:,0])*(f[:,1] if oy else 1-f[:,1])
                    result[valid]+=t[y[valid],x[valid]]*weight[valid,None]
            return result
        rgba=sample(lo)*(1-frac)+sample(hi)*frac
        alpha=rgba[:,3].clip(0,1)
        rgb=np.divide(rgba[:,:3],alpha[:,None],out=np.zeros_like(rgba[:,:3]),where=alpha[:,None]>1e-12)
        return rgb.astype(np.float32),alpha.astype(np.float32),dict(mip_low=lo,mip_high=hi,mip_fraction=frac)

    def sampling(self,matrix,camera):
        area,invalid=self.screen_area(matrix,camera)
        du=float(1/math.sqrt(float(np.median(area))))
        if abs(du-1)<1e-5:du=1.  # numerical identity of declared reference grid
        key=float(du)
        record=dict(du=du,median_screen_area_per_source_UV=float(np.median(area)),area=stats(area),
            invalid_differentials=invalid,viewport=[int(camera.image_width),int(camera.image_height)],
            retained_rows=int(self.retained.sum()),canonical_chart_rows=int(self.chart.sum()),
            persistent_SF_hash=self.persistent_SF_hash,pose_or_edit_state_patch=False,
            peak_opacity=.99,opacity_area_compensation=False,
            sigma_formula='.6 * du * S_camera_z_reference / reference_fx',
            grid_formula='UV=(integer_index+.5)*du-.5',
            spline_source_local_maxabs_m=self.spline_source_local_maxabs,
            spline_source_sigma_maxabs_m=self.spline_source_sigma_maxabs)
        if du==1:
            record.update(canonical_grid_reused_exact=True,chart_rows=int(self.chart.sum()),render_rows=len(self.base_asset.xyz),mip_low=0,mip_high=0,mip_fraction=0.)
            return None,record
        if key not in self._cache:
            lo=self.uv.min(0)-3*du;hi=self.uv.max(0)+3*du
            u=np.arange(math.floor((lo[0]+.5)/du-.5),math.ceil((hi[0]+.5)/du-.5)+1)*du+.5*du-.5
            v=np.arange(math.floor((lo[1]+.5)/du-.5),math.ceil((hi[1]+.5)/du-.5)+1)*du+.5*du-.5
            xx,yy=np.meshgrid(u,v);uv=np.column_stack((xx.ravel(),yy.ravel()))
            rgb,alpha,mip=self.material(uv,du)
            supported=(uv[:,0]>=self.tu[3])&(uv[:,0]<=self.tu[-4])&(uv[:,1]>=self.tv[3])&(uv[:,1]<=self.tv[-4])
            outside=(alpha>1e-12)&~supported
            keep=(alpha>1e-12)&supported
            uv=uv[keep];rgb=rgb[keep];alpha=alpha[keep]
            local,z=self.surface(uv)
            self._cache[key]=dict(uv=uv,local=local.astype(np.float32),sigma=(.6*du*z/self.K[0,0]).astype(np.float32),rgb=rgb,
                opacity=(np.float32(.99)*alpha).astype(np.float32),alpha=alpha,mip=mip,
                texture_positive_outside_spline_count=int(outside.sum()))
        sample=self._cache[key]
        record.update(sample['mip'],canonical_grid_reused_exact=False,chart_rows=len(sample['local']),
            render_rows=len(sample['local'])+int(self.retained.sum()),
            coverage=stats(sample['alpha']),texture_positive_outside_spline_count=sample['texture_positive_outside_spline_count'])
        return sample,record

    @_no_grad
    def rows(self,matrix,camera,pipe,channels,precomputed=False):
        import torch
        if isinstance(channels,bool) or not isinstance(channels,(int,np.integer)) or channels<1:raise ValueError('channels must be a positive integer')
        base=self.base_asset.rows(matrix,camera,pipe,channels,precomputed=precomputed)
        sample,self.last_lod_record=self.sampling(matrix,camera)
        if sample is None:
            self._render_xyz=self.base_asset.xyz;self._render_rgb=self.base_asset.rgb
            self._render_kind=self.base_asset.representation_kind
            return base
        device=matrix.device;T=lambda a:torch.as_tensor(a,device=device,dtype=base['means3D'].dtype)
        local=T(sample['local']);rgb=T(sample['rgb']);pose=matrix.to(local.dtype)
        world=local@pose[:3,:3].T+pose[:3,3]
        s=T(sample['sigma'])*pose[:3,:3].norm(dim=0).mean()
        depth=world@camera.world_view_transform[:3,2:3]+camera.world_view_transform[3,2]
        coefs=world.new_zeros((len(world),int(channels),3));coefs[:,0]=(rgb-.5)/float(C0)
        cov=None;scales=s[:,None].expand(-1,3).contiguous();rotation=world.new_zeros((len(world),4));rotation[:,0]=1
        if pipe.compute_cov3D_python:
            cov=world.new_zeros((len(world),6));cov[:,[0,3,5]]=s.square()[:,None];scales=rotation=None
        chart=dict(means3D=world,means2D=torch.zeros_like(world),shs=None if precomputed else coefs,
            colors_precomp=rgb if precomputed else None,features=torch.cat((depth,torch.ones_like(depth)),1),
            opacities=T(sample['opacity'])[:,None],scales=scales,rotations=rotation,cov3D_precomp=cov,
            mask=torch.ones(len(world),device=device,dtype=torch.bool))
        retained=select(base,self.retained)
        self._render_xyz=torch.cat((self.base_asset.xyz[self.retained],local))
        self._render_rgb=torch.cat((self.base_asset.rgb[self.retained],rgb))
        self._render_kind=torch.cat((self.base_asset.representation_kind[self.retained],self.base_asset.representation_kind.new_ones(len(local))))
        result=merge(retained,chart)
        assert len(result['means3D'])==len(self.xyz)
        return result

    def assert_persistent_unchanged(self):
        if self.zref!=self._zref_snapshot or self.reference_hw!=self._reference_hw_snapshot:raise RuntimeError('Frozen surface scalar metadata changed')
        assert hashes(self._persistent)==self.persistent_SF_hash
        assert hashlib.sha256(self.posterior_path.read_bytes()).hexdigest()==self.posterior_sha256
        assert self.base_asset.assert_geometry_unchanged()
        return True
