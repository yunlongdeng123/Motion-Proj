"""Camera-frequency sampling of one immutable C2 surface and RGBA material.

SurfaceLODAsset(base_asset, posterior_npz) implements the legacy asset.rows API.
S is the frozen reference-UV cubic depth spline. F is the final canonical chart
RGB and opacity/.99 on that same UV grid, not a new image or fitted texture.
The camera only chooses du=1/sqrt(median(abs(det d(screenXY)/d(UV)))).
No edit/delete state, brightness mask, area-opacity factor, or global renderer
change enters this operator. Native neg_fov affects the unchanged raster only.
"""
import hashlib,math
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
import torch

C0=np.float32(.28209479177387814)

def arr(x):return x.detach().cpu().numpy().copy()
def hashes(values):
    h=hashlib.sha256()
    for a in values:
        a=np.ascontiguousarray(a);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()
def stats(a):
    a=np.asarray(a).ravel();q=np.quantile(a,[0,.1,.5,.9,1])
    return dict(count=len(a),min=float(q[0]),p10=float(q[1]),median=float(q[2]),p90=float(q[3]),max=float(q[4]),mean=float(a.mean()))
def select(rows,mask):return {k:None if v is None else v[mask].contiguous() for k,v in rows.items()}
def merge(a,b):return {k:None if a[k] is None else torch.cat((a[k],b[k]),0) for k in a}


class SurfaceLODAsset:
    def __init__(self,base_asset,posterior_npz):
        self.base_asset=base_asset;self.posterior_path=Path(posterior_npz)
        self.posterior_sha256=hashlib.sha256(self.posterior_path.read_bytes()).hexdigest()
        with np.load(self.posterior_path,allow_pickle=False) as z:
            self.tu=z['u_knots_pixels'].copy();self.tv=z['v_knots_pixels'].copy()
            self.coef=z['control_delta_z_m'].reshape(len(self.tu)-4,len(self.tv)-4).copy()
            self.zref=float(z['z_reference_m']);self.K=z['reference_K'].copy()
            self.camera_from_local=z['reference_camera_from_local_m'].copy()
        self.invR=np.linalg.inv(self.camera_from_local[:3,:3]);self.invK=np.linalg.inv(self.K)
        self.chart=base_asset.representation_kind==1;self.retained=~self.chart
        self.uv=arr(base_asset.resample_reference_pixel_xy[self.chart]).astype(np.float64)
        self.local=arr(base_asset.xyz[self.chart]).astype(np.float64)
        self.reference_hw=(1280,1920)
        assert len(self.uv) and np.isfinite(self.uv).all()
        assert bool(torch.all(base_asset.coefficients[:,1:]==0)),'F requires fixed SH0 material'
        xyz,zu,zv,z=self.surface(self.uv,derivatives=True)
        self._dlocal=np.stack((zu,zv),-1)
        self.spline_source_local_maxabs=float(np.abs(xyz.astype(np.float32)-self.local).max())
        assert self.spline_source_local_maxabs<2e-6
        sigma=(.6*z/self.K[0,0]).astype(np.float32)
        self.spline_source_sigma_maxabs=float(abs(sigma-arr(base_asset.sigma[self.chart])).max())
        assert self.spline_source_sigma_maxabs<2e-7
        alpha=arr(base_asset.opacity[self.chart])/np.float32(.99)
        assert np.all((alpha>=0)&(alpha<=1+2e-7));alpha=np.minimum(alpha,1)
        rgb=arr(base_asset.rgb[self.chart]).clip(min=0)
        ij=np.rint(self.uv).astype(np.int64);assert np.array_equal(ij,self.uv)
        assert len(np.unique(ij[:,1]*1920+ij[:,0]))==len(ij)
        tex=np.zeros((1280,1920,4),np.float32)
        tex[ij[:,1],ij[:,0],:3]=rgb*alpha[:,None]
        tex[ij[:,1],ij[:,0],3]=alpha
        self.mips=[tex]
        while min(self.mips[-1].shape[:2])>1:
            a=self.mips[-1];h,w=a.shape[:2]
            a=np.pad(a,((0,h%2),(0,w%2),(0,0)))
            self.mips.append(a.reshape(a.shape[0]//2,2,a.shape[1]//2,2,4).mean((1,3)))
        self._persistent=[self.tu,self.tv,self.coef,self.K,self.camera_from_local,self.uv,*self.mips]
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
        u=BSpline(self.tu,np.eye(len(self.tu)-4),3,extrapolate=False)
        v=BSpline(self.tv,np.eye(len(self.tv)-4),3,extrapolate=False)
        bu=u(uv[:,0]);bv=v(uv[:,1])
        z=self.zref+np.sum((bu@self.coef)*bv,axis=1)
        rays=np.column_stack((uv+.5,np.ones(len(uv))))@self.invK.T
        xyz=(rays*z[:,None]-self.camera_from_local[:3,3])@self.invR.T
        assert np.isfinite(xyz).all() and (z>0).all()
        if not derivatives:return xyz,z
        dzdu=np.sum((u(uv[:,0],nu=1)@self.coef)*bv,axis=1)
        dzdv=np.sum((bu@self.coef)*v(uv[:,1],nu=1),axis=1)
        dcdu=rays*dzdu[:,None];dcdu[:,0]+=z/self.K[0,0]
        dcdv=rays*dzdv[:,None];dcdv[:,1]+=z/self.K[1,1]
        return xyz,dcdu@self.invR.T,dcdv@self.invR.T,z

    def screen_area(self,matrix,camera):
        pose=arr(matrix).astype(np.float64);P=arr(camera.full_proj_transform).astype(np.float64)
        world=self.local@pose[:3,:3].T+pose[:3,3]
        clip=np.column_stack((world,np.ones(len(world))))@P
        d_world=np.einsum('ij,njk->nik',pose[:3,:3],self._dlocal)
        dq=np.einsum('ji,njk->nik',P[:3,:],d_world)
        w=clip[:,3]
        dx=(dq[:,0,:]*w[:,None]-clip[:,0,None]*dq[:,3,:])/w[:,None]**2*(camera.image_width/2)
        dy=(dq[:,1,:]*w[:,None]-clip[:,1,None]*dq[:,3,:])/w[:,None]**2*(camera.image_height/2)
        area=abs(dx[:,0]*dy[:,1]-dx[:,1]*dy[:,0])
        valid=np.isfinite(area)&(area>1e-12)&(w>1e-5)
        assert valid.any(),'No finite front-facing chart differential'
        return area[valid],int((~valid).sum())

    def material(self,uv,du):
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
        if abs(du-1)<1e-5:du=1.  # numerical identity of reference1920 grid
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

    @torch.no_grad()
    def rows(self,matrix,camera,pipe,channels,precomputed=False):
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
        assert hashes(self._persistent)==self.persistent_SF_hash
        assert hashlib.sha256(self.posterior_path.read_bytes()).hexdigest()==self.posterior_sha256
        assert self.base_asset.assert_geometry_unchanged()
        return True
