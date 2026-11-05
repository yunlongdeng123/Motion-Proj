"""Native fixed-geometry sparse alpha*T design, without a research-model chain."""
import ctypes as C
from pathlib import Path
import sys
import math
import numpy as np
import torch
from scipy.sparse import coo_matrix

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
sys.path.insert(0,str(CAMPAIGN/'scene008_a6_ray_support_20261105/private_cuda'))
from a6_clip_rasterization import _C

PN=C.c_void_p;INT=C.c_int;FLOAT=C.c_float;TILE=16

def layout(buffer,fields):
    offset=0;address=buffer.data_ptr()
    for count,dtype,width in fields:
        offset+=(-(address+offset))%128;end=offset+width*count
        assert end<=len(buffer)
        yield buffer[offset:end].view(dtype);offset=end

class Kernel:
    def __init__(self):
        self.lib=C.CDLL(str(CAMPAIGN/'scene008_a6_depth_order_20261105/libreplay_quantile.so'))
        common=[INT,INT,INT]+[PN]*10+[INT,FLOAT]
        self.lib.count_pass.argtypes=common+[PN]
        self.lib.gather_pass.argtypes=common+[PN,PN,PN,PN,PN]
        for name in ('count_pass','gather_pass'):getattr(self.lib,name).restype=INT
    def call(self,name,*args):
        actual=[x.data_ptr() if isinstance(x,torch.Tensor) else x for x in args]
        status=getattr(self.lib,name)(*actual,torch.cuda.current_stream().cuda_stream)
        if status:raise RuntimeError(f'{name}: CUDA status {status}')

def padded(rows):
    value=rows['features'];out=value.new_zeros((len(value),24));out[:,:value.shape[1]]=value
    assert not bool(out[:,6:].any()),'Complete-asset renderer has no inherited clip metadata'
    return dict(rows,features=out)

def native(camera,pipe,degree,rows,background):
    rows=padded(rows);empty=rows['means3D'].new_empty(0)
    value=lambda name:rows[name] if rows[name] is not None else empty
    result=_C.rasterize_gaussians(background,rows['means3D'],value('colors_precomp'),rows['features'],rows['opacities'],
        value('scales'),value('rotations'),1.,value('cov3D_precomp'),rows['mask'],camera.world_view_transform,
        camera.full_proj_transform,math.tan(-.5 if pipe.neg_fov else camera.FoVx*.5),
        math.tan(-.5 if pipe.neg_fov else camera.FoVy*.5),camera.image_height,camera.image_width,
        value('shs'),degree,camera.camera_center,False,False)
    return result,rows

@torch.no_grad()
def design(camera,pipe,rows,mask):
    """Return CSR W (premultiplied RGB), selected pixels, and conservation data."""
    kernel=Kernel();background=rows['means3D'].new_zeros(3)
    result,rows=native(camera,pipe,0,rows,background)
    rendered,contrib,rgb,features,radii,geom,binning,pixel_buffer=result
    n=len(rows['features']);f,i,u=torch.float32,torch.int32,torch.uint8
    depth,_,_,xy,_,conic,_=layout(geom,((n,f,4),(3*n,u,1),(n,i,4),(2*n,f,4),(6*n,f,4),(4*n,f,4),(3*n,f,4)))
    tile_ids,_,_=layout(binning,((rendered,i,4),(rendered,i,4),(rendered,torch.int64,8)))
    h,w=rgb.shape[1:];p=h*w;_,_,ranges=layout(pixel_buffer,((p,f,4),(p,i,4),(2*p,i,4)))
    pixels=torch.where(mask.flatten())[0].to(torch.int32);count=torch.empty_like(pixels)
    kind=torch.full((n,),3,device='cuda',dtype=torch.int32)
    terms=rows['features'].new_zeros((n,9));ray=terms.new_zeros(4)
    args=(len(pixels),w,(w+15)//16,pixels,ranges,tile_ids,xy,conic,depth,rows['features'],kind,terms,ray,0,0.)
    kernel.call('count_pass',*args,count)
    offsets=torch.cat((count.new_zeros(1),count.cumsum(0,dtype=torch.int32)))
    hit_count=int(offsets[-1]);keys=torch.empty(hit_count,device='cuda',dtype=torch.int64)
    ids=torch.empty(hit_count,device='cuda',dtype=torch.int32);alpha=torch.empty(hit_count,device='cuda')
    if hit_count:kernel.call('gather_pass',*args,offsets,keys,ids,alpha)
    ray_index=torch.arange(len(pixels),device='cuda').repeat_interleave(count.long(),output_size=hit_count)
    a=alpha.double();logs=torch.cat((a.new_zeros(1),torch.log1p(-a).cumsum(0)))
    before=(logs[:-1]-logs[offsets.long()[ray_index]]).exp()
    weight=before*a*(before*(1-a)>=.0001)
    use=weight>0
    matrix=coo_matrix((weight[use].cpu().numpy(),(ray_index[use].cpu().numpy(),ids[use].cpu().numpy())),shape=(len(pixels),n)).tocsr()
    index=pixels.long();alpha_ref=features[1].flatten()[index].cpu().numpy()
    rgb_ref=rgb.flatten(1)[:,index].T.cpu().numpy()
    alpha_error=float(np.max(np.abs(np.asarray(matrix.sum(1)).ravel()-alpha_ref))) if len(index) else 0.
    assert alpha_error<5e-6,alpha_error
    colour=rows['colors_precomp']
    if colour is None:raise ValueError('Appearance design uses explicit per-point RGB')
    colour_error=float(np.max(np.abs(matrix@colour.cpu().numpy()-rgb_ref))) if len(index) else 0.
    assert colour_error<5e-6,colour_error
    return dict(matrix=matrix,pixels=index.cpu().numpy(),alpha=alpha_ref,rgb=rgb_ref,
                alpha_maxabs=alpha_error,colour_maxabs=colour_error,hits=hit_count,
                raster_rgb=rgb,raster_alpha=features[1:2])
