"""CAPA只接收build图像和校准的稀疏轴向深度，不读取额外时刻标签。"""
import torch


def build_capa_condition(scene):
    images=torch.stack([view['image'] for view in scene['views']])
    height,width=images.shape[-2:]
    depth=[]; counts=[]
    for view in scene['views']:
        pixels=view['uv'].round().long()
        z=view['z_m'].float()
        valid=(pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)&torch.isfinite(z)&(z>0)
        index=pixels[valid,1]*width+pixels[valid,0]
        flat=torch.full((height*width,),float('inf'))
        # 量化到同一像素时保留最近的真实测量，避免后方点覆盖前方深度。
        flat.scatter_reduce_(0,index,z[valid],reduce='amin',include_self=True)
        mask=torch.isfinite(flat)
        depth.append(torch.where(mask,flat,torch.zeros_like(flat)).reshape(height,width))
        counts.append(int(mask.sum()))
    depth=torch.stack(depth)
    return images,depth,depth>0,counts
