# Adapted from NVIDIA CAPA alignment, Copyright (c) 2026 NVIDIA Corporation.
# Original and this derivative: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/
# Source: https://github.com/nv-dvl/capa/blob/main/capa/utils/alignment.py
"""分块枚举同一组仿射锚点，复用官方L1求解与全局选择，不减少对齐数据。"""
import torch


def chunked_align_depth_affine(depth_src,depth_tgt,weight,anchor_chunk=128):
    from capa.utils.alignment import align,scatter_min
    shape,n=depth_src.shape[:-1],depth_src.shape[-1]
    src=depth_src.reshape(-1,n); tgt=depth_tgt.reshape(-1,n); weights=weight.reshape(-1,n)
    batch,anchor=torch.where(weights>0)
    losses=[]; indices=[]
    with torch.no_grad():
        for start in range(0,len(anchor),anchor_chunk):
            b=batch[start:start+anchor_chunk]; a=anchor[start:start+anchor_chunk]
            x=src[b]-src[b,a,None]; y=tgt[b]-tgt[b,a,None]
            _,loss,index=align(x,y,weights[b])
            losses.append(loss); indices.append(index)
        _,winner=scatter_min(size=len(src),dim=0,index=batch,src=torch.cat(losses))
        first=anchor[winner]; second=torch.cat(indices)[winner]
    row=torch.arange(len(src),device=src.device)
    x1=src[row,first]; x2=src[row,second]; y1=tgt[row,first]; y2=tgt[row,second]
    scale=(y2-y1)/torch.where(x2!=x1,x2-x1,1e-7)
    shift=y1-scale*x1
    return scale.reshape(shape),shift.reshape(shape)
