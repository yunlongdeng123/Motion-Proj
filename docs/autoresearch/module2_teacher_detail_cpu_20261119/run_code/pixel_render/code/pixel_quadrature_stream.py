"""固定pixel box的各向异性流式UV；不加载Torch/native，不截断/缩域。"""
import numpy as np


def axis_counts(jacobian,density=2.):
    J=np.asarray(jacobian,np.float64)
    return np.maximum(1,np.ceil(float(density)*np.linalg.norm(J,axis=-2)).astype(np.int64))


def iter_pixel_chunks(center_uv,jacobian_uv,pixel_ids=None,*,density=2.,point_budget=131072):
    """返回(ids,UV[n,q,2],weights[q],meta)，完整pixel行归一化不可拆成多个add。

    同(nx,ny)分组后按整行分块，超预算的单pixel独立生成，保持其完整den。
    """
    center=np.asarray(center_uv,np.float64).reshape(-1,2);J=np.asarray(jacobian_uv,np.float64).reshape(-1,2,2)
    ids=np.arange(len(center),dtype=np.int64)if pixel_ids is None else np.asarray(pixel_ids,np.int64)
    sizes=axis_counts(J[ids],density);groups=np.unique(sizes,axis=0)
    for nx,ny in groups:
        use=ids[(sizes[:,0]==nx)&(sizes[:,1]==ny)];q=int(nx*ny)
        t=np.arange(q,dtype=np.int64);offset=np.stack(((t%nx+.5)/nx-.5,(t//nx+.5)/ny-.5),-1)
        weight=np.full(q,1/q,np.float64);batch=max(1,int(point_budget)//q)
        for start in range(0,len(use),batch):
            selected=use[start:start+batch]
            uv=center[selected,None]+np.einsum('nij,qj->nqi',J[selected],offset)
            yield selected,uv,weight,dict(nx=int(nx),ny=int(ny),points=int(len(selected)*q),
                complete_pixel_rows=True,oversized_singleton=q>point_budget,density=float(density))


def one_pixel_samples(center,jacobian,*,density=2.):
    nx,ny=axis_counts(np.asarray(jacobian)[None],density)[0];q=int(nx*ny);k=np.arange(q)
    offset=np.stack(((k%nx+.5)/nx-.5,(k//nx+.5)/ny-.5),-1)
    return np.asarray(center)[None]+offset@np.asarray(jacobian).T,np.full(q,1/q),dict(nx=int(nx),ny=int(ny),points=q)
