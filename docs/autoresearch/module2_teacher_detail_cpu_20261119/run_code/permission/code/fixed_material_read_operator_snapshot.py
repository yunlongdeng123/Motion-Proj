"""固定几何/许可下的精确数值降本：B+K_L L+K_H H。

系数等价于给定quadrature的原始valid和mH分开双线性采样；不改变材质模型、
过滤核或训练目标。仅编译一次静态读取，低/高频仍是世界网格参数。
"""
from pathlib import Path
import json
import numpy as np
from scipy import sparse

def corners(uv,shape):
    H,W=shape;u=np.asarray(uv,np.float64);lo=np.floor(u).astype(np.int64);t=u-lo
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        x=lo[...,0]+dx;y=lo[...,1]+dy
        valid=(x>=0)&(x<W)&(y>=0)&(y<H)
        b=(t[...,0]if dx else 1-t[...,0])*(t[...,1]if dy else 1-t[...,1])
        yield (np.clip(y,0,H-1)*W+np.clip(x,0,W-1)),b*valid

def bilinear(value,uv):
    a=np.asarray(value);a=a[...,None]if a.ndim==2 else a
    result=np.zeros((*uv.shape[:-1],a.shape[-1]),np.float64);flat=a.reshape(-1,a.shape[-1])
    for index,w in corners(uv,a.shape[:2]):result+=flat[index]*w[...,None]
    return result

def coefficient_matrix(uv,point_mass,shape):
    n,q=uv.shape[:2];rows=np.repeat(np.arange(n,dtype=np.int32),q)
    data=[];row=[];col=[]
    for index,b in corners(uv,shape):
        weight=(point_mass*b).ravel();ok=weight!=0
        data.append(weight[ok]);row.append(rows[ok]);col.append(index.ravel()[ok])
    out=sparse.coo_matrix((np.concatenate(data),(np.concatenate(row),np.concatenate(col))),shape=(n,int(np.prod(shape)))).tocsr()
    out.sum_duplicates();out.eliminate_zeros();out.sort_indices();return out

class ReadCompiler:
    def __init__(self,base,valid,gates,*,low_stride=6,low_shape=None):
        self.base=np.asarray(base,np.float64);self.valid=np.asarray(valid,np.float64);self.gates={k:np.asarray(v,np.float64)for k,v in gates.items()}
        self.low_stride=low_stride;self.high_shape=self.valid.shape
        self.low_shape=low_shape or tuple(int(np.ceil((n-1)/low_stride))+1 for n in self.high_shape)
        if self.base.shape!=(*self.high_shape,3)or any(g.shape!=self.high_shape for g in self.gates.values()):raise ValueError('共同场shape不一致')
        self.parts=[]

    def add(self,uv,weights,geometric_valid=None):
        u=np.asarray(uv,np.float64);n,q=u.shape[:2]
        w=np.broadcast_to(np.asarray(weights,np.float64),(n,q)).copy()
        if not np.isfinite(u).all()or not np.isfinite(w).all()or np.any(w<0):raise ValueError('非finite/负积分权重')
        if geometric_valid is not None:w*=np.asarray(geometric_valid,bool)
        v=bilinear(self.valid,u)[...,0];mass=(w*v).sum(1)
        norm=np.divide(w,mass[:,None],out=np.zeros_like(w),where=mass[:,None]>1e-12)
        base=(norm[...,None]*bilinear(self.base*self.valid[...,None],u)).sum(1)
        KL=coefficient_matrix(u/self.low_stride,norm*v,self.low_shape)
        KH={key:coefficient_matrix(u,norm*v*bilinear(g,u)[...,0],self.high_shape)for key,g in self.gates.items()}
        self.parts.append((base,mass,KL,KH))

    def finish(self,row_ids,metadata):
        if not self.parts:raise ValueError('没有读取样本')
        out=FixedReadOperator(np.concatenate([p[0]for p in self.parts]),np.concatenate([p[1]for p in self.parts]),
            sparse.vstack([p[2]for p in self.parts],format='csr'),
            {k:sparse.vstack([p[3][k]for p in self.parts],format='csr')for k in self.gates},
            self.low_shape,self.high_shape,np.asarray(row_ids),dict(metadata))
        if len(out.base)!=len(out.row_ids):raise ValueError('读取行身份数不同')
        return out

class FixedReadOperator:
    def __init__(self,base,mass,KL,KH,low_shape,high_shape,row_ids,metadata):
        self.base=base;self.mass=mass;self.KL=KL;self.KH=KH;self.low_shape=tuple(low_shape);self.high_shape=tuple(high_shape)
        self.row_ids=row_ids;self.metadata=metadata;self._torch_cache={}

    def numpy_query(self,low,high,gate):
        return np.clip(self.base+self.KL@np.asarray(low).reshape(-1,3)+self.KH[gate]@np.asarray(high).reshape(-1,3),0,1)

    def torch_query(self,low,high,gate):
        import torch
        if tuple(low.shape[:2])!=self.low_shape or tuple(high.shape[:2])!=self.high_shape:raise ValueError('系数与field shape不同')
        key=(str(low.device),gate)
        if key not in self._torch_cache:
            def tensor(matrix):
                # COO在本轮CPU验收；未来CUDA SparseABI必须在真实smoke验证。
                m=matrix.tocoo();idx=torch.as_tensor(np.stack((m.row,m.col)),device=low.device,dtype=torch.long)
                values=torch.as_tensor(m.data,device=low.device,dtype=torch.float64)
                return torch.sparse_coo_tensor(idx,values,m.shape,device=low.device,check_invariants=True).coalesce()
            self._torch_cache[key]=(torch.as_tensor(self.base,device=low.device,dtype=torch.float64),tensor(self.KL),tensor(self.KH[gate]))
        b,l,h=self._torch_cache[key]
        return (b+torch.sparse.mm(l,low.reshape(-1,3).double())+torch.sparse.mm(h,high.reshape(-1,3).double())).clamp(0,1).float()

    def save(self,root):
        root=Path(root);root.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(root/'constants.npz',base=self.base,mass=self.mass,row_ids=self.row_ids)
        sparse.save_npz(root/'low.npz',self.KL)
        for key,m in self.KH.items():
            if not key.replace('_','').isalnum():raise ValueError('非法gate名')
            sparse.save_npz(root/f'high_{key}.npz',m)
        (root/'read.json').write_text(json.dumps(dict(schema='fixed_linear_world_material_read.v1',
            equation='B+KL*L+KH*H; KH already includes separately interpolated mH, never apply gate again',
            low_shape=self.low_shape,high_shape=self.high_shape,gates=list(self.KH),metadata=self.metadata,
            rows=len(self.base),nonzeros_low=self.KL.nnz,nonzeros_high={k:v.nnz for k,v in self.KH.items()},
            no_independent_trainable_row_RGB=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,root):
        root=Path(root);d=json.loads((root/'read.json').read_text(encoding='utf-8'))
        with np.load(root/'constants.npz')as z:base=z['base'];mass=z['mass'];ids=z['row_ids']
        return cls(base,mass,sparse.load_npz(root/'low.npz'),{k:sparse.load_npz(root/f'high_{k}.npz')for k in d['gates']},d['low_shape'],d['high_shape'],ids,d['metadata'])
