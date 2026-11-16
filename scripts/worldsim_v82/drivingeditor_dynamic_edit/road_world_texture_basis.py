"""纯NumPy世界XY颜色basis；不导入Torch、scene或renderer。"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def bilinear_basis(world_xy, origin_xy, spacing_m, shape):
    xy=np.asarray(world_xy,np.float64);origin=np.asarray(origin_xy,np.float64)
    if xy.ndim!=2 or xy.shape[1]!=2 or origin.shape!=(2,) or not np.isfinite(xy).all():
        raise ValueError("世界XY必须是有限N×2米制坐标")
    height,width=map(int,shape)
    if min(height,width)<2 or not np.isfinite(spacing_m) or spacing_m<=0:raise ValueError("图集尺寸/米制分辨率无效")
    uv=(xy-origin)/float(spacing_m)
    inside=((uv>=-1e-10)&(uv<=[width-1+1e-10,height-1+1e-10])).all(axis=1)
    safe=np.clip(uv,[0,0],[width-1,height-1])
    lo=np.minimum(np.floor(safe).astype(np.int64),[width-2,height-2]);fraction=safe-lo
    x,y=lo.T;u,v=fraction.T
    ids=np.stack([y*width+x,y*width+x+1,(y+1)*width+x,(y+1)*width+x+1],axis=1)
    weights=np.stack([(1-u)*(1-v),u*(1-v),(1-u)*v,u*v],axis=1)
    weights[~inside]=0
    return ids,weights,inside,uv


@dataclass
class ColorBasisPlan:
    world_xyz: np.ndarray
    world_row_ids: np.ndarray
    base_srgb: np.ndarray
    requested_eligible: np.ndarray
    protected_rows: np.ndarray
    eligible: np.ndarray
    origin_xy: np.ndarray
    spacing_m: float
    shape: tuple
    corner_ids: np.ndarray
    weights: np.ndarray
    mutable_texels: np.ndarray
    scope_source: str
    source_role: str

    def save(self,path):
        np.savez_compressed(path,**{name:getattr(self,name) for name in self.__dataclass_fields__})

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:values={name:data[name].copy() for name in cls.__dataclass_fields__}
        values["spacing_m"]=float(values["spacing_m"]);values["shape"]=tuple(map(int,values["shape"]))
        values["scope_source"]=str(values["scope_source"]);values["source_role"]=str(values["source_role"])
        result=cls(**values)
        ids,weights,inside,_=bilinear_basis(result.world_xyz[:,:2],result.origin_xy,result.spacing_m,result.shape)
        if not inside.all() or not np.array_equal(ids,result.corner_ids) or not np.array_equal(weights,result.weights):
            raise ValueError("保存的世界位置与双线性查询basis不一致")
        return result


def make_color_basis_plan(world_xyz,base_srgb,row_eligible,protected_rows=None,*,scope_source,source_role,spacing_m=.04,world_row_ids=None):
    world=np.asarray(world_xyz,np.float64).copy();base=np.asarray(base_srgb,np.float32).copy()
    requested=np.asarray(row_eligible,bool).copy();n=len(world)
    protected=np.zeros(n,bool) if protected_rows is None else np.asarray(protected_rows,bool).copy()
    if not isinstance(scope_source,str) or not scope_source or not isinstance(source_role,str) or not source_role:
        raise ValueError("必须显式提供common scope来源和角色，旧scope不可默认升级为新target许可")
    if world.shape!=(n,3) or base.shape!=(n,3) or requested.shape!=(n,) or protected.shape!=(n,):raise ValueError("道路颜色/位置/行域尺寸不一致")
    if not n or not np.isfinite(world).all() or not np.isfinite(base).all() or np.any((base<0)|(base>1)):
        raise ValueError("道路位置与初始化RGB须有限，RGB在[0,1]")
    if not np.isfinite(spacing_m) or spacing_m<=0:raise ValueError("纹理分辨率须为正米数")
    ids=np.arange(n,dtype=np.int64) if world_row_ids is None else np.asarray(world_row_ids,np.int64).copy()
    if ids.shape!=(n,) or len(np.unique(ids))!=n or np.any(ids<0):raise ValueError("世界row ID须唯一非负")
    origin=np.floor(world[:,:2].min(axis=0)/spacing_m)*spacing_m
    size=np.maximum(np.ceil((world[:,:2].max(axis=0)-origin)/spacing_m).astype(np.int64)+1,2)
    shape=(int(size[1]),int(size[0]))
    corners,weights,inside,_=bilinear_basis(world[:,:2],origin,spacing_m,shape)
    if not inside.all():raise ValueError("自动纹理格未覆盖固定世界点")
    mutable=np.zeros(shape[0]*shape[1],bool)
    selected=requested&~protected
    mutable[corners[selected][weights[selected]>0]]=True
    # primary只按共同row范围回退c0；不冻结相邻texel而额外约束可写row。
    # raw共享修正场可在guard位置非零，有效shader始终显式row override。
    controllable=(weights*mutable[corners]).sum(axis=1)>0
    eligible=selected&controllable
    # 两arm都使用该共同可控行域；K/非授权row有显式c0持久override。
    return ColorBasisPlan(world,ids,base,requested,protected,eligible,origin,float(spacing_m),shape,corners,weights,mutable,scope_source,source_role)


def basis_statistics(plan,*,row_mass=None):
    """真实固定世界点的共享/权重/秩界；不运行任何真实场景光栅。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import structural_rank
    selected=np.flatnonzero(plan.eligible);columns=np.flatnonzero(plan.mutable_texels)
    mapping=np.full(len(plan.mutable_texels),-1,np.int64);mapping[columns]=np.arange(len(columns))
    local=mapping[plan.corner_ids[selected]];weights=plan.weights[selected]
    okay=(local>=0)&(weights>0)
    matrix=coo_matrix((weights[okay],(np.broadcast_to(np.arange(len(selected))[:,None],local.shape)[okay],local[okay])),shape=(len(selected),len(columns))).tocsr()
    counts=np.diff(matrix.tocsc().indptr)
    weights_sum=np.asarray(matrix.sum(0)).ravel()
    # 确定性局部窗口的实际数值秩；结构秩是全局数值秩上界，二者不混称。
    order=np.argsort(np.asarray(matrix.sum(1)).ravel(),kind="stable")
    windows=[]
    for start in (0,max(len(order)//2-32,0),max(len(order)-64,0)):
        rows=order[start:start+64]
        if len(rows):
            sub=matrix[rows];used=np.unique(sub.indices);dense=sub[:,used].toarray()
            windows.append(dict(rows=len(rows),columns=len(used),numeric_rank=int(np.linalg.matrix_rank(dense,tol=1e-10))))
    mass=np.ones(len(plan.world_xyz),np.float64) if row_mass is None else np.asarray(row_mass,np.float64)
    if mass.shape!=(len(plan.world_xyz),) or not np.isfinite(mass).all() or np.any(mass<0):raise ValueError("静态row mass应有限非负")
    texel_mass=np.asarray(matrix.T@mass[selected]).ravel()
    return dict(spacing_m=plan.spacing_m,shape=list(plan.shape),requested_rows=int(plan.requested_eligible.sum()),
        scope_source=plan.scope_source,source_role=plan.source_role,scope_requalified_for_new_teacher=plan.source_role=="qualified_common_actual_alphaT",
        common_eligible_rows=len(selected),additional_rows_blocked_by_basis=int((plan.requested_eligible&~plan.protected_rows&~plan.eligible).sum()),
        independent_RGB_scalars=3*len(selected),shared_RGB_scalars=3*len(columns),
        positive_weight_entries=int(matrix.nnz),weight_row_sum_min=float(np.asarray(matrix.sum(1)).min()) if len(selected) else 0.,
        weight_row_sum_max=float(np.asarray(matrix.sum(1)).max()) if len(selected) else 0.,
        shared_texels_with_multiple_rows=int((counts>=2).sum()),shared_texels_with_multiple_rows_fraction=float(np.mean(counts>=2)) if len(counts) else 0.,
        row_contributions_per_texel_quantiles=np.quantile(counts,[0,.5,.95,1]).tolist() if len(counts) else [],
        coefficient_weight_sum_quantiles=np.quantile(weights_sum,[0,.5,.95,1]).tolist() if len(columns) else [],
        static_mass_per_texel_quantiles=np.quantile(texel_mass,[0,.5,.95,1]).tolist() if len(columns) else [],
        structural_rank_upper_bound=int(structural_rank(matrix)),global_numeric_rank="not computed; structural rank is not numerical rank",
        fixed_local_numeric_rank_windows=windows,all_CPU=True,GPU_executed=False)
