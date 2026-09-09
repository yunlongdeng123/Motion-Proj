"""有限数量的开放局部曲面片；支持分配与同一UV三角面用于训练和硬读出。"""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F

from .spatial_queries import ActorSpatialQueryDecoder
from .surface_seeds import farthest_indices


@torch.no_grad()
def initial_chart_normals(initial, ids):
    # PCA只构造当前步只读局部坐标架；不反传重根不稳定的特征向量。
    points=initial.detach().float().cpu().numpy().astype(np.float64)
    selected=ids.cpu().numpy()
    neighbor=cKDTree(points).query(points[selected],k=min(16,len(points)))[1]
    neighbor=np.asarray(neighbor).reshape(len(ids),-1)
    local=points[neighbor]; local=local-local.mean(1,keepdims=True)
    values,vectors=np.linalg.eigh(np.swapaxes(local,1,2)@local)
    normals=vectors[:,:,0]
    fallback=values[:,-1]<1e-10
    radial=points[selected]
    length=np.linalg.norm(radial,axis=-1,keepdims=True)
    radial=np.divide(radial,np.maximum(length,1e-12))
    radial[length[:,0]<1e-12]=[0,0,1]
    normals[fallback]=radial[fallback]
    normals*=np.where((normals*radial).sum(-1)<0,-1.,1.)[:,None]
    return torch.as_tensor(normals,device=initial.device,dtype=torch.float32)


class ActorOpenChartQueryDecoder(ActorSpatialQueryDecoder):
    def __init__(self,chart_count=64,chart_resolution=4,chart_scale=.15,**kwargs):
        super().__init__(**kwargs)
        self.chart_count=chart_count
        self.chart_resolution=chart_resolution
        self.chart_scale=chart_scale
        del self.patch_bend
        hidden=self.source.embedding_dim
        self.chart_height=nn.Sequential(nn.Linear(hidden+2,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.chart_height[-1].weight)
        nn.init.zeros_(self.chart_height[-1].bias)
        nn.init.zeros_(self.normal.weight)
        nn.init.zeros_(self.normal.bias)
        grid=torch.cartesian_prod(torch.linspace(-1,1,chart_resolution),torch.linspace(-1,1,chart_resolution))
        faces=[]
        for row in range(chart_resolution-1):
            for col in range(chart_resolution-1):
                a=row*chart_resolution+col
                faces.extend([[a,a+1,a+chart_resolution],[a+1,a+chart_resolution+1,a+chart_resolution]])
        self.register_buffer('chart_grid',grid)
        self.register_buffer('chart_faces',torch.tensor(faces,dtype=torch.long))

    def forward(self,build_points_actor_m,size_lwh_m,features,camera_from_actor,intrinsics,
                image_hw,camera_ids,time_offsets_s,use_spatial=True,use_visual=True,completion_seeds=None,
                camera_weights=None,valid_image_rect=None):
        x,h,source,observed,initial=self.encode_queries(
            build_points_actor_m,size_lwh_m,features,camera_from_actor,intrinsics,
            image_hw,camera_ids,time_offsets_s,use_spatial,use_visual,completion_seeds,
            camera_weights,valid_image_rect)
        evidence_count=min(len(build_points_actor_m),self.evidence_queries)
        assigned=min(evidence_count,self.chart_count//2)
        evidence_ids=(farthest_indices(initial[:evidence_count],assigned) if assigned else
                      torch.empty(0,device=x.device,dtype=torch.long))
        completion_ids=farthest_indices(initial[evidence_count:],self.chart_count-assigned)+evidence_count
        ids=torch.cat([evidence_ids,completion_ids])
        centers=x[ids].float(); state=h[ids]
        seed_normal=initial_chart_normals(initial,ids)
        # 残差范数小于1，避免单位PCA法向被抵消；它是参数化约束，不是可见性门控。
        normal=F.normalize(seed_normal+.5*self.normal(state).float().tanh(),dim=-1)
        reference=torch.zeros_like(normal); reference[:,2]=1
        vertical=normal[:,2].abs()>.9
        reference[vertical,2]=0; reference[vertical,1]=1
        tangent=F.normalize(torch.cross(normal,reference,dim=-1),dim=-1)
        bitangent=torch.cross(normal,tangent,dim=-1)
        halfspan=self.chart_scale*size_lwh_m.float().prod().pow(1/3)
        uv=self.chart_grid[None].expand(len(ids),-1,-1)
        code=torch.cat([state[:,None].expand(-1,len(self.chart_grid),-1),uv],dim=-1)
        height=.5*halfspan*self.chart_height(code).float().squeeze(-1).tanh()
        vertices=(centers[:,None]+halfspan*tangent[:,None]*uv[:,:,0:1]
                  +halfspan*bitangent[:,None]*uv[:,:,1:2]+normal[:,None]*height[:,:,None])
        faces=(self.chart_faces[None]+torch.arange(len(ids),device=x.device)[:,None,None]*len(self.chart_grid)).reshape(-1,3)
        return {'centers_actor_m':centers,'vertices_actor_m':vertices.reshape(-1,3),'faces':faces,
                'normals_actor':normal,'source':source[ids],'visual_observed':observed[ids],
                'hidden':state,'context_actor_m':x,'chart_query_indices':ids,'chart_halfspan_m':halfspan}
