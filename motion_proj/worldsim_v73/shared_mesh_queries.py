"""Q-v2：观测查询驱动共享顶点网格，所有物理目标读取同一组显式三角面。"""
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import torch
from torch import nn

from .spatial_queries import ActorSpatialQueryDecoder, neighborhood


def sphere_template(level=3):
    """只对解析二十面体建立初始拓扑，不对观测点取凸包或伪造闭合真值。"""
    phi=(1+5**.5)/2
    vertices=[]
    for axis in range(3):
        for a in [-1.,1.]:
            for b in [-phi,phi]:
                v=[0.,0.,0.]; v[(axis+1)%3]=a; v[(axis+2)%3]=b
                vertices.append(v)
    vertices=np.asarray(vertices,dtype=np.float64)
    vertices/=np.linalg.norm(vertices,axis=1,keepdims=True)
    faces=ConvexHull(vertices).simplices.copy()
    for face in faces:
        a,b,c=vertices[face]
        if np.dot(np.cross(b-a,c-a),a)<0: face[1],face[2]=face[2],face[1]
    for _ in range(level):
        verts=vertices.tolist(); mids={}; refined=[]
        def middle(i,j):
            edge=tuple(sorted((int(i),int(j))))
            if edge not in mids:
                point=(vertices[i]+vertices[j])/2
                point/=np.linalg.norm(point)
                mids[edge]=len(verts); verts.append(point.tolist())
            return mids[edge]
        for a,b,c in faces:
            ab,bc,ca=middle(a,b),middle(b,c),middle(c,a)
            refined.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
        vertices=np.asarray(verts); faces=np.asarray(refined)
    adjacency=[set() for _ in vertices]
    for a,b,c in faces:
        adjacency[a].update([int(b),int(c)])
        adjacency[b].update([int(a),int(c)])
        adjacency[c].update([int(a),int(b)])
    width=max(map(len,adjacency)); ids=np.zeros((len(vertices),width),dtype=np.int64)
    mask=np.zeros_like(ids,dtype=bool)
    for i,neighbors in enumerate(adjacency):
        neighbors=sorted(neighbors); ids[i,:len(neighbors)]=neighbors; mask[i,:len(neighbors)]=True
    return (torch.tensor(vertices,dtype=torch.float32),torch.tensor(faces,dtype=torch.long),
            torch.tensor(ids),torch.tensor(mask))


def nearest_context(x,context,k):
    """离散观测关联只选索引；所选原生深度坐标和隐状态保留真实梯度。"""
    n=min(k,len(context))
    ids=cKDTree(context.detach().float().cpu().numpy()).query(x.detach().float().cpu().numpy(),k=n)[1]
    return torch.as_tensor(np.asarray(ids).reshape(len(x),n),device=x.device,dtype=torch.long)


class ActorSharedMeshQueryDecoder(ActorSpatialQueryDecoder):
    """固定球面拓扑不保证无自交；其收益必须通过相同硬首交点与缺失率验证。"""
    def __init__(self,mesh_level=3,**kwargs):
        super().__init__(**kwargs)
        hidden=self.source.embedding_dim
        # 旧patch缓冲仅供同信息PCA基线；网格没有独立法向、bend或透明度头。
        del self.normal
        del self.patch_bend
        old_source=self.source
        self.source=nn.Embedding(3,hidden)
        with torch.no_grad(): self.source.weight[:2].copy_(old_source.weight)
        vertices,faces,ids,mask=sphere_template(mesh_level)
        self.register_buffer('template_vertices',vertices)
        self.register_buffer('mesh_faces',faces)
        self.register_buffer('mesh_neighbors',ids)
        self.register_buffer('mesh_neighbor_mask',mask)

    def forward(self,build_points_actor_m,size_lwh_m,features,camera_from_actor,intrinsics,
                image_hw,camera_ids,time_offsets_s,use_spatial=True,use_visual=True,completion_seeds=None,
                camera_weights=None,valid_image_rect=None):
        count=min(len(build_points_actor_m),self.evidence_queries)
        ids=torch.linspace(0,max(len(build_points_actor_m)-1,0),count,device=build_points_actor_m.device).long()
        evidence=build_points_actor_m[ids]
        completion=self.coarse*size_lwh_m if completion_seeds is None else completion_seeds+self.coarse*.1
        context=torch.cat([evidence,completion]); context_count=len(context)
        mesh=self.template_vertices*size_lwh_m/2
        x=torch.cat([context,mesh])
        source=torch.cat([torch.zeros(count,device=x.device,dtype=torch.long),
                          torch.ones(len(completion),device=x.device,dtype=torch.long),
                          torch.full((len(mesh),),2,device=x.device,dtype=torch.long)])
        h=self.position(x/size_lwh_m.clamp_min(.1))+self.source(source)
        observed=torch.zeros(len(x),device=x.device,dtype=torch.bool)
        for read,message,update,displacement in zip(self.reads,self.messages,self.updates,self.displacements):
            if use_spatial:
                # 观测隐状态交互与顶点的拓扑邻接分开，避免稀疏观测被密集网格挤出近邻。
                cx=x[:context_count]; ch=h[:context_count]
                ci=neighborhood(cx,self.neighbors)
                relative=cx[ci]-cx[:,None]
                clocal=message(torch.cat([ch[ci],relative,relative.norm(dim=-1,keepdim=True)],-1)).mean(1)
                mx=x[context_count:]; mh=h[context_count:]
                mi=self.mesh_neighbors
                relative=mx[mi]-mx[:,None]
                edge_message=message(torch.cat([mh[mi],relative,relative.norm(dim=-1,keepdim=True)],-1))
                mask=self.mesh_neighbor_mask
                topological=(edge_message*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
                ci=nearest_context(mx,cx,self.neighbors)
                relative=cx[ci]-mx[:,None]
                evidence_message=message(torch.cat([ch[ci],relative,relative.norm(dim=-1,keepdim=True)],-1)).mean(1)
                local=torch.cat([clocal,(topological+evidence_message)/2])
            else:
                local=message(torch.cat([h,x.new_zeros(len(x),4)],-1))
            if use_visual:
                visual,available=read(x,h,features,camera_from_actor,intrinsics,image_hw,camera_ids,time_offsets_s,
                                      camera_weights=camera_weights,valid_image_rect=valid_image_rect)
            else:
                visual=torch.zeros_like(h); available=torch.zeros(len(x),device=x.device,dtype=torch.bool)
            h=h+update(torch.cat([h,local,visual],-1))
            x=x+displacement(h)
            observed|=available
        vertices=x[context_count:]
        return {'centers_actor_m':vertices,'vertices_actor_m':vertices,'faces':self.mesh_faces,
                'source':source[context_count:],'visual_observed':observed[context_count:],
                'hidden':h[context_count:],'context_actor_m':x[:context_count]}
