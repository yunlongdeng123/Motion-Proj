"""Actor 局部空间查询：先投影局部采样，再更新米制位置和连通曲面片。"""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F


def neighborhood(x, k):
    """建图走空间索引；离散邻接不反传，邻接内坐标与消息正常反传。"""
    count = min(k+1, len(x))
    _, ids = cKDTree(x.detach().float().cpu().numpy()).query(x.detach().float().cpu().numpy(), k=count)
    ids = np.asarray(ids).reshape(len(x),count)
    return torch.as_tensor(ids[:,1:] if count>1 else ids, device=x.device, dtype=torch.long)


class ProjectedLocalRead(nn.Module):
    def __init__(self, feature_channels=(256,256,256,256), hidden=192, samples=4, query_chunk=128):
        super().__init__()
        self.samples=samples
        self.query_chunk=query_chunk
        self.projections=nn.ModuleList([nn.Conv2d(c,hidden,1) for c in feature_channels])
        self.query=nn.Linear(hidden,hidden)
        self.key=nn.Linear(hidden,hidden)
        self.geometry=nn.Sequential(nn.Linear(8,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.offsets=nn.Linear(hidden,len(feature_channels)*samples*2)
        self.camera=nn.Embedding(6,hidden)
        self.scale=nn.Embedding(len(feature_channels),hidden)
        self.output=nn.Linear(hidden,hidden)
        nn.init.zeros_(self.offsets.weight)
        with torch.no_grad():
            angles=torch.arange(samples)*2*torch.pi/samples
            pattern=torch.stack([angles.cos(),angles.sin()],-1)
            self.offsets.bias.copy_(pattern.repeat(len(feature_channels),1).reshape(-1))

    def forward(self,x,h,features,camera_from_actor,intrinsics,image_hw,camera_ids,time_offsets_s):
        height,width=image_hw
        maps=[proj(feat) for proj,feat in zip(self.projections,features)]
        outputs=[]
        visibility=[]
        for start in range(0,len(x),self.query_chunk):
            xyz=x[start:start+self.query_chunk]
            state=h[start:start+self.query_chunk]
            camera_xyz=torch.einsum('vij,qj->qvi',camera_from_actor[:,:3,:3],xyz)+camera_from_actor[None,:,:3,3]
            homogeneous=torch.einsum('vij,qvj->qvi',intrinsics,camera_xyz)
            uv=homogeneous[...,:2]/homogeneous[...,2:].clamp_min(1e-5)
            uv=torch.stack([2*uv[...,0]/(width-1)-1,2*uv[...,1]/(height-1)-1],-1)
            base_valid=(camera_xyz[...,2]>.05)&(uv.abs()<=1).all(-1)
            offset=self.offsets(state).reshape(len(xyz),len(maps),self.samples,2).tanh()*4
            values=[]
            masks=[]
            for level,feature in enumerate(maps):
                fh,fw=feature.shape[-2:]
                unit=xyz.new_tensor([2/max(fw-1,1),2/max(fh-1,1)])
                grid=uv[:,:,None,:]+offset[:,None,level]*unit
                valid=base_valid[:,:,None]&(grid.abs()<=1).all(-1)
                sampled=F.grid_sample(feature,grid.permute(1,0,2,3).to(feature.dtype),align_corners=True)
                sampled=sampled.permute(2,0,3,1)
                direction=F.normalize(camera_xyz,dim=-1)
                aux=torch.cat([direction, camera_xyz[...,2:3]/20,
                    time_offsets_s[None,:,None].expand(len(xyz),-1,1),uv,
                    xyz.new_full((len(xyz),len(camera_ids),1),float(level))],-1)
                embedding=self.geometry(aux)+self.camera(camera_ids)[None]+self.scale.weight[level]
                values.append(sampled+embedding[:,:,None,:])
                masks.append(valid)
            value=torch.stack(values,dim=2).flatten(1,3)
            valid=torch.stack(masks,dim=2).flatten(1,3)
            logits=(self.query(state)[:,None]*self.key(value)).sum(-1)/(state.shape[-1]**.5)
            # 无有效视觉的查询保留LiDAR路径，避免全-inf softmax产生NaN。
            logits=logits.masked_fill(~valid,-1e4)
            weight=logits.softmax(-1)*valid
            weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
            observed=valid.any(-1)
            read=self.output((value*weight[:,:,None]).sum(1))*observed[:,None]
            outputs.append(read)
            visibility.append(observed)
        return torch.cat(outputs),torch.cat(visibility)


class ActorSpatialQueryDecoder(nn.Module):
    """证据和补全查询共享更新，无可学习opacity/existence/半径逃避通道。"""
    def __init__(self,feature_channels=(256,256,256,256),hidden=192,evidence_queries=1024,
                 completion_queries=512,layers=3,neighbors=16,patch_spacing_m=.06):
        super().__init__()
        self.evidence_queries=evidence_queries
        self.neighbors=neighbors
        self.patch_spacing_m=patch_spacing_m
        self.coarse=nn.Parameter(torch.rand(completion_queries,3)-.5)
        self.position=nn.Sequential(nn.Linear(3,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.source=nn.Embedding(2,hidden)
        self.reads=nn.ModuleList([ProjectedLocalRead(feature_channels,hidden) for _ in range(layers)])
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(hidden+4,hidden),nn.GELU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.updates=nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden*3),nn.Linear(hidden*3,hidden),nn.GELU()) for _ in range(layers)])
        self.displacements=nn.ModuleList([nn.Linear(hidden,3) for _ in range(layers)])
        self.normal=nn.Linear(hidden,3)
        self.patch_bend=nn.Linear(hidden,9)
        for head in self.displacements:
            nn.init.normal_(head.weight,std=.002)
            nn.init.zeros_(head.bias)
        grid=torch.cartesian_prod(torch.arange(-1,2),torch.arange(-1,2)).float()
        self.register_buffer('patch_grid',grid)
        faces=[]
        for y in range(2):
            for z in range(2):
                a=y*3+z
                faces.extend([[a,a+1,a+3],[a+1,a+4,a+3]])
        self.register_buffer('patch_faces',torch.tensor(faces,dtype=torch.long))

    def forward(self,build_points_actor_m,size_lwh_m,features,camera_from_actor,intrinsics,
                image_hw,camera_ids,time_offsets_s,use_spatial=True,use_visual=True,completion_seeds=None):
        count=min(len(build_points_actor_m),self.evidence_queries)
        ids=torch.linspace(0,max(len(build_points_actor_m)-1,0),count,device=build_points_actor_m.device).long()
        evidence=build_points_actor_m[ids]
        # 表面种子并不限制后续位置更新；原生depth生成的种子保留梯度。
        completion=self.coarse*size_lwh_m if completion_seeds is None else completion_seeds+self.coarse*.1
        x=torch.cat([evidence,completion])
        source=torch.cat([torch.zeros(count,device=x.device,dtype=torch.long),
                          torch.ones(len(completion),device=x.device,dtype=torch.long)])
        h=self.position(x/size_lwh_m.clamp_min(.1))+self.source(source)
        observed=torch.zeros(len(x),device=x.device,dtype=torch.bool)
        for read,message,update,displacement in zip(self.reads,self.messages,self.updates,self.displacements):
            if use_spatial:
                ids=neighborhood(x,self.neighbors)
                relative=x[ids]-x[:,None]
                local=message(torch.cat([h[ids],relative,relative.norm(dim=-1,keepdim=True)],-1)).mean(1)
            else:
                # 等容量逐点控制：同一message MLP只处理自身，不读取其他查询。
                local=message(torch.cat([h,x.new_zeros(len(x),4)],-1))
            if use_visual:
                visual,available=read(x,h,features,camera_from_actor,intrinsics,image_hw,camera_ids,time_offsets_s)
            else:
                visual=torch.zeros_like(h)
                available=torch.zeros(len(x),device=x.device,dtype=torch.bool)
            h=h+update(torch.cat([h,local,visual],-1))
            x=x+displacement(h)
            observed|=available
        normal=F.normalize(self.normal(h)+F.normalize(x,dim=-1),dim=-1,eps=1e-5)
        reference=torch.zeros_like(normal)
        reference[:,2]=1
        alternative=normal[:,2].abs()>.9
        reference[alternative,2]=0
        reference[alternative,1]=1
        tangent=F.normalize(torch.cross(normal,reference,dim=-1),dim=-1)
        bitangent=torch.cross(normal,tangent,dim=-1)
        grid=self.patch_grid*self.patch_spacing_m
        bend=self.patch_bend(h).tanh()*self.patch_spacing_m
        vertices=(x[:,None]+tangent[:,None]*grid[None,:,0:1]+
                  bitangent[:,None]*grid[None,:,1:2]+normal[:,None]*bend[:,:,None])
        faces=(self.patch_faces[None]+torch.arange(len(x),device=x.device)[:,None,None]*9).reshape(-1,3)
        return {'centers_actor_m':x,'vertices_actor_m':vertices.reshape(-1,3),'faces':faces,
                'normals_actor':normal,'source':source,'visual_observed':observed,'hidden':h}
