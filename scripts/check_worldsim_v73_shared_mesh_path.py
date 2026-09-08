"""一次必要的Q-v2接口验证：共享拓扑、字面首交点及视觉/原生种子梯度。"""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.shared_mesh_queries import ActorSharedMeshQueryDecoder
from motion_proj.worldsim_v73.surface_readout import first_triangle_intersection,closest_surface_points

torch.manual_seed(7304); torch.set_num_threads(4)
model=ActorSharedMeshQueryDecoder().cuda()
size=torch.tensor([4.,2.,1.6],device='cuda')
vertices=model.template_vertices*size/2
faces=model.mesh_faces
edges=torch.cat([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]).sort(-1).values
_,counts=torch.unique(edges,dim=0,return_counts=True)
assert bool((counts==2).all())
origins=torch.tensor([[4.,0.,0.],[0.,3.,0.],[0.,0.,2.]],device='cuda')
directions=-torch.eye(3,device='cuda')
depth,_=first_triangle_intersection(vertices,faces,origins,directions)
assert torch.allclose(depth,depth.new_tensor([2.,2.,1.2]),atol=1e-5)
points=torch.rand(32,3,device='cuda')*size-size/2
seeds=(torch.rand(512,3,device='cuda')*size-size/2).requires_grad_()
# 现有offset上限为4个特征像素；4×4图会使最粗层所有候选越界，不能用来要求各层梯度。
features=[torch.randn(1,256,n,n,device='cuda',requires_grad=True) for n in [12,24,48,96]]
pose=torch.eye(4,device='cuda')[None]; pose[:,2,3]=8
k=torch.tensor([[[48.,0.,31.5],[0.,48.,31.5],[0.,0.,1.]]],device='cuda')
with torch.autocast('cuda',dtype=torch.bfloat16):
    result=model(points,size,features,pose,k,(64,64),torch.zeros(1,device='cuda',dtype=torch.long),
                 torch.zeros(1,device='cuda'),completion_seeds=seeds)
surface=result['vertices_actor_m'].float()
targets=vertices[::16]*.9
nearest=closest_surface_points(surface,faces,targets)
loss=(nearest-targets).square().mean()
loss.backward()
feature_grads=[v.grad.norm().item() for v in features]
seed_grad=seeds.grad.norm().item()
print(json.dumps({'feature_gradient_norms':feature_grads,'native_seed_gradient_norm':seed_grad}),flush=True)
assert all(v>0 for v in feature_grads) and seed_grad>0
assert bool(torch.isfinite(surface).all()) and bool(torch.isfinite(loss))
print(json.dumps({'kind':'single implementation check, synthetic inputs, not research quality evidence',
    'mesh_vertices':len(vertices),'mesh_faces':len(faces),'edge_incidence':counts.unique().tolist(),
    'template_axis_first_ranges_m':depth.tolist(),'feature_gradient_norms':feature_grads,
    'native_seed_gradient_norm':seed_grad,'query_trainable_parameters':sum(p.numel() for p in model.parameters()),
    'loss':loss.item(),'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30}))
