"""一次解析miss梯度及真实FIT联合反传核对，不是新质量实验。"""
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.ray_support import constructive_ray_support_loss,ray_conditioned_surface_points
from motion_proj.worldsim_v73.surface_readout import closest_surface_points,first_triangle_intersection
from motion_proj.worldsim_v73.first_event import FirstSurfaceEventLoss
from motion_proj.worldsim_v73.open_chart_queries import ActorOpenChartQueryDecoder
from motion_proj.worldsim_v73.surface_seeds import farthest_indices
from motion_proj.worldsim_v73.surface_visibility import BeamTubeFreeSpaceLoss

start=time.monotonic(); torch.set_num_threads(4); torch.manual_seed(7304)
v=torch.tensor([[.5,-.5,1.5],[1.5,-.5,2.5],[.5,.5,2.5]],dtype=torch.float64,requires_grad=True)
f=torch.tensor([[0,1,2]]); o=torch.zeros(1,3,dtype=v.dtype); d=torch.tensor([[0.,0.,1.]],dtype=v.dtype); r=torch.tensor([2.],dtype=v.dtype)
target=o+r[:,None]*d
iso=ray_conditioned_surface_points(v,f,target,d,lateral_ratio=1.)
iso_error=(iso-closest_surface_points(v,f,target)).abs().max().item()
loss,stats=constructive_ray_support_loss(v,f,o,d,r)
gradient=torch.autograd.grad(loss,v)[0]
step=1e-5; plus=v.detach().clone(); minus=plus.clone(); plus[0,0]+=step; minus[0,0]-=step
finite_difference=(constructive_ray_support_loss(plus,f,o,d,r)[0]-constructive_ray_support_loss(minus,f,o,d,r)[0])/(2*step)
fd_error=abs(finite_difference.item()-gradient[0,0].item())
after,after_stats=constructive_ray_support_loss(v.detach()-.001*gradient,f,o,d,r)
cv=v.detach().float().cuda().requires_grad_(); cf=f.cuda(); co=o.float().cuda(); cd=d.float().cuda(); cr=r.float().cuda()
hard,_=first_triangle_intersection(cv,cf,co,cd)
event,event_stats=FirstSurfaceEventLoss()(cv,cf,co,cd,cr)
event_gradient=torch.autograd.grad(event,cv)[0]
assert iso_error<1e-9 and fd_error<1e-5 and gradient.norm()>0 and after<loss
assert not torch.isfinite(hard).any() and event_gradient.norm()==0
synthetic={'literal_miss':True,'event_nll':event.item(),'event_gradient_norm':event_gradient.norm().item(),
    'event_statistics':event_stats,'isotropic_nearest_error_m':iso_error,'finite_difference_absolute_error':fd_error,
    'ray_loss_before_m':loss.item(),'ray_loss_after_m':after.item(),'ray_gradient_norm':gradient.norm().item(),
    'lateral_before_m':stats['lateral_m'],'lateral_after_m':after_stats['lateral_m'],
    'boundary':'one missed, off-axis triangle; local derivative away from face switches, not a guarantee on arbitrary meshes'}

base=Path('/root/autodl-tmp/runs/worldsim_v73')
data=base/'WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2'
targets=base/'WS-V73-M2-EXTENDED-TARGETS-01/20260907T200000Z__fit-track-measurements-r2'
entry=next(x for x in json.loads((data/'index.json').read_text())['cases'] if x['role']=='fit' and x['status']=='ready')
case=torch.load(data/entry['file'],map_location='cpu',weights_only=True)
label=torch.load(targets/(case['metadata']['scene']+'__'+case['metadata']['owner']+'.pt'),map_location='cpu',weights_only=True)
torch.manual_seed(7304); generator=torch.Generator(device='cuda').manual_seed(7305)
model=ActorOpenChartQueryDecoder().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
points=case['points_actor_m'].cuda(); size=case['size_lwh_m'].cuda(); seeds=points[farthest_indices(points,len(model.coarse))]
def predict():
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return model(points,size,None,case['camera_from_actor'].cuda(),case['intrinsics'].cuda(),case['image_hw'],
            case['camera_ids'].cuda(),case['time_offsets_s'].cuda(),use_visual=False,completion_seeds=seeds)
surface=predict(); vertices=surface['vertices_actor_m'].float(); faces=surface['faces']
chosen=label['target_points_actor_m'].cuda(); chosen=chosen[torch.randperm(len(chosen),device='cuda')[:1024]]
frames=label['target_rays']; origins=torch.cat([x['origins_actor_m'] for x in frames]).cuda()
directions=torch.cat([x['directions_actor'] for x in frames]).cuda(); ranges=torch.cat([x['observed_first_range_m'] for x in frames]).cuda()
ids=torch.randperm(len(ranges),device='cuda')[:512]
coverage=(closest_surface_points(vertices,faces,chosen)-chosen).norm(dim=-1).mean()
free=BeamTubeFreeSpaceLoss(width_m=.03,resolution=32,penalty='range')(vertices,faces,origins[ids],directions[ids],ranges[ids])
envelope=(vertices.abs()-size/2-.25).clamp_min(0).square().mean()
owned=torch.nonzero(torch.cat([x['positive_actor'] for x in frames]).cuda(),as_tuple=False).flatten()
selected=owned[torch.randperm(len(owned),device='cuda',generator=generator)[:1024]]
ray,ray_stats=constructive_ray_support_loss(vertices,faces,origins[selected],directions[selected],ranges[selected])
groups=[('normal',model.normal),('height',model.chart_height),('position',model.displacements)]
ray_grad={}
for name,module in groups:
    gradients=torch.autograd.grad(ray,list(module.parameters()),retain_graph=True,allow_unused=True)
    ray_grad[name]=torch.stack([g.norm() for g in gradients if g is not None]).norm().item()
combined=coverage+.5*free+.05*envelope+ray
combined.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
assert torch.isfinite(norm) and all(x>0 for x in ray_grad.values())
optimizer.step()
with torch.no_grad(): changed=(predict()['vertices_actor_m']-vertices.detach()).abs().max().item()
assert changed>0
result={'status':'done','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'synthetic':synthetic,'real_fit':{'actor':entry,'owned_returns':len(owned),'sampled_owned_returns':len(selected),
    'coverage_m':coverage.item(),'free_m':free.item(),'ray_support_m':ray.item(),'ray_statistics':ray_stats,
    'ray_only_gradient_norms':ray_grad,'combined_gradient_norm_before_clip':norm.item(),
    'combined_loss':combined.item(),'optimizer_updates':1,'vertex_max_change_m':changed},
    'wall_s':time.monotonic()-start,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
    'failure_ledger_refs':['V73-F02','V73-F03'],'failure_ledger_delta':'none; new gradient path check only',
    'boundary':'no DEV or new-log evaluation, no visual training; scratch optimizer not reused by formal training'}
out=ROOT/'docs/autoresearch/worldsim_v73/ray_support/path_check_r1.json'; out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); print(json.dumps(result),flush=True)
