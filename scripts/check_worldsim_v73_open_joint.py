"""一次真实FIT的开放曲面视觉重接检查；表面梯度与原生辅助项分开。"""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.native_pyramid import NativeGeometryPyramid
from motion_proj.worldsim_v73.native_data import sample_depth
from motion_proj.worldsim_v73.open_chart_queries import ActorOpenChartQueryDecoder
from motion_proj.worldsim_v73.surface_seeds import native_surface_seeds
from motion_proj.worldsim_v73.surface_readout import closest_surface_points
from motion_proj.worldsim_v73.surface_visibility import BeamTubeFreeSpaceLoss
from motion_proj.worldsim_v73.ray_support import constructive_ray_support_loss

start=time.monotonic(); torch.set_num_threads(4)
base=Path('/root/autodl-tmp/runs/worldsim_v73')
native=base/'WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01/20260907T161500Z__native-dpt-surround25-dev6-s7301-r3'
data=base/'WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2'
targets=base/'WS-V73-M2-EXTENDED-TARGETS-01/20260907T200000Z__fit-track-measurements-r2'
entry=next(x for x in json.loads((data/'index.json').read_text())['cases'] if x['role']=='fit' and x['status']=='ready')
case=torch.load(data/entry['file'],map_location='cpu',weights_only=True)
label=torch.load(targets/(entry['scene']+'__'+entry['owner']+'.pt'),map_location='cpu',weights_only=True)
scene=next(s for s in torch.load(native/'build_observations.pt',map_location='cpu',weights_only=False,mmap=True) if s['scene_id']==entry['scene'])
scale=json.loads((native/'metric_scales.json').read_text())[entry['scene']]
pyramid=NativeGeometryPyramid(native,scene,case['view_indices'],token_device='cpu')
torch.manual_seed(7304); generator=torch.Generator(device='cuda').manual_seed(7305)
model=ActorOpenChartQueryDecoder().cuda()
parameters=[*pyramid.head.parameters(),*model.parameters()]
optimizer=torch.optim.AdamW(parameters,lr=1e-5)
before=pyramid.head.projects[0].weight.detach().clone()
points=case['points_actor_m'].cuda(); size=case['size_lwh_m'].cuda()
matrices=case['camera_from_actor'].cuda(); calibration=case['intrinsics'].cuda()
features,depth=pyramid(include_depth=True)
seeds,support=native_surface_seeds(depth,scale,matrices,calibration,size,points,len(model.coarse),
                                  valid_image_rect=case.get('valid_image_rect_xyxy'))
with torch.autocast('cuda',dtype=torch.bfloat16):
    surface=model(points,size,features,matrices,calibration,case['image_hw'],case['camera_ids'].cuda(),
        case['time_offsets_s'].cuda(),completion_seeds=seeds,
        camera_weights=case.get('camera_embedding_weights'),valid_image_rect=case.get('valid_image_rect_xyxy'))
vertices=surface['vertices_actor_m'].float(); faces=surface['faces']
chosen=label['target_points_actor_m'].cuda()
chosen=chosen[torch.randperm(len(chosen),device='cuda')[:1024]]
coverage=(closest_surface_points(vertices,faces,chosen)-chosen).norm(dim=-1).mean()
frames=label['target_rays']
origins=torch.cat([x['origins_actor_m'] for x in frames]).cuda()
directions=torch.cat([x['directions_actor'] for x in frames]).cuda()
ranges=torch.cat([x['observed_first_range_m'] for x in frames]).cuda()
ids=torch.randperm(len(ranges),device='cuda')[:512]
free=BeamTubeFreeSpaceLoss(width_m=.03,resolution=32,penalty='range')(vertices,faces,origins[ids],directions[ids],ranges[ids])
envelope=(vertices.abs()-size/2-.25).clamp_min(0).square().mean()
owned=torch.nonzero(torch.cat([x['positive_actor'] for x in frames]).cuda(),as_tuple=False).flatten()
selected=owned[torch.randperm(len(owned),device='cuda',generator=generator)[:1024]]
ray,ray_stats=constructive_ray_support_loss(vertices,faces,origins[selected],directions[selected],ranges[selected],kind='first_surface')
geometry=coverage+.5*free+.05*envelope+ray
# 只用几何目标辨别通路，不能让辅助深度梯度冒充表面到DPT的梯度。
feature_grad=torch.autograd.grad(geometry,features,retain_graph=True,allow_unused=True)
feature_norms=[g.norm().item() if g is not None else 0. for g in feature_grad]
head_grad=torch.autograd.grad(geometry,pyramid.head.projects[0].weight,retain_graph=True)[0].norm().item()
terms=[]; count=0
for image,i in zip(depth,case['view_indices']):
    view=scene['views'][i]; owned_view=torch.tensor([owner==entry['owner'] for owner in view['owners']],dtype=torch.bool)
    uv=view['uv'][owned_view]; target=view['z_m'][owned_view]
    if len(uv):
        terms.append(torch.nn.functional.smooth_l1_loss(sample_depth(image,uv.cuda())*scale,target.cuda(),beta=.2,reduction='sum'))
        count+=len(uv)
native_loss=torch.stack(terms).sum()/count if terms else vertices.sum()*0
combined=geometry+native_loss
combined.backward(); norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
assert head_grad>0 and all(x>0 for x in feature_norms) and torch.isfinite(norm)
optimizer.step(); change=(pyramid.head.projects[0].weight.detach()-before).abs().max().item()
assert change>0
result={'status':'done','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'actor':entry,'actual_dpt_views':len(case['view_indices']),'native_support':support,
    'geometry_loss':geometry.item(),'ray_statistics':ray_stats,'native_auxiliary_m':native_loss.item(),
    'native_observations':count,'geometry_only_feature_gradient_norms':feature_norms,
    'geometry_only_dpt_project_gradient_norm':head_grad,'combined_gradient_norm_before_clip':norm.item(),
    'dpt_project_max_change':change,'optimizer_updates':1,'wall_s':time.monotonic()-start,
    'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
    'failure_ledger_refs':['V73-F02','V73-F03','V73-F09'],'failure_ledger_delta':'none; gradient path not quality result',
    'boundary':'one real FIT only; no DEV/new logs, no upper adaptation, no reuse of scratch optimizer'}
out=ROOT/'docs/autoresearch/worldsim_v73/ray_support/open_joint_check_r1.json'
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); print(json.dumps(result),flush=True)
