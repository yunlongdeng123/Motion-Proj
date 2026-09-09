"""一次真实FIT对象的表面梯度与固定解码器恢复检查；不是质量实验。"""
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from motion_proj.worldsim_v73.open_chart_queries import ActorOpenChartQueryDecoder
from motion_proj.worldsim_v73.surface_seeds import farthest_indices
from motion_proj.worldsim_v73.surface_readout import closest_surface_points
from motion_proj.worldsim_v73.surface_visibility import BeamTubeFreeSpaceLoss
from evaluate_worldsim_v73_fixed_actors import make_query_decoder

base=Path('/root/autodl-tmp/runs/worldsim_v73')
data=base/'WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2'
targets=base/'WS-V73-M2-EXTENDED-TARGETS-01/20260907T200000Z__fit-track-measurements-r2'
started=time.monotonic(); torch.set_num_threads(4); torch.manual_seed(7304)
entry=next(x for x in json.loads((data/'index.json').read_text())['cases'] if x['role']=='fit' and x['status']=='ready')
case=torch.load(data/entry['file'],map_location='cpu',weights_only=True)
target=torch.load(targets/(case['metadata']['scene']+'__'+case['metadata']['owner']+'.pt'),map_location='cpu',weights_only=True)
model=ActorOpenChartQueryDecoder().cuda()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
points=case['points_actor_m'].cuda(); size=case['size_lwh_m'].cuda()
seeds=points[farthest_indices(points,len(model.coarse))]

def predict(module):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return module(points,size,None,case['camera_from_actor'].cuda(),case['intrinsics'].cuda(),
                      case['image_hw'],case['camera_ids'].cuda(),case['time_offsets_s'].cuda(),
                      use_visual=False,completion_seeds=seeds)

surface=predict(model); vertices=surface['vertices_actor_m'].float(); faces=surface['faces']
chosen=target['target_points_actor_m'].cuda()
chosen=chosen[torch.randperm(len(chosen),device='cuda')[:1024]]
frames=target['target_rays']
origins=torch.cat([x['origins_actor_m'] for x in frames]).cuda()
directions=torch.cat([x['directions_actor'] for x in frames]).cuda()
ranges=torch.cat([x['observed_first_range_m'] for x in frames]).cuda()
ids=torch.randperm(len(ranges),device='cuda')[:512]
coverage=(closest_surface_points(vertices,faces,chosen)-chosen).norm(dim=-1).mean()
free=BeamTubeFreeSpaceLoss(width_m=.03,resolution=32,penalty='range')(vertices,faces,origins[ids],directions[ids],ranges[ids])
envelope=(vertices.abs()-size/2-.25).clamp_min(0).square().mean()
loss=coverage+.5*free+.05*envelope
loss.backward()
grad={name:float(torch.stack([p.grad.norm() for p in module.parameters() if p.grad is not None]).norm())
      for name,module in [('normal',model.normal),('chart_height',model.chart_height),('displacements',model.displacements)]}
norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
if not torch.isfinite(norm): raise FloatingPointError('真实FIT曲面梯度非有限')
optimizer.step()
with torch.no_grad(): after=predict(model)['vertices_actor_m'].clone()
config={'query_surface':'open_charts','chart_count':64,'chart_resolution':4,'chart_scale':.15}
path=Path('/root/autodl-tmp/controller_logs/open_chart_path_check_state.pt')
torch.save({'config':config,'query_decoder':model.cpu().state_dict()},path)
state=torch.load(path,map_location='cpu',weights_only=True)
restored=make_query_decoder(state['config']).cuda()
restored.load_state_dict(state['query_decoder'])
with torch.no_grad(): restored_vertices=predict(restored)['vertices_actor_m']
result={'status':'done','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'actor':entry,'seed':7304,'optimizer_updates':1,'mode':'lidar_only','config':config,
        'vertices':len(vertices),'faces':len(faces),'charts':len(surface['centers_actor_m']),
        'chart_halfspan_m':surface['chart_halfspan_m'].item(),'context_queries':len(surface['context_actor_m']),
        'sampled_target_points':len(chosen),'sampled_original_rays':len(ids),
        'coverage_m':coverage.item(),'free_objective_m':free.item(),'loss':loss.item(),
        'gradient_groups_before_clip':grad,'gradient_norm_before_clip':norm.item(),
        'vertex_change_max_m':(after-vertices.detach()).abs().max().item(),
        'fixed_factory_restore_max_error_m':(restored_vertices-after).abs().max().item(),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'saved_bytes':path.stat().st_size,'wall_s':time.monotonic()-started,
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        'failure_ledger_refs':['V73-F01','V73-F02','V73-F03'], 'failure_ledger_delta':'none; implementation path only',
        'boundary':'one real FIT observation check; no DEV evaluation, no new logs, no visual gradient claim, no optimizer-state restore claim'}
out=ROOT/'docs/autoresearch/worldsim_v73/open_charts/path_check_r1.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result),flush=True)
