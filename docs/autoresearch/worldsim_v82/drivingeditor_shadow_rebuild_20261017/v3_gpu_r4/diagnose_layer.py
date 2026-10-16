"""对冻结5k模型做分层、外泄贡献和梯度诊断；不更新参数。"""
from pathlib import Path
import sys,json,math,argparse
parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,default=5000);parser.add_argument('--run',default='E0_full_s0')
parser.add_argument('--implementation',default='implementation_r3');parser.add_argument('--suffix',default='r1');args=parser.parse_args()
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(root/args.implementation/'shadow_rebuild_v3'))
import torch,numpy as np
from scipy.ndimage import distance_transform_edt
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state,write,objective
from shadow_rebuild.raster import cuda_raster
from shadow_rebuild.math3d import quaternion_matrix
from shadow_rebuild.evaluate import save_png
torch.set_num_threads(2)
saved=torch.load(root/args.run/'latest.pt',map_location='cpu',weights_only=False)
if saved['global_step']!=args.step:raise ValueError('diagnostic frozen step differs')
raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);config=Config(**raw).validate()
model=SceneModel(seed_from_state(saved['state']),config).cuda();model.load_state_dict(saved['state']);model.requires_grad_(False)
dataset=Dataset(root/'data_r2','cuda',True);out=root/f'diagnostic_{args.step:06d}_{args.suffix}';out.mkdir(exist_ok=False)
evidence=json.loads((root/'actor_evidence_r2/actor_evidence.json').read_text());entries=[r for r in evidence['actors'] if r['eligible_delete']]
records=[]
for entry in entries:
 actor=entry['actor_index'];fid=Path(entry['preview']).stem.split('_',1)[1]
 index=next(i for i,r in enumerate(dataset.rows) if r['frame_id']==fid);frame=dataset.frame(index)
 with torch.no_grad():g,p=model.buffers(frame,len(dataset.actor_ids),'cuda',isolated=False)
 rows=g['rows'][g['owner']==actor];selected=g['owner']==actor
 means,cov,opacity=g['xyz'][selected],g['cov'][selected],g['opacity'][selected]
 feature=torch.ones(len(means),1,device='cuda',requires_grad=True)
 raw_alpha=cuda_raster(frame.camera,means,cov,opacity,feature,10)[...,0]
 mask=frame.masks[...,actor];union=(frame.masks&frame.mask_known[None,None]).any(-1)
 distance=torch.as_tensor(distance_transform_edt((~union).cpu().numpy()),device='cuda')
 far=(distance>6)&(~mask)
 gradient=torch.autograd.grad(raw_alpha[far].sum(),feature)[0][:,0].detach()
 total=gradient.sum();top=gradient.topk(min(128,len(gradient))).indices
 labels=json.loads((Path(dataset.manifest['annotations'])/(fid+'.json')).read_text())['laser_labels']
 box=next(x['box'] for x in labels if x['id']==dataset.actor_ids[actor]);height=box['height']
 seed=model.field.seed_xyz[rows];scales=model.field.scales[rows];normal=quaternion_matrix(model.field.quaternion[rows])[...,2]
 near_bottom=seed[:,2] < -height/2+.15
 known_count=int(frame.mask_known.sum());body_counts=frame.masks[...,frame.mask_known].sum((0,1));outside_total=known_count*mask.numel()-int(body_counts.sum())
 visible=p['visible_labels'][...,actor];static_visible=(p['alpha']-p['visible_labels'].sum(-1)).clamp_min(0)
 record=dict(actor=actor,frame_id=fid,body_pixels=int(mask.sum()),body_fraction_of_union=float(mask.sum()/union.sum()),known_actor_count=known_count,
             per_actor_outer_gradient_scale_vs_global=float(outside_total/(~mask).sum()),
             raw_inside=float(raw_alpha[mask].mean()),visible_actor_inside=float(visible[mask].mean()),static_visible_inside=float(static_visible[mask].mean()),
             outside_raw_alpha_mass=float(raw_alpha[~mask].sum()),far_background_raw_mass=float(raw_alpha[far].sum()),
             outside_visible_actor_mass=float(visible[~mask].sum()),
             far_background_contribution_near_seed_bottom_fraction=float(gradient[near_bottom].sum()/total.clamp_min(1e-9)),
             top128_far_contribution_fraction=float(gradient[top].sum()/total.clamp_min(1e-9)),
             scale_max=float(scales.max()),seed_bottom_z=float(seed[:,2].min()),annotation_bottom_z=-height/2,
             top_rows=[dict(row=int(rows[j]),far_alpha_contribution=float(gradient[j]),seed_xyz=seed[j].tolist(),xyz=model.field.xyz[rows[j]].tolist(),
                           scales=scales[j].tolist(),opacity=float(opacity[j,0]),local_normal=normal[j].tolist()) for j in top[:16]])
 save_png(out/f'a{actor:02d}_far_outside_alpha.png',raw_alpha.detach()[...,None].expand(-1,-1,3)*far[...,None])
 records.append(record);del raw_alpha,feature,g,p
 # 单点诊断不把offscreen/遮挡区域的无body标签全认证成真实道路。
write(out/'layer_contribution.json',dict(global_step=args.step,records=records,parameters_updated=False,
      far_domain='more than6pixels from all known visible bodies; auxiliary unknown may remain; not all certified road',human_verdict=None))
actor=37;entry=next(r for r in entries if r['actor_index']==actor);fid=Path(entry['preview']).stem.split('_',1)[1]
frame=dataset.frame(next(i for i,r in enumerate(dataset.rows) if r['frame_id']==fid))
model.set_stage('geometry')
loss,terms,_=objective(model,frame,len(dataset.actor_ids),'cuda','geometry',torch.Generator().manual_seed(0))
photometric=sum(terms[k]*getattr(config.weights,k) for k in ['rgb','reconstruction','ssim','body_rgb'])
outer=terms['mask_outside']*config.weights.mask_outside
gp=torch.autograd.grad(photometric,model.field.opacity_logit,retain_graph=True)[0].detach()[:,0]
go=torch.autograd.grad(outer,model.field.opacity_logit)[0].detach()[:,0]
select=model.field.owner==actor
gp,go=gp[select],go[select]
gradient_record=dict(actor=actor,frame_id=fid,photometric_opacity_grad_L1=float(gp.abs().sum()),outer_opacity_grad_L1=float(go.abs().sum()),
                     cosine=float((gp*go).sum()/(gp.norm()*go.norm()).clamp_min(1e-12)),
                     opacity_rows_photometric_push_up=int((gp<0).sum()),opacity_rows_outer_push_down=int((go>0).sum()),
                     weights=dict(mask_inside=config.weights.mask_inside,mask_outside=config.weights.mask_outside,body_rgb=config.weights.body_rgb),
                     instance_balanced_supervision=getattr(config,'instance_balanced_supervision',False),parameters_updated=False)
write(out/'gradient_balance.json',gradient_record)
print(json.dumps(dict(records=[{k:v for k,v in r.items() if k!='top_rows'} for r in records],gradients=gradient_record)),flush=True)
