"""在现有v3检查点输出固定七车证据帧；不读旧模型、不做GT裁剪渲染。"""
from pathlib import Path
import sys,json,argparse
parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True)
parser.add_argument('--run',default='E0_full_s0');parser.add_argument('--implementation',default='implementation_r3');args=parser.parse_args()
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(root/args.implementation/'shadow_rebuild_v3'))
import torch,numpy as np
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state,write
from shadow_rebuild.multi_evaluate import actor_only
from shadow_rebuild.evaluate import save_png
torch.set_num_threads(2)
checkpoint=torch.load(root/args.run/'latest.pt',map_location='cpu',weights_only=False)
if checkpoint['global_step']!=args.step:raise ValueError('checkpoint step differs')
raw=checkpoint['config'].copy();raw['weights']=LossWeights(**raw['weights']);config=Config(**raw).validate()
model=SceneModel(seed_from_state(checkpoint['state']),config).cuda();model.load_state_dict(checkpoint['state']);model.requires_grad_(False)
dataset=Dataset(root/'data_r2','cuda',True)
evidence=json.loads((root/'actor_evidence_r2/actor_evidence.json').read_text())
selected=[a for a in evidence['actors'] if a['eligible_delete']]
out=root/f'{args.run.split("_")[0]}_preview_{args.step:06d}';out.mkdir(exist_ok=False);records=[]
with torch.no_grad():
 for a in selected:
  fid=Path(a['preview']).stem.split('_',1)[1]
  i=next(i for i,r in enumerate(dataset.rows) if r['frame_id']==fid)
  frame=dataset.frame(i);g,p=model.buffers(frame,len(dataset.actor_ids),'cuda',isolated=False)
  dynamic,alpha=actor_only(frame.camera,g,'cuda',config.max_features)
  body,body_alpha=actor_only(frame.camera,g,'cuda',config.max_features,a['actor_index'])
  _,static=model.buffers(frame,len(dataset.actor_ids),'cuda',{k:{'delete':True} for k in range(len(dataset.actor_ids))},isolated=False)
  images=dict(GT=frame.rgb,canonical=p['canonical'],static=static['canonical'],dynamic=dynamic,alpha=alpha[...,None].expand_as(dynamic),target=body,target_alpha=body_alpha[...,None].expand_as(body))
  for name,value in images.items():save_png(out/f'a{a["actor_index"]:02d}_{fid}_{name}.png',value)
  mask=frame.masks[...,a['actor_index']];y,x=mask.nonzero(as_tuple=True)
  lo=[max(int(x.min())-10,0),max(int(y.min())-10,0)];hi=[min(int(x.max())+11,480),min(int(y.max())+11,320)]
  error=(p['canonical']-frame.rgb).abs()
  records.append(dict(actor=a['actor_index'],frame_id=fid,body_pixels=int(mask.sum()),body_linear_rgb_mae=float(error[mask].mean()),
                      raw_alpha_inside=float(body_alpha[mask].mean()),raw_alpha_outside=float(body_alpha[~mask].mean()),
                      coverage_mean=float(p['alpha'].mean()),gaussians=len(model.field.xyz),crop_box=lo+hi))
write(out/'preview.json',dict(status='REQUIRES_ROOT_VISUAL_REVIEW',global_step=checkpoint['global_step'],records=records,quality_validated=False,light_stage_not_started=checkpoint['stage']=='geometry'))
write(out/'crop_boxes.json',records)
print('PREVIEW_FINISHED',str(out),flush=True)
