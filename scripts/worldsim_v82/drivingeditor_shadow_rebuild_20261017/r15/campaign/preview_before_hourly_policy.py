from pathlib import Path
import sys,json,argparse
import torch
from PIL import Image,ImageDraw
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'backend_restore_r15';sys.path.insert(0,str(R/'implementation_r15_train_v3/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.model import SceneModel
from shadow_rebuild.data import Dataset
from shadow_rebuild.train import seed_from_state,write
from shadow_rebuild.multi_evaluate import actor_only
from shadow_rebuild.math3d import linear_to_srgb
from shadow_rebuild.losses import pixel_valid,masked_mean
torch.set_num_threads(2);saved=torch.load(D/'main/latest.pt',map_location='cpu',weights_only=False);raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate();model=SceneModel(seed_from_state(saved['state']),cfg).cuda();model.load_state_dict(saved['state']);model.requires_grad_(False);data=Dataset(D/'idsplat_frontend/data_IDSplat','cuda',True)
step=saved['global_step'];O=D/'previews'/f'{step:06d}';O.mkdir(parents=True,exist_ok=False);rows=[];metrics=[]
def png(t,scalar=False):
    x=t if scalar else linear_to_srgb(t)
    if x.ndim==2:x=x[...,None].expand(-1,-1,3)
    return Image.fromarray((x.clamp(0,1)*255).round().byte().cpu().numpy())
with torch.no_grad():
    for actor,index in [(5,18),(6,16),(12,48),(18,48),(25,48),(26,44),(37,32)]:
        f=data.frame(index);g,p=model.buffers(f,len(data.actor_ids),'cuda',isolated=True);dynamic,alpha=actor_only(f.camera,g,'cuda',cfg.max_features,actor);edited,deleted=model.buffers(f,len(data.actor_ids),'cuda',{actor:{'delete':True}},isolated=False)
        model.set_stage(saved['stage']);image=model.train_image(p,g);deletion=model.train_image(deleted,edited)
        if saved['stage']!='geometry':image,_=model.full_image(p,g);deletion,_=model.full_image(deleted,edited)
        ims=[png(f.rgb),png(image),png(dynamic),png(alpha,True),png(deletion)];canvas=Image.new('RGB',(2400,340),(15,22,30));dr=ImageDraw.Draw(canvas)
        for j,(im,label) in enumerate(zip(ims,['real RGB','full scene','isolated Actor RGB','raw alpha','delete edit'])):canvas.paste(im,(480*j,20));dr.text((480*j+5,3),f'a{actor} raw{index+89} {label}',fill='white')
        canvas.save(O/f'a{actor:02d}.png');rows.append(canvas)
        body=f.masks[...,actor]&pixel_valid(f)[...,actor];metrics.append(dict(actor=actor,raw=index+89,valid_body_pixels=int(body.sum()),body_MAE=float(masked_mean((image-f.rgb).abs(),body)) if body.any() else None,unknown_mask_pixels=int((~pixel_valid(f)[...,actor]).sum())))
contact=Image.new('RGB',(2400,340*len(rows)))
for i,row in enumerate(rows):contact.paste(row,(0,340*i))
contact.save(O/'contact.png');write(O/'summary.json',dict(step=step,stage=saved['stage'],actors=metrics,actual_shadow_rendering=True,training_shadow_grid_stride=model._shadow_stride,opacity_reset_at_checkpoint=(step%cfg.opacity_reset_interval==0 and step<cfg.densify_until),quality_accepted=False,not_a_stop_gate=True));print(json.dumps({'preview':str(O),'metrics':metrics}),flush=True)
