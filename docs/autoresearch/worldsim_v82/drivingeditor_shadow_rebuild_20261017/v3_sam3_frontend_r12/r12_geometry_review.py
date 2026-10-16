from pathlib import Path
import argparse,sys,json,types,time
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);a=p.parse_args()
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=r/'ownership_controls_r12_sam3';sys.path.insert(0,str(r/'implementation_r12/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.model import SceneModel
from shadow_rebuild.data import Dataset
from shadow_rebuild.train import seed_from_state,write
from shadow_rebuild.multi_evaluate import evaluate_many
torch.set_num_threads(2);saved=torch.load(root/a.arm/'latest.pt',map_location='cpu',weights_only=False)
if saved['global_step']!=3000 or saved['stage']!='geometry':raise ValueError('finite control checkpoint differs')
cfg=saved['config'].copy();cfg['weights']=LossWeights(**cfg['weights']);cfg=Config(**cfg).validate();model=SceneModel(seed_from_state(saved['state']),cfg).cuda();model.load_state_dict(saved['state']);model.requires_grad_(False)
class MetricDataset(Dataset):
 def frame(self,index):
  f=super().frame(index);orig=np.load(r/'data_r2'/self.rows[index]['supervision']);f.masks=torch.as_tensor(orig['masks'],device=f.rgb.device);f.mask_known=torch.as_tensor(orig['mask_known'],device=f.rgb.device);return f
data=MetricDataset(root/('data_A_box' if a.arm=='A_box' else 'data_B_semantic_motion'),'cuda',True)
if saved['input_contract']!=data.manifest:raise ValueError('data contract differs')
def canonical_image(self,package,geometry,chunk=2048,shadows=True):return package['canonical'],torch.ones_like(package['alpha'])
model.full_image=types.MethodType(canonical_image,model);start=time.monotonic();out=root/a.arm/'geometry49_003000';protocol=json.loads((r/'comparison_protocol.json').read_text());evaluate_many(model,data,out,[5,6,12,18,25,26,37],protocol=protocol)
write(out/'scope.json',dict(arm=a.arm,global_step=3000,geometry_only=True,metrics_body_mask_source='original unmodified data_r2 DEVA labels, evaluation only; training all3 shares actual native SAM3 bundle with explicit fallbacks',
 same49_cameras_and_RGB=True,QUERY_raw=[101,113,125],raw_alpha_not_GT_masked=True,shadow_off_identity='by canonical diagnostic construction, not physical shadow validation',seconds=time.monotonic()-start))
print('FULL49_CONTROL_REVIEW_COMPLETE',a.arm,flush=True)
