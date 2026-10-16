"""49帧65流几何/归属诊断；不伪称完成材质光照或阴影训练。"""
from pathlib import Path
import sys,json,argparse,types,time
import torch
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--step',type=int,required=True)
p.add_argument('--implementation',required=True);p.add_argument('--data',default='data_r2');p.add_argument('--output',required=True);a=p.parse_args()
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/a.implementation/'shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state,write
from shadow_rebuild.multi_evaluate import evaluate_many
torch.set_num_threads(2);saved=torch.load(r/a.run/'latest.pt',map_location='cpu',weights_only=False)
if saved['global_step']!=a.step:raise ValueError('frozen step differs')
cfg=saved['config'].copy();cfg['weights']=LossWeights(**cfg['weights']);cfg=Config(**cfg).validate()
model=SceneModel(seed_from_state(saved['state']),cfg).cuda();model.load_state_dict(saved['state']);model.requires_grad_(False)
def canonical_image(self,package,geometry,chunk=2048,shadows=True):
 return package['canonical'],torch.ones_like(package['alpha'])
model.full_image=types.MethodType(canonical_image,model)
data=Dataset(r/a.data,'cuda',True);protocol=json.loads((r/'comparison_protocol.json').read_text())
start=time.monotonic();evaluate_many(model,data,r/a.output,[5,6,12,18,25,26,37],protocol=protocol)
write(r/a.output/'scope.json',dict(run=a.run,global_step=a.step,geometry_only=True,
      rendering='canonical reconstruction; physical material/light/shadow stage NOT yet trained',
      shadow_off_identity='by diagnostic construction; NOT evidence shadow module is solved',
      checkpoint_mutated=False,train_test_metrics_separate=True,elapsed_seconds=time.monotonic()-start))
print(json.dumps(dict(status='GEOMETRY_DIAGNOSTIC_RENDERED_REVIEW_REQUIRED',output=a.output,seconds=time.monotonic()-start)),flush=True)
