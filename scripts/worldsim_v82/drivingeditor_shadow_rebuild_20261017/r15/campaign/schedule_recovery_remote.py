"""First real R15 update and nonempty Adam/model restore, counted in main45k."""
from pathlib import Path
import sys,json,torch,gc,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'backend_restore_r15';sys.path.insert(0,str(R/'implementation_r15_train_v3/shadow_rebuild_v3'))
from shadow_rebuild.config import Config
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import run,seed_from_state,write
torch.set_num_threads(2);assert json.loads((D/'main/progress.json').read_text())['global_step']==1
proof=json.loads((D/'GPU_selfcheck.json').read_text());assert proof['status']=='ACTUAL_PRODUCTION_CAPACITY_GRADIENTS_AND_RESTORE_PASSED'
cfg=Config.load(D/'config.json');data=Dataset(D/'data','cuda',True);model,status=run(data,cfg,D/'main','cuda','cuda',resume=D/'main/latest.pt',max_updates=1)
saved=torch.load(D/'main/latest.pt',map_location='cpu',weights_only=False);assert saved['global_step']==2 and saved['optimizer']['state']
other=SceneModel(seed_from_state(saved['state']),cfg).cuda();other.load_state_dict(saved['state']);optimizer=other.optimizer('geometry');optimizer.load_state_dict(saved['optimizer'])
for k,v in model.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],rtol=0,atol=0)
restored=optimizer.state_dict()
for key,state in saved['optimizer']['state'].items():
    for name,value in state.items():
        if torch.is_tensor(value):torch.testing.assert_close(value,restored['state'][key][name].cpu(),rtol=0,atol=0)
with torch.no_grad():
    frame=data.frame(18);g,p=model.buffers(frame,len(data.actor_ids),'cuda',isolated=False);first=model.train_image(p,g);g2,p2=other.buffers(frame,len(data.actor_ids),'cuda',isolated=False);second=other.train_image(p2,g2);difference=float((first-second).abs().max());assert difference<=1e-6
write(D/'schedule_recovery_verified.json',dict(status='FIRST_MAIN_UPDATE_AND_ACTUAL_NONEMPTY_ADAM_RESTORE_PASSED',actual_main_updates=2,adopted_step1_without_early_topology=True,discarded_extra_updates_range=[199,248],extra_probe_SGD=0,render_max_error=difference,model_bitwise_equal=True,Adam_moments_bitwise_equal=True,checkpoint='main/latest.pt',implementation='implementation_r15_train_v3',at=time.time()))
print('R15_SCHEDULE_REPAIR_STEP2_AND_RESTORE_VERIFIED',flush=True)
