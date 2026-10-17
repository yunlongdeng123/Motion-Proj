"""Actual production-capacity forward/backward/restore checks; no SGD updates."""
from pathlib import Path
import sys,json,time,importlib.util,gc
import numpy as np,torch
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'backend_restore_r15';sys.path.insert(0,str(R/'implementation_r15_train_v2/shadow_rebuild_v3'))
from shadow_rebuild.config import Config
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import preflight,capture,seed_from_state,save,write
from shadow_rebuild.objective import objective
from shadow_rebuild.gpu_probe import probe
from shadow_rebuild.sgcm_adapter import make_controller,refine
torch.set_num_threads(2);cfg=Config.load(D/'config.json');data=Dataset(D/'data','cpu',True);receipt=preflight(data,cfg);write(D/'input_preflight.json',receipt)
assert len(data.seed['xyz'][data.seed['owner']<0])>=500000 and cfg.env_map_res==1024 and cfg.sh_degree==3
assert torch.cuda.is_available();abi=probe(cfg);data.device='cuda';model=SceneModel(data.seed,cfg).cuda();generator=torch.Generator().manual_seed(0)
# Exact native old cube kernel and coordinate transform comparison, same weights.
source=R.parent/'rigid_source/scene0008_089_137/scene/envlight.py';spec=importlib.util.spec_from_file_location('original_envlight_probe',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);old=module.EnvLight(1024).cuda()
with torch.no_grad():old.base.copy_(model.environment.base)
directions=torch.randn(1,32,32,3,device='cuda');a=old(directions);b=model.environment(directions);error=float((a-b).abs().max());assert error==0
del old,a,b;gc.collect();torch.cuda.empty_cache();records=[]
for stage,fi in [('geometry',18),('material',44),('joint',48)]:
    model.set_stage(stage);model.field.active_sh_degree.fill_(3);model.zero_grad(set_to_none=True);frame=data.frame(fi);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    loss,terms,package=objective(model,frame,len(data.actor_ids),'cuda',stage,generator,None,6000);loss.backward();torch.cuda.synchronize();seconds=time.monotonic()-start
    assert torch.isfinite(loss)
    norms={}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        assert torch.isfinite(p.grad).all(),name;norms[name]=float(p.grad.norm())
    assert norms['field.albedo_logit']>0 and norms['field.features_dc']>0 and norms['field.features_rest']>0 and norms['environment.base']>0
    if stage=='geometry':
        assert norms['field.xyz']>0
        # Actual no-clean-reference road points still receive measured RGB gradient.
        missing=model.field.road&(model.field.reference_confidence==0);active=int((model.field.albedo_logit.grad[missing].abs().sum(-1)>0).sum());assert active>0
    else:active=None
    records.append(dict(stage=stage,seconds_forward_backward=seconds,peak_GiB=torch.cuda.max_memory_allocated()/2**30,loss=float(loss),gradient_norms=norms,no_reference_road_points_with_RGB_gradient=active,SGD_updates=0))
    write(D/'GPU_selfcheck_partial.json',dict(stages=records,SGD_updates=0))
    del loss,terms,package,frame;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
model.set_stage('geometry');opt=model.optimizer('geometry');model.schedule(opt,0)
# Actual SGCM adapter call on a disposable model; no point crop may affect Actors.
frame=data.frame(18);g,p=model.buffers(frame,len(data.actor_ids),'cuda');before=model.field.xyz[model.field.owner>=0].detach().clone();ctl=make_controller(data,cfg);sgcm=refine(ctl,model.field,opt,p,550);torch.testing.assert_close(model.field.xyz[model.field.owner>=0],before,rtol=0,atol=0)
del g,p,frame,before;gc.collect();torch.cuda.empty_cache()
saved=capture(model,opt,'geometry',0,0,generator,None);saved['input_contract']=data.manifest;save(D/'probe_restore_only.pt',saved)
other=SceneModel(seed_from_state(saved['state']),cfg).cuda();other.load_state_dict(saved['state']);o=other.optimizer('geometry');o.load_state_dict(saved['optimizer'])
for k,v in model.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],rtol=0,atol=0)
write(D/'GPU_selfcheck.json',dict(status='ACTUAL_PRODUCTION_CAPACITY_GRADIENTS_AND_RESTORE_PASSED',steps=0,ABI=abi,cube_old_new_max_error=error,stages=records,SGCM=sgcm,Actor_unchanged_by_static_SGCM=True,checkpoint_model_optimizer_loaded=True,production_static_seeds=receipt['gaussians']-int((data.seed['owner']>=0).sum()),full_RGB_all_stages=True,legacy_SH3=True,cube_shape=list(model.environment.base.shape)))
print(json.dumps(records,indent=2));print('R15_GPU_SELFCHECK_PASSED_ZERO_SGD',flush=True)
