"""Append real-visible BG to the verified new field, then replay the new policy."""
from pathlib import Path
import argparse,copy,json,os,socket,time,uuid,sys
import numpy as np
import torch
import train_teacher_gaussians as d

ROOT=Path(__file__).resolve().parent.parent

def run(phase):
    a=argparse.Namespace(config=ROOT/'config/training.json',runtime_contract=ROOT/'config/runtime.json',execution_contract=ROOT/'config/execution.json',
        runtime_module=ROOT/'code/teacher_gaussian_runtime.py',objective_module=ROOT/'code/training_objective.py',lifecycle_module=ROOT/'code/full_gaussian_lifecycle.py',teacher_targets=None,mask_root=None)
    cfg,rc,ex,target,masks,binds=d.preflight(a)
    life=d.module(a.lifecycle_module,'_bg_lifecycle');module=d.module(a.runtime_module,'_bg_runtime');objective=d.module(a.objective_module,'_bg_objective')
    d.seed_all(torch,cfg['seed']);out=ROOT/'training_v1';stage=out/'probe';stage.mkdir(parents=True,exist_ok=True)
    origin=dict(phase=phase,pid=os.getpid(),host=socket.gethostname(),nonce=uuid.uuid4().hex)
    rt=module.TeacherGaussianRuntime(a.runtime_contract,ROOT/(phase+'_runtime'))
    cache_root=ROOT/'fixed_training_cache_v1'
    if phase=='prepare':objective.build_cache(rt,target,masks,cache_root,cfg)
    cache=objective.load_cache(torch,cache_root);cache_pin=d.pin(cache_root/'manifest.json')
    binds.update(cache_manifest=cache_pin);rt.frozen_assets['cache_manifest']=cache_pin['sha256'];binds['frozen_assets']=copy.deepcopy(rt.frozen_assets)
    opt=rt.make_optimizer(cfg);sampler=life.BalancedSampler(d.TRAIN,cfg['seed'],d.GROUPS)
    def envelope(step,ledger,active,replay,record=None,event=None):
        return dict(schema=d.SCHEMA,recovery_safe=True,complete_state=life.snapshot(rt,opt,sampler,step,binds),execution_ledger=ledger.state(),
            unique_training_seconds=active,origin=origin,replay_verification=replay,record=life.cpu(record),event=life.cpu(event),bindings=copy.deepcopy(binds))
    def save(path,payload):torch.save(payload,path)
    if phase=='prepare':
        source_path=d.pinned(ex['parent_checkpoint']);source=torch.load(source_path,map_location='cpu',weights_only=False)
        state=source['complete_state'];assert state['logical_step']==2000 and source['replay_verification']['passed']
        old_frozen=state['frozen_assets'];allowed={'candidate_pool','cache_manifest','real_background_manifest','retained_fragments'}
        assert {k:v for k,v in old_frozen.items()if k not in allowed}=={k:v for k,v in rt.frozen_assets.items()if k not in allowed}
        rebound=copy.deepcopy(state);rebound['bindings']=copy.deepcopy(binds);rebound['frozen_assets']=copy.deepcopy(rt.frozen_assets)
        opt=life.restore(rt,opt,sampler,rebound,binds)
        assert life.equal(rt.capture_model_state(),state['model'])
        before=life.cpu(rt.field.capture());before_opt=life.cpu(opt.state_dict());n0=len(rt.field.xyz)
        seed_path=d.pinned(ex['background_seed']);seed=module.load(seed_path);added=rt.field.append(seed,opt,born_step=2000)
        rt.used_candidate_ids.update(module.candidate_ids(seed).tolist())
        assert added==len(seed['xyz'])
        for key,old in before['parameters'].items():assert torch.equal(rt.field.capture()['parameters'][key][:n0].cpu(),old)
        after_opt=opt.state_dict()
        for key,old_state in before_opt['state'].items():
            for name,value in old_state.items():
                actual=after_opt['state'][key][name].detach().cpu()
                if value.ndim and value.shape[0]==n0:
                    assert torch.equal(actual[:n0],value)and not bool(actual[n0:].any())
                else:assert torch.equal(actual,value)
        data=copy.deepcopy(source['execution_ledger']);data['unique_budget']=2600;ledger=life.OptimizerEventLedger(data,unique_budget=2600)
        active=float(source['unique_training_seconds']);new_seconds=0.;records=[]
        transition=dict(parent_checkpoint=d.pin(source_path),parent_unique_updates=2000,road_parameters_preserved_exact=True,road_Adam_moments_preserved_exact=True,
            sampler_and_all_RNG_restored=True,background_seed=d.pin(seed_path),background_rows_added=added,old_rows=n0,new_rows=len(rt.field.xyz),
            input_policy_change='Append legitimate real-visible BG and increase residual-driven densification; keep all five groups trainable',
            frozen_binding_changes=sorted(k for k in set(old_frozen)|set(rt.frozen_assets)if old_frozen.get(k)!=rt.frozen_assets.get(k)))
        d.write(stage/'transition.json',transition);save(stage/'checkpoint002000_with_BG.pt',envelope(2000,ledger,active,None))
        curve=[dict(step=2000,rows=d.inference(rt,life,lambda:objective.evaluate(rt,cache,cfg['fixed_eval_frames'])))]
        for step in range(2001,2017):
            record,event,seconds=d.checked_step(rt,opt,sampler,cache,cfg,objective,step)
            active+=seconds;new_seconds+=seconds;ledger.record_update(step,topology=event);records.append(dict(step=step,active_seconds=seconds,record=record,event=event))
            if step==2015:save(stage/'checkpoint002015.pt',envelope(step,ledger,active,None,record,event))
            if step==2016:save(stage/'expected002016.pt',envelope(step,ledger,active,None,record,event))
        curve.append(dict(step=2016,rows=d.inference(rt,life,lambda:objective.evaluate(rt,cache,cfg['fixed_eval_frames']))))
        d.inference(rt,life,lambda:d.readout(rt,stage/'actual_step002016',2016,cfg['readout_frames']))
        d.write(stage/'losses.json',records);d.write(stage/'fixed_eval_curve.json',curve)
        d.write(stage/'receipt.json',dict(complete=True,phase='continuation_probe',logical_step=2016,new_unique_updates=16,bindings=binds,
            projected_unique_training_seconds=new_seconds/16*600,transition=transition,execution_ledger=ledger.state(),original_native_parameters_frozen=True))
        print(json.dumps(dict(phase=phase,logical_step=2016,point_count=len(rt.field.xyz),new_unique_updates=16)),flush=True)
    else:
        source=torch.load(stage/'checkpoint002015.pt',map_location='cpu',weights_only=False);expected=torch.load(stage/'expected002016.pt',map_location='cpu',weights_only=False)
        assert source['origin']['pid']!=os.getpid()
        opt=life.restore(rt,opt,sampler,source['complete_state'],binds)
        record,event,seconds=d.checked_step(rt,opt,sampler,cache,cfg,objective,2016)
        actual=life.snapshot(rt,opt,sampler,2016,binds)
        replay=d.compare_replay(life,expected['complete_state'],actual,expected['record'],record,expected['event'],event,cfg)
        replay.update(fresh_process=True,step=2016,engineering_seconds=seconds)
        ledger=life.OptimizerEventLedger(expected['execution_ledger'],unique_budget=2600);ledger.record_update(2016,replay=True,topology=event)
        payload=envelope(2016,ledger,expected['unique_training_seconds'],replay,record,event)
        save(stage/'verified_checkpoint002016.pt',payload);d.write(stage/'step2016_replay.json',replay)
        print(json.dumps(replay),flush=True)
    rt.assert_fixed()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','verify'],required=True);run(p.parse_args().phase)
