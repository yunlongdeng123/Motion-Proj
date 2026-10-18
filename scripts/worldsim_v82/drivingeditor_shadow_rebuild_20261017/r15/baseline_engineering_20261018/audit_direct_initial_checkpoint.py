"""Read-only check of the real first update of the same instance30K budget."""
from pathlib import Path
import os,sys,json,datetime
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
import torch
torch.set_num_threads(2)
E=Path(__file__).resolve().parent;D=E/'direct_instance30k'
sys.path.insert(0,str(E/'direct_implementation'))
from training_policy import validate_resume_checkpoint

def main():
    p=D/'checkpoint_000001.pt';s=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True)
    assert s['step']==1
    effective=json.loads((D/'effective_config.json').read_text())
    schedule=[1,5000,10000,15000,20000,25000,30000]
    assert effective['arguments']['checkpoint_steps']==schedule
    policy=effective['training_policy']
    contract=validate_resume_checkpoint(s,policy,30000,expected_step=1)
    spec=s['spec'];bg=Path(spec['background_checkpoint']);env=Path(spec['environment_checkpoint'])
    assert bg.name=='chkpnt30000.pth' and env.name=='env_light_chkpnt30000.pth'
    assert bg.parent==env.parent and bg.parent.name.endswith('GN11_resume20k')
    assert spec['baseline_extension']['identities_match_both_stages'] is True
    assert spec['baseline_extension']['load_old_trained_actors'] is False
    assert spec['baseline_extension']['load_R15_parameters'] is False
    count=[0]
    def finite(x):
        if isinstance(x,torch.Tensor):
            assert torch.isfinite(x).all();count[0]+=1
        elif isinstance(x,dict):
            for y in x.values():finite(y)
        elif isinstance(x,(tuple,list)):
            for y in x:finite(y)
    for key in ['background','instances','environment']:finite(s[key])
    out=dict(status='PASSED_ACTUAL_INSTANCE_STEP1_COMPLETE_CHECKPOINT_CPU_AUDIT',
        at=datetime.datetime.now(datetime.timezone.utc).isoformat(),saved_step=1,target_step=30000,
        actual_checkpoint_bytes=p.stat().st_size,checkpoint_steps=schedule,
        source_background=str(bg),source_environment=str(env),same_five_identity_set=True,
        instance_ids=s['instances']['instance_ids'],all_saved_model_Adam_tensors_finite=True,
        tensor_fields_checked=count[0],sampler_steps=s['sampler']['steps'],
        strict_resume_contract=contract,background_exclusion=s['background_exclusion'],
        saved_state_audit_only=True,GPU_restore_at_step1_verified=False,
        audit_training_updates=0,first_training_update_is_inside_original30K_budget=True,human_verdict=None)
    (E/'direct_initial_checkpoint_receipt.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:out[k] for k in ['status','saved_step','actual_checkpoint_bytes','checkpoint_steps','tensor_fields_checked','sampler_steps','background_exclusion']}))
if __name__=='__main__':main()
