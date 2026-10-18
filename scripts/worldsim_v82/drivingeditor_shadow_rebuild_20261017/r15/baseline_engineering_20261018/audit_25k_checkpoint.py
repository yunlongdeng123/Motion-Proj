"""Read-only durability check of the first new5K checkpoint; no GPU or updates."""
from pathlib import Path
import os,json,sys,importlib.util,random,datetime
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
import numpy as np,torch
torch.set_num_threads(2)
E=Path(__file__).resolve().parent
NEW=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018')
RUN=NEW/'runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k'

def main():
    step=25000;event=json.loads((RUN/'checkpoint_events/iteration_025000.json').read_text())
    assert event['status']=='complete' and event['iteration']==step
    assert set(event['checkpoints'])=={'static','dynamic','environment','training_state'}
    rows=[]
    for kind,row in event['checkpoints'].items():
        p=Path(row['path']);assert p.stat().st_size==row['bytes']>0
        if kind=='training_state':continue
        params,n=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True);assert n==step
        opts=[x for x in params if isinstance(x,dict) and 'param_groups' in x and 'state' in x]
        assert len(opts)==1 and opts[0]['state']
        for v in opts[0]['state'].values():
            for value in v.values():
                if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
        rows.append(dict(component=kind,bytes=row['bytes'],optimizer_groups=len(opts[0]['param_groups']),
                         optimizer_states=len(opts[0]['state']),Adam_tensors_finite=True))
        del params,opts
    p=Path(event['checkpoints']['training_state']['path'])
    state=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True)
    assert state['iteration']==step and state['sgcm'] is not None
    cfg=state['config'];schedule=list(range(5000,30001,5000))
    assert cfg['checkpoint_iterations']==cfg['save_iterations']==schedule
    random.Random().setstate(state['python_rng']);np.random.RandomState().set_state(state['numpy_rng'])
    torch.Generator().set_state(state['torch_rng'])
    assert len(state['cuda_rng'])==1 and state['hybrid_sampler']['steps']==step
    source=NEW/'training_source/scene0008_089_137/utils/paper_checkpoint.py'
    spec=importlib.util.spec_from_file_location('read_only_checkpoint_25k',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module._validate_resume_configuration(cfg,dict(cfg,model_path=str(RUN)+'_read_only_resume25k_audit',start_checkpoint=str(RUN/'chkpnt25000.pth')))
    out=dict(status='PASSED_COMPLETE_25K_DURABILITY_READONLY',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        step=step,components=rows,training_state_bytes=p.stat().st_size,all_four_components_match_complete_event=True,
        Python_NumPy_Torch_rng_valid=True,cuda_rng_count=1,hybrid_sampler_steps=step,SGCM_state_present=True,
        checkpoint_iterations=schedule,strict_native_resume_configuration_valid=True,
        evaluation_frames=len(event['frames']),actual_GPU_restore_at25K_verified=False,
        boundary='Saved-state CPU validation only. Running training was not restored or modified.',training_updates=0,human_verdict=None)
    (E/'GN11_25k_durability_receipt.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
