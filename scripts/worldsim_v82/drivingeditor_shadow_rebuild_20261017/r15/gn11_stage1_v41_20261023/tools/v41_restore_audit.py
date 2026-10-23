"""Read-only checks immediately after actual CUDA restore, before the next update.

This distinguishes state restoration from numerical divergence between fresh
CUDA trajectories. It never changes RNG, optimizer, sampling, or native files.
"""
import json
from pathlib import Path
import torch
from compare_v1_states import differences


def install(output):
    from utils import paper_checkpoint as native
    from utils.v41_road import DenseRoadTraining
    original_native, original_road = native.restore_training_state, DenseRoadTraining.restore
    output = Path(output)

    def restore_native(path, iteration, scene, models, args, sgcm=None):
        stack, ema = original_native(path, iteration, scene, models, args, sgcm=sgcm)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        current = native.capture_training_state(iteration, scene, stack, models, args, sgcm=sgcm, ema=ema)
        for record in (saved,current):
            record['config'] = {k:v for k,v in record['config'].items() if k not in ('model_path','start_checkpoint')}
        errors = differences(saved,current,'continuation')
        for name,prefix in (('static','chkpnt'),('dynamic','dynamic_chkpnt'),('environment','env_light_chkpnt')):
            expected,step = torch.load(Path(path).parent/f'{prefix}{iteration}.pth',map_location='cpu',weights_only=False)
            assert step == iteration
            errors.extend(differences(expected,models[name].capture(),name))
        receipt = dict(status='PASS_EXACT_CUDA_RESTORE' if not errors else 'FAIL_EXACT_CUDA_RESTORE',
            iteration=iteration,optimizer_updates=0,differences=errors,
            includes='static/dynamic/environment parameters and optimizers; Python/NumPy/Torch/CUDA RNG; sampler; SGCM; runtime; EMA',
            excluded_config=['model_path','start_checkpoint'])
        with (output/f'native_restore_audit{iteration}.json').open('x') as f:json.dump(receipt,f,indent=2)
        if errors:raise RuntimeError('native restore audit failed: '+str(errors))
        return stack,ema

    def restore_road(self,path,iteration):
        original_road(self,path,iteration)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        errors=differences(saved,self.state(iteration),'road')
        receipt=dict(status='PASS_EXACT_ROAD_RESTORE' if not errors else 'FAIL_EXACT_ROAD_RESTORE',
            iteration=iteration,optimizer_updates=0,differences=errors,
            includes='all road layer tensors, Adam, independent RNG/sampler, stack, contract and cumulative time')
        with (output/f'road_restore_audit{iteration}.json').open('x') as f:json.dump(receipt,f,indent=2)
        if errors:raise RuntimeError('road restore audit failed: '+str(errors))

    native.restore_training_state=restore_native
    DenseRoadTraining.restore=restore_road
