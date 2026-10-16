"""No-update rollback of unverified extra candidates after visual rejection of union."""
import argparse,json,traceback
import numpy as np
import torch
from run_union_refinement import UnionExperiment
from run_experiments import BASE,write
import renderer

class SafeExperiment(UnionExperiment):
    def __init__(self,args):
        super().__init__(args)
        previous=torch.load(BASE/'runs/deferred_1000_v1/R4/addon_001000.pt',map_location='cpu',weights_only=False)
        keep=np.isin(self.selection['artifact_ids'],previous['road_addon']['artifact_ids'].numpy())
        for name in ['artifact_ids','artifact_xyz_anchor','artifact_base_opacity']:self.selection[name]=self.selection[name][keep]
        self.warm=torch.load(BASE/'runs/union_1000_v1/R4/addon_001000.pt',map_location='cpu',weights_only=False)
        self.contract.update(inference_variant='verified_receiver_expansion_only',
            added_unverified_other_candidates_reverted_to_parent=True,
            receiver_count=len(self.selection['receiver_ids']),other_count=len(self.selection['artifact_ids']),
            training_updates=0,rollback_reason='all49 broad candidate freeze produced localized black shards outside accepted road support')
        write(self.out/'runtime_contract.json',self.contract)
    def run(self):
        self.gradient_smoke();(self.out/'R4').mkdir();self.activate('R4',self.shadow())
        metrics=self.evaluate('R4')
        torch.save({'schema':'drivingeditor_shadow_local_inference_v1','runtime_contract':self.contract,
            'road_addon':self.active.addon_state(),'shadow_state':self.active.shadow.state_dict(),
            'optimizer':None,'training_updates_in_this_variant':0,'source_addon':str(BASE/'runs/union_1000_v1/R4/addon_001000.pt')},self.out/'inference_bundle.pt')
        self.verify_sources();self.update(state='complete',phase='zero_update_safe_coverage_rollback',metrics=metrics,
            training_updates=0,parent_checkpoint_unchanged=True,source_parameters_unchanged=True)
        print(json.dumps(metrics))

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();a.steps=0;a.warmup=1;e=None
try:e=SafeExperiment(a);e.run()
except BaseException as error:
    if e:e.update(state='failed',error=str(error),traceback=traceback.format_exc())
    raise
finally:
    if e:renderer.prepare_static=e.original_prepare;e.slot.close()
