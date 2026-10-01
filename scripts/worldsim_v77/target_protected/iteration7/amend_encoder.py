"""发现实质加载错误后停原控制；登记只改encoder的同r6强控制。"""
from pathlib import Path
import json,shutil
TASK=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');ROOT=TASK/'r7'
def main():
    p=ROOT/'protocol_amendment_encoder.json'
    if p.exists():return
    state=json.loads((ROOT/'native_spatial/training/state.json').read_text())
    amendment={'trigger':'r6 training/logs 106 missing first_stage_model.encoder weights; inference 0missing cannot certify train initialization',
               'confirmed_prior_unsafe_training':{'r6_steps':160,'missing_encoder_tensors':106,'frozen_random_encoder':True},
               'interrupted_arm':{'id':'native_spatial','steps':state['steps'],'state':'stopped_engineering_invalid','power_action':False},
               'cancelled_unrun_arm':'native_contextual; cannot infer data/module causality before fixing latent target',
               'new_arms':[{'id':'encoder_fixed_lowres','size':[320,576],'steps':160,'modules':'same80spatial','changed':'only restore106 official SVD target encoder weights'},
                           {'id':'encoder_fixed_native','size':[576,1024],'steps':160,'modules':'same80spatial','changed':'only resolution vs encoder_fixed_lowres; execute if residual diagnosis needed'}],
               'fixed':'same r6 data, train/split, seeds, steps, optimizer, loss, original architecture; no regeneration/relabeling',
               'baseline_report_correction':'r6 0missing claim belongs to evaluation, not training; loss drop invalid as training success evidence',
               'stop':'finish bounded repaired comparison; no module/loss grid without evidence'}
    p.write_text(json.dumps(amendment,ensure_ascii=False,indent=2)+'\n')
    state.update(stage='stopped_engineering_invalid',reason=amendment['trigger']);(ROOT/'native_spatial/training/state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    for c in amendment['new_arms']:
        folder=ROOT/c['id'];folder.mkdir();shutil.copy2(TASK/'r6/dataset_catalog.json',folder/'dataset_catalog.json')
    print(json.dumps(amendment,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
