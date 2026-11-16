import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82')
from drivingeditor_dynamic_edit.road_texture_training import differences

p=argparse.ArgumentParser()
p.add_argument('--left',required=True,type=Path);p.add_argument('--right',required=True,type=Path)
p.add_argument('--output',required=True,type=Path);a=p.parse_args()
left=torch.load(a.left,map_location='cpu',weights_only=False);right=torch.load(a.right,map_location='cpu',weights_only=False)
diff=differences(left,right)
delta=float((left['delta_raw']-right['delta_raw']).abs().max())
moments={key:float((left['optimizer']['state'][0][key]-right['optimizer']['state'][0][key]).abs().max())
    for key in ['exp_avg','exp_avg_sq']}
other_left={key:value for key,value in left.items() if key not in ['delta_raw','optimizer']}
other_right={key:value for key,value in right.items() if key not in ['delta_raw','optimizer']}
nonfloat=differences(other_left,other_right)
report=dict(schema='world_texture_next_step_replay.v1',left=str(a.left),right=str(a.right),
    full_next_state_bitexact=not diff,differences=diff,delta_max_abs_difference=delta,
    adam_moment_max_abs_difference=moments,RNG_sampler_step_and_binding_exact=not nonfloat,
    full_state_restore_required_before_each_step=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
if delta>1e-5 or max(moments.values())>1e-6 or nonfloat:
    raise RuntimeError('颜色阶段下一步重放不满足数值恢复约束')
