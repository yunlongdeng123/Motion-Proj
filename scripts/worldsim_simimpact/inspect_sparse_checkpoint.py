"""运行环境准备期间仅核查作者 checkpoint 的配置与初始化数组。"""
import json
from pathlib import Path
import torch
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1')
p=R/'assets/sparsedrive_stage2.pth';c=torch.load(p,map_location='cpu');s=c['state_dict']
keys=['head.det_head.instance_bank.anchor','head.map_head.instance_bank.anchor','head.motion_plan_head.motion_anchor','head.motion_plan_head.plan_anchor']
result={'checkpoint_bytes':p.stat().st_size,'top_level_keys':list(c),'state_tensor_count':len(s),'initialization_parameters':{k:{'shape':list(s[k].shape),'dtype':str(s[k].dtype),'finite':bool(torch.isfinite(s[k]).all())} for k in keys},'metadata_keys':list(c.get('meta',{})),'forward_count':0}
for k in ['epoch','iter','mmcv_version','mmdet_version']:
    if k in c.get('meta',{}):result[k]=str(c['meta'][k])
if isinstance(c.get('meta',{}).get('config'),str):(R/'checkpoint_training_config.py').write_text(c['meta']['config'])
(R/'checkpoint_structure.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
