"""冻结公开源列表与开发/保留角色；不读取新的生成结果或选择模型失败。"""
import json
import re
import zipfile
from datetime import datetime,timezone
from pathlib import Path
from huggingface_hub import HfApi
from common import SCENE

OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-COHORT-01/20260920-r1')
OUT.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(SCENE) as z:
    inventory=[{'name':i.filename,'compressed_bytes':i.compress_size,'bytes':i.file_size} for i in z.infolist()]
(OUT/'existing_scene_members.json').write_text(json.dumps(inventory,indent=2)+'\n')
api=HfApi()
info=api.repo_info('nvidia/omni-dreams-scenes',repo_type='dataset')
items=[]
for f in api.list_repo_tree('nvidia/omni-dreams-scenes',repo_type='dataset',revision=info.sha,path_in_repo='scenes',recursive=False):
    if not hasattr(f,'size'):
        continue
    items.append({'path':f.path,'bytes':f.size})
pattern=re.compile(r'^scenes/clipgt-([0-9a-f-]{36})[.]usdz$')
default=sorted([x for x in items if pattern.match(x['path'])],key=lambda x:x['path'])
old=SCENE.name
fresh=[x for x in default if Path(x['path']).name!=old]
for i,x in enumerate(fresh[:6]):
    x['role']='development' if i<3 else 'heldout_reserved'
    x['scene_uuid']=pattern.match(x['path']).group(1)
result={'task_id':'WS-V75-COHORT-01','run_id':'20260920-r1',
        'status':'cohort_ready' if len(fresh)>=6 else 'insufficient_independent_scenes',
        'required_new_scenes':6,'available_new_scenes':len(fresh),
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'repo_id':'nvidia/omni-dreams-scenes',
        'revision':info.sha,'all_files':items,'default_scene_count':len(default),
        'rule':'Exclude the already-viewed scene; lexicographically first 3 default UUIDs for development, next 3 reserved. Weather variants are not independent sources.',
        'selected':fresh[:6] if len(fresh)>=6 else [],
        'unselected_available':fresh if len(fresh)<6 else [],
        'heldout_output_viewed':False,'model_generation_calls':0,
        'claim_boundary':'目录不足是数据限制，不是模型失败。天气版本不算独立来源。'}
path=OUT/'cohort_manifest.json'
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','default_scene_count','available_new_scenes','selected']},indent=2),flush=True)
