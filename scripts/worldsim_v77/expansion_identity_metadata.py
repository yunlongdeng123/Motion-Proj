"""用保存的原始文件名前缀分组日志，防止同段别名跨集合。"""
from pathlib import Path
import sys,json,itertools
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928/r1');reg=read(ROOT/'registration.json');batch=read('/root/autodl-tmp/data/worldsim_v5/manifests/m1_development_raw_batch_v1.json');mapping={str(s['scene_index']):(s['scene_name'],s['manifest']) for s in batch['scenes']};mapping['191']=('scene-0242','/root/autodl-tmp/data/dynamic_editing_v2/manifests/scene-0242_raw_manifest.json');rows=[]
for sid,(name,manifest) in mapping.items():
    m=read(manifest);files=m['files'];prefixes=sorted({Path(x['filename']).name.split('__')[0] for x in files if '__CAM_' in x['filename']});s=next((s for s in reg['new_scenes'] if s['processed_id']==sid),None);rows.append(dict(processed_id=sid,scene_name=name,raw_log_prefixes=prefixes,split=s['split'] if s else 'training_pool_candidate',manifest=manifest))
leaks=[]
for a,b in itertools.combinations(rows,2):
    shared=set(a['raw_log_prefixes'])&set(b['raw_log_prefixes'])
    if shared and a['split']!=b['split']:leaks.append(dict(a=a['scene_name'],b=b['scene_name'],logs=sorted(shared)))
dump(ROOT/'identity_metadata.json',dict(rows=rows,log_overlap_between_splits=leaks,scope='Saved raw filename log prefixes; pretrained corpus overlap unknown; v77 holdout does not erase historical V5 source exposure.'))
print(json.dumps(dict(rows=rows,leaks=leaks),ensure_ascii=False))
