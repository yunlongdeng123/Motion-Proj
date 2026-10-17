"""R15资源与数据存在性检查；记录最新人类授权，不读取密钥。"""
from pathlib import Path
import datetime,importlib.util,json,os,subprocess
import numpy as np
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
D=R/'backend_restore_r15';C=D/'r15_closure_20261018'
record={'at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'latest_human_instruction':'GPU opened; continue bounded R15 closure; shutdown AutoDL when no tasks remain; report progress',
 'original_45k_resume_allowed':False,'bounded_experiments_allowed':True,
 'shutdown_authorized':True,'shutdown_condition':'save all results/code, verify no GPU or CPU worker/controller remains, then shutdown',
 'subagents_allowed':False,'gpu_budget_minutes_cap':180,'stage_minutes_caps':[60,45,75],
 'triton_available':importlib.util.find_spec('triton') is not None,
 'shutdown_path':str(Path('/usr/bin/shutdown').resolve()),'inputs':{}}
for name in ['body_bank_a06.npz','body_bank_a37.npz','source_candidate_evidence_a06.npz',
             'source_candidate_evidence_a37.npz','actual_world_road_support.npz']:
 p=D/name
 if p.exists():
  with np.load(p,allow_pickle=False) as data:
   record['inputs'][name]={k:list(data[k].shape) for k in data.files}
 else:record['inputs'][name]='MISSING'
for name in ['gates/003000/owner/owner.json','gates/003000/split/split.json']:
 p=D/name
 if p.exists():record['inputs'][name]=json.loads(p.read_text())
(C/'execution_policy.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in record.items() if k!='inputs'},ensure_ascii=False))
print(json.dumps({k:v for k,v in record['inputs'].items() if k.endswith('.npz')},ensure_ascii=False))
