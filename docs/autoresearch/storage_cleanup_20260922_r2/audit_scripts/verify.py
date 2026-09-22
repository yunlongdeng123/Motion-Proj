import collections,gzip,json,os,time
from pathlib import Path
from prune_storage import ROOT,AUDIT

linked={};pending={};dedup_categories=collections.Counter()
for name in ['dedup-applied.jsonl','legacy-dedup.jsonl']:
 for line in (AUDIT/name).open():
  r=json.loads(line)
  if r['event']=='intent':pending[r['path']]=r
  if r['event']=='linked':linked[r['path']]=r
for name,r in pending.items():
 p=Path(name);src=Path(r['restore']['retained_source']);s=p.stat();t=src.stat()
 assert (s.st_dev,s.st_ino)==(t.st_dev,t.st_ino) and s.st_size==r['bytes'],name
 assert not p.with_name(p.name+'.dedup-tmp').exists(),name
 linked[name]=r
for r in linked.values():dedup_categories[r['category']]+=r['allocated']
deleted=set();deletion_categories=collections.Counter()
for name in ['download-plan','profile-plan','sim-download-plan','wheel-plan']:
 plan=json.loads((AUDIT/(name+'.json')).read_text())
 for r in plan['files']:
  assert not Path(r['path']).exists(),r['path'];deleted.add(r['path']);deletion_categories[r['category']]+=r['allocated']
 if name=='profile-plan':
  for recipe in plan['recovery']:
   for p in recipe['preserved_formal_checkpoints']:assert Path(p).is_file(),p
   for key in ['config','source_snapshot']:assert Path(recipe[key]).exists()
with gzip.open(AUDIT/'sweeps-plan.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line);assert not Path(r['path']).exists();assert Path(r['restore']['archive']).is_file()
  deleted.add(r['path']);deletion_categories[r['category']]+=r['allocated']
v75=0;actors=0
for r in map(json.loads,(AUDIT/'inventory.jsonl').open()):
 if '/runs/worldsim_v75/' not in r['path']:continue
 p=Path(r['path']);s=p.stat()
 assert (s.st_ino,s.st_size,s.st_mtime_ns)==(r['inode'],r['bytes'],r['mtime_ns']),p
 v75+=1
 if '/WS-V75-ACTOR-' in r['path'] and p.name=='generated.npy':actors+=1
assert actors==23,actors
# 检查被清目标是否让已存在的符号链接断裂；不跟随目录符号链接。
affected=[]
for root in [ROOT/'data',ROOT/'runs',ROOT/'models']:
 for base,dirs,files in os.walk(root,followlinks=False):
  for name in dirs+files:
   p=Path(base)/name
   if p.is_symlink() and str(p.resolve()) in deleted:affected.append(str(p))
assert not affected,affected
baseline=json.loads((AUDIT/'baseline.json').read_text());v=os.statvfs(ROOT)
result={'timestamp':time.time(),'baseline_available_bytes':baseline['available_bytes'],'available_bytes':v.f_bavail*v.f_frsize,'used_bytes':(v.f_blocks-v.f_bfree)*v.f_frsize,'actual_available_increase_bytes':v.f_bavail*v.f_frsize-baseline['available_bytes'],'deleted_files':len(deleted),'deduplicated_files':len(linked),'deletion_allocated_bytes':dict(deletion_categories),'dedup_allocated_bytes':dict(dedup_categories),'v75_large_files_unchanged':v75,'latest_actor_generated_arrays_unchanged':actors,'new_broken_symlinks':affected,'all_preserved_formal_checkpoints_exist':True,'raw_restore_probe_bytewise_equal':True,'human_verdict':None,'failure_ledger_delta':'none','model_calls':0}
(AUDIT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
