"""按已验证公共原包来源的清单清除展开 sweeps；仅清单文件，逐批记账。"""
import gzip,itertools,json,os,time
from pathlib import Path
from prune_storage import ROOT,AUDIT,open_paths

base=ROOT/'data/worldsim_v4/drivestudio_raw_trainval/sweeps'
plan=AUDIT/'sweeps-plan.jsonl.gz';assert json.loads((AUDIT/'restore-probe-result.json').read_text())['bytewise_equal']

def check(r,active):
 p=Path(r['path']);assert p.is_relative_to(base) and p.resolve()==p and not p.is_symlink()
 s=p.stat();assert (s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns,s.st_nlink)==tuple(r[k] for k in ['inode','device','bytes','mtime_ns','links'])
 assert s.st_nlink==1 and str(p) not in active
 assert Path(r['restore']['archive']).is_file() and Path(r['restore']['manifest']).is_file()
 return p

active=open_paths();count=0;total=0
with gzip.open(plan,'rt') as f:
 for line in f:check(json.loads(line),active)
before=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
with gzip.open(plan,'rt') as f,(AUDIT/'sweeps-deletions.jsonl').open('x') as log:
 while batch:=list(itertools.islice(f,256)):
  rows=[json.loads(line) for line in batch];active=open_paths()
  paths=[check(row,active) for row in rows]
  log.write(json.dumps({'event':'delete_batch_intent','paths':[str(p) for p in paths]})+'\n');log.flush();os.fsync(log.fileno())
  for row,p in zip(rows,paths):
   check(row,active).unlink();count+=1;total+=row['allocated']
  log.write(json.dumps({'event':'batch_deleted','count':len(rows),'total':count})+'\n');log.flush();os.fsync(log.fileno())
  if count%4096==0:print(json.dumps({'deleted':count,'GiB':total/2**30}),flush=True)
result={'deleted_files':count,'released_allocated_bytes':total,'available_before':before,'available_after':os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize,'public_archives_retained':True}
(AUDIT/'sweeps-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
