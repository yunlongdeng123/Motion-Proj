"""合并旧实验中的相同缓存与完整 checkpoint，所有原路径保持可读。"""
import collections,json,os
from pathlib import Path
from prune_storage import ROOT,AUDIT,describe,open_paths

def equal(a,b):
 with a.open('rb') as x,b.open('rb') as y:
  if x.read(4096)!=y.read(4096):return False
  while True:
   b=x.read(1024*1024);c=y.read(1024*1024)
   if b!=c:return False
   if not b:return True

groups=collections.defaultdict(list); linked=0;total=0;scanned=0;active=open_paths()
before=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
with (AUDIT/'legacy-dedup.jsonl').open('x') as log:
 for row in map(json.loads,(AUDIT/'inventory.jsonl').open()):
  p=Path(row['path']);parts=p.parts
  if not p.is_file() or p.is_symlink():continue
  if '/runs/' not in str(p) or '/worldsim_v75/' in str(p):continue
  if p.suffix=='.npy' and any('/'+v+'/' in str(p) for v in ['worldsim_v6','worldsim_v63','worldsim_v64','worldsim_v65','worldsim_v67']):
   scene=next((x for x in parts if x.startswith('scene-')),'')
   key=(p.name,p.parent.name,scene)
  elif p.name in {'env.pth','checkpoint_final.pth','step-000030000.ckpt','latest.pt','final.pt'}:key=(p.name,)
  else:continue
  s=p.stat();key=key+(s.st_size,);scanned+=1
  matched=False
  for src in groups[key]:
   t=src.stat()
   if (s.st_dev,s.st_ino)==(t.st_dev,t.st_ino):matched=True;break
   if not equal(src,p):continue
   matched=True
   if s.st_nlink!=1:break
   if linked%50==0:active=open_paths()
   assert str(src) not in active and str(p) not in active
   record=describe(p,'legacy_exact_copy',{'retained_source':str(src),'operation':'恢复独立副本时复制源文件；读取无需恢复'})
   assert p.resolve()==p and src.resolve()==src
   assert p.stat().st_ino==s.st_ino and src.stat().st_ino==t.st_ino
   tmp=p.with_name(p.name+'.dedup-tmp');assert not tmp.exists()
   log.write(json.dumps(dict(record,event='intent'),ensure_ascii=False)+'\n');log.flush();os.fsync(log.fileno())
   os.link(src,tmp);os.replace(tmp,p);assert p.stat().st_ino==src.stat().st_ino
   log.write(json.dumps(dict(record,event='linked'),ensure_ascii=False)+'\n');log.flush()
   linked+=1;total+=s.st_blocks*512
   if linked%25==0:print(json.dumps({'linked':linked,'GiB':total/2**30}),flush=True)
   break
  if not matched:groups[key].append(p)
result={'scanned':scanned,'linked':linked,'released_allocated_bytes':total,'available_before':before,'available_after':os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize}
(AUDIT/'legacy-dedup-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
