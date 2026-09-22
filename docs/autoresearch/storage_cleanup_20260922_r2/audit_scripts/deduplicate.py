"""逐字节比较冻结数据副本，原子替换为硬链接；保留路径与内容。"""
import argparse,collections,json,os,stat,time
from pathlib import Path
from prune_storage import ROOT,AUDIT,open_paths,describe

def equal(a,b):
 with a.open('rb') as x,b.open('rb') as y:
  if x.read(4096)!=y.read(4096):return False
  while True:
   p=x.read(1024*1024); q=y.read(1024*1024)
   if p!=q:return False
   if not p:return True

def candidates():
 d=ROOT/'data/worldsim_v4'
 # 原始图像与处理图像的尺寸/内容相同时共享存储，不依赖重新预处理。
 for base in ['drivestudio_raw_trainval','drivestudio_processed_10Hz','drivestudio_processed_10Hz_10Hz']:
  for folder,dirs,files in os.walk(d/base,followlinks=False):
   for name in sorted(files):
    p=Path(folder)/name
    if p.suffix in {'.jpg','.bin','.json'} and not p.is_symlink(): yield p,'v4_data'
 # 大型旧 checkpoint 只合并完全相同的文件，不合并张量近似相同的权重。
 for row in map(json.loads,(AUDIT/'inventory.jsonl').open()):
  p=Path(row['path'])
  if p.is_file() and p.is_relative_to(ROOT/'runs') and p.name in {'env.pth','checkpoint_final.pth','step-000030000.ckpt','latest.pt','final.pt'} and 'worldsim_v75' not in str(p):yield p,'legacy_checkpoint'

ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
by_size=collections.defaultdict(list); counts=collections.Counter(); rows=[]; scanned=0; active=open_paths()
before=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
logpath=AUDIT/('dedup-applied.jsonl' if args.apply else 'dedup-candidates.jsonl')
if logpath.exists():
 for r in map(json.loads,logpath.open()):
  if r['event']=='linked':rows.append(r);counts[r['category']]+=r['allocated']
with logpath.open('a') as log:
 for p,category in candidates():
  s=p.lstat(); scanned+=1
  if s.st_size<65536:continue
  # 直接使用少量原始字节缩小比较范围；无哈希，最终仍逐字节核验全文件。
  with p.open('rb') as peek:prefix=peek.read(1536)
  key=(category,s.st_size,prefix)
  if scanned%10000==0:print(json.dumps({'scanned':scanned,'linked':len(rows)}),flush=True)
  if '/drivestudio_raw_trainval/' in str(p):
   by_size[key].append(p);continue
  matched=False
  for src in by_size[key]:
   t=src.stat()
   if (s.st_dev,s.st_ino)==(t.st_dev,t.st_ino):matched=True;break
   if not equal(src,p):continue
   matched=True
   if s.st_nlink!=1:break
   record=describe(p,category,{'operation':'copy retained_source to independent file if needed','retained_source':str(src)})
   record['source_inode']=t.st_ino
   if args.apply:
    if len(rows)%100==0:active=open_paths()
    assert str(src) not in active and str(p) not in active
    assert src.resolve()==src and p.resolve()==p and p.is_relative_to(ROOT)
    now=p.stat();new=src.stat()
    assert (now.st_ino,now.st_size,now.st_mtime_ns,now.st_nlink)==(s.st_ino,s.st_size,s.st_mtime_ns,1)
    assert (new.st_ino,new.st_size,new.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns)
    tmp=p.with_name(p.name+'.dedup-tmp');assert not tmp.exists()
    log.write(json.dumps(dict(record,event='intent'),ensure_ascii=False)+'\n');log.flush()
    if len(rows)%100==0:os.fsync(log.fileno())
    os.link(src,tmp);os.replace(tmp,p)
    assert p.stat().st_ino==src.stat().st_ino
   log.write(json.dumps(dict(record,event='linked' if args.apply else 'candidate'),ensure_ascii=False)+'\n');log.flush()
   counts[category]+=s.st_blocks*512;rows.append(record)
   if len(rows)%1000==0:print(json.dumps({'scanned':scanned,'linked':len(rows),'GiB':sum(counts.values())/2**30}),flush=True)
   break
  if not matched:by_size[key].append(p)
 log.flush();os.fsync(log.fileno())
after=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
result={'apply':args.apply,'scanned':scanned,'linked':len(rows),'released_allocated_bytes':dict(counts),'available_before':before,'available_after':after,'available_increase':after-before}
(AUDIT/('dedup-result.json' if args.apply else 'dedup-preview.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
