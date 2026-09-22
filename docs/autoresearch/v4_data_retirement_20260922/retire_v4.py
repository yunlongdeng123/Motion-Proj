"""按用户要求撤下 V4 常驻数据；保留轻量来源记录和预处理权重。"""
import argparse,collections,gzip,json,os,shutil,stat,tarfile,time
from pathlib import Path

ROOT=Path('/root/autodl-tmp')
TARGET=ROOT/'data/worldsim_v4'
AUDIT=ROOT/'cleanup_manifests/20260922-v4-retire'
WEIGHTS=ROOT/'models/legacy_v4_preprocess'
PUBLIC=Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval')

def active_paths():
 paths=set()
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:paths.add(os.readlink(fd).removesuffix(' (deleted)'))
    except OSError:pass
   for line in (proc/'maps').read_text().splitlines():
    fields=line.split(maxsplit=5)
    if len(fields)==6:paths.add(fields[5])
  except OSError:pass
 return [p for p in paths if p.startswith(str(TARGET)+'/')]

def prepare():
 assert TARGET.resolve()==TARGET and TARGET.is_dir() and not TARGET.is_symlink()
 assert not active_paths(),active_paths()
 assert not WEIGHTS.exists()
 AUDIT.mkdir(parents=True,exist_ok=False)
 v=os.statvfs(ROOT)
 (AUDIT/'baseline.json').write_text(json.dumps({'available':v.f_bavail*v.f_frsize,'used':(v.f_blocks-v.f_bfree)*v.f_frsize,'time':time.time()}))
 sources=[]
 for p in sorted(PUBLIC.glob('v1.0-trainval*.tgz')):
  with tarfile.open(p,'r|gz') as tar: first=next(iter(tar))
  sources.append({'path':str(p),'bytes':p.stat().st_size,'first_member':first.name})
 assert len(sources)>=11
 # 保留清单及小型适配元数据；不把全部派生数据换个目录存一遍。
 shutil.copytree(TARGET/'manifests',AUDIT/'manifests')
 small=AUDIT/'adapter_records';small.mkdir()
 for p in TARGET.rglob('adapter_manifest.json'):
  dest=small/p.relative_to(TARGET);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  for name in ['partition.json']:
   if (p.parent/name).is_file():shutil.copy2(p.parent/name,dest.parent/name)
 scene=TARGET/'nuscenes_meta/v1.0-trainval/scene.json'
 shutil.copy2(scene,AUDIT/'scene.json')
 groups=collections.Counter(); inodes={};count=0;symlinks=0
 with gzip.open(AUDIT/'removed-layout.jsonl.gz','xt',encoding='utf8',compresslevel=1) as out:
  for base,dirs,files in os.walk(TARGET,followlinks=False):
   for name in dirs+files:
    p=Path(base)/name;s=p.lstat();rel=p.relative_to(TARGET)
    if stat.S_ISDIR(s.st_mode):continue
    row={'path':str(rel),'bytes':s.st_size,'inode':s.st_ino,'device':s.st_dev,'mtime_ns':s.st_mtime_ns,'nlink':s.st_nlink,'allocated':s.st_blocks*512,'kind':'symlink' if p.is_symlink() else 'file'}
    if p.is_symlink():row['target']=os.readlink(p);symlinks+=1
    else:
     count+=1;groups[rel.parts[0]]+=s.st_blocks*512
     key=(s.st_dev,s.st_ino)
     if key not in inodes:inodes[key]=[0,s.st_nlink,s.st_blocks*512,rel.parts[0]=='model_staging']
     inodes[key][0]+=1
    out.write(json.dumps(row,ensure_ascii=False)+'\n')
 expected=sum(v[2] for v in inodes.values() if v[0]==v[1] and not v[3])
 links=[]
 for root in [ROOT/'data',ROOT/'runs',ROOT/'models',ROOT/'third_party']:
  for base,dirs,files in os.walk(root,followlinks=False):
   dirs[:]=[d for d in dirs if Path(base)/d!=TARGET]
   for name in dirs+files:
    p=Path(base)/name
    if not p.is_symlink():continue
    resolved=p.resolve()
    if resolved.is_relative_to(TARGET):
     assert 'worldsim_v75' not in str(p),f'current dependency: {p}'
     links.append({'path':str(p),'resolved':str(resolved),'link':os.readlink(p),'weights':resolved.is_relative_to(TARGET/'model_staging')})
 result={'target':str(TARGET),'files':count,'symlinks':symlinks,'allocated_by_directory':dict(groups),'estimated_released_bytes':expected,'public_sources':sources,'affected_historical_symlinks':links,'preserved_weights_to':str(WEIGHTS)}
 (AUDIT/'plan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps({'files':count,'expected_GiB':expected/2**30,'historical_symlinks':len(links),'plan':str(AUDIT/'plan.json')}),flush=True)

def apply():
 plan=json.loads((AUDIT/'plan.json').read_text())
 assert plan['target']==str(TARGET) and TARGET.resolve()==TARGET and not TARGET.is_symlink()
 assert TARGET.parent.resolve()==ROOT/'data' and not active_paths()
 assert not WEIGHTS.exists()
 # 源代码和旧预处理仍引用这些官方权重；移到模型目录，避免以后再次下载。
 (TARGET/'model_staging').rename(WEIGHTS)
 for row in plan['affected_historical_symlinks']:
  if not row['weights']:continue
  p=Path(row['path']);assert p.is_symlink() and os.readlink(p)==row['link']
  dest=WEIGHTS/Path(row['resolved']).relative_to(TARGET/'model_staging')
  assert dest.exists();tmp=p.with_name(p.name+'.relocate-tmp');assert not tmp.exists()
  tmp.symlink_to(dest,target_is_directory=dest.is_dir());os.replace(tmp,p)
 with (AUDIT/'delete-intent.json').open('x') as f:
  json.dump({'target':str(TARGET),'time':time.time(),'authorization':'用户要求直接删除 V4 数据；仅保留来源记录及预处理权重'},f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
 # 精确指定的已审计目录，rmtree 不跟随内部目录符号链接。
 shutil.rmtree(TARGET)
 assert not TARGET.exists()
 for p in [ROOT/'data/av2/sensor',ROOT/'models/worldsim_v75',ROOT/'runs/worldsim_v75',WEIGHTS]:assert p.is_dir(),p
 baseline=json.loads((AUDIT/'baseline.json').read_text());v=os.statvfs(ROOT)
 result={'status':'complete','removed':str(TARGET),'available_bytes':v.f_bavail*v.f_frsize,'used_bytes':(v.f_blocks-v.f_bfree)*v.f_frsize,'actual_available_increase_bytes':v.f_bavail*v.f_frsize-baseline['available'],'preserved_weights':str(WEIGHTS),'historical_symlinks_needing_restore':[r['path'] for r in plan['affected_historical_symlinks'] if not r['weights']],'model_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
 (AUDIT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)

ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
apply() if args.apply else prepare()
