import gzip,json,os,tarfile,collections
from pathlib import Path
from prune_storage import ROOT,AUDIT,describe,open_paths
from restore_v4_sweeps import PUBLIC

root=ROOT/'data/worldsim_v4';raw=root/'drivestudio_raw_trainval'; seen=set();total=0;counts=collections.Counter();not_covered=[]
archive_checks=[]
for source in sorted(PUBLIC.glob('v1.0-trainval??_blobs.tgz')):
 with tarfile.open(source,'r|gz') as tar:
  first=next(iter(tar)); assert first.name
 archive_checks.append({'source':str(source),'bytes':source.stat().st_size,'first_tar_member_readable':True})
assert len(archive_checks)==10
active=open_paths()
with gzip.open(AUDIT/'sweeps-plan.jsonl.gz','xt',encoding='utf8',compresslevel=1) as out:
 for manifest in sorted((root/'manifests').glob('scene-*_raw_manifest_v4.json')):
  for row in json.loads(manifest.read_text())['files']:
   name=row['filename']
   if not name.startswith('sweeps/') or name in seen:continue
   seen.add(name);p=raw/name
   if not p.is_file():continue
   assert p.resolve()==p and p.stat().st_size==row['bytes'] and str(p) not in active
   assert (PUBLIC/row['shard']).is_file()
   item=describe(p,'v4_extracted_sweep',{'manifest':str(manifest),'archive':str(PUBLIC/row['shard']),'member':name})
   # 已共享 inode 的文件不删：并不释放空间，且有其他旧入口引用。
   if item['links']!=1:continue
   out.write(json.dumps(item,ensure_ascii=False)+'\n');total+=item['allocated'];counts[manifest.stem]+=1
for folder,dirs,files in os.walk(raw/'sweeps'):
 for name in files:
  p=Path(folder)/name
  if str(p.relative_to(raw)) not in seen:not_covered.append(str(p))
result={'files':sum(counts.values()),'allocated_bytes':total,'scene_counts':dict(counts),'archives':archive_checks,'unmapped_files_kept':len(not_covered),'archive_check_scope':'每个公共原包可读且能解析首个 tar 成员；依据原提取清单确认逐文件来源；不宣称重新读取全部 294 GiB 原包'}
(AUDIT/'sweeps-plan-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False),flush=True)
