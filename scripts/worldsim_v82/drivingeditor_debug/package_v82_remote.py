"""Snapshot existing evidence/code only; does not import training code or use CUDA."""
from pathlib import Path
import json, hashlib, tarfile, os, datetime, subprocess

B=Path('/root/autodl-tmp/drivingeditor_repro')
D=B/'campaign/vgpu_bugfix_20261012'
O=D/'delivery_v82_20261012'
O.mkdir(exist_ok=True)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
task=D/'task_manifest.json'
old=task.read_bytes()
backup=O/'task_manifest_before_handoff.json'
if not backup.exists(): backup.write_bytes(old)
j=json.loads(old)
j.update(status='paused_by_user_evidence_handoff', updated_utc=now,
 objective='Deliver all collected evidence and push scoped code to user Motion-Proj research/worldsim-v8.2-drivingeditor-debug; no further experiments',
 next_experiment=None, gpu_owner=None,
 user_handoff={'automation':'PAUSED','whole_debug_task_complete':False,'new_experiments':False,'shutdown_requested':False,'repository':'https://github.com/yunlongdeng123/Motion-Proj','branch':'research/worldsim-v8.2-drivingeditor-debug','at':now})
task.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
code_ext={'.py','.sh','.yaml','.yml','.toml','.patch','.diff','.cpp','.cc','.c','.cu','.cuh','.h','.hpp','.cmake'}
meta_ext={'.json','.jsonl','.txt','.md','.csv','.tsv','.log','.exit','.pid'}
excluded={'.git','__pycache__','deps','node_modules','build','CMakeFiles','.pytest_cache'}
roots=[B/'campaign',B/'runs',B/'reports',B/'DrivingEditor-paper',B/'DrivingEditor-run',B/'data',B/'datasets']
inventory=[];selected=[]
for root in roots:
 if not root.exists(): continue
 for dirpath,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[x for x in dirs if x not in {'.git','__pycache__'}]
  pdir=Path(dirpath)
  if O==pdir: dirs[:]=[];continue
  for name in files:
   p=pdir/name; rel=p.relative_to(B); st=p.lstat()
   row={'path':str(p),'relative_to_base':str(rel),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
   if p.is_symlink(): row.update(kind='symlink',target=os.readlink(p));inventory.append(row);continue
   parts=set(rel.parts)
   is_code=(p.suffix in code_ext or name in ('CMakeLists.txt','LICENSE','LICENSE_gaussian_splatting.md'))
   is_meta=p.suffix in meta_ext
   eligible=(not parts.intersection(excluded) and (is_code or is_meta) and st.st_size<=32*1024**2
     and not any(k in name.lower() for k in ('credential','token','secret','id_rsa','id_ed25519')))
   # Snapshot code and evidence in the task campaign, runs/reports and actual repo variants.
   row['kind']='code' if is_code else ('evidence_metadata' if is_meta else 'large_or_binary_artifact')
   row['included_in_light_archive']=eligible
   if eligible:
    row['sha256']=sha(p);selected.append(p)
   inventory.append(row)
inv={'schema':1,'created_utc':now,'host':'wm-vgpu-1011','base':str(B),'scope':[str(x) for x in roots if x.exists()],
 'note':'Metadata inventory, not a complete model/data backup. Binary media, tensors, checkpoints and private dependency installations remain on server. Existing canonical receipt hashes are retained; large originals are not rehashed.',
 'files':inventory}
(O/'remote_inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2)+'\n')
archive=O/'drivingeditor_v82_remote_code_evidence.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as t:
 for p in selected:t.add(p,arcname='remote/'+str(p.relative_to(B)),recursive=False)
 t.add(O/'remote_inventory.json',arcname='remote_inventory.json')
 t.add(backup,arcname='task_manifest_before_handoff.json')
receipt={'created_utc':now,'files_in_inventory':len(inventory),'selected_files':len(selected),
 'selected_bytes':sum(p.stat().st_size for p in selected),'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
 'new_experiments':False,'shutdown_requested':False,'task_manifest_sha256':sha(task)}
(O/'package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
