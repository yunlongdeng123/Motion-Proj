from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
p=argparse.ArgumentParser(description='Install the R4 workbench patch and rebuild candidate previews from local data.')
p.add_argument('--platform-root',type=Path,required=True);p.add_argument('--legacy-root',type=Path,required=True);p.add_argument('--results-root',type=Path,required=True);p.add_argument('--backup-dir',type=Path,required=True);a=p.parse_args()
here=Path(__file__).resolve().parent;manifest=json.loads((here/'changes_manifest.json').read_text(encoding='utf8'));zfile=here/'workbench_changes_patch.zip';old=a.results_root/'actor_screen_existing/existing_normalized_v1';hf=a.results_root/'actor_screen_horizonforge'
for f in [a.platform_root/'style.css',a.platform_root/'workspace.css',a.platform_root/'14705303724557273004_3105_000_3125_000/assets/reconstruction_30000/rgb/101.png']:
 if not f.is_file():raise FileNotFoundError('Required existing platform asset: '+str(f))
if not old.is_dir() or not hf.is_dir():raise FileNotFoundError('Need both actor_screen_existing/existing_normalized_v1 and actor_screen_horizonforge under results-root')
with tempfile.TemporaryDirectory(prefix='r4_workbench_') as temp:
 stage=Path(temp);pack=stage/'package';pack.mkdir()
 with zipfile.ZipFile(zfile) as z:
  expected={f['package_path']:f for f in manifest['files']}
  if set(z.namelist())!=set(expected):raise ValueError('Patch archive contents differ from manifest')
  for name in z.namelist():
   dest=pack/name
   if not dest.resolve().is_relative_to(pack.resolve()):raise ValueError('Unsafe path: '+name)
   content=z.read(name)
   if hashlib.sha256(content).hexdigest()!=expected[name]['sha256']:raise ValueError('Hash mismatch: '+name)
   dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
 candidate=stage/'candidate';subprocess.run([sys.executable,str(here/'build_actor_page.py'),str(old),str(candidate),'--additional-root',str(hf)],check=True)
 data=json.loads((candidate/'data.js').read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'))
 if len(data)!=15 or sum(len(d['actors']) for d in data)!=1228 or sum(bool(x['preview_image']) for d in data for x in d['actors'])!=90:raise ValueError('Candidate inputs do not match the 15-scene release')
 targets=[]
 for item in manifest['files']:
  name=Path(item['package_path']);target=(a.platform_root if name.parts[0]=='shared' else a.legacy_root).joinpath(*name.parts[1:]);targets.append((pack/name,target,Path(name.parts[0]).joinpath(*name.parts[1:])))
 for src in candidate.rglob('*'):
  if src.is_file():
   rel=src.relative_to(candidate);targets.append((src,a.platform_root/'actor_candidates'/rel,Path('shared/actor_candidates')/rel))
 a.backup_dir.mkdir(parents=True,exist_ok=True)
 for src,dst,rel in targets:
  if dst.exists():
   backup=a.backup_dir/rel;backup.parent.mkdir(parents=True,exist_ok=True)
   if backup.exists():raise FileExistsError('Backup already exists: '+str(backup))
 for src,dst,rel in targets:
  if dst.exists():
   backup=a.backup_dir/rel;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,backup)
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 print('Installed',len(targets),'files after verifying archive hashes, preview inputs, and preserving existing files in',a.backup_dir)
