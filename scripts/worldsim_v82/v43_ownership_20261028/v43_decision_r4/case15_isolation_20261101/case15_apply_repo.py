from pathlib import Path
import tarfile,shutil,subprocess,json
H=Path('/root/autodl-tmp/codex_tmp/v43_case15_isolation_20261101_r1');R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');archive=H/'repo_patch.tar.gz';stage=H/'repo_candidate';stage.mkdir(exist_ok=True)
for name in ['RESEARCH_STATUS.md','EXPERIMENTS.md']:
 if (R/'docs'/name).read_bytes()!=(H/'repo_guard'/name).read_bytes():raise RuntimeError('Concurrent document change; do not overwrite '+name)
bridge=(R/'scripts/run_worldsim_v72_e3_render_bridge.py').read_bytes()
allowed=('docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/autoresearch/v43_ownership_20261028/case15_isolation_20261101/','scripts/worldsim_v82/v43_ownership_20261028/v43_decision_r4/case15_isolation_20261101/')
with tarfile.open(archive) as z:
 for m in z.getmembers():
  p=Path(m.name)
  if p.is_absolute() or '..' in p.parts or m.issym() or m.islnk():raise RuntimeError('Invalid archive member')
  name=str(p)
  if m.isfile() and not any(name==a or name.startswith(a) for a in allowed):raise RuntimeError(name)
 z.extractall(stage)
for a in allowed[2:]:
 if (R/a).exists():raise RuntimeError('New result directory already exists; preserve it')
backup=H/'repo_before_CPU_closeout';backup.mkdir(exist_ok=True)
for name in ['RESEARCH_STATUS.md','EXPERIMENTS.md']:
 target=backup/name
 if target.exists():raise RuntimeError('Backup already exists; do not overwrite')
 shutil.copy2(R/'docs'/name,target)
for p in stage.rglob('*'):
 if p.is_file():
  target=R/p.relative_to(stage);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
assert bridge==(R/'scripts/run_worldsim_v72_e3_render_bridge.py').read_bytes()
print(json.dumps({'applied_files':sum(p.is_file() for p in stage.rglob('*')),'original_bridge_preserved':True,'backup':str(backup)}))
