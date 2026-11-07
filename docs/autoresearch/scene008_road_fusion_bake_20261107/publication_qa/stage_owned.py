from pathlib import Path
import json,subprocess
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
OWN='docs/autoresearch/scene008_road_fusion_bake_20261107'
files=[OWN,'docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V82-F13.md']
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True)
before=git('diff','--cached','--name-only').strip()
if before:raise RuntimeError('Preserve another staged change: '+before)
git('add','--',*files)
ignored=git('ls-files','--others','--ignored','--exclude-standard','--',OWN).splitlines()
for rel in ignored:
    assert rel.startswith(OWN+'/') and Path(rel).suffix in ('.npz','.pt'),rel
if ignored:git('add','-f','--',*ignored)
subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
staged=git('diff','--cached','--name-only').splitlines()
assert all(p.startswith(OWN+'/') or p in files[1:] for p in staged),staged
manifest=json.loads((REPO/OWN/'publication_manifest.json').read_text())
import hashlib
for rel,item in manifest.items():
    blob=subprocess.check_output(['git','show',':'+OWN+'/'+rel],cwd=REPO)
    assert len(blob)==item['bytes'] and hashlib.sha256(blob).hexdigest()==item['sha256'],rel
print(json.dumps(dict(staged_files=len(staged),forced_own_state_files=len(ignored),scientific_packet_blobs_exact=len(manifest),
    diff_check_passed=True,remaining_unstaged=git('diff','--name-only').splitlines()),ensure_ascii=False))
print(git('diff','--cached','--shortstat'))
