"""Only explicitly enumerated task paths; preserve all unrelated working tree changes."""
from pathlib import Path
import json,subprocess
W=Path(__file__).parent;P=W.parent/'r15_closure_run/publish_repo'
manifest=W.parent/'r15_repair_4k/final_publish_paths.json'
if not manifest.is_file():manifest=W/'sync_publish_paths.json'
paths=json.loads(manifest.read_text(encoding='utf-8'))
for force in [False,True]:
    group=[p for p in paths if p.endswith('.mp4')==force]
    for i in range(0,len(group),15):
        subprocess.run(['git','add',*(['-f'] if force else []),'--',*group[i:i+15]],cwd=P,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=P,check=True)
subprocess.run(['git','diff','--cached','--stat'],cwd=P,check=True)
print(json.dumps(dict(manifest=str(manifest),explicit_paths=len(paths))))
