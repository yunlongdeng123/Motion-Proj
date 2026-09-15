import json,tarfile
from pathlib import Path
from PIL import Image
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1');O=R/'context_export';O.mkdir(exist_ok=True)
for f in ['registration.json','selection_summary.json','causal_status_amendment.json']:(O/f).write_text((R/f).read_text())
for p in (R/'inputs').glob('*.json'):
 inp=json.loads(p.read_text());d=O/p.stem;d.mkdir(exist_ok=True)
 im=Image.open(inp['views'][0]['image']);im.thumbnail((960,600));im.save(d/'front.jpg',quality=90)
 (d/'input.json').write_text(p.read_text());(d/'log_reference.json').write_text((R/'lidar_policy'/f'{p.stem}_log_reference.json').read_text())
with tarfile.open(R/'context_export.tar.gz','w:gz') as t:
 for p in O.rglob('*'):
  if p.is_file():t.add(p,arcname=p.relative_to(O).as_posix())
