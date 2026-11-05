"""SHA ledger for the retrievable sphere control preview and final assets."""
import hashlib,json
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105/output_adapter/sphere_highres')
files=[]
for p in ROOT.rglob('*'):
    if not p.is_file() or p.name in ('design.npz','inputs.npz','raster_hr.npz','preview_manifest.json') or p.suffix=='.pt':continue
    files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(ROOT/'preview_manifest.json').write_text(json.dumps(dict(files=sorted(files,key=lambda f:f['path'])),indent=2)+'\n')
print(json.dumps(dict(files=len(files))))
