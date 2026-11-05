"""Each model-facing archive contains ONE case only."""
import tarfile,hashlib,json
from pathlib import Path
RUN=Path(__file__).resolve().parent.parent;SOURCE=RUN/'appearance_inputs_v2';DEST=RUN/'appearance_packages'
DEST.mkdir(exist_ok=False);result={}
for case in ('full','left','right','top'):
    path=DEST/f'appearance_inputs_{case}.tar.gz'
    with tarfile.open(path,'w:gz') as t:
        t.add(SOURCE/'geometry_only.npz',arcname='geometry_only.npz')
        for p in sorted((SOURCE/case).iterdir()):
            if p.is_file():t.add(p,arcname=p.name)
    with tarfile.open(path) as t:
        names=t.getnames();assert len(names)==82 and all('/' not in n for n in names)
    result[case]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),files=82,bytes=path.stat().st_size)
(DEST/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
