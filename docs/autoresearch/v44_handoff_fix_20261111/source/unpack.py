from pathlib import Path,PurePosixPath
import hashlib,json,shutil,tarfile
H=Path(__file__).resolve().parent;receipt=json.loads((H/'package_receipt.json').read_text());p=H/'viewer.tar.gz'
assert p.stat().st_size==receipt['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['sha256']
D=H/'delivery_r1';D.mkdir(exist_ok=False)
with tarfile.open(p) as tar:
    members=tar.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
    for m in members:
        parts=PurePosixPath(m.name);assert m.isfile() and not parts.is_absolute() and '..' not in parts.parts and '\\' not in m.name and ':' not in m.name
    for m in members:
        dest=D/m.name;dest.parent.mkdir(parents=True,exist_ok=True)
        with tar.extractfile(m) as src,dest.open('wb') as out:shutil.copyfileobj(src,out)
mf=json.loads((D/'manifest.json').read_text());assert set(names)=={'manifest.json'}|{r['path'] for r in mf['files']}
print(D)
