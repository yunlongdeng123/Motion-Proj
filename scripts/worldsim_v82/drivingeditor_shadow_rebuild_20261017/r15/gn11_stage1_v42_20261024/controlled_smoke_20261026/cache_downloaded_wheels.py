from pathlib import Path
import zipfile,shutil,os,json
dest=Path('/root/autodl-tmp/v42_ready_wheels');dest.mkdir(exist_ok=True);rows=[]
for p in Path('/tmp').glob('pip-unpack-*/*.whl'):
    try:
        with zipfile.ZipFile(p) as z:assert any(n.endswith('/METADATA') for n in z.namelist())
    except (zipfile.BadZipFile,AssertionError):continue
    target=dest/p.name
    if not target.exists():
        try:os.link(p,target)
        except OSError:shutil.copy2(p,target)
    rows.append(dict(file=p.name,bytes=p.stat().st_size))
(dest/'receipt.json').write_text(json.dumps(rows,indent=2));print(len(rows))
