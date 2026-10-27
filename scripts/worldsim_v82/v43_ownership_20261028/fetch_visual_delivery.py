"""Checked inbound diagnostic artifacts; never models or unsafe archive paths."""
from pathlib import Path
import subprocess,json,hashlib,tarfile,sys
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';R='/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028';name=sys.argv[1]
receipt=json.loads(subprocess.check_output(['ssh','wm-vgpu-1011','cat '+R+'/'+name+'_receipt.json']));archive=O/(name+'.tar.gz');tmp=archive.with_suffix('.download')
if not archive.exists():subprocess.run(['scp','wm-vgpu-1011:'+receipt['archive'],str(tmp)],check=True);tmp.replace(archive)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
with tarfile.open(archive) as t:
    for m in t.getmembers():assert (O/m.name).resolve().is_relative_to(O.resolve()) and (m.isfile() or m.isdir()) and Path(m.name).suffix not in ['.pth','.pt','.ckpt'],m.name
    t.extractall(O)
for rel,d in json.loads((O/name/'manifest.json').read_text()).items():
    p=O/name/rel;assert p.stat().st_size==d['bytes'] and sha(p)==d['sha256']
(O/name/'delivery_validation.json').write_text(json.dumps(dict(archive_and_files_verified=True,receipt=receipt,models_downloaded=False),indent=2))
print(json.dumps(dict(name=name,archive_bytes=receipt['bytes'],archive_and_files_verified=True,models_downloaded=False)))
