"""Verified diagnostic delivery; no checkpoints or Adam state downloaded."""
from pathlib import Path
import subprocess,json,hashlib,tarfile,shutil
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';O.mkdir(exist_ok=True)
R='/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028';name='semantic_delivery_v1'
receipt=json.loads(subprocess.check_output(['ssh','wm-vgpu-1011','cat '+R+'/'+name+'_receipt.json']))
archive=O/(name+'.tar.gz');tmp=archive.with_suffix('.download')
if not archive.exists():
    subprocess.run(['scp','wm-vgpu-1011:'+receipt['archive'],str(tmp)],check=True)
    assert tmp.stat().st_size==receipt['bytes'] and hashlib.sha256(tmp.read_bytes()).hexdigest()==receipt['sha256'];tmp.replace(archive)
assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
with tarfile.open(archive) as t:
    for m in t.getmembers():
        p=(O/m.name).resolve();assert p.is_relative_to(O.resolve()) and (m.isfile() or m.isdir()),m.name
    t.extractall(O)
folder=O/name;manifest=json.loads((folder/'manifest.json').read_text());checked=0
for rel,r in manifest.items():
    p=folder/rel;assert p.is_relative_to(folder) and p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],rel;checked+=1
report=json.loads((folder/'report.json').read_text())
for item in report['contexts']:
    source=folder/item['context'];dest=B/'outputs/GS_Scene_Diagnostics_20261027'/item['context']/'gpu_semantics';dest.mkdir(exist_ok=True)
    for p in source.iterdir():
        if p.suffix=='.js':shutil.copy2(p,dest/p.name)
(O/'delivery_receipt.json').write_text(json.dumps(dict(archive=receipt,verified_files=checked,models_downloaded=False),indent=2))
print(json.dumps(report,ensure_ascii=True))
