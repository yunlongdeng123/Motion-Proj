from pathlib import Path
import json,subprocess,tarfile,hashlib,shutil
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';R='/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028';name='depth_delivery_v1';receipt=json.loads(subprocess.check_output(['ssh','wm-vgpu-1011','cat '+R+'/'+name+'_receipt.json']));archive=O/(name+'.tar.gz');tmp=archive.with_suffix('.download')
if not archive.exists():
    subprocess.run(['scp','wm-vgpu-1011:'+receipt['archive'],str(tmp)],check=True);assert tmp.stat().st_size==receipt['bytes'] and hashlib.sha256(tmp.read_bytes()).hexdigest()==receipt['sha256'];tmp.replace(archive)
with tarfile.open(archive) as t:
    for m in t.getmembers():assert (O/m.name).resolve().is_relative_to(O.resolve()) and (m.isdir() or m.isfile())
    t.extractall(O)
folder=O/name
for rel,r in json.loads((folder/'manifest.json').read_text()).items():
    p=folder/rel;assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
for c in json.loads((folder/'report.json').read_text()):
    target=B/'outputs/GS_Scene_Diagnostics_20261027'/c['context']/'gpu_semantics'
    for p in (folder/c['context']).iterdir():shutil.copy2(p,target/p.name)
    p=target/'index.js';s=p.read_text();s+='\nwindow.GPU_SEMANTIC_INDEX.depth_available=true;' if 'depth_available=true' not in s else '';p.write_text(s)
print(json.dumps(dict(verified=True,report=json.loads((folder/'report.json').read_text()))))
