"""Gather actual receipts, failure accounting and review bundle; never invent passes."""
import json,time,tarfile,hashlib,io
from pathlib import Path
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026')
read=lambda p:json.loads(p.read_text())
runs=[]
for folder in ['observed_smoke','restore_check','edit_smoke','edit_smoke_preserve','edit_restore','edit_visibility','visibility_restore']:
    for f in sorted((C/folder).glob('*/receipt_*.json')):
        d=read(f);d['updates']=d.get('updates',d.get('updates_executed',0));runs.append(dict(path=str(f.relative_to(C)),**d))
ledger=dict(updated=time.time(),runs=runs,engineering_updates=sum(r['updates'] for r in runs),native_B_updates=0,formal_updates=0,
    authorization='2026-10-26 GPU open; both controlled routes and bounded smoke approved; latest steering prioritizes real199 evidence and static occlusion.',
    amendments=[
      'Three hard static gates were zero-update destructive diagnostic controls, not trusted ghost classifiers; retained failed outputs.',
      'Initial isolated-road1000 per arm tests only colour/opacity plumbing, not complete V41 objective.',
      'Three-view joint-edit1000 full199 with gate .001 caused outside damage; adaptive repeat with .999 on both arms exposed a colour-fitting shortcut.',
      'After zero centre road contribution was measured, adaptive edit_visibility adds0.1 negative-log static transmittance on fixed full199-alpha>=.5 AND exact crossview-valid rays to both1000 arms. Other parameters unchanged. Not a preregistered main result.',
      'Zero-update qualification rescue keeps original points and adds only temporally block-checked colour consensus; not global relaxation.',
      'FreeInpaint frozen1pass and max3pass per actor; default opacity stop causes actual1pass. No forced STR or optimizer training.',
      'First FreeInpaint forward saved state before gsplat rendering failed. Exact pinned source compiled with CUDA12.8 headers, saved result recovered; separate-process RGB/depth/alpha restore exact.',
      'CUDA training restores load parameters/optimizer/RNG/sampler exactly but post-step traces differ at1e-7..1e-6; retained, not called exact continuation.'
    ],
    data_scope='199frames BUILD, three-frame edit development102/129/137;49-view check overlaps scene and input sources. No hidden-region clean GT or unseen-scene claim.',
    memory=read(C/'gpu_memory_peak.json'),free_probe=read(C/'free_probe_status.json'),quality_accepted=False,human_verdict=None)
(C/'execution_ledger.json').write_text(json.dumps(ledger,indent=2,ensure_ascii=False))
visibility=[]
for arm in ['window49','full199']:
    for r in read(C/'edit_visibility'/arm/'evaluation1000.json'):visibility.append(dict(arm=arm,**r))
(C/'visibility_result.json').write_text(json.dumps(visibility,indent=2))
files=[]
for folder in ['coverage_audit','qualification_rescue','edit49','visibility49','freeinpaint_pass3']:
    for f in (C/folder).rglob('*'):
        if f.is_file() and f.suffix in ['.json','.png','.jpg','.npz','.npy']:files.append(f)
for folder in ['observed_smoke','restore_check','edit_smoke','edit_smoke_preserve','edit_restore','edit_visibility','visibility_restore']:
    for f in (C/folder).rglob('*'):
        if f.is_file() and f.suffix in ['.json','.png']:files.append(f)
for f in C.glob('*.json'):
    if f.name!='review_manifest.json':files.append(f)
files+=list((C/'crossview').glob('*.json'))
files+=list((C/'freeinpaint_pass1/a37/object-removal/restored').glob('*.json'))
manifest={}
with tarfile.open(C/'controlled_review.tar.gz','w:gz') as t:
    # Use the exact same bytes for metadata and archive, including live NVML JSON.
    for f in sorted(set(files)):
        name=str(f.relative_to(C));data=f.read_bytes()
        manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
    payload=json.dumps(manifest,indent=2).encode();(C/'review_manifest.json').write_bytes(payload)
    info=tarfile.TarInfo('review_manifest.json');info.size=len(payload);t.addfile(info,io.BytesIO(payload))
print(json.dumps(dict(updates=ledger['engineering_updates'],files=len(manifest),review_bytes=(C/'controlled_review.tar.gz').stat().st_size,sha256=hashlib.sha256((C/'controlled_review.tar.gz').read_bytes()).hexdigest())))
