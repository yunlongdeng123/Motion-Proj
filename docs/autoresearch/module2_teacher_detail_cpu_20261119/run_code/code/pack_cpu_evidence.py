from pathlib import Path
import json,tarfile,hashlib
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
D=Q/'delivery';D.mkdir(exist_ok=True);selected={}
for p in sorted(Q.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(Q)
    if any(x in ('publication','registration','__pycache__')for x in rel.parts):continue
    if p.name in ['cpu_evidence.tar.gz','bundle_manifest.json']:continue
    if p.suffix not in ['.py','.json','.md','.svg','.csv','.png','.txt','.log','.xml']:continue
    if p.stat().st_size>2_500_000:continue
    selected[str(rel)]=p
manifest=dict(schema='teacher_detail_CPU_light_evidence.v1',
    files=[dict(path=k,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for k,p in selected.items()],
    full_models_arrays_read_operators_and_parent_scene_not_included=True,
    full_inputs_and_operators_preserved_remote=str(Q),GPU_execution=False)
(D/'bundle_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with tarfile.open(D/'cpu_evidence.tar.gz','w:gz',compresslevel=3)as tar:
    for name,path in selected.items():tar.add(path,arcname=name,recursive=False)
    tar.add(D/'bundle_manifest.json',arcname='delivery/bundle_manifest.json',recursive=False)
print(json.dumps(dict(files=len(selected),bytes=sum(x.stat().st_size for x in selected.values()),archive_bytes=(D/'cpu_evidence.tar.gz').stat().st_size)))
