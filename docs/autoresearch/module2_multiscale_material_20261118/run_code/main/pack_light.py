from pathlib import Path
import json,tarfile,hashlib

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
D=P/'delivery';selected={}
for src in sorted(P.rglob('*')):
    if not src.is_file():continue
    rel=src.relative_to(P)
    if any(x in ('publication','__pycache__','registration') for x in rel.parts):continue
    if src.name in ('light_bundle.tar','light_bundle_receipt.json'):continue
    if src.suffix not in ('.py','.json','.md','.svg','.csv','.webp','.png','.txt','.log'):continue
    if src.stat().st_size>2_500_000:continue
    if '.before' in src.name or 'codexbak' in src.name:continue
    selected[str(rel)]=src
receipt=dict(schema='multiscale_material_light_delivery.v1',
    files=[dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in selected.items()],
    includes_full_models_or_checkpoint_or_full_shared_field=False,
    full_original_inputs_new_assets_grids_optimizer_states_and_precision_arrays_retained_remotely=True)
(D/'light_bundle_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
with tarfile.open(D/'light_bundle.tar','w') as tar:
    for name,path in selected.items():tar.add(path,arcname=name,recursive=False)
    tar.add(D/'light_bundle_receipt.json',arcname='delivery/light_bundle_receipt.json',recursive=False)
print(json.dumps(dict(files=len(selected),bytes=sum(p.stat().st_size for p in selected.values()),tar_bytes=(D/'light_bundle.tar').stat().st_size)))
