"""Package completed PNG/JSON identity artifacts only; never checkpoints or FP32 arrays."""
from pathlib import Path
import json,tarfile
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
T=ROOT/'identity_completion_20261018'
Q=ROOT/'shadow_rebuild_20261017_v3/backend_restore_r15/repair_4k_20261018'
old=json.loads((T/'renders_old5/render_receipt.json').read_text())
tri=json.loads((T/'tristate/projection/projection_receipt.json').read_text())
assert old['checkpoint_step']==30000 and old['frames']==49 and len(old['records'])==245
assert old['parameter_updates']==0 and old['checkpoint_file_unchanged']
assert tri['frames']==49 and tri['parameter_updates']==0
target=T/'completed_identity_pngs.tar.gz';temporary=target.with_suffix('.tmp')
with tarfile.open(temporary,'w:gz',compresslevel=3) as tar:
    for folder in ['renders_old5','tristate/projection']:
        for p in sorted((T/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.png','.json'):tar.add(p,arcname=p.relative_to(T),recursive=False)
    for name in ['queue_status.json','tristate_queue_status.json','train5/status.json']:
        tar.add(T/name,arcname='metadata_completed/'+name)
    for name in ['queue_status.json','smoke_receipt.json','train/progress.json']:
        if (Q/name).is_file():tar.add(Q/name,arcname='repair_status/'+name)
temporary.replace(target)
print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,old5_records=len(old['records']),projection_records=len(tri['records']))))
