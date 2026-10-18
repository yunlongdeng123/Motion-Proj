"""CPU-only package of completed repair evaluation, branches, and audit receipts."""
from pathlib import Path
import json,tarfile
Q=Path(__file__).resolve().parent;T=Q.parent.parent.parent/'identity_completion_20261018'
read=lambda p:json.loads(p.read_text())
assert read(Q/'queue_status.json')['state']=='ALL_REPAIR_GPU_WORK_COMPLETE_AWAITING_REVIEW_AND_DELIVERY'
assert (Q/'renders_repair4k/render_receipt.json').is_file()
assert (T/'branches_repair4k/branch_receipt.json').is_file()
target=Q/'final_evidence_png_json.tar.gz'
with tarfile.open(target.with_suffix('.tmp'),'w:gz',compresslevel=3) as tar:
    for folder,arc in [(Q/'renders_repair4k','renders_repair4k'),(T/'branches_repair4k','branches_repair4k')]:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in ['.png','.json']:tar.add(p,arcname=Path(arc)/p.relative_to(folder),recursive=False)
    for name in ['queue_status.json','train_receipt.json','smoke_receipt.json','geometry_comparison.json','experiment_contract.json']:
        tar.add(Q/name,arcname=Path('metadata')/name,recursive=False)
    for name in ['progress.json','metrics.jsonl','resolved_config.json','topology_latest.json','sgcm_latest.json','reference_refresh.json','experiment.json']:
        p=Q/'train'/name
        if p.is_file():tar.add(p,arcname=Path('metadata/train')/name,recursive=False)
    for step in [1500,2500,3500]:
        folder=Q/'train/ownership'/f'{step:06d}'
        for p in folder.rglob('*.json'):tar.add(p,arcname=Path('metadata/train/ownership')/f'{step:06d}'/p.relative_to(folder),recursive=False)
target.with_suffix('.tmp').replace(target)
print(json.dumps(dict(path=str(target),bytes=target.stat().st_size)))
