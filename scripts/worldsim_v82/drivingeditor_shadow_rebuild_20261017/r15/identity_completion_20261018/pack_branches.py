"""Package only completed, actual branch PNGs/receipts for local synchronized video."""
import tarfile
from identity_common import *
models=[n for n in ['old5','r15','repair4k'] if (TASK/('branches_'+n)/'branch_receipt.json').is_file()]
assert 'old5' in models and 'r15' in models
target=TASK/('branch_pngs_'+('_'.join(models))+'.tar.gz')
with tarfile.open(target.with_suffix('.tmp'),'w:gz',compresslevel=3) as tar:
    for n in models:
        for p in sorted((TASK/('branches_'+n)).rglob('*')):
            if p.is_file() and p.suffix in ['.png','.json']:tar.add(p,arcname=p.relative_to(TASK),recursive=False)
    q=R15/'repair_4k_20261018/renders_repair4k'
    if (q/'render_receipt.json').is_file():
        for p in sorted(q.rglob('*')):
            if p.is_file() and p.suffix in ['.png','.json']:tar.add(p,arcname=Path('renders_repair4k')/p.relative_to(q),recursive=False)
target.with_suffix('.tmp').replace(target)
print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,models=models)))
