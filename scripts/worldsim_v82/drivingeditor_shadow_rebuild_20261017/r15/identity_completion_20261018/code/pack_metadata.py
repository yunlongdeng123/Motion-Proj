"""CPU archive of reproducibility inputs, never checkpoints or raw RGB/FP32 arrays."""
from identity_common import *
import tarfile

def main():
    names=[p for p in TASK.glob('*.json')]+[p for p in (TASK/'infra').glob('*.py')]
    names+=list((TASK/'native_masks_for_review').glob('*'))
    names+=list((TASK/'prepared5').glob('*.json'))
    names+=list((TASK/'prepared5/instance_masks').glob('*.png'))
    for model in ['old3','old5','r15']:
        p=TASK/('renders_'+model)/'render_receipt.json'
        if p.exists():names.append(p)
    target=TASK/'metadata_snapshot.tar.gz'
    with tarfile.open(target,'w:gz') as t:
        for p in names:
            if p.is_file():t.add(p,arcname=str(p.relative_to(TASK)))
    print(json.dumps(dict(path=str(target),files=len(names),bytes=target.stat().st_size)))

if __name__=='__main__':main()
