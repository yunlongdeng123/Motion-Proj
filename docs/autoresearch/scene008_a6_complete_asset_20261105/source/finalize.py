"""Finite closeout: text/link/image/asset integrity; no new fitting or shutdown."""
import json,re,hashlib,shutil,tarfile
from html.parser import HTMLParser
from pathlib import Path
from PIL import Image
from full_pvd import RUN,FixedAsset

DOC=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/docs/autoresearch/scene008_a6_complete_asset_20261105')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    delivery=RUN/'delivery'
    # A first failed smoke exercised an ABI tuple mismatch; preserve its log.
    shutil.copyfile(RUN/'smoke_abi_failure.log',delivery/'evidence/smoke_first_ABI_failure.log')
    fit=json.loads((RUN/'appearance/fit_receipt.json').read_text())
    evaluation=json.loads((RUN/'evaluation/metrics.json').read_text())
    smoke=json.loads((RUN/'smoke/receipt.json').read_text())
    ledger=dict(first_smoke='ABI tuple unpack failure, corrected before any fitting; log retained',
       successful_real_GPU_smoke_seconds=smoke['seconds'],appearance_solve_seconds=fit['seconds'],
       all49_evaluation_seconds=evaluation['seconds'],geometry_updates=0,neural_training_steps=0,
       long_run_started=False,old_branch='Closed with incomplete formal evaluation; originals and partial outputs preserved',
       autodl='KEEP_ON',shutdown_performed=False)
    (delivery/'evidence/execution_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    shutil.copyfile(RUN/'source/report.py',delivery/'source/report.py')
    shutil.copyfile(RUN/'source/finalize.py',delivery/'source/finalize.py')
    # Normalize only generated text in the new deliverable, preserving binary evidence.
    suffixes={'.py','.json','.html','.svg','.md','.txt','.csv','.log','.jsonl','.sha256'}
    for path in delivery.rglob('*'):
        if path.is_file() and path.suffix in suffixes:
            raw=path.read_bytes()
            if b'\r\n' in raw:path.write_bytes(raw.replace(b'\r\n',b'\n'))
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            self.links.extend(value for key,value in attrs if key in ('href','src') and value)
    links=[]
    for page in delivery.rglob('*.html'):
        parser=Links();parser.feed(page.read_text())
        for link in parser.links:
            if link.startswith(('http:','https:','#','data:')):continue
            path=page.parent/link
            assert path.is_file(),(page,link)
            links.append(str(path.relative_to(delivery)))
    photos=list((delivery/'frames').glob('*.jpg'))+list((delivery/'frames').glob('*.png'))
    for path in photos:
        with Image.open(path) as im:im.verify()
    assert len(list((delivery/'frames').glob('*_objects.jpg')))==49
    asset=FixedAsset.from_geometry(delivery/'assets/complete_a6_fitted.npz')
    assert asset.assert_geometry_unchanged()
    qa=dict(status='PASS',html_links_checked=len(links),images_checked=len(photos),
            frames=49,asset_geometry_hash=asset.geometry_hash,
            visual_review='raw113 black and street reviewed; all frame files decoded',
            scientific_quality_accepted=False)
    (delivery/'evidence/delivery_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    # Copy only the scoped deliverable; preserve README and unrelated repository work.
    shutil.copytree(delivery,DOC,dirs_exist_ok=True)
    # Repack after final text and QA updates.
    with tarfile.open(RUN/'delivery.tar.gz','w:gz') as archive:archive.add(delivery,arcname='delivery')
    line=sha(RUN/'delivery.tar.gz')+'  delivery.tar.gz\n'
    (RUN/'delivery.tar.gz.sha256').write_text(line);(DOC/'delivery.tar.gz.sha256').write_text(line)
    print(json.dumps(dict(qa=qa,archive_sha256=line.split()[0]),ensure_ascii=False))

if __name__=='__main__':main()
