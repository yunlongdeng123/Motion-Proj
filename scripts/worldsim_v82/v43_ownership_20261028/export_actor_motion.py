"""Export completed first-scene native edits and actual available fixed backend PNGs."""
import argparse,json,shutil,hashlib,tarfile
from segmentation import ROOT,CONTEXTS,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--context',default=CONTEXTS[0]);a=p.parse_args()
    source=ROOT/'actor_motion_v1'/a.context;summary=json.loads((source/'summary.json').read_text());assert summary['frames_per_case']==31
    dest=ROOT/a.name;dest.mkdir(exist_ok=False);out=dest/a.context;out.mkdir();write(out/'summary.json',summary);deliver=[]
    for case in summary['cases']:
        native=source/case/'native_rgb';assert len(list(native.glob('*.png')))==31
        target=out/case;shutil.copytree(native,target/'native_rgb');shutil.copytree(source/case/'mask',target/'mask')
        backend=source/case/'propainter/summary.json';generated=False
        if backend.exists():
            d=json.loads(backend.read_text());assert d['status']=='complete' and d['frames']==31
            assert len(list((backend.parent/'model_png').glob('*.png')))==31
            shutil.copytree(backend.parent/'model_png',target/'generated');write(target/'backend_summary.json',d);generated=True
        deliver.append(dict(case=case,frames=31,generated=generated))
    write(dest/'index.json',dict(context=a.context,cases=deliver))
    files={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dest.rglob('*') if p.is_file()};write(dest/'manifest.json',files)
    archive=ROOT/(a.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as t:t.add(dest,arcname=a.name)
    write(ROOT/(a.name+'_receipt.json'),dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),models_exported=False));print(json.dumps(deliver))
if __name__=='__main__':main()
