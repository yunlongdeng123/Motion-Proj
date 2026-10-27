"""Export first-scene native diagnostic artifacts only; no weights or held scenes."""
import argparse,json,time,shutil,tarfile,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from segmentation import ROOT,write
from export_diagnostics import pack

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['ablation','opacity'],required=True);p.add_argument('--boundary-only',action='store_true');a=p.parse_args();R=ROOT/'first_gpu_review_v1';name='first_review_'+('boundary' if a.boundary_only else a.part)+'_delivery_v1';out=ROOT/name;out.mkdir(exist_ok=False)
    if a.part=='ablation':
        src=R/'semantic_ablation';assert json.loads((src/'status.json').read_text())['phase']=='complete';summary=json.loads((src/'summary.json').read_text());rows=[]
        for radius in [0,5]:
            for mode in ['hide_unknown','hide_vehicle','hide_both']:
                source=src/(str(radius)+'px')/mode;target=out/(str(radius)+'px')/mode;target.mkdir(parents=True)
                for im in source.glob('*.png'):
                    shutil.copy2(im,target/im.name);raw=int(im.stem);d=np.load(source/('%03d_maps.npz'%raw));alpha=d['alpha'];Image.fromarray((np.clip(alpha,0,1)*255).round().astype('uint8')).save(target/(im.stem+'_alpha.png'))
                rows.append(dict(radius=radius,mode=mode,frames=31))
        # Exact re-render baseline consistency, same source pixels as prior files.
        original=ROOT/'first_badcase_diagnostics_v1/no_edit/native_rgb';same=max(int(np.abs(np.array(Image.open(im)).astype('int16')-np.array(Image.open(original/im.name)).astype('int16')).max()) for im in (src/'0px/all').glob('*.png'))
        write(out/'index.json',dict(part=a.part,groups=rows,all_control_byte_max=same));write(out/'summary.json',summary)
    else:
        state=json.loads((R/('boundary_controller.json' if a.boundary_only else 'opacity_controller.json')).read_text());assert state['phase']=='complete';index=[]
        for arm in (['boundary'] if a.boundary_only else ['legacy','balanced','field']):
            source=R/'opacity'/arm/'formal';s=json.loads((source/'status.json').read_text());assert s['step']==10000 and s['phase']=='complete';ev=source/'evaluation/010000';target=out/arm;shutil.copytree(ev,target)
            for case in ['turn_left15','translate_lateral_1m']:
                for f in (target/case).glob('*_maps.npz'):
                    raw=int(f.name.split('_')[0]);data=np.load(f);key='evidence/v43_review_maps/'+arm+'/'+case+'/'+str(raw)+'.js';blob=pack({k:(data[k]*255).clip(0,255).round().astype('uint8') for k in data.files});f.with_suffix('.js').write_text('window.EVIDENCE_MAPS['+json.dumps(key)+']='+json.dumps(blob,separators=(',',':'))+';');f.unlink()
            for key in ['contract.json','frozen_verification.json','status.json']:shutil.copy2(source/key,target/key)
            shutil.copy2(R/'opacity'/arm/'restore/restore_verification.json',target/'restore_verification.json');index.append(dict(arm=arm,step=10000,frames=49))
        write(out/'index.json',dict(part=a.part,arms=index,controller=state))
    files={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in out.rglob('*') if f.is_file()};write(out/'manifest.json',files);archive=ROOT/(name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=name)
    write(ROOT/(name+'_receipt.json'),dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),models_exported=False));print(json.dumps(dict(name=name,bytes=archive.stat().st_size,files=len(files))))

if __name__=='__main__':main()
