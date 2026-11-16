"""只输出小型可视化；float 与无损原图保留远端。"""
from pathlib import Path
import json,hashlib,tarfile,time
import numpy as np
from PIL import Image
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene065_closedloop_20261116_r1/handoff_coverage_r2/contribution_diagnostics_20261116')
def main():
    src=BASE/'diagnostic_delivery';out=BASE/'light'/ 'diagnostic_delivery';out.mkdir(parents=True,exist_ok=False)
    data=json.loads((src/'manifest.json').read_text());total=0
    stops=data['legend']['static_alphaT']['stops']
    color=lambda s:[int(s[i:i+2],16) for i in (1,3,5)]
    palette=np.uint8(np.stack([np.interp(np.linspace(0,1,256),[v[0] for v in stops],[color(v[1])[c] for v in stops]) for c in range(3)],-1)+.5)
    for row in data['frames']:
        raw=row['raw']
        for role in ['road_gap','static_alphaT']:
            suffix='.webp' if role=='road_gap' else '.png';dest=out/role/(f'{raw:03d}'+suffix);dest.parent.mkdir(exist_ok=True)
            if role=='road_gap':
                Image.open(src/row[role]['path']).save(dest,quality=90,method=6)
            else:
                with np.load(BASE/'float'/f'{raw:03d}.npz') as z:values=np.uint8(np.clip(z['static_alphaT'],0,1)*255+.5)
                im=Image.fromarray(values,mode='P');im.putpalette(palette.flatten().tolist());im.save(dest,optimize=True)
            row[role]=dict(path=str(dest.relative_to(out)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest());total+=dest.stat().st_size
    data['preview_encoding']=dict(road_gap='WebP quality90, visual overlay only; authoritative masks and counts remain float/domain data',
                                  static_alphaT='256-level indexed PNG; fixed colormap; alphaT rounding error <= 1/510',
                                  original_images_and_float_kept_remote=True)
    (out/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    archive=BASE/'diagnostic_delivery_light.tar'
    with tarfile.open(archive,'w') as tar:tar.add(out,arcname='diagnostic_delivery')
    receipt=dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,preview_bytes=total)
    (BASE/'delivery_ready_light.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
