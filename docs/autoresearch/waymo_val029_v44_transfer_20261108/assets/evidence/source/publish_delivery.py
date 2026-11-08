"""Build compact all-frame videos and a reviewable evidence package, preserving failures."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,shutil,subprocess,tarfile,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent.parent
PREP=Path('/root/autodl-tmp/codex_tmp/waymo_val029_cpu_20261108')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def main():
    import imageio_ffmpeg
    dest=ROOT/'delivery';dest.mkdir(exist_ok=False);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);media=dest/'media';media.mkdir()
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();commands=[(media/'full198_comparison.mp4',(1920,348)),(media/'event119_197_crops.mp4',(2040,528))];procs=[]
    for path,size in commands:
        procs.append(subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','10','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE))
    for raw in range(198):
        held=79<=raw<=118;final=ROOT/('final_heldout_eval' if held else 'final_all_eval')/f'{raw:03d}';native=ROOT/('native_heldout_eval' if held else 'formal10k_eval')/f'{raw:03d}';ims=[Image.open(final/'observed.png'),Image.open(native/'keep.png'),Image.open(final/'keep.png'),Image.open(final/'delete.png')];im=Image.new('RGB',(1920,348),'#111827');d=ImageDraw.Draw(im)
        for i,(pic,label) in enumerate(zip(ims,['Observed FRONT','Native reconstruction','Editable scene KEEP','Editable scene DELETE red sedan'])):im.paste(pic,(i*480,28));d.text((i*480+8,5),label,fill='white',font=font)
        d.text((8,322),f'raw {raw:03d} | '+('HELD OUT from all fitting' if held else 'BUILD observation'),fill='#ffe6a0',font=font);procs[0].stdin.write(im.tobytes())
        if raw>=119:
            crop=Image.new('RGB',(2040,528),'#111827');draw=ImageDraw.Draw(crop)
            for i,(pic,label) in enumerate(zip([ims[0],ims[2],ims[3]],['Observed','KEEP','DELETE red sedan'])):crop.paste(pic.crop((280,125,450,250)).resize((680,500),Image.Resampling.NEAREST),(i*680,28));draw.text((i*680+8,5),f'{label} | raw {raw:03d}',fill='white',font=font)
            procs[1].stdin.write(crop.tobytes())
    for p in procs:p.stdin.close();assert p.wait()==0
    stages=['formal10k_eval','semantic_all_eval','semantic_asset_all_r2','final_all_eval']
    for raw in [130,155,160,170,190,197]:
        for stage in stages:
            for name in ['keep.png','delete.png','observed.png']:
                p=ROOT/stage/f'{raw:03d}'/name
                if p.exists():copy(p,dest/'frames'/f'{raw:03d}'/stage/name)
        copy(PREP/'inputs/image/image_0'/f'0029{raw:03d}.png',dest/'frames'/f'{raw:03d}/source_full.png')
    # Key-frame complete float layers, all-frame scalar checks and immutable hashes.
    manifest=[]
    for stage in ['formal10k_eval','semantic_all_eval','semantic_asset_all_r2','final_all_eval','native_heldout_eval','final_heldout_eval']:
        for p in (ROOT/stage).rglob('*.npz'):
            manifest.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
            if p.parent.name in ['130','155','160','190'] and stage in ['formal10k_eval','semantic_all_eval','final_all_eval']:copy(p,dest/'key_float'/stage/p.parent.name/p.name)
        summary=ROOT/stage/'summary.json'
        if summary.exists():copy(summary,dest/'evidence'/stage/'summary.json')
    (dest/'all_render_float_manifest.json').write_text(json.dumps(manifest,indent=2))
    for folder in ['analysis','source','revealed_prior','revealed_surface','revealed_material_r2','world_road','semantic_all158','road_gate32_r2','road_gates','smoke_restore','smoke_eval_r2','formal10k','heldout']:
        p=ROOT/folder
        if not p.exists():continue
        for q in p.rglob('*'):
            if not q.is_file() or '__pycache__' in q.parts:continue
            if (folder in ['source','analysis'] and q.suffix in ['.py','.json','.csv','.png']) or q.suffix=='.json':copy(q,dest/'evidence'/q.relative_to(ROOT))
    for p in ROOT.glob('*.json'):copy(p,dest/'evidence'/p.name)
    for p in ROOT.glob('*.log'):copy(p,dest/'logs'/p.name)
    for raw in [130,155,160,190]:copy(PREP/'masks_all_resolved_r2'/f'{raw:03d}/labels.png',dest/'masks'/f'{raw:03d}.png')
    archive=ROOT/'review_delivery.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:t.add(dest,arcname='evidence_package')
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),all_render_float_files=len(manifest),videos=[dict(path=str(p.relative_to(dest)),frames=n) for p,n in [(commands[0][0],198),(commands[1][0],79)]],full_float_layers='all retained remotely and SHA listed; key4 frames copied here; full recovery models and source bank separately archived',quality_accepted=False,human_verdict=None);(ROOT/'review_delivery_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
