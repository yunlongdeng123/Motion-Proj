"""封存必要模型状态、49帧新结果和全部关键诊断；不复制原始大数据集。"""
from pathlib import Path
import json,time,tarfile,hashlib,subprocess,shutil
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
import numpy as np

RUN=Path(__file__).resolve().parent.parent
PARENT=RUN.parent/'scene008_a37_C2_20261103_r1'

def main():
    start=time.monotonic();ev=RUN/'final_all49_closeout'
    s=json.loads((ev/'summary.json').read_text());assert s['frames']==49 and s['parent_unchanged']
    media=RUN/'delivery_media';media.mkdir(exist_ok=True)
    comp=media/'comparison_frames';comp.mkdir(exist_ok=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',17);small=ImageFont.truetype('DejaVuSans.ttf',12)
    except OSError:font=small=ImageFont.load_default()
    sheet=Image.new('RGB',(7*300,7*122+32),'#edf2f7');sd=ImageDraw.Draw(sheet)
    sd.text((10,7),'All49: each cell = strong r3 | old C2 | new local3D; frozen proposal, inferred appearance',font=font,fill='#183047')
    for i,raw in enumerate(range(89,138)):
        f=ev/'frames'/f'{raw:03d}'
        canvas=Image.new('RGB',(1440,352),'#edf2f7');d=ImageDraw.Draw(canvas)
        view=np.load(PARENT/'inputs/a6_views'/f'{raw:03d}.npz')
        yy,xx=np.where(view['a6_box_candidate_mask'])
        box=(max(0,int(xx.min())-12),max(0,int(yy.min())-10),min(480,int(xx.max())+13),min(320,int(yy.max())+18))
        for col,(method,label) in enumerate([('r3','Strong baseline'),('oldC2','Old C2'),('new','New local3D')]):
            im=Image.open(f/(method+'_delete.png')).convert('RGB')
            canvas.paste(im,(col*480,32));d.text((col*480+10,7),f'{label} / raw{raw}',font=font,fill='#183047')
            crop=im.crop(box);factor=min(96/crop.width,96/crop.height)
            crop=crop.resize((max(1,round(crop.width*factor)),max(1,round(crop.height*factor))),Image.Resampling.NEAREST)
            x=(i%7)*300+col*100+(100-crop.width)//2;y=(i//7)*122+32
            sheet.paste(crop,(x,y+17));sd.text((x,y),f'{raw} {method}',font=small,fill='#183047')
        canvas.save(comp/f'{i:03d}.png')
    sheet.save(media/'comparison_all49.jpg',quality=86,subsampling=0)
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    video=media/'comparison_all49.mp4'
    subprocess.run([ff,'-y','-loglevel','error','-framerate','10','-start_number','0','-i',str(comp/'%03d.png'),'-frames:v','49',
        '-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p',str(video)],check=True)
    it=imageio_ffmpeg.read_frames(str(video));meta=next(it);count=sum(1 for _ in it);assert count==49
    (media/'media_verification.json').write_text(json.dumps(dict(video_frames=count,video_fps=meta['fps'],contactsheet_frames=49),indent=2)+'\n')
    (media/'render_config_final.json').write_text(json.dumps(s['config'],indent=2)+'\n')
    paths=[]
    def add(p):
        p=Path(p)
        if p.is_file() and '__pycache__' not in p.parts:paths.append(p)
    for name in ['task_contract.json','controller_status.json'] :add(RUN/name)
    for folder in ['source','inputs']:
        for p in (RUN/folder).rglob('*'):add(p)
    for folder in ['formal_r1','smoke32_r1','restore2_r1']:
        for name in ['receipt.json','contract.json','restore.json','training.jsonl','fit_vehicle.py','vehicle_runtime.py']:add(RUN/folder/name)
    add(RUN/'formal_r1/checkpoint001000.pt')
    for name in ['summary.json','evaluation_source.py','progress.json']:add(ev/name)
    for raw in range(89,138):
        for name in ['new_delete.png','new_lower_alpha.png']:add(ev/'frames'/f'{raw:03d}'/name)
    for raw in [105,113,130]:
        for name in ['r3_delete.png','oldC2_delete.png','native_full.png','gt_observed.png']:add(ev/'frames'/f'{raw:03d}'/name)
    for p in (RUN/'evidence').rglob('*'):
        if p.suffix in ['.json','.log','.npz']:add(p)
    for folder in ['peel_r1','peel_r2']:
        for p in (RUN/folder).iterdir():
            if p.suffix in ['.json','.npz','.py','.txt']:add(p)
    for name in ['comparison_all49.jpg','comparison_all49.mp4','media_verification.json','render_config_final.json']:add(media/name)
    paths=sorted(set(paths));out=Path('/root/autodl-tmp/codex_tmp/c2_vehicle_20261104_delivery.tar.gz')
    temp=out.with_name(out.name+'.tmp')
    with tarfile.open(temp,'w:gz') as t:
        for p in paths:t.add(p,arcname=p.relative_to(RUN).as_posix(),recursive=False)
    temp.replace(out)
    h=hashlib.sha256()
    with out.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    receipt=dict(path=str(out),bytes=out.stat().st_size,sha256=h.hexdigest(),files=len(paths),seconds=time.monotonic()-start,
        latest_checkpoint=True,all49_new_PNG=True,full_run_mirror=False,parents='Use preserved B10k/r3/PVD prior backups; no raw dataset duplication',
        video_frames=count,scientific_quality_accepted=False,human_verdict=None)
    Path(str(out)+'.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
