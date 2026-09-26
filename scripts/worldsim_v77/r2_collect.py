import pathlib,json,tarfile,subprocess
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation
ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-PIPELINE-R2-20260926/r1');OLD=ROOT.parents[1]/'WS-V77-EXPLICIT-POC-20260926/r1'
review=ROOT/'review';review.mkdir(exist_ok=True);summary={'task_id':'WS-V77-PIPELINE-R2-20260926','training_steps':0,'human_verdict':None,'scenes':[]}
for s,c,n,refs in [('scene_0230',2,50,[0,5,10]),('scene_0255',3,100,list(range(0,100,10)))]:
    out=review/s;out.mkdir(exist_ok=True);deltas=[];outside=[]
    for f in range(n):
        raw=np.array(Image.open(OLD/s/'rgb'/f'cam{c}'/f'{f:05d}.jpg').convert('RGB'))
        old=np.array(Image.open(OLD/s/'inpaint'/f'cam{c}'/'frames'/f'{f:04d}.png').convert('RGB'))
        new=np.array(Image.open(ROOT/s/'inpaint'/f'cam{c}'/'frames'/f'{f:04d}.png').convert('RGB'))
        mask=binary_dilation(np.array(Image.open(OLD/s/'masks'/f'cam{c}'/f'{f:05d}.png'))>0,iterations=4)
        diff=np.abs(new.astype(float)-old.astype(float));err=np.abs(new.astype(float)-raw.astype(float))
        if mask.any():deltas.append(float(diff[mask].mean()))
        outside.append(float(err[~mask].max()))
        if f in refs:
            strip=Image.new('RGB',(688*3,414),'#eef3f7');d=ImageDraw.Draw(strip)
            for i,(name,rgb) in enumerate([('RGB',raw),('R1 short context',old),('R2 full 196 frames',new)]):
                strip.paste(Image.fromarray(rgb),(i*688,30));d.text((i*688+10,8),f'{s} f{f:03d} CAM{c} | {name}',fill='#15384c')
            strip.save(out/f'f{f:03d}.jpg',quality=93)
        # Local user review keeps complete original evaluation windows.
        for key,rgb in [('raw',raw),('old_delete',old),('new_delete',new)]:
            folder=out/key;folder.mkdir(exist_ok=True);Image.fromarray(rgb).save(folder/f'{f:04d}.jpg',quality=94)
    for key in ['raw','old_delete','new_delete']:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-framerate','10','-i',str(out/key/'%04d.jpg'),'-c:v','libx264','-threads','4','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{key}.mp4')],check=True)
    summary['scenes'].append({'name':s,'camera':c,'evaluation_video_frames':n,'comparison_frames':refs,'masked_old_new_mean_absolute_difference_0_255':float(np.mean(deltas)),'outside_removal_region_max_change_0_255':max(outside),'quality_metric_warning':'旧新像素差仅度量变化量，不是无真值背景质量分数','human_verdict':None})
(review/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with tarfile.open(ROOT/'r2_review.tar.gz','w:gz') as t:
    for p in review.rglob('*'):
        if p.is_file() and (p.parent.parent==review or p.parent==review):t.add(p,arcname=str(p.relative_to(review)))
print(json.dumps(summary,ensure_ascii=False),flush=True)
