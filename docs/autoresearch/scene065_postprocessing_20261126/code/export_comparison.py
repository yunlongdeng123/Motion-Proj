"""Archive exact common-domain PNGs, 10-fps videos and paired diagnostics."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, '/root/autodl-tmp/_wsjob')
from vace_run_clip import CLIPS, OUT, R5, SRC
from scene065_propainter_20261126_r5 import region_stats

out = Path(OUT)/'delivery_codex'
out.mkdir(exist_ok=True)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
report = dict(schema='scene065.postprocess.paired_comparison.20261126.v1',
    training_updates=0, gaussian_updates=0, width=480, height=320, fps=10,
    mask_tag='w8_swept_cd18', seed=2026, vace_sampling_steps=25,
    vace_reference_images=False, common_masked_RGB_value=127, clips={},
    metric_notes=['All RGB/texture/seam statistics are diagnostics without hidden-road ground truth.',
                  'Temporal delta is unwarped adjacent-frame RGB change, not a motion-compensated flicker score.',
                  'Peak memory is nvidia-smi device memory sampled every 0.5 seconds.',
                  'Primary ProPainter control sees the same 21 frames as VACE; full199 run is supplemental.'])
pilot=Path(OUT).parent/'scene065_vace13b_20261126_r1'
report['engineering_replays']={}
for c in CLIPS:
    p=pilot/c/'summary_steps25_none.json'
    if p.exists():report['engineering_replays'][c]=dict(run=json.loads(p.read_text())['run'],
        valid_primary_comparison=False,reason='Unknown RGB was not gray-neutralized for VACE inpainting')

def encode(folder, lo, n, destination):
    cmd=[ffmpeg,'-y','-loglevel','error','-framerate','10','-start_number',str(lo),
         '-i',str(folder/'%05d.png'),'-frames:v',str(n),'-c:v','libx264','-preset','fast',
         '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    subprocess.run(cmd,check=True)
    cap=cv2.VideoCapture(str(destination)); got=0
    while cap.read()[0]: got+=1
    cap.release(); assert got==n,(destination,got,n)

for clip,(lo,hi) in CLIPS.items():
    base=Path(OUT)/clip
    dest=out/clip; dest.mkdir(exist_ok=True)
    pp=json.loads((base/'matched_propainter/run_info.json').read_text())
    va=json.loads((base/'summary_steps25_none.json').read_text())
    rows={k:[] for k in ('input','propainter','propainter_tone','vace')}
    paths={k:[] for k in rows}
    prev={}
    for raw in range(lo,hi+1):
        k=raw-lo
        mask=cv2.imread(str(base/f'src_mask/{k:05d}.png'),0)>0
        K=np.load(f'{SRC}/domains/{raw:03d}.npz')['K'].astype(bool)
        sources=dict(input=Path(R5)/f'input_frames/{raw:05d}.png',
            propainter=base/f'matched_propainter/none/final/{raw:05d}.png',
            propainter_tone=base/f'matched_propainter/medirq/final/{raw:05d}.png',
            vace=base/f'composite/{raw:05d}.png')
        original=cv2.imread(str(sources['input']))
        assert original.shape==(320,480,3)
        for name,source in sources.items():
            image=cv2.imread(str(source)); assert image is not None and image.shape==original.shape
            diff=np.abs(image.astype(np.int16)-original.astype(np.int16)).max(2)
            assert not diff[~mask].any() and not diff[K].any(),(name,raw)
            folder=dest/'frames'/name; folder.mkdir(parents=True,exist_ok=True)
            target=folder/f'{raw:05d}.png'; shutil.copy2(source,target); paths[name].append(target)
            row=dict(raw=raw,mask_pixels=int(mask.sum()),retained_pixels=int(K.sum()),
                outside_changed=0,retained_changed=0,**region_stats(image,mask))
            if name in prev:
                old,oldmask=prev[name]; overlap=oldmask&mask
                row['unwarped_temporal_RGB_delta']=float(np.abs(image.astype(np.float32)-old).mean(2)[overlap].mean())
            prev[name]=(image.astype(np.float32),mask)
            rows[name].append(row)
        masks=dest/'masks';masks.mkdir(exist_ok=True)
        shutil.copy2(base/f'src_mask/{k:05d}.png',masks/f'{raw:05d}.png')
        common=dest/'common_masked_input';common.mkdir(exist_ok=True)
        shutil.copy2(base/f'src_video/{k:05d}.png',common/f'{raw:05d}.png')
    for name in rows:
        encode(dest/'frames'/name,lo,hi-lo+1,dest/f'{name}.mp4')
    cmd=[ffmpeg,'-y','-loglevel','error']
    for name in ('input','propainter','vace'): cmd+=['-i',str(dest/f'{name}.mp4')]
    cmd+=['-filter_complex','[0:v][1:v][2:v]hstack=inputs=3[v]','-map','[v]',
          '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'comparison.mp4')]
    subprocess.run(cmd,check=True)
    for raw in (int(clip),):
        sheet=Image.new('RGB',(1440,346),(10,17,28)); draw=ImageDraw.Draw(sheet)
        for col,name in enumerate(('input','propainter','vace')):
            draw.text((col*480+8,6),f'{name} | raw {raw}',fill=(230,240,250))
            sheet.paste(Image.open(dest/f'frames/{name}/{raw:05d}.png').convert('RGB'),(col*480,26))
        sheet.save(dest/f'keyframe_{raw:03d}.png')
    sample_raws=[lo+2,lo+6,int(clip),hi-5,hi-1]
    sheet=Image.new('RGB',(900,len(sample_raws)*216),(10,17,28));draw=ImageDraw.Draw(sheet)
    for row,raw in enumerate(sample_raws):
        for col,name in enumerate(('input','propainter','vace')):
            draw.text((col*300+5,row*216+3),f'{name} raw {raw}',fill=(230,240,250))
            pic=Image.open(dest/f'frames/{name}/{raw:05d}.png').convert('RGB').resize((300,200))
            sheet.paste(pic,(col*300,row*216+16))
    sheet.save(dest/'sequence.png')
    aggs={name:{key:float(np.mean([r[key] for r in rr if key in r]))
        for key in ('luma_delta','grad_ratio_patch_vs_ring','seam_inner_over_far','hist_chi2_bgr','unwarped_temporal_RGB_delta')}
        for name,rr in rows.items()}
    report['clips'][clip]=dict(raw_range=[lo,hi],frames=hi-lo+1,propainter_run=pp,vace_run=va['run'],
        common_mask_mean=float(np.mean([r['mask_pixels'] for r in rows['input']])),
        retained_pixel_frame_denominator=sum(r['retained_pixels'] for r in rows['input']),
        metrics=aggs,per_frame=rows,protection_PNG_exact=True)

manifest=json.loads((Path(OUT)/'clips_manifest.json').read_text())
for clip,entry in manifest['clips'].items():
    entry['reference_images_used']=False
(out/'clips_manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.copy2(Path(OUT)/'smoke_087/smoke_info.json',out/'smoke_info.json')
shutil.copy2(Path(R5)/'stage_report_w8_swept_cd18.json',out/'mask_stage_report.json')
code=out/'code';code.mkdir(exist_ok=True)
for name in ('vace_run_clip.py','vace_smoke.py','run_matched_propainter.py','scene065_propainter_20261126_r5.py','export_comparison.py','execute_comparison.sh','stage_official_inpainting.py'):
    shutil.copy2(Path('/root/autodl-tmp/_wsjob')/name,code/name)
official=Path('/root/autodl-tmp/third_party/Wan2.1_vace_official')
patches=code/'official_adaptations';patches.mkdir(exist_ok=True)
for rel,name in [('generate.py','wan_generate.py'),('wan/configs/__init__.py','wan_config.py'),
                 ('wan/vace.py','wan_vace.py'),('wan/utils/vace_processor.py','wan_vace_processor.py'),
                 ('wan/modules/attention.py','wan_attention.py')]:
    shutil.copy2(official/rel,patches/name)
    old=Path('/root/autodl-tmp/_wsjob/backups_postprocessing_20261126')/name
    if old.exists():shutil.copy2(old,patches/(name+'.before'))
versions=subprocess.check_output([sys.executable,'-c','import torch,transformers,numpy,cv2;print(torch.__version__,transformers.__version__,numpy.__version__,cv2.__version__)'],text=True)
(out/'environment.json').write_text(json.dumps(dict(versions_torch_transformers_numpy_cv2=versions.strip(),
    python=sys.executable,official_git=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()),indent=2))
shutil.copytree(Path(OUT)/'smoke_087/attempts',out/'smoke_failed_attempts',dirs_exist_ok=True)
for clip in CLIPS:
    for p in (Path(OUT)/clip).glob('vace_stdout*.log'):shutil.copy2(p,out/f'{clip}_{p.name}')
(out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
files=[]
for p in sorted(out.rglob('*')):
    if p.is_file() and p.name!='files.json': files.append(dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,
                                     sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(out/'files.json').write_text(json.dumps(files,indent=2)+'\n')
archive=Path(OUT)/'delivery_codex.tar.gz'
subprocess.run(['tar','czf',str(archive),'-C',str(out),'.'],check=True)
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    runs={k:dict(pp=v['propainter_run'],vace=v['vace_run'])for k,v in report['clips'].items()}),indent=2))
