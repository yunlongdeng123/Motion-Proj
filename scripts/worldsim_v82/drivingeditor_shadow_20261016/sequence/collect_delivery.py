"""Read-only collection, standards-compatible video and inspectable report."""
import os,json,subprocess,shutil,sys,hashlib,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'outputs/shadow_sequence_v2';WORK=BASE/'work/shadow_sequence_v2'
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2'
def fetch(remote,local,required=True):
    local=Path(local);local.parent.mkdir(exist_ok=True,parents=True)
    if local.exists() and local.stat().st_size:return True
    temporary=local.with_name(local.name+'.partial')
    result=subprocess.run(['scp','wm-vgpu-1011:'+remote,str(temporary)],capture_output=True,text=True)
    if result.returncode==0:temporary.replace(local)
    elif temporary.exists():temporary.unlink()
    if required and result.returncode:raise RuntimeError(result.stderr)
    return result.returncode==0
def collect():
    OUT.mkdir(exist_ok=True,parents=True)
    for directory,names in {
        'diagnostic49':['summary.json','per_frame.json','videos.json','runtime_contract.json'],
        'surface_guarded_data':['summary.json','atlas_rgb.png','atlas_support.png','protection_boxes.json'],
        'joint_v1':['runtime_contract.json','actor_contamination.json','caster_parts.json','gradient_smoke.json','initial_metrics.json','validation_curve.json','background_restore_check.json','shadow_restore_check.json','joint_restore_check.json','full49_metrics.json','training.jsonl','status.json','final.pt'],
        'final49_v1':['summary.json','per_frame.json','runtime_contract.json','status.json']}.items():
        for name in names:fetch(REMOTE+'/'+directory+'/'+name,OUT/directory/name,required=directory!='final49_v1')
    for name in ['diagnostic49','joint_v1','finish_pipeline']:
        fetch(REMOTE+'/'+name+'_launch.json',OUT/'evidence'/(name+'_launch.json'))
    fetch(REMOTE+'/finish_pipeline_status.json',OUT/'evidence/finish_pipeline_status.json',False)
    for name in ['old_metric_visibility_audit.json','ownership_audit.json','shadow_math_validation.json','away_reference_audit.json','away_reference_coverage.png']:
        fetch(REMOTE+'/'+name,OUT/'evidence'/name)
    (OUT/'scripts').mkdir(exist_ok=True)
    for file in WORK.glob('*.py'):shutil.copy2(file,OUT/'scripts'/file.name)
    ffmpeg=shutil.which('ffmpeg');assert ffmpeg
    for directory in ['diagnostic49','final49_v1']:
        summary=OUT/directory/'summary.json'
        if not summary.exists():continue
        manifest=json.loads(summary.read_text())
        for video in manifest.get('videos',manifest.get('video',[])):
            if not video:continue
            name=Path(video['file']).name;dest=OUT/'videos'/directory/name
            raw=WORK/'encoded_source'/directory/name
            fetch(video['file'],raw)
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True)
                subprocess.run([ffmpeg,'-v','error','-y','-i',str(raw),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
    for phase in ['initial','background','shadow','joint']:
        for fid in ['0008089','0008101','0008137']:
            for mode in ['observed','static','delete_target']:
                fetch(REMOTE+'/joint_v1/previews/'+phase+'/'+fid+'_'+mode+'.png',OUT/'previews'/phase/(fid+'_'+mode+'.png'),False)
    for fid in ['0008089','0008095','0008101','0008107','0008113','0008125','0008137']:
        fetch(REMOTE+'/diagnostic49/review_'+fid+'.jpg',OUT/'diagnostic49'/('review_'+fid+'.jpg'))
    # Full49 visual review: every frame of the side-by-side video appears in
    # contact sheets. This is checked visually by the primary agent separately.
    video=OUT/'videos/final49_v1/comparison.mp4'
    if video.exists():
        frames=WORK/'review_frames';frames.mkdir(exist_ok=True)
        subprocess.run([ffmpeg,'-v','error','-y','-i',str(video),str(frames/'%03d.png')],check=True)
        from PIL import Image,ImageDraw
        paths=sorted(frames.glob('*.png'));assert len(paths)==49
        for page in range(4):
            subset=paths[page*14:(page+1)*14]
            if not subset:continue
            sheet=Image.new('RGB',(960, (len(subset)+1)//2*368),'white')
            d=ImageDraw.Draw(sheet)
            for k,p in enumerate(subset):
                im=Image.open(p);im.thumbnail((480,344));x=k%2*480;y=k//2*368
                sheet.paste(im,(x,y+24));d.text((x+5,y+5),'Frame '+str(page*14+k+1),fill='black')
            sheet.save(OUT/('full49_review_'+str(page+1)+'.jpg'),quality=93)
        shutil.copy2(paths[12],OUT/'comparison_raw101.png')
        shutil.copy2(paths[-1],OUT/'comparison_raw137.png')
    print('Collected to',OUT)
if __name__=='__main__':collect()
