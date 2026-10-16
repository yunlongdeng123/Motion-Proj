import argparse,json,subprocess,shutil
from pathlib import Path
from collect_delivery import OUT,WORK,REMOTE,fetch
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--train',action='append',default=[]);parser.add_argument('--evaluation',action='append',default=[]);a=parser.parse_args()
    for directory in a.train:
        for name in ['runtime_contract.json','actor_contamination.json','caster_parts.json','gradient_smoke.json','initial_metrics.json','validation_curve.json','background_restore_check.json','shadow_restore_check.json','joint_restore_check.json','full49_metrics.json','training.jsonl','status.json','final.pt','ownership_receipt.json','actor_responsibility_check.json']:
            fetch(REMOTE+'/'+directory+'/'+name,OUT/directory/name,required=name!='actor_responsibility_check.json')
        fetch(REMOTE+'/'+directory+'_launch.json',OUT/'evidence'/(directory+'_launch.json'))
        fetch(REMOTE+'/source_snapshots/'+directory+'/sha256.json',OUT/directory/'source_sha256.json')
        for phase in ['initial','background','shadow','joint']:
            for fid in ['0008089','0008101','0008137']:
                for mode in ['observed','static','delete_target']:
                    fetch(REMOTE+'/'+directory+'/previews/'+phase+'/'+fid+'_'+mode+'.png',OUT/'previews'/directory/phase/(fid+'_'+mode+'.png'))
    for directory,names in {
        'ownership_probe':['comparisons.json','runtime_contract.json'],
        'contribution_audit':['rounds.json','base_selection.json','zero_update_metrics.json','runtime_contract.json','selection.pt'],
        'contribution_guarded_audit':['rounds.json','base_selection.json','zero_update_metrics.json','runtime_contract.json','selection.pt'],
        'guard_refinement_v2':['summary.json','overrides.npz','foreground_proxies.npz','fallback_boxes.json','early_shadow_box_matches.json','guard_review_0008089.png','guard_review_0008101.png','guard_review_0008137.png']}.items():
        for name in names:fetch(REMOTE+'/'+directory+'/'+name,OUT/directory/name,required=False)
    for name in ['body_endpoint_audit.json','guard_pixel_audit.json']:
        fetch(REMOTE+'/'+name,OUT/'evidence'/name,False)
    ffmpeg=shutil.which('ffmpeg');assert ffmpeg
    from PIL import Image,ImageDraw
    for directory in a.evaluation:
        for name in ['summary.json','per_frame.json','runtime_contract.json','status.json']:
            fetch(REMOTE+'/'+directory+'/'+name,OUT/directory/name)
        summary=json.loads((OUT/directory/'summary.json').read_text())
        for row in summary['videos']:
            file=Path(row['file']).name;raw=WORK/'encoded_source'/directory/file;dest=OUT/'videos'/directory/file
            fetch(row['file'],raw)
            if not dest.exists():
                dest.parent.mkdir(exist_ok=True,parents=True)
                subprocess.run([ffmpeg,'-v','error','-y','-i',str(raw),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
        frames=WORK/'review_frames'/directory;frames.mkdir(exist_ok=True,parents=True)
        subprocess.run([ffmpeg,'-v','error','-y','-i',str(OUT/'videos'/directory/'comparison.mp4'),str(frames/'%03d.png')],check=True)
        paths=sorted(frames.glob('*.png'));assert len(paths)==49
        review=OUT/'reviews'/directory;review.mkdir(exist_ok=True,parents=True)
        for page in range(4):
            subset=paths[page*14:(page+1)*14];sheet=Image.new('RGB',(960,(len(subset)+1)//2*368),'white');draw=ImageDraw.Draw(sheet)
            for k,p in enumerate(subset):
                image=Image.open(p);image.thumbnail((480,344));x=k%2*480;y=k//2*368;sheet.paste(image,(x,y+24));draw.text((x+5,y+5),'Frame '+str(page*14+k+1),fill='black')
            sheet.save(review/('page_%d.jpg'%(page+1)),quality=93)
        for i in [0,12,24,36,48]:shutil.copy2(paths[i],review/('comparison_%03d.png'%(i+1)))
        # Native edit views at strongest edit angles and beginning/end.
        for stream in ['translate_smooth','yaw_smooth_15','yaw_smooth_30','fixed_static','fixed_translate','confidence','delete_target']:
            selected=review/stream;selected.mkdir(exist_ok=True)
            subprocess.run([ffmpeg,'-v','error','-y','-i',str(OUT/'videos'/directory/(stream+'.mp4')),'-vf',"select=eq(n\,0)+eq(n\,12)+eq(n\,24)+eq(n\,36)+eq(n\,48)",'-vsync','0',str(selected/'%02d.png')],check=True)
    (OUT/'scripts').mkdir(exist_ok=True)
    for p in WORK.glob('*.py'):shutil.copy2(p,OUT/'scripts'/p.name)
    print('Collected',a.train,a.evaluation)
if __name__=='__main__':main()
