"""Build a local-file compatible, frame-synchronized branch/delete video player."""
from pathlib import Path
import argparse,json,subprocess,shutil
import numpy as np
from PIL import Image
W=Path(__file__).parent;OUT=W.parents[1]/'outputs/identity_comparison_20261018'
ACTORS=[6,37,5,26,18]

def encode(source,filename,target):
    if target.is_file():return
    ff=shutil.which('ffmpeg')
    if not ff:raise FileNotFoundError('ffmpeg')
    pattern=str((source/'0008%03d'/filename).resolve())
    for raw in range(89,138):
        if not (source/f'0008{raw:03d}'/filename).is_file():raise FileNotFoundError((source,raw,filename))
    subprocess.run([ff,'-hide_banner','-loglevel','error','-threads','2','-filter_threads','1','-framerate','10','-start_number','89',
        '-i',pattern,'-frames:v','49','-c:v','libx264','-threads','2','-g','1','-keyint_min','1','-sc_threshold','0',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-pending',action='store_true');a=ap.parse_args()
    videos=OUT/'branch_videos';videos.mkdir(exist_ok=True);models={}
    for name,label in [('old5','旧 GN11 30K + 五实例 30K'),('r15','原 R15 · 冻结 4K'),('repair4k','修复 R15 · 4K')]:
        src=OUT/'assets'/name;branch=W/'branches'/('branches_'+name);receipt=branch/'branch_receipt.json'
        if not src.is_dir():continue
        ready=receipt.is_file()
        if not ready and not a.allow_pending:
            if name in ['old5','r15']:raise FileNotFoundError('Missing actual branch render: '+str(receipt))
            continue
        if ready:
            r=json.loads(receipt.read_text(encoding='utf-8'));assert r['frames']==49 and r['parameter_updates']==0 and len(r['records'])==245
        item=dict(label=label,ready=ready,base={},actors={})
        def add(key,folder,file,actor=None):
            if not (folder/'0008089'/file).is_file():return
            dest=videos/(name+'_'+('a%02d_'%actor if actor is not None else '')+key+'.mp4')
            encode(folder,file,dest)
            (item['base'] if actor is None else item['actors'].setdefault(str(actor),{}))[key]='branch_videos/'+dest.name
        for key,file in [('observed','gt.png'),('full','full.png'),('full_no_shadow','full_no_shadow.png')]:add(key,src,file)
        if ready:
            for key in ['dynamic','dynamic_alpha','static','static_alpha']:add(key,branch,key+'.png')
            if name!='old5':add('static_shadow',branch,'static_shadow.png')
        elif name=='old5':add('static',src,'static.png')
        for actor in ACTORS:
            for key,file in [('isolated',f'a{actor:02d}_isolated.png'),('isolated_alpha',f'a{actor:02d}_alpha.png'),('delete',f'a{actor:02d}_delete.png'),('delete_no_shadow',f'a{actor:02d}_delete_no_shadow.png')]:add(key,src,file,actor)
            if ready:
                add('dynamic_delete',branch,f'a{actor:02d}_dynamic_delete.png',actor)
                add('dynamic_delete_alpha',branch,f'a{actor:02d}_dynamic_delete_alpha.png',actor)
                if name!='old5':add('static_delete_shadow',branch,f'a{actor:02d}_static_delete_shadow.png',actor)
        models[name]=item
    # Same observed-input ROI for every model and every branch, including deletion.
    # It only zooms the display; the encoded videos are complete, unmasked frames.
    rois={str(a):[] for a in ACTORS};visible={str(a):[] for a in ACTORS}
    for raw in range(89,138):
        fid=f'0008{raw:03d}';labels=np.asarray(Image.open(W/'metadata/prepared5/instance_masks'/(fid+'.png')))
        for actor in ACTORS:
            idx=[5,26,18,6,37].index(actor)+1;y,x=np.where(labels==idx);visible[str(actor)].append(len(x))
            if len(x):
                pad=max(45,int(max(x.max()-x.min(),y.max()-y.min())*.35))
                box=[max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(480,int(x.max())+pad+1),min(320,int(y.max())+pad+1)]
            else:box=[0,0,480,320]
            rois[str(actor)].append(box)
    data=dict(models=models,rois=rois,visible_input_pixels=visible,
        best_frame={k:89+int(np.argmax(v)) for k,v in visible.items()},first=89,last=137,fps=10,actors=ACTORS)
    template=Path(__file__).with_name('sync_player_template.html').read_text(encoding='utf-8')
    page=template.replace('@@DATA@@',json.dumps(data,separators=(',',':'),ensure_ascii=False))
    required_ready=all(n in models and models[n]['ready'] for n in ['old5','r15'])
    (OUT/('sync_player.html' if required_ready else 'sync_player.preview.html')).write_text(page,encoding='utf-8')
    # Preserve the report as a supplementary page; the useful video interface is primary.
    if not (OUT/'diagnostics.html').is_file() and (OUT/'index.html').is_file():shutil.copy2(OUT/'index.html',OUT/'diagnostics.html')
    if required_ready:(OUT/'index.html').write_text(page,encoding='utf-8')
    (OUT/'player_status.json').write_text(json.dumps(dict(models={n:m['ready'] for n,m in models.items()},frames=49,actors=ACTORS,
        actual_branch_render_required=True,all_branch_videos_full_frame=True,
        composite_exports_use_shared_observed_ROI=True,shared_frame_axis=True),indent=2),encoding='utf-8')
    print(json.dumps(dict(models={n:m['ready'] for n,m in models.items()},videos=len(list(videos.glob('*.mp4'))))))

if __name__=='__main__':main()
