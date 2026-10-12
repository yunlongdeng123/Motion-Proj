#!/usr/bin/env python3
"""CPU-only, atomic checkpoint-event to H264 comparison report builder.

No checkpoint is loaded and no GPU render is performed. The training producer
must publish status=complete only after all component saves and matching-state
evaluation. Missing/incomplete events stay pending. Metadata is embedded in the
HTML for offline viewing; HTTP viewing polls comparison_manifest.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from PIL import Image, ImageOps

CHANNELS=('gt','composite','static','dynamic')
VERSION=1
_event_cache={}

def now(): return datetime.now(timezone.utc).isoformat()

def digest(data):
    if not isinstance(data,bytes): data=json.dumps(data,sort_keys=True,ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def atomic_json(path,data):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)

def resolve_path(value,run):
    if isinstance(value,dict): value=value.get('path')
    if not value: return None
    path=Path(value)
    return path if path.is_absolute() else run/path

def read_json(path,default=None):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError): return default

def ffmpeg_binary():
    executable=shutil.which('ffmpeg')
    if executable: return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def image_fingerprint(path):
    stat=path.stat()
    return [str(path),stat.st_size,stat.st_mtime_ns]

def encode_video(paths,destination,fps,resize_width=None):
    with Image.open(paths[0]) as image: width,height=image.size
    if resize_width and width>resize_width:
        height=round(height*resize_width/width);width=resize_width
    video_width=width+width%2;video_height=height+height%2
    temporary=destination.with_name(destination.stem+'.encoding.mp4')
    command=[ffmpeg_binary(),'-hide_banner','-loglevel','error','-y',
             '-f','rawvideo','-pix_fmt','rgb24','-s',f'{video_width}x{video_height}',
             '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2',
             '-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        for path in paths:
            with Image.open(path) as source:
                frame=source.convert('RGB')
                if resize_width:
                    frame=frame.resize((width,height),Image.Resampling.LANCZOS)
                elif frame.size!=(width,height):
                    raise ValueError('Frame dimensions changed: '+str(path))
                if video_width!=width or video_height!=height:
                    frame=ImageOps.expand(frame,border=(0,0,video_width-width,video_height-height))
                process.stdin.write(frame.tobytes())
        process.stdin.close();process.stdin=None
        _,error=process.communicate(timeout=120)
        if process.returncode: raise RuntimeError(error.decode('utf-8',errors='replace'))
        if not temporary.is_file() or temporary.stat().st_size<100:
            raise RuntimeError('Encoder produced no playable output')
        os.replace(temporary,destination)
    except Exception:
        process.kill();process.communicate()
        raise
    return {'path':str(destination),'frames':len(paths),'fps':fps,'width':video_width,'height':video_height,
            'duration_seconds':len(paths)/fps,'codec':'H264','pixel_format':'yuv420p','bytes':destination.stat().st_size}

def inspect_event(event_path,run,iteration,camera):
    event=read_json(event_path)
    if event is None or event.get('status')!='complete': return None
    if int(event.get('iteration',-1))!=iteration: raise ValueError('Event iteration differs from requested node')
    components=event.get('checkpoints') or {}
    convenience=event.get('checkpoint_paths') or {}
    for name,alias in [('static','static'),('dynamic','dynamic'),('environment','env'),('training_state','training_state')]:
        record=components.get(name) or convenience.get(alias)
        path=resolve_path(record,run)
        if not path or not path.is_file() or path.stat().st_size==0:
            raise ValueError('Completed event is missing '+name+' checkpoint')
        if isinstance(record,dict) and record.get('bytes') is not None and path.stat().st_size!=record['bytes']:
            raise ValueError('Checkpoint size differs from completed event: '+name)
    frames=event.get('frames')
    if not frames: raise ValueError('Completed event has no fixed evaluation frame manifest')
    # Filter a single physical camera; mixing views into a video would falsify time.
    frames=[row for row in frames if row.get('camera',camera)==camera]
    if not frames: raise ValueError('Requested camera has no evaluation frames: '+camera)
    frame_ids=[str(row.get('frame_id') or row.get('image_name')) for row in frames]
    if any(name in ('None','') for name in frame_ids) or len(set(frame_ids))!=len(frame_ids):
        raise ValueError('Frame IDs are absent or duplicated')
    if any(float(frames[i].get('timestamp',i))>float(frames[i+1].get('timestamp',i+1)) for i in range(len(frames)-1)):
        raise ValueError('Producer frame sequence is not chronological')
    paths={channel:[resolve_path(row.get('channels',{}).get(channel),run) for row in frames] for channel in CHANNELS}
    for channel,items in paths.items():
        for path in items:
            if path is None or not path.is_file(): raise ValueError('Missing evaluation channel '+channel+': '+str(path))
    signature=digest({'event':event,'inputs':{key:[image_fingerprint(path) for path in values] for key,values in paths.items()},'version':VERSION})
    cached=_event_cache.get((str(event_path),camera))
    if cached and cached['signature']==signature: return cached
    with Image.open(paths['gt'][0]) as image: dimensions=image.size
    for channel,items in paths.items():
        for path in items:
            with Image.open(path) as image:
                if image.size!=dimensions: raise ValueError('Inconsistent evaluation channel dimensions: '+str(path))
    gt_hash=hashlib.sha256()
    for path in paths['gt']:
        with Image.open(path) as image: gt_hash.update(image.convert('RGB').tobytes())
    metrics=event.get('metrics')
    if isinstance(metrics,(str,dict)):
        if isinstance(metrics,str) or 'path' in metrics: metrics=read_json(resolve_path(metrics,run),{})
    if not isinstance(metrics,dict): metrics={}
    paper=event.get('paper_metrics')
    if isinstance(paper,str) or isinstance(paper,dict) and 'path' in paper:
        paper=read_json(resolve_path(paper,run),{})
    result={'event':event,'frames':frames,'frame_ids':frame_ids,'paths':paths,'metrics':metrics,
            'paper_metrics':paper or {},'signature':signature,'gt_hash':gt_hash.hexdigest(),'dimensions':list(dimensions)}
    _event_cache[(str(event_path),camera)]=result
    return result

def build_once(args):
    run=Path(args.run).resolve();output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True);(output/'videos').mkdir(exist_ok=True)
    old=read_json(output/'comparison_manifest.json',{})
    old_nodes={node['iteration']:node for node in old.get('nodes',[])}
    progress=read_json(run/'progress_status.json',{})
    template=Path(__file__).with_name('checkpoint_comparison_template.html').read_text(encoding='utf-8')
    report={'schema_version':VERSION,'run':str(run),'title':args.title,'camera':args.camera,'fps':args.fps,
            'template_sha256':digest(template.encode('utf-8')),
            'steps':args.steps,'nodes':[],'progress':progress,'history_url':args.history_url,
            'notes':['所有节点使用同一相机、同一帧顺序；视频按10fps播放，模型时间另行记录。',
                     '移除动态视图直接使用独立静态分支；Static与Removed是同一个结果，不是两次实验。',
                     '全图 PSNR / SSIM / LPIPS 始终评估 Composite，与视频显示模式无关；指标来自训练视角，单场景不能代替论文四场景平均结果。',
                     'PSNR* ROI 使用 Composite 的动态区域；PSNR-S 使用 Composite 的静态区域，PSNR-SR 使用移除动态后的静态区域。ROI 按有效像素归一化；论文未明确归一化方式，另提供全图零填充分母结果。没有历史无遮挡真值，因此这些指标不评估删除孔洞的补全质量。',
                     '视频由已完成评估图片CPU编码；不重新加载或修改训练状态。'],
            'frames':[],'common':{},'warnings':[]}
    reference=None
    for iteration in args.steps:
        node={'iteration':iteration,'status':'pending','label':f'{iteration//1000}k' if iteration>=1000 else str(iteration)}
        event_path=run/'checkpoint_events'/f'iteration_{iteration:06d}.json'
        try:
            inspected=inspect_event(event_path,run,iteration,args.camera)
            if inspected is None:
                report['nodes'].append(node);continue
            if reference is None:
                reference=inspected
                report['frames']=[{key:row.get(key) for key in ['uid','frame_id','camera','image_name','timestamp','source_image','width','height']} for row in inspected['frames']]
                report['frame_count']=len(report['frames'])
                report['duration_seconds']=len(report['frames'])/args.fps
                old_common=old.get('common',{})
                if old_common.get('gt_hash')==inspected['gt_hash'] and old_common.get('gt',{}).get('fps')==args.fps and (output/'videos/ground_truth.mp4').is_file():
                    report['common']=old_common
                else:
                    video=encode_video(inspected['paths']['gt'],output/'videos/ground_truth.mp4',args.fps)
                    video['path']='videos/ground_truth.mp4'
                    report['common']={'gt':video,'gt_hash':inspected['gt_hash'],'dimensions':inspected['dimensions']}
                originals=[resolve_path(row.get('source_image'),run) for row in inspected['frames']]
                if all(path and path.is_file() for path in originals):
                    original_signature=digest([image_fingerprint(path) for path in originals])
                    if old_common.get('original_signature')==original_signature and old_common.get('original',{}).get('fps')==args.fps and (output/'videos/original.mp4').is_file():
                        report['common']['original']=old_common['original'];report['common']['original_signature']=original_signature
                    else:
                        video=encode_video(originals,output/'videos/original.mp4',args.fps,resize_width=960)
                        video['path']='videos/original.mp4';report['common']['original']=video
                        report['common']['original_signature']=original_signature
            if inspected['frame_ids']!=reference['frame_ids'] or inspected['gt_hash']!=reference['gt_hash']:
                raise ValueError('This node has different frames/GT pixels and cannot join synchronized comparison')
            previous=old_nodes.get(iteration,{})
            encoding_signature=digest({'input':inspected['signature'],'fps':args.fps,'camera':args.camera})
            videos={}
            if previous.get('signature')==encoding_signature and all((output/record['path']).is_file() for record in previous.get('videos',{}).values()) and len(previous.get('videos',{}))==3:
                videos=previous['videos']
            else:
                for channel in ('composite','static','dynamic'):
                    relative=f'videos/iteration_{iteration:06d}_{channel}.mp4'
                    video=encode_video(inspected['paths'][channel],output/relative,args.fps)
                    video['path']=relative;videos[channel]=video
            node.update(status='ready',signature=encoding_signature,videos=videos,
                        metrics=inspected['metrics'],paper_metrics=inspected['paper_metrics'],
                        frames=len(inspected['frames']),event_file=str(event_path),
                        checkpoint_saved_at=inspected['event'].get('completed_at') or inspected['event'].get('saved_at'),
                        source_provenance=inspected['event'].get('provenance',{}))
        except Exception as error:
            node.update(status='error',message=str(error))
            report['warnings'].append(f'{iteration}: {error}')
        report['nodes'].append(node)
    if not reference:
        report['frame_count']=0;report['duration_seconds']=0
    report['revision']=digest(report)
    report['updated_at']=old.get('updated_at') if old.get('revision')==report['revision'] else now()
    if old.get('revision')!=report['revision'] or not (output/'index.html').exists():
        embedded=json.dumps(report,ensure_ascii=False).replace('</','<\\/')
        html=template.replace('__COMPARISON_MANIFEST_JSON__',embedded)
        temporary=output/'index.html.tmp';temporary.write_text(html,encoding='utf-8');os.replace(temporary,output/'index.html')
        atomic_json(output/'comparison_manifest.json',report)
    print(json.dumps({'status':'updated','output':str(output),'ready':[n['iteration'] for n in report['nodes'] if n['status']=='ready'],
                      'pending':[n['iteration'] for n in report['nodes'] if n['status']=='pending'],
                      'warnings':report['warnings'],'training_state':progress.get('state')},ensure_ascii=False),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--steps',default='10000,20000,30000,40000,50000')
    parser.add_argument('--fps',type=int,default=10);parser.add_argument('--camera',default='image_0')
    parser.add_argument('--title',default='DrivingEditor · 50k 场景迭代')
    parser.add_argument('--history-url',default='history/baseline30k.html')
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--once',action='store_true');modes.add_argument('--watch',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=30)
    args=parser.parse_args();args.steps=[int(step) for step in args.steps.split(',')]
    if args.steps!=sorted(set(args.steps)) or any(step<=0 for step in args.steps): parser.error('steps must be positive, unique and increasing')
    if args.fps<=0 or args.poll_seconds<1: parser.error('fps and poll interval must be positive')
    while True:
        report=build_once(args)
        state=report['progress'].get('state')
        if not args.watch or state=='failed' or state=='completed' and all(node['status']=='ready' for node in report['nodes']): break
        time.sleep(args.poll_seconds)

if __name__=='__main__': main()
