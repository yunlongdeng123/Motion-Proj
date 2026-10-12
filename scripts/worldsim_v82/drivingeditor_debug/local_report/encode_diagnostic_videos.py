"""Encode real fixed49 diagnostic frames, then decode every frame for validation."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/drivingeditor_vgpu_debug/assets'
OUT.mkdir(parents=True,exist_ok=True)
records=[]
for case in ('original30k_to30300','opacity_outside30k_to30300'):
    root=HERE/'continuation_diagnostic'/case
    status_path=root/'status.json'
    eval_path=root/'extra_eval/30300/diagnostic_eval.json'
    if not status_path.exists() or not eval_path.exists():continue
    status=json.loads(status_path.read_text())
    data=json.loads(eval_path.read_text())
    if status['status']!='complete' or status['actual_exit_code']!=0:continue
    frame_ids=[r['frame_id'] for r in data['frames']]
    assert len(frame_ids)==49 and frame_ids==sorted(frame_ids)
    for channel in ('gt','composite','static','dynamic'):
        paths=[root/'extra_eval/30300'/channel/(Path(f).name+'.png') for f in frame_ids]
        digest=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in paths)).hexdigest()
        dest=OUT/(case+'_'+channel+'.mp4')
        receipt=dest.with_suffix('.json')
        if not (dest.exists() and receipt.exists() and json.loads(receipt.read_text()).get('input_digest')==digest):
            temp=dest.with_suffix('.part')
            with av.open(str(temp),'w',format='mp4',options={'movflags':'+faststart'}) as container:
                stream=container.add_stream('libx264',rate=10)
                stream.width=480;stream.height=320;stream.pix_fmt='yuv420p'
                stream.options={'crf':'18','preset':'medium'}
                for index,path in enumerate(paths):
                    arr=np.asarray(Image.open(path).convert('RGB'))
                    assert arr.shape==(320,480,3)
                    frame=av.VideoFrame.from_ndarray(arr,format='rgb24')
                    frame.pts=index;frame.time_base=Fraction(1,10)
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            temp.replace(dest)
        with av.open(str(dest)) as container:
            stream=container.streams.video[0]
            fps=float(stream.average_rate)
            pts=[]
            for frame in container.decode(video=0):
                assert (frame.width,frame.height)==(480,320)
                frame.to_ndarray(format='rgb24')
                pts.append(float(frame.pts*frame.time_base))
        assert fps==10 and len(pts)==49 and all(abs(p-i/10)<1e-9 for i,p in enumerate(pts))
        row={'case':case,'channel':channel,'input_digest':digest,'frame_ids':frame_ids,'frame_count':49,'fps':10,'duration_seconds':4.9,'pts_seconds':pts,'width':480,'height':320,'decode_all_frames':'pass','sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'file':dest.name,'UI_synchronized_playback_verified':False}
        receipt.write_text(json.dumps(row,indent=2),encoding='utf8')
        records.append(row)
(OUT.parent/'video_validation.json').write_text(json.dumps({'videos':records,'count':len(records),'actual_media_decoded':True,'UI_synchronized_playback_verified':False},indent=2),encoding='utf8')
print(json.dumps({'count':len(records),'all_frames_decoded':True,'each':'49 frames,480x320,10fps,4.9s','UI_synchronized_playback_verified':False}))
