"""One video contains both actual runs, so panels share exactly one timeline."""
import hashlib,json
from fractions import Fraction
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/drivingeditor_vgpu_debug/assets'
cases=['original30k_to30300','opacity_outside30k_to30300']
roots=[HERE/'continuation_diagnostic'/name for name in cases]
for root in roots:
    if not (root/'status.json').exists():
        raise SystemExit('Pair not complete; no placeholder created')
    d=json.loads((root/'status.json').read_text())
    if d['status']!='complete' or d['actual_exit_code']!=0:
        raise SystemExit('Pair not complete; no placeholder created')
evals=[json.loads((root/'extra_eval/30300/diagnostic_eval.json').read_text()) for root in roots]
ids=[[f['frame_id'] for f in e['frames']] for e in evals]
assert ids[0]==ids[1] and len(ids[0])==49
channels=['gt','composite','static','dynamic']
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
width,height=1920,704
dest=OUT/'paired30300_actual49.mp4'
temp=dest.with_suffix('.part')
input_pins={}
with av.open(str(temp),'w',format='mp4',options={'movflags':'+faststart'}) as container:
    stream=container.add_stream('libx264',rate=10)
    stream.width=width;stream.height=height;stream.pix_fmt='yuv420p'
    stream.options={'crf':'18','preset':'medium'}
    for index,frame_id in enumerate(ids[0]):
        canvas=Image.new('RGB',(width,height),(8,14,23))
        draw=ImageDraw.Draw(canvas)
        for row,root in enumerate(roots):
            title='Original objective' if row==0 else 'Outside-opacity hypothesis'
            for col,channel in enumerate(channels):
                path=root/'extra_eval/30300'/channel/(Path(frame_id).name+'.png')
                image=Image.open(path).convert('RGB')
                assert image.size==(480,320)
                canvas.paste(image,(480*col,352*row+32))
                draw.text((480*col+8,352*row+3),f'{title} / {channel}',font=font,fill=(225,235,246))
                input_pins[str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
        frame=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')
        frame.pts=index;frame.time_base=Fraction(1,10)
        for packet in stream.encode(frame):container.mux(packet)
    for packet in stream.encode():container.mux(packet)
temp.replace(dest)
with av.open(str(dest)) as container:
    assert float(container.streams.video[0].average_rate)==10
    pts=[]
    for f in container.decode(video=0):
        assert(f.width,f.height)==(width,height)
        f.to_ndarray(format='rgb24')
        pts.append(float(f.pts*f.time_base))
assert len(pts)==49 and all(abs(t-i/10)<1e-9 for i,t in enumerate(pts))
receipt={'status':'all49_frames_decoded','file':dest.name,'frame_ids':ids[0],'fps':10,'frames':49,'duration_seconds':4.9,'dimensions':[width,height],'pts':pts,'panels_synchronized_by_single_encoded_frame':True,'all_panels_actual_same_camera_and_iteration':30300,'input_png_sha256':input_pins,'mp4_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'metrics_computed_from_raw_images_not_lossy_video':True,'browser_playback_checked':False}
(OUT.parent/'paired_video_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps({k:receipt[k] for k in ['status','file','frames','fps','duration_seconds','dimensions','panels_synchronized_by_single_encoded_frame','browser_playback_checked']}))
