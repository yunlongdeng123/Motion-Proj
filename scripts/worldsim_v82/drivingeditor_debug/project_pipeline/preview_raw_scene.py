"""Create an annotated FRONT contact sheet, video, and timestamp map from one raw scene."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess

from PIL import Image, ImageDraw, ImageFont
from waymo_open_dataset import dataset_pb2

parser=argparse.ArgumentParser()
parser.add_argument('--tfrecord',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--file-index',type=int,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
(args.output/'front_samples').mkdir()
rows=[]; thumbs=[]; first_timestamp=None
ffmpeg=shutil.which('ffmpeg')
bundled_ffmpeg=Path('/root/autodl-tmp/envs/motionproj/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
if ffmpeg is None and bundled_ffmpeg.is_file(): ffmpeg=str(bundled_ffmpeg)
process=None
if ffmpeg:
    process=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','480x320','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','24','-pix_fmt','yuv420p',
        '-threads','2','-movflags','+faststart',str(args.output/'front_fullscene_10fps.mp4')],stdin=subprocess.PIPE)
font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
with args.tfrecord.open('rb') as stream:
    index=0
    while True:
        header=stream.read(12)
        if not header: break
        length=struct.unpack('<Q',header[:8])[0]
        frame=dataset_pb2.Frame.FromString(stream.read(length));stream.read(4)
        front=next(im for im in frame.images if im.name==1)
        if first_timestamp is None: first_timestamp=frame.timestamp_micros
        seconds=(frame.timestamp_micros-first_timestamp)/1e6
        basename=f'0{args.file_index:03d}{index:03d}'
        row={'frame_index':index,'filename':basename+'.png','frame_timestamp_micros':frame.timestamp_micros,
             'elapsed_seconds':seconds,'front_pose_timestamp':front.pose_timestamp,
             'front_camera_trigger_time':front.camera_trigger_time,'front_camera_readout_done_time':front.camera_readout_done_time}
        rows.append(row)
        im=Image.open(io.BytesIO(front.image)).convert('RGB')
        if process:
            small=im.resize((480,320),Image.Resampling.LANCZOS)
            d=ImageDraw.Draw(small);d.rectangle((0,0,310,24),fill=(0,0,0));d.text((6,2),f'{basename} | +{seconds:.2f}s',font=font,fill=(255,255,255))
            process.stdin.write(small.tobytes())
        if index%10==0:
            thumbs.append((index,seconds,im.resize((320,213),Image.Resampling.LANCZOS)))
            (args.output/'front_samples'/(basename+'.jpg')).write_bytes(front.image)
        final_thumb=(index,seconds,im.resize((320,213),Image.Resampling.LANCZOS))
        index+=1
if thumbs[-1][0]!=final_thumb[0]:
    thumbs.append(final_thumb)
    (args.output/'front_samples'/(basename+'.jpg')).write_bytes(front.image)
if process:
    process.stdin.close()
    if process.wait()!=0: raise RuntimeError('ffmpeg preview failed')
with (args.output/'frame_camera_timestamps.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
cols=4;tile_w=332;tile_h=243
sheet=Image.new('RGB',(cols*tile_w,((len(thumbs)+cols-1)//cols)*tile_h),(235,238,242))
draw=ImageDraw.Draw(sheet)
for n,(index,seconds,im) in enumerate(thumbs):
    x=(n%cols)*tile_w+6;y=(n//cols)*tile_h+6
    sheet.paste(im,(x,y))
    draw.text((x,y+215),f'frame {index:03d} | +{seconds:.2f}s',font=font,fill=(10,10,10))
sheet.save(args.output/'front_fullscene_contact.jpg',quality=90)
meta={'raw_path':str(args.tfrecord),'context_name':frame.context.name,'frame_count':len(rows),'first_timestamp_micros':first_timestamp,
      'last_timestamp_micros':rows[-1]['frame_timestamp_micros'],'sample_every_frames':10,'preview_fps':10,'camera':'FRONT',
      'note':'Video is a diagnostic resized FRONT preview, not a DrivingEditor output. Contact tiles sample every10 frames plus final.'}
(args.output/'preview_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
