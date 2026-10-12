"""CPU decode and idempotence check of a generated comparison report."""
import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import imageio_ffmpeg
import build_checkpoint_comparison as builder

p=argparse.ArgumentParser();p.add_argument('report');a=p.parse_args()
root=Path(a.report);data=json.loads((root/'comparison_manifest.json').read_text())
result={'report':str(root),'videos':[]}
before={str(path):[path.stat().st_mtime_ns,path.stat().st_size] for path in (root/'videos').glob('*.mp4')}
records=[data['common'][key] for key in ('gt','original') if key in data['common']]
records += [video for node in data['nodes'] for video in node.get('videos',{}).values()]
for record in records:
    reader=imageio_ffmpeg.read_frames(str(root/record['path']),pix_fmt='rgb24')
    metadata=next(reader);count=sum(1 for _ in reader)
    assert count==record['frames'],(record,count)
    assert abs(metadata['fps']-record['fps'])<1e-8
    assert list(metadata['size'])==[record['width'],record['height']]
    assert metadata['codec']=='h264'
    result['videos'].append({'path':record['path'],'decoded_frames':count,'fps':metadata['fps'],'dimensions':metadata['size'],'codec':metadata['codec']})
assert data['nodes'] and all(node['status']=='ready' for node in data['nodes'])
for node in data['nodes']:
    event=json.loads(Path(node['event_file']).read_text())
    frames=[row for row in event['frames'] if row['camera']==data['camera']]
    for channel,record in node['videos'].items():
        reader=imageio_ffmpeg.read_frames(str(root/record['path']),pix_fmt='rgb24');metadata=next(reader)
        maes=[]
        for index,decoded in enumerate(reader):
            if index in (0,len(frames)//2,len(frames)-1):
                pixels=np.frombuffer(decoded,dtype=np.uint8).reshape(metadata['size'][1],metadata['size'][0],3).astype(float)
                source=np.asarray(Image.open(frames[index]['channels'][channel]).convert('RGB'),dtype=float)
                mae=float(np.abs(pixels[:source.shape[0],:source.shape[1]]-source).mean()/255)
                assert mae<.03,(channel,index,mae)
                maes.append({'frame':index,'mae_rgb01':mae})
        result.setdefault('source_alignment_checks',[]).append({'iteration':node['iteration'],'channel':channel,'samples':maes})
args=SimpleNamespace(run=data['run'],output=str(root),steps=data['steps'],fps=data['fps'],camera=data['camera'],title=data['title'],history_url=data['history_url'])
with contextlib.redirect_stdout(io.StringIO()): new=builder.build_once(args)
after={str(path):[path.stat().st_mtime_ns,path.stat().st_size] for path in (root/'videos').glob('*.mp4')}
assert before==after,'Second run unexpectedly re-encoded videos'
assert data['revision']==new['revision'],'Unchanged run unexpectedly changed manifest'
result['idempotence']='passed: all video sizes/mtimes and manifest revision unchanged'
result['status']='passed'
(root/'cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
