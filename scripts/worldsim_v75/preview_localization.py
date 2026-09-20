"""同一参考裁剪展示五条件，不按生成目标重新居中；保留真实时间。"""
import argparse
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from select_target import P1

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    a=p.parse_args()
    variants=[('clean','Clean'),('negative_persistent','-0.5m persistent'),('negative_restore','-0.5m then restore'),
              ('positive_persistent','+0.5m persistent'),('positive_restore','+0.5m then restore')]
    videos={}
    for case,_ in variants:
        assert json.loads((P1/'rollouts'/f'{case}-seed{a.seed}.json').read_text())['status']=='complete'
        videos[case]=np.load(P1/'rollouts'/f'{case}-seed{a.seed}.npy',mmap_mode='r')
    projection=json.loads((P1/'target_projections.json').read_text())['clean']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    rows=[]
    with av.open(str(P1/f'comparison-seed{a.seed}.mp4'),mode='w') as writer:
        stream=writer.add_stream('libx264',rate=30)
        stream.width,stream.height,stream.pix_fmt=1280,250,'yuv420p'
        stream.options={'crf':'16'}
        for f in range(54):
            b=np.asarray(projection[f]['bounds'])
            cx,cy=(b[:2]+b[2:])/2
            width=max(80,b[2]-b[0]+60)
            height=max(60,b[3]-b[1]+50,width*0.75)
            width=max(width,height/0.75)
            crop=tuple(map(int,[cx-width/2,cy-height/2,cx+width/2,cy+height/2]))
            canvas=Image.new('RGB',(1280,250),'#eef2f6')
            d=ImageDraw.Draw(canvas)
            for i,(case,label) in enumerate(variants):
                d.text((i*256+8,5),label,fill='#153347',font=font)
                im=Image.fromarray(videos[case][f]).crop(crop).resize((256,192))
                canvas.paste(im,(i*256,30))
            phase='clean prefix' if f<5 else ('biased input' if f<37 else 'restored input in restore columns')
            d.text((10,226),f'seed {a.seed} | t={f/30:.2f}s | {phase} | shared reference crop and zoom',fill='#153347',font=small)
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                writer.mux(packet)
            if f in [21,36,45,53]:
                rows.append(canvas)
        for packet in stream.encode():
            writer.mux(packet)
    sheet=Image.new('RGB',(1280,250*len(rows)),'white')
    for i,row in enumerate(rows):
        sheet.paste(row,(0,i*250))
    sheet.save(P1/f'comparison-seed{a.seed}.jpg',quality=96)
    print(f'seed{a.seed} paired preview saved',flush=True)

if __name__=='__main__':
    main()
