"""生成对象移除实验的组件图和一眼可读主图；不改变或筛选结果。"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
DEV = ROOT/'WS-V75-ACTOR-REMOVAL-GENERATION-01/20260921-r1'
CONF = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-GENERATION-01/20260921-r1'
QUAL = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01/20260921-r3'
ONSET = ROOT/'WS-V75-ACTOR-REMOVAL-ONSET-01/20260921-r1'
OUT = ONSET/'figures'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def arrow(draw, a, b, color='#46657a', width=6):
    draw.line([a,b], fill=color, width=width)
    x,y=b; draw.polygon([(x,y),(x-16,y-10),(x-16,y+10)],fill=color)


def architecture():
    im=Image.new('RGB',(2300,650),'white');d=ImageDraw.Draw(im)
    d.text((70,38),'Counterfactual WorldSim: reconstruction errors are testable only after reference editability passes',font=f(35,True),fill='#172a3a')
    boxes=[(70,'Observed prefix\nRGB + cameras'),(415,'Reconstructed state\nactors + map'),(780,'Semantic edit\nremove actor A'),
           (1125,'Condition encoder\nstate raster'),(1470,'Generative world model\ninitial image + history'),(1880,'Sensor observations\nvideo / detections')]
    for x,label in boxes:
        d.rounded_rectangle((x,155,x+285,285),radius=18,fill='#edf4f7',outline='#5f7f91',width=4)
        lines=label.split('\n')
        for i,line in enumerate(lines):
            size=20 if line.startswith('Generative') else 22
            w=d.textbbox((0,0),line,font=f(size,i==0))[2]
            d.text((x+(285-w)/2,185+i*40),line,font=f(size,i==0),fill='#183447')
    for (x,_),(nx,__) in zip(boxes[:-1],boxes[1:]): arrow(d,(x+285,220),(nx-18,220))
    gates=[(600,'G0  Raster changes'),(1060,'G1  Reference state obeys edit'),(1530,'G2  Reconstruction arms differ'),(1940,'G3  Decision/outcome fidelity')]
    colors=['#2b7a78','#c23b3b','#a66a00','#5c4aa5']
    for (x,label),color in zip(gates,colors):
        d.rounded_rectangle((x,390,x+310,485),radius=15,fill='#fffaf0',outline=color,width=4)
        d.text((x+16,420),label,font=f(17 if label.startswith(('G2','G3')) else 19,True),fill=color)
    d.text((600,525),'Stop reconstruction ranking when G1 fails: generator/interface error dominates the experiment.',font=f(28,True),fill='#c23b3b')
    d.text((600,575),'Only G0 → G1 → G2 → repair/confirmation may justify a reconstruction claim.',font=f(24),fill='#344e5c')
    im.save(OUT/'architecture-components.png')


def panel(image, title, subtitle='', boxes=()):
    image=Image.fromarray(np.asarray(image)).convert('RGB') if not isinstance(image,Image.Image) else image.convert('RGB')
    image=image.resize((480,264),Image.Resampling.LANCZOS);d=ImageDraw.Draw(image)
    for box,color,label in boxes:
        b=np.asarray(box,dtype=float)*np.array([480/1280,264/704,480/1280,264/704])
        d.rectangle(b.tolist(),outline=color,width=3);d.text((b[0]+3,max(3,b[1]-20)),label,font=f(17,True),fill=color,stroke_width=2,stroke_fill='black')
    frame=Image.new('RGB',(500,350),'white');frame.paste(image,(10,62));dd=ImageDraw.Draw(frame)
    dd.text((10,10),title,font=f(24,True),fill='#152c3b')
    dd.text((10,38),subtitle,font=f(17),fill='#4e6573')
    return frame


def main_figure():
    q=json.loads((QUAL/'result.json').read_text());e=json.loads((CONF/'evaluation.json').read_text());o=json.loads((ONSET/'evaluation.json').read_text())
    base=Path(q['selected']['base']); initial=Image.open(base/'initial_rgb.png')
    row0=next(x for x in e['rows'] if x['arm']=='reference' and x['variant']=='removed' and x['frame']==5)
    boxes=[]
    for key,color,label in [('projected_actor','#f04d43','A'),('projected_behind','#00b9a7','B')]:
        item=row0['assignment'][key]
        if item: boxes.append((item['bounds'],color,label))
    uncond=np.load(CONF/'reference-unedited/conditions.npy',mmap_mode='r')
    onsetcond=np.load(ONSET/'conditions.npy',mmap_mode='r')
    ungen=np.load(CONF/'reference-unedited/generated.npy',mmap_mode='r')
    f5gen=np.load(CONF/'reference-removed/generated.npy',mmap_mode='r')
    f0gen=np.load(ONSET/'generated.npy',mmap_mode='r')
    dev_eval=json.loads((DEV/'evaluation.json').read_text());dev_q=Path(json.loads((DEV/'protocol.json').read_text())['qualification'])
    dev_info=json.loads((dev_q/'result.json').read_text())
    dev_base=Path(dev_info['selected']['base']);dev_initial=Image.open(dev_base/'initial_rgb.png')
    dev_ref=np.load(DEV/'reference-removed/generated.npy',mmap_mode='r');dev_cls=np.load(DEV/'class_prior-removed/generated.npy',mmap_mode='r')
    panels=[
        panel(initial,'(a) Independent source: initial RGB','A is visible; B is partly hidden',boxes),
        panel(uncond[4],'(b) Unedited state raster, f=4','A remains in structured state'),
        panel(onsetcond[4],'(c) Remove A from state at f=0','18.2k raster pixels changed'),
        panel(ungen[4],'(d) Generated: unedited, f=4','A visible'),
        panel(f0gen[4],'(e) Generated: removed from f=0','A still visible; 10/10 samples',boxes),
        panel(f5gen[29],'(f) Generated: removed from f=5','A still visible; reference gate fails'),
        panel(dev_initial,'(g) Development source','Same fixed edit; exposed source'),
        panel(dev_ref[29],'(h) Dev / reference state','A removed; B revealed (pass)'),
        panel(dev_cls[29],'(i) Dev / ordinary class prior','A persists; signal did not replicate'),
    ]
    canvas=Image.new('RGB',(1540,1190),'#eef3f5');d=ImageDraw.Draw(canvas)
    d.text((30,18),'Actor removal exposes two different bottlenecks',font=f(38,True),fill='#132d3d')
    d.text((30,68),'Independent source: initial-image anchor dominates all reconstruction states.  Development source: editability is state-sensitive.',font=f(23),fill='#405d6d')
    for i,p in enumerate(panels): canvas.paste(p,(20+(i%3)*510,115+(i//3)*350))
    d.rounded_rectangle((35,1160-50,1505,1160),radius=12,fill='#fff3e9',outline='#c23b3b',width=3)
    d.text((60,1121),'Conclusion: reference editability must pass before reconstruction errors can be ranked.',font=f(25,True),fill='#a12f2f')
    canvas.save(OUT/'actor-removal-main-figure.png')


def response_plot():
    # 固定时刻的条件响应比；仅作描述，不作为事前门槛。
    s=np.load(CONF/'reference-unedited/conditions.npy',mmap_mode='r');c=np.load(ONSET/'conditions.npy',mmap_mode='r')
    g=np.load(CONF/'reference-unedited/generated.npy',mmap_mode='r');h=np.load(ONSET/'generated.npy',mmap_mode='r')
    frames=[0,4,5,13,29,61,116];ratios=[]
    for frame in frames:
        mask=np.any(s[frame]!=c[frame],axis=-1)
        cd=np.abs(s[frame].astype(np.float32)-c[frame].astype(np.float32)).mean(-1)[mask].mean()
        gd=np.abs(g[frame].astype(np.float32)-h[frame].astype(np.float32)).mean(-1)[mask].mean()
        ratios.append(float(gd/cd))
    im=Image.new('RGB',(1280,600),'white');d=ImageDraw.Draw(im)
    d.text((55,30),'Generated response inside the edited raster region',font=f(34,True),fill='#183447')
    d.text((55,78),'MAE(generated pair) / MAE(condition pair); descriptive post-hoc trace',font=f(22),fill='#526b78')
    left,top,right,bottom=100,140,1210,500
    d.line((left,bottom,right,bottom),fill='#607784',width=3);d.line((left,top,left,bottom),fill='#607784',width=3)
    for k in range(6):
        y=bottom-(bottom-top)*k/5;d.line((left,y,right,y),fill='#dce5e9',width=1);d.text((45,y-10),f'{k/10:.1f}',font=f(18),fill='#607784')
    pts=[]
    for i,(frame,value) in enumerate(zip(frames,ratios)):
        x=left+(right-left)*i/(len(frames)-1);y=bottom-(bottom-top)*value/.5;pts.append((x,y))
        d.ellipse((x-8,y-8,x+8,y+8),fill='#c23b3b');d.text((x-15,bottom+15),str(frame),font=f(18),fill='#405967');d.text((x-22,y-34),f'{value*100:.1f}%',font=f(17,True),fill='#a12f2f')
    d.line(pts,fill='#c23b3b',width=5);d.text((520,550),'frame',font=f(20),fill='#405967')
    im.save(OUT/'edit-response-ratio.png')
    return dict(zip(map(str,frames),ratios))


def main():
    OUT.mkdir(exist_ok=True)
    architecture();main_figure();ratios=response_plot()
    (OUT/'figure-data.json').write_text(json.dumps({'onset_generated_to_condition_mae_ratio':ratios,'human_verdict':None},indent=2)+'\n')
    print(json.dumps({'status':'complete','figures':[p.name for p in OUT.iterdir()],'ratios':ratios}))


if __name__=='__main__':
    main()
