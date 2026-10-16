"""Raster QA of the simple authored SVG components diagram on Windows fonts."""
from pathlib import Path
from xml.etree import ElementTree as ET
import re,math
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[2];svg=root/'outputs/shadow_sequence_v2/architecture.svg'
tree=ET.parse(svg).getroot();im=Image.new('RGB',(1120,540),'#f5f8fc');d=ImageDraw.Draw(im)
for e in tree:
    tag=e.tag.rsplit('}',1)[-1]
    if tag=='rect':
        x,y,w,h=[float(e.attrib.get(k,0)) for k in ['x','y','width','height']]
        d.rounded_rectangle((x,y,x+w,y+h),radius=12 if e.attrib.get('class')=='box' else 0,fill=e.attrib.get('fill','#fff'),outline='#90a5b9' if e.attrib.get('class') else None,width=1)
    elif tag=='path':
        tokens=re.findall(r'[MHVL]|-?\d+(?:\.\d+)?',e.attrib['d']);i=0;points=[];point=(0.,0.)
        while i<len(tokens):
            cmd=tokens[i];i+=1
            if cmd in ['M','L']:point=(float(tokens[i]),float(tokens[i+1]));i+=2
            elif cmd=='H':point=(float(tokens[i]),point[1]);i+=1
            elif cmd=='V':point=(point[0],float(tokens[i]));i+=1
            points.append(point)
        d.line(points,fill='#526377',width=2)
        if len(points)>1:
            a,b=points[-2:];dx=b[0]-a[0];dy=b[1]-a[1];n=math.hypot(dx,dy);dx/=n;dy/=n
            d.polygon([b,(b[0]-9*dx+4*dy,b[1]-9*dy-4*dx),(b[0]-9*dx-4*dy,b[1]-9*dy+4*dx)],fill='#526377')
    elif tag=='text':
        style=e.attrib.get('style','');size=23 if '23px' in style else 14 if e.attrib.get('class')=='small' else 17
        font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if 'bold' in style else 'C:/Windows/Fonts/msyh.ttc',size)
        d.text((float(e.attrib['x']),float(e.attrib['y'])),e.text,font=font,fill='#526377' if e.attrib.get('class')=='small' else '#1c2e42',anchor='ls')
im.save(root/'outputs/shadow_sequence_v2/architecture.png')
