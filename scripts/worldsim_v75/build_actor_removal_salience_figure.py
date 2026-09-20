"""三固定来源的 reference actor-removal 显著性候选图；只描述，不拟合。"""
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
OUT=ROOT/'WS-V75-ACTOR-REMOVAL-G1-GENERATION-01/20260921-r1'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(n,b=False):return ImageFont.truetype(BOLD if b else FONT,n)


def main():
    points=[
      {'name':'Exposed dev','area':3853.626264333224,'depth':47.96738421440668,'residual':1,'color':'#2b7a78'},
      {'name':'New mid-distance','area':5859.371807993581,'depth':41.879992463345275,'residual':7,'color':'#c98b16'},
      {'name':'Independent near','area':18229.067214930343,'depth':20.26024864135077,'residual':10,'color':'#c23b3b'}]
    im=Image.new('RGB',(1400,760),'white');d=ImageDraw.Draw(im)
    d.text((60,35),'Reference-state editability vs initial actor salience',font=font(38,True),fill='#173246')
    d.text((60,85),'Three fixed sources; input area/depth co-vary, so this is a candidate trend, not a causal fit.',font=font(22),fill='#526a78')
    left,top,right,bottom=140,160,1320,640
    d.line((left,bottom,right,bottom),fill='#5e7582',width=3);d.line((left,top,left,bottom),fill='#5e7582',width=3)
    for yv in range(0,11,2):
        y=bottom-(bottom-top)*yv/10;d.line((left,y,right,y),fill='#dde5e8',width=1);d.text((85,y-12),str(yv),font=font(19),fill='#5e7582')
    for xv in [0,5000,10000,15000,20000]:
        x=left+(right-left)*xv/20000;d.line((x,top,x,bottom),fill='#eef2f3',width=1);d.text((x-28,bottom+14),f'{xv//1000}k',font=font(18),fill='#5e7582')
    ordered=[]
    for row in points:
        x=left+(right-left)*row['area']/20000;y=bottom-(bottom-top)*row['residual']/10;ordered.append((x,y))
    d.line(ordered,fill='#8a9ca5',width=4)
    for row,(x,y) in zip(points,ordered):
        d.ellipse((x-13,y-13,x+13,y+13),fill=row['color'],outline='white',width=3)
        tx=x+18 if x<right-260 else x-360
        d.text((tx,y-42),row['name'],font=font(21,True),fill=row['color'])
        d.text((tx,y-14),f'{row["area"]:,.0f} px² · {row["depth"]:.1f} m · A={row["residual"]}/10',font=font(18),fill='#405a69')
    d.text((520,700),'initial projected actor area (px²)',font=font(22),fill='#405a69')
    d.text((145,125),'removed A detections / 10',font=font(18),fill='#405a69')
    im.save(OUT/'salience-candidate.png')
    (OUT/'salience-candidate.json').write_text(json.dumps({'points':points,'interpretation':'candidate visual-salience editability envelope; no causal or monotonic claim','human_verdict':None,'failure_ledger_delta':'none'},indent=2)+'\n')
    print(json.dumps({'status':'complete','points':points}))


if __name__=='__main__':main()
