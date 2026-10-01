"""只呈现已完成四臂的固定帧；不写评分，不干涉推理。"""
from pathlib import Path
import sys
import numpy as np,cv2
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).parent))
from temporal_factory import T,read
from render_pairs import font
O=T/'r10'

def main():
    cv2.setNumThreads(1);dest=O/'review/effect_contacts';dest.mkdir(parents=True,exist_ok=True)
    completed=[]
    for c in read(O/'evaluation_plan.json')['cases']:
        src=O/'evaluation'/c['eval_id']
        if not all((src/(a+'_metrics.json')).exists() for a in ['base','r7','r8','r10']):continue
        dst=dest/(c['eval_id']+'_f05.jpg')
        if dst.exists() and '--refresh' not in sys.argv:continue
        i=5;load=lambda r:np.asarray(Image.open(src/r/f'{i:05}.png').convert('RGB'))
        h=np.asarray(Image.open(src/'mask'/f'{i:05}.png'))>0
        target=load('GT') if c['kind']=='synthetic' else load('input').copy();inp=load('input').copy()
        if c['kind']!='synthetic':
            core=np.asarray(Image.open(sorted((Path(c['folder'])/'core').glob('*.png'))[c['frames'][i]]))>0
            yy,xx=np.where(core)
            if len(xx):cv2.rectangle(target,(int(xx.min()),int(yy.min())),(int(xx.max()),int(yy.max())),(255,220,30),2)
            tint=np.zeros_like(inp);tint[:,:,2]=255;inp[h]=np.rint(.75*inp[h]+.25*tint[h]).astype('uint8')
        panel={'target':target,'input':inp,**{a:load(a) for a in ['base','r7','r8','r10']}}
        yy,xx=np.where(h);box=np.array([xx.min()-20,yy.min()-20,xx.max()+21,yy.max()+21]);box[[0,2]]=box[[0,2]].clip(0,1024);box[[1,3]]=box[[1,3]].clip(0,576)
        sheet=Image.new('RGB',(2400,900),(15,22,33));d=ImageDraw.Draw(sheet)
        d.text((8,8),f'{c["eval_id"]} | {c.get("receiver_scene",c.get("scene"))} | f5 | {c["suite"]}',font=font(22),fill='white')
        for j,(role,arr) in enumerate(panel.items()):
            d.text((j*400+8,42),role,font=font(),fill='white');im=Image.fromarray(arr);sheet.paste(im.resize((400,225)),(j*400,72));crop=im.crop(tuple(box));scale=min(396/crop.width,550/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS);sheet.paste(crop,(j*400+(400-crop.width)//2,330))
        sheet.save(dst,quality=96);completed.append(c['eval_id'])
    print('new fixed-frame contacts',completed)

if __name__=='__main__':main()
