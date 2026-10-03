"""固定f5原图、遮洞、三组原生输出的等比例对照，供助手实看。"""
from common import *
import numpy as np
from PIL import Image, ImageDraw


def main():
    for c in read(O/'manifest.json')['cases']:
        if c['split']=='train':continue
        cid=c['case_id'];out=O/'inspection'/f'{cid}_f05.jpg'
        if out.exists():continue
        arms=['adapter_off','all_unknown','conditioned']
        if not all((O/'evaluation'/cid/a/'result.json').exists() for a in arms):continue
        x=images(c,'rgb')[5];hole=images(c,'hole')[5]>0
        orig=images(c,'target')[5] if c['kind']=='synthetic' else x
        masked=x.copy();masked[hole]=127
        pics=[orig,masked]+[np.asarray(Image.open(O/'evaluation'/cid/a/'native/00005.png')) for a in arms]
        yy,xx=np.where(hole);x0=max(0,int(xx.min())-65);x1=min(1024,int(xx.max())+66)
        y0=max(0,int(yy.min())-65);y1=min(576,int(yy.max())+66)
        sheet=Image.new('RGB',(1920,670),(20,27,36));draw=ImageDraw.Draw(sheet)
        for j,(name,pic) in enumerate(zip(['original_Y_or_RGB','masked_input']+arms,pics)):
            draw.text((j*384+6,5),cid+' / '+name,fill='white')
            full=Image.fromarray(pic);full.thumbnail((384,240));sheet.paste(full,(j*384,25))
            crop=Image.fromarray(pic[y0:y1,x0:x1]);crop.thumbnail((384,370));sheet.paste(crop,(j*384+(384-crop.width)//2,285+(370-crop.height)//2))
        out.parent.mkdir(exist_ok=True);sheet.save(out,quality=95)
        print(cid,flush=True)


if __name__=='__main__':main()
