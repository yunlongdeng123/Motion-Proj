"""Common input ROI sheets for visual review; underlying full-frame renders unchanged."""
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
W=Path(__file__).parent;O=W.parents[1]/'outputs/identity_comparison_20261018'
for actor in [6,37]:
    canvas=Image.new('RGB',(7*300,3*245),(10,19,31));d=ImageDraw.Draw(canvas)
    for j,raw in enumerate([105,113,121]):
        fid=f'0008{raw:03d}';idx=[5,26,18,6,37].index(actor)
        mask=np.asarray(Image.open(W/'metadata/prepared5/instance_masks'/(fid+'.png')))==idx+1
        yy,xx=np.where(mask);roi=(max(0,int(xx.min())-30),max(0,int(yy.min())-35),min(480,int(xx.max())+30),min(320,int(yy.max())+50)) if len(xx) else (0,0,480,320)
        items=[('old5','gt.png','Observed RGB'),('old5',f'a{actor:02d}_isolated.png','Old5 isolated'),('r15',f'a{actor:02d}_isolated.png','Original R15 isolated'),
            ('old5',f'a{actor:02d}_alpha.png','Old5 isolated alpha'),('r15',f'a{actor:02d}_alpha.png','Original R15 alpha'),
            ('old5',f'a{actor:02d}_delete.png','Old5 delete'),('r15',f'a{actor:02d}_delete_no_shadow.png','Original R15 delete, no shadow')]
        for k,(model,file,label) in enumerate(items):
            im=Image.open(O/'assets'/model/fid/file).convert('RGB').crop(roi);im.thumbnail((296,214))
            canvas.paste(im,(k*300+(296-im.width)//2,j*245+25));d.text((k*300+4,j*245+5),f'a{actor} raw{raw} / {label}',fill='white')
    canvas.save(O/f'a{actor:02d}_completed_old5_review.png')
print('review sheets built')
