from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;sheet=Image.new('RGB',(1200,700),(25,34,45));d=ImageDraw.Draw(sheet)
for i,(f,c) in enumerate([(0,0),(46,2)]):
    orig=np.array(Image.open(R/'actor12'/f'source_f{f:03}_cam{c}.png').convert('RGB'));ims=[orig];alphas=[]
    for yaw in [0,180]:
        rgba=np.array(Image.open(R/f'factual12_yaw{yaw}/actor_layers'/f'f{f:03}_cam{c}.png').convert('RGBA'));a=rgba[...,3:4]/255;ims.append(np.rint(rgba[...,:3]*a+orig*(1-a)).astype('uint8'));alphas.append(rgba[...,3]>0)
    y,x=np.where(alphas[0]|alphas[1]);box=[max(0,int(x.min())-25),max(0,int(y.min())-25),min(688,int(x.max())+26),min(384,int(y.max())+26)]
    for j,im in enumerate(ims):
        pic=Image.fromarray(im).crop(box);pic.thumbnail((400,310));pic=pic.resize((min(400,pic.width*2),min(310,pic.height*2)));sheet.paste(pic,(j*400,i*350+28));d.text((j*400+5,i*350+6),f'f{f} CAM{c} '+['Original RGB','canonical yaw0','canonical yaw180'][j],fill='white')
sheet.save(R/'actor12_pose.jpg',quality=95)
