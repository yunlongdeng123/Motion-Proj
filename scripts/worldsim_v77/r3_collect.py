"""导出悬浮框/贴地包络的可复核图；不修改源图或GLB。"""
import argparse,json,pathlib,sys
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from r3_visibility import close_box_to_ground
from geometry import transform,resized_intrinsics
from actor_command_audit import box

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=pathlib.Path,default=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-PIPELINE-R3-20260926/r1'));p.add_argument('--data-root',type=pathlib.Path,default=pathlib.Path('/root/autodl-tmp/data/v76_vadgs'));args=p.parse_args()
 edges=[(i,j) for i in range(8) for j in range(i+1,8) if bin(i^j).count('1')==1]
 signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
 for scene,aid,f,c,ref in [('scene_0230','22',45,5,5),('scene_0255','25',165,5,20)]:
  data=args.data_root/scene;inst=json.loads((data/'instances/instances_info.json').read_text());pose,size=box(inst[aid],f);rp,rs=box(inst[aid],ref)
  summary=json.loads((args.run/f'{scene}_visibility.json').read_text());plane=summary['plane']['plane_z_ax_by_c'];closed,cs=close_box_to_ground(pose,size,rp,plane)
  v=np.loadtxt(data/'intrinsics'/f'{c}.txt');K=resized_intrinsics([[v[0],0,v[2]],[0,v[1],v[3]],[0,0,1]],(1600,900),(384,688));c2w=np.loadtxt(data/'extrinsics'/f'{f:03}_{c}.txt')
  def project(q):
   cp=transform(q,np.linalg.inv(c2w));a=cp@K.T;return a[:,:2]/a[:,2:]
  uvraw=project(transform(signs*size/2,pose));uvclosed=project(transform(signs*cs/2,closed));z=np.load(args.run/f'{scene}_visibility.npz');q=project(z['query_world'][z['core']])
  x0,y0=np.minimum(uvraw.min(0),uvclosed.min(0));x1,y1=np.maximum(uvraw.max(0),uvclosed.max(0));margin=20;rect=(max(0,int(x0-margin)),max(0,int(y0-margin)),min(688,int(x1+margin)),min(384,int(y1+margin)))
  source=Image.open(data/'images'/f'{f:03}_{c}.jpg').convert('RGB').resize((688,384),Image.Resampling.BICUBIC)
  panels=[]
  for title,uv,color in [('SOURCE / same actor',None,None),('RAW GT BOX / floor gap',uvraw,'#ed9e41'),('GROUND-CLOSED / conservative',uvclosed,'#20d5d0')]:
   image=source.copy();draw=ImageDraw.Draw(image)
   if uv is not None:
    for a,b in edges:draw.line([tuple(uv[a]),tuple(uv[b])],fill=color,width=1)
    for x,y in q:draw.ellipse((x-.6,y-.6,x+.6,y+.6),fill='#eb60aa')
   crop=image.crop(rect);crop.thumbnail((688,340),Image.Resampling.LANCZOS)
   # Enlarge modest source crop for inspection without claiming added detail.
   ratio=min(688/crop.width,340/crop.height);crop=crop.resize((int(crop.width*ratio),int(crop.height*ratio)),Image.Resampling.NEAREST)
   panel=Image.new('RGB',(688,410),'#102b3a');panel.paste(crop,((688-crop.width)//2,45+(340-crop.height)//2));ImageDraw.Draw(panel).text((16,15),title,fill='white');panels.append(panel)
  result=Image.new('RGB',(2064,410));[result.paste(im,(i*688,0)) for i,im in enumerate(panels)];result.save(args.run/f'{scene}_box_control.jpg',quality=94)
  d=ImageDraw.Draw(source);d.rectangle(rect,outline='#20d5d0',width=2);d.rectangle((0,0,420,27),fill='#102b3a');d.text((8,7),f'{scene} actor {aid} / f{f:03} CAM{c} / box-control context',fill='white');source.save(args.run/f'{scene}_box_context.jpg',quality=94)

if __name__=='__main__':main()
