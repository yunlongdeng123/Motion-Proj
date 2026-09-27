import argparse,json,numpy as np
from PIL import Image
from delete_full_common import ROOT
from delete_full_query import load_rgb,compose_actor,label
p=argparse.ArgumentParser();p.add_argument('scene');p.add_argument('frame',type=int);p.add_argument('camera',type=int);a=p.parse_args()
b=ROOT/a.scene;f=a.frame;c=a.camera;out=b/'background_world'/f'{f:03}'
source=load_rgb(b/f'cam{c}/rgb/{f:05}.png');direct=load_rgb(b/f'cam{c}/background/{f:05}.png');pure=load_rgb(out/f'geometry_cam{c}.png');hybrid=load_rgb(out/f'hybrid_cam{c}.png')
rgba=np.array(Image.open(b/'actor_layers'/f'f{f:03}_cam{c}.png'));az=np.load(b/'actor_layers'/f'f{f:03}_cam{c}_depth.npy');bz=np.load(out/f'z_cam{c}.npy')
factual,stats=compose_actor(hybrid,rgba,az,bz);unoccluded,_=compose_actor(hybrid,rgba,az,np.full_like(bz,np.inf))
im=Image.new('RGB',(1376,1152))
for i,(rgb,title) in enumerate([(source,'SOURCE'),(direct,'DriveEditor 2D'),(pure,'Omega pure B_t'),(hybrid,'Omega hybrid DELETE'),(unoccluded,'GLB no depth test (diagnostic only)'),(factual,'Factual with depth test')]):im.paste(label(rgb,f'{a.scene} f{f} CAM{c} | {title}'),((i%2)*688,(i//2)*384))
path=ROOT/f'peek_{a.scene}_{f}_{c}.jpg';im.save(path,quality=94);print(path,stats)
