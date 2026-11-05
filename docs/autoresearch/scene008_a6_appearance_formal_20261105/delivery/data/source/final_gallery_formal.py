"""Actual frozen-object/scene renders, shared crops and all49 frame controls."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign');RUN=CAM/'scene008_a6_appearance_formal_20261105';OLD=CAM/'scene008_a6_asset_controls_20261105'
sys.path.insert(0,str(OLD/'source'))
from sh_asset import SHAsset
from scene_runtime import Runtime,tagged,raster
sys.path.insert(0,str(RUN/'source'))
from highres_fit import camera_hr,RAW,HELD,write

def array(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
def rgb_image(x):
    a=array(x)
    if a.shape[0]==3:a=a.transpose(1,2,0)
    return Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))
def strip(images,names,path,height=280):
    pieces=[]
    for im,name in zip(images,names):
        p=im.resize((round(im.width*height/im.height),height),Image.Resampling.LANCZOS)
        q=Image.new('RGB',(p.width,height+32),(239,245,248));q.paste(p,(0,32));ImageDraw.Draw(q).text((7,9),name,fill=(19,41,55));pieces.append(q)
    out=Image.new('RGB',(sum(p.width for p in pieces),height+32));x=0
    for p in pieces:out.paste(p,(x,0));x+=p.width
    out.save(path,quality=96)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--sphere',required=True);p.add_argument('--surface',required=True);args=p.parse_args()
    dest=RUN/'final_gallery_formal_surface';dest.mkdir(exist_ok=False);rt=Runtime();start=time.time()
    paths={'M2':OLD/'m2/complete_a6.npz','HR observed':RUN/'highres/formal/complete_a6.npz',
           'Thin + appearance':RUN/'generated_fit/strong_hr_hrfit/complete_a6.npz','Sphere + appearance':Path(args.sphere),
           'Surface + appearance /40 BUILD':Path(args.surface)}
    assets={n:SHAsset.from_geometry(path).cuda() for n,path in paths.items()}
    records=[]
    for raw in range(89,138):
        d=rt.raw(raw);cam=camera_hr(d['cam']);images=[];domain=None
        for name,asset in assets.items():
            out=raster(cam,rt.pipe,3,tagged(asset.rows(d['pose'],cam,rt.pipe,16),'asset'),rt.n.bg.new_zeros(3))
            images.append(rgb_image(out['rgb']));mask=array(out['alpha'][0])>.01;domain=mask if domain is None else domain|mask
        y,x=np.where(domain);crop=(max(0,int(x.min())-18),max(0,int(y.min())-18),min(1920,int(x.max())+19),min(1280,int(y.max())+19))
        original=np.asarray(Image.open(RAW/'image/image_0'/f'0008{raw:03d}.png').convert('RGB'))
        visible=np.repeat(np.repeat(array(d['a6_visible']).astype(bool),4,axis=0),4,axis=1)
        observed=np.zeros_like(original);observed[visible]=original[visible]
        strip([Image.fromarray(observed).crop(crop)]+[im.crop(crop) for im in images],['Observed RGB']+list(assets),dest/f'{raw:03d}_objects.jpg')
        best=assets['Surface + appearance /40 BUILD'];keep=rt.render_scene(raw,best);delete=rt.render_scene(raw,best,delete=True)
        assert keep['meta']['native_a6_rows_input']==0 and delete['meta']['native_a6_rows_input']==0
        strip([rgb_image(keep['rgb']),rgb_image(delete['rgb'])],['Same asset / a37 retained','Same asset / delete a37'],dest/f'{raw:03d}_street.jpg',320)
        records.append(dict(raw=raw,split='HELD' if raw in HELD else 'BUILD',crop_xyxy=crop))
        print(json.dumps(dict(stage='final_gallery',raw=raw)),flush=True)
        del d,cam,out,keep,delete;rt._data.clear();torch.cuda.empty_cache()
    assert rt.modelunchanged() and all(a.assert_geometry_unchanged() for a in assets.values())
    write(dest/'receipt.json',dict(status='PASS',frames=records,assets={n:str(p) for n,p in paths.items()},
        display='Same actual1920 camera/crop for all asset columns; original480 native joint street for context',
        no_image_overlay=True,same_asset_keep_delete=True,native_a6_rows_input=0,
        road_native_environment_frozen=True,model_weight_updates=0,seconds=time.time()-start))
if __name__=='__main__':main()
