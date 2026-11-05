"""Compare historical and current training configurations on aligned pilot views."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,sys,time
import numpy as np
import torch
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent.parent
C=ROOT.parent
ASSET=C/'scene008_a6_asset_controls_20261105'
sys.path.insert(0,str(ASSET/'source'))
from scene_runtime import Runtime,raster,prepare_rasterization
from sh_asset import SHAsset
import editor_common

BOXES={'left_road':(0,205,215,320),'right_curb':(315,177,480,290),
       'building':(365,85,480,185),'trees':(15,65,160,180)}
def cpu(x):return x.detach().cpu().numpy()
def blue(x):return x[2]-.5*(x[0]+x[1])
def stats(x,ref,m):
    d=(x-ref)[:,m].astype(np.float64)
    return dict(pixels=int(m.sum()),rgb_mae=float(np.abs(d).mean()),blue_error=float(blue(d).mean()))
def image(x):return Image.fromarray(np.uint8(np.clip(x.transpose(1,2,0),0,1)*255+.5))

@torch.no_grad()
def main():
    start=time.monotonic();out=ROOT/'audit';out.mkdir(exist_ok=False)
    rt=Runtime();asset=SHAsset.from_geometry(ASSET/'m2/complete_a6.npz').cuda()
    current=editor_common.B
    editor_common.B=C/'scene008_a37_only_B10k_20261103_r1/formal10k'
    old=editor_common.Native();editor_common.B=current
    config_diff={k:dict(old=old.config.get(k),new=rt.n.config.get(k)) for k in set(old.config)|set(rt.n.config)
                 if old.config.get(k)!=rt.n.config.get(k)}
    rows=[];yy,xx=np.indices((320,480))
    for raw in (105,113,130):
        d=rt.raw(raw);cam=d['cam'];ocam=old.camera(raw)
        assert torch.allclose(cam.world_view_transform,ocam.world_view_transform,atol=1e-6,rtol=0)
        assert abs(cam.timestamp-ocam.timestamp)<1e-6
        ref=cpu(d['rgb']);actors=cpu(d['actor_labels'])>0
        packages={'old_native':old.render(cam,dynamic=True),'current_native':rt.n.render(cam,dynamic=True),
                  'm2_keep':rt.render_scene(raw,asset),'m2_delete':rt.render_scene(raw,asset,delete=True)}
        kw=prepare_rasterization(cam,rt.n.static,rt.pipe);kw['means2D']=torch.zeros_like(kw['means3D'])
        p=raster(cam,rt.pipe,rt.n.static.active_sh_degree,kw,rt.n.bg)
        p['rgb']=p['rgb']+(1-p['alpha'])*d['env'];packages['current_static_only']=p
        regions={name:(xx>=a)&(xx<c)&(yy>=b)&(yy<e)&~actors for name,(a,b,c,e) in BOXES.items()}
        regions['road']=cpu(d['road']).astype(bool)&~actors
        row=dict(raw=raw,metrics={name:{roi:stats(cpu(p['rgb']),ref,m) for roi,m in regions.items()} for name,p in packages.items()})
        rows.append(row)
        tiles=[('Raw RGB',ref)]+[(name,cpu(p['rgb'])) for name,p in packages.items()]
        canvas=Image.new('RGB',(480*3,348*2),'white')
        for i,(name,value) in enumerate(tiles):
            x=(i%3)*480;y=(i//3)*348;canvas.paste(image(value),(x,y+28));ImageDraw.Draw(canvas).text((x+8,y+7),name,fill='black')
        canvas.save(out/f'{raw:03d}_comparison.jpg',quality=96,subsampling=0)
        rt._data.clear()
    assert rt.modelunchanged() and old.unchanged() and asset.assert_geometry_unchanged()
    result=dict(status='PASS_ALIGNED_HISTORY_AUDIT',rows=rows,config_diff=config_diff,seconds=time.monotonic()-start,
                current_native=str(current),historical_native=str(C/'scene008_a37_only_B10k_20261103_r1/formal10k'),
                m2_asset=str(ASSET/'m2/complete_a6.npz'),training_updates=0)
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
